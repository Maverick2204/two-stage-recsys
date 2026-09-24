"""Offline A/B *replay* with CUPED. This is NOT a measured production lift.

Test-window users are split 50/50 by a salted hash of their id. Each user contributes only their own arm's
metric, computed offline against what they actually interacted with in the test window (logged data, so the
recommendations never influenced behaviour). Covariates are pre_* columns, measured strictly before the test window.
Because both arms can be scored offline for every user, the paired difference is reported as a sanity check.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from recsys.metrics import cuped, diff_ci


def assign(ids, salt="ab-2026"):
    return np.array([int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest(), 16) % 2 == 1 for u in ids])


def ci_dict(d, lo, hi, base):
    return {"diff": d, "ci95": [lo, hi], "rel_lift": d / base, "rel_ci95": [lo / base, hi / base]}


def main(a):
    df = pd.read_csv(a.per_user)
    ctrl, trt = df[f"{a.control}_{a.metric}"].to_numpy(float), df[f"{a.treatment}_{a.metric}"].to_numpy(float)
    X = df[[c for c in df if c.startswith("pre_")]].to_numpy(float)
    treat = assign(df.user_id)
    y = np.where(treat, trt, ctrl)
    adj = cuped(y, X)
    base = y[~treat].mean()
    raw, cup = diff_ci(y, treat), diff_ci(adj, treat)
    pd_ = trt - ctrl
    paired = pd_.mean(), 1.96 * pd_.std(ddof=1) / np.sqrt(len(pd_))

    # A/A: control vs control under 1000 random splits -> how often does CUPED "find" an effect? (should be ~5%)
    rng, fp = np.random.default_rng(0), 0
    for _ in range(1000):
        t = rng.random(len(ctrl)) < 0.5
        _, lo, hi, _ = diff_ci(cuped(ctrl, X), t)
        fp += lo > 0 or hi < 0

    out = {"label": "OFFLINE REPLAY on logged test-window data, not production lift",
           "metric": a.metric, "control": a.control, "treatment": a.treatment,
           "n_control": int((~treat).sum()), "n_treatment": int(treat.sum()),
           "control_mean": base, "covariates": [c for c in df if c.startswith("pre_")],
           "raw": ci_dict(*raw[:3], base), "cuped": ci_dict(*cup[:3], base),
           "variance_reduction": 1 - adj.var() / y.var(), "se_raw": raw[3], "se_cuped": cup[3],
           "paired_all_users": ci_dict(paired[0], paired[0] - paired[1], paired[0] + paired[1], ctrl.mean()),
           "aa_false_positive_rate": fp / 1000}
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    f = lambda r: f"{r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  ({r['rel_lift']:+.1%} [{r['rel_ci95'][0]:+.1%}, {r['rel_ci95'][1]:+.1%}])"
    print(f"{out['label']}\n{a.metric}: {a.treatment} vs {a.control}, n={out['n_control']}/{out['n_treatment']}")
    print(f"  raw    {f(out['raw'])}\n  CUPED  {f(out['cuped'])}   variance -{out['variance_reduction']:.1%}")
    print(f"  paired {f(out['paired_all_users'])}   A/A false-positive rate {out['aa_false_positive_rate']:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-user", default="reports/per_user.csv")
    p.add_argument("--control", default="popular")
    p.add_argument("--treatment", default="two_stage")
    p.add_argument("--metric", default="ndcg10")
    p.add_argument("--out", default="reports/ab.json")
    main(p.parse_args())
