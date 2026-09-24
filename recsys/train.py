"""End-to-end offline pipeline on H&M: weekly time split -> two-tower -> candidates -> LightGBM ranker -> eval.

Protocol (no leakage), time in days, weeks as in the Kaggle competition:
  train: [T1 - WEEKS*7, T1)   val: [T1, T2)   test: [T2, T2 + 7)
  * two-tower trained on train; ranker trained on val-week purchases with candidates/features as of T1
  * two-tower refit on train+val (same hyperparameters), features as of T2, evaluated on the test week
The T2 model is what gets served.
"""
import json
import os
import subprocess
import time
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from recsys.core import FEATURES, ITEM_CATS, N_CAND, R, USER_CATS, Model, item_vec, user_key
from recsys.metrics import mean_ci, ndcg_at_k, recall_at_k

DATA, ART, REP = (os.environ.get(k, d) for k, d in [("DATA", "data/hm"), ("ART", "artifacts"), ("REP", "reports")])
WEEKS = int(os.environ.get("WEEKS", 26))         # history window used for training and features
EPOCHS = int(os.environ.get("EPOCHS", 3))
RANKER_USERS = int(os.environ.get("RANKER_USERS", 60000))
DIM, TAU, BATCH, ID_DROPOUT = 64, 0.05, 2048, 0.1
AGE_BINS = [0, 20, 25, 30, 40, 50, 60, 200]
EPOCH0 = pd.Timestamp("2018-01-01")


def download():
    os.makedirs(DATA, exist_ok=True)
    for f in ("transactions_train.csv", "customers.csv", "articles.csv"):
        if not os.path.exists(f"{DATA}/{f}"):
            subprocess.run(["kaggle", "competitions", "download", "-c", "h-and-m-personalized-fashion-recommendations",
                            "-f", f, "-p", DATA], check=True)
            zipfile.ZipFile(f"{DATA}/{f}.zip").extractall(DATA)
            os.remove(f"{DATA}/{f}.zip")


def load():
    if not os.path.exists(f"{DATA}/transactions.parquet"):
        download()
        tx = pd.read_csv(f"{DATA}/transactions_train.csv", dtype={"article_id": np.int32, "price": np.float32,
                                                                   "sales_channel_id": np.int8})
        tx["day"] = (pd.to_datetime(tx.t_dat) - EPOCH0).dt.days.astype(np.int16)
        tx.drop(columns="t_dat").to_parquet(f"{DATA}/transactions.parquet")
    tx = pd.read_parquet(f"{DATA}/transactions.parquet")
    cu = pd.read_csv(f"{DATA}/customers.csv")
    ar = pd.read_csv(f"{DATA}/articles.csv")

    cu["key"] = np.array([user_key(c) for c in cu.customer_id], dtype=np.uint64)
    assert cu.key.is_unique
    cu = cu.sort_values("key").reset_index(drop=True)
    cu["age_b"] = np.where(cu.age.isna(), 0, np.digitize(cu.age.fillna(0), AGE_BINS[1:-1]) + 1)
    cu["club"] = pd.factorize(cu.club_member_status.fillna("NA"), sort=True)[0]
    cu["news"] = pd.factorize(cu.fashion_news_frequency.fillna("NONE").str.upper(), sort=True)[0]
    cu["fn"], cu["active"] = cu.FN.notna().astype(int), cu.Active.notna().astype(int)

    ar = ar.sort_values("article_id").reset_index(drop=True)
    for c in ITEM_CATS:
        ar[c] = pd.factorize(ar[c], sort=True)[0]
    ar["title"] = ar.prod_name.str.strip() + " · " + ar.colour_group_name.str.strip()

    ids = tx.customer_id.unique()
    pos = np.searchsorted(cu.key.to_numpy(np.uint64), np.array([user_key(c) for c in ids], dtype=np.uint64))
    tx["u"] = tx.customer_id.map(pd.Series(pos, index=ids)).astype(np.int32)
    tx["i"] = np.searchsorted(ar.article_id.to_numpy(), tx.article_id.to_numpy()).astype(np.int32)
    return tx.drop(columns=["customer_id", "article_id"]), cu, ar


def last_rows(h, n_users):
    """Per user, their last R purchases in h (most recent first), -1 padded: (items, days)."""
    h = h.sort_values(["u", "day"], ascending=[True, False], kind="stable")
    r = h.groupby("u").cumcount().to_numpy()
    h, r = h[r < R], r[r < R]
    items, days = np.full((n_users, R), -1, np.int32), np.full((n_users, R), -1, np.float32)
    items[h.u.to_numpy(), r], days[h.u.to_numpy(), r] = h.i.to_numpy(), h.day.to_numpy()
    return items, days


def prev_items(hu, hd, hi):
    """Row j -> up to R of the user's purchases from strictly earlier days (-1 = none). Rows sorted by (u, day)."""
    out = np.full((len(hu), R), -1, np.int32)
    for lag in range(1, R + 1):
        ok = (hu[lag:] == hu[:-lag]) & (hd[lag:] > hd[:-lag])  # same user, earlier day (same basket excluded)
        out[lag:, lag - 1] = np.where(ok, hi[:-lag], -1)
    return out


def fit_two_tower(hist, cu, ar, epochs=EPOCHS, seed=0):
    """In-batch sampled softmax with logQ correction. Item-id dropout trains the 'unknown' slot for new articles."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ni = len(ar)
    h = hist.drop_duplicates(["u", "i", "day"]).sort_values(["u", "day"], kind="stable")
    hu, hd, hi = h.u.to_numpy(), h.day.to_numpy(), h.i.to_numpy()
    prev = torch.as_tensor(prev_items(hu, hd, hi)).long()
    tu, ti = torch.as_tensor(hu).long(), torch.as_tensor(hi).long()
    sold = np.bincount(hi, minlength=ni)
    iid = torch.as_tensor(np.where(sold > 0, np.arange(ni) + 1, 0))
    ucat = torch.as_tensor(cu[USER_CATS].to_numpy()).long()
    icat = torch.as_tensor(ar[ITEM_CATS].to_numpy()).long()
    emb = lambda n: torch.nn.Parameter(torch.randn(n, DIM) * 0.05)
    P = {"u_hist": emb(ni), "i_id": emb(ni + 1),
         **{f"u_{c}": emb(int(cu[c].max()) + 1) for c in USER_CATS},
         **{f"i_{c}": emb(int(ar[c].max()) + 1) for c in ITEM_CATS}}
    opt = torch.optim.Adam(P.values(), lr=3e-3)
    logq = torch.as_tensor(np.log(np.maximum(sold, 1) / sold.sum())).float()
    for ep in range(epochs):
        perm, tot, t0 = torch.as_tensor(rng.permutation(len(hu))), 0.0, time.time()
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            u, i, pv = tu[b], ti[b], prev[b]
            m = (pv >= 0).float()[..., None]
            U = (P["u_hist"][pv.clamp(min=0)] * m).sum(1) / m.sum(1).clamp(min=1)
            for j, c in enumerate(USER_CATS):
                U = U + P[f"u_{c}"][ucat[u, j]]
            I = P["i_id"][iid[i] * (torch.rand(len(b)) > ID_DROPOUT)]
            for j, c in enumerate(ITEM_CATS):
                I = I + P[f"i_{c}"][icat[i, j]]
            logits = F.normalize(U, dim=-1) @ F.normalize(I, dim=-1).T / TAU - logq[i][None]
            dup = (i[None] == i[:, None]) & ~torch.eye(len(b), dtype=torch.bool)  # same item twice isn't a negative
            loss = F.cross_entropy(logits.masked_fill(dup, -1e9), torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / len(hu):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {k: v.detach().numpy() for k, v in P.items()}, iid.numpy()


def make_state(tx, cutoff, cu, ar, W, iid):
    """Everything the online path needs, from purchases strictly before `cutoff` (window for stats, full for age)."""
    nu, ni = len(cu), len(ar)
    past = tx[tx.day < cutoff]
    win = past[past.day >= cutoff - WEEKS * 7]
    bc = lambda x, w=None, n=ni: np.bincount(x, weights=w, minlength=n)
    since = lambda d: win[win.day >= cutoff - d]
    ic, uc = bc(win.i), bc(win.u, n=nu)
    first = past.groupby("i").day.min().reindex(range(ni)).to_numpy(np.float32)
    last = past.groupby("i").day.max().reindex(range(ni)).to_numpy(np.float32)
    s1w = bc(since(7).i)
    hist, hist_day = last_rows(win, nu)
    return {
        "cutoff": np.float32(cutoff),
        "user_key": cu.key.to_numpy(np.uint64), "user_cat": cu[USER_CATS].to_numpy(np.int16),
        "user_age": cu.age.to_numpy(np.float32), "user_n": uc.astype(np.float32),
        "user_last_day": win.groupby("u").day.max().reindex(range(nu)).to_numpy(np.float32),
        "user_avg_price": np.where(uc > 0, bc(win.u, win.price, nu) / np.maximum(uc, 1), np.nan).astype(np.float32),
        "user_hist": hist, "user_hist_day": hist_day,
        "item_ids": ar.article_id.to_numpy(np.int64), "item_title": ar.title.to_numpy(str),
        "item_cat": ar[ITEM_CATS].to_numpy(np.int32),
        "sales_1w": s1w.astype(np.float32), "sales_4w": bc(since(28).i).astype(np.float32),
        "sales_all": ic.astype(np.float32),
        "item_price": np.where(ic > 0, bc(win.i, win.price) / np.maximum(ic, 1), np.nan).astype(np.float32),
        "item_online_share": np.where(ic > 0, bc(win.i, win.sales_channel_id == 2) / np.maximum(ic, 1),
                                      np.nan).astype(np.float32),
        "item_first_day": first, "item_last_day": last,
        # ANN catalogue: sold in the last 12 weeks, or never sold before (= new arrivals). Long-dead articles are out.
        "indexed": np.flatnonzero((last >= cutoff - 84) | np.isnan(first)).astype(np.int64),
        "pop_top": np.argsort(-s1w, kind="stable")[:200].astype(np.int64),
        "item_vec": item_vec(W, iid, ar[ITEM_CATS].to_numpy()).astype(np.float32),
    }


def truth_of(df):
    return {u: set(g.tolist()) for u, g in df.groupby("u").i}


def score_lists(lists, truth):
    rows = []
    for u, t in truth.items():
        r = list(lists[u])
        rows.append({"u": u, "ndcg10": ndcg_at_k(r, t), "hits10": len(set(r[:10]) & t),
                     **{f"recall{k}": recall_at_k(r, t, k) for k in (10, 50, 100)}})
    return pd.DataFrame(rows).set_index("u")


def ranker_data(m, truth, users):
    X, y, grp = [], [], []
    for u in users:
        c = m.candidates(u, m.query_vec(u))
        lab = np.isin(c[0], list(truth[u]))
        if lab.any():  # groups without a positive give lambdarank no signal
            X.append(m.features(u, *c)), y.append(lab), grp.append(len(lab))
    return np.vstack(X), np.concatenate(y).astype(int), grp


def fit_ranker(m1, val_truth):
    users = np.random.default_rng(0).permutation(sorted(val_truth))[:RANKER_USERS]
    cut = int(0.8 * len(users))
    ds = lambda d: lgb.Dataset(d[0], d[1], group=d[2], feature_name=FEATURES)
    tr, va = ranker_data(m1, val_truth, users[:cut]), ranker_data(m1, val_truth, users[cut:])
    print(f"  ranker rows: train {len(tr[1]):,} ({len(tr[2]):,} users with a positive), holdout {len(va[1]):,}",
          flush=True)
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "verbose": -1, "seed": 0, "num_threads": os.cpu_count()}
    return lgb.train(params, ds(tr), 2000, valid_sets=[ds(va)],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def evaluate(m, truth):
    """Every policy's top-N_CAND per user. Candidates are computed once and shared by the stages built on them."""
    pop = m.popular(N_CAND).tolist()
    out = {k: {} for k in ("popular", "repurchase_pop", "two_tower", "candidates", "two_stage")}
    for u in truth:
        rep = list(dict.fromkeys(m.history(u).tolist()))
        out["popular"][u] = pop
        out["repurchase_pop"][u] = list(dict.fromkeys(rep + pop))[:N_CAND]
        q = m.query_vec(u)
        out["two_tower"][u] = m.ann(q, N_CAND)
        c = m.candidates(u, q)
        out["candidates"][u] = c[0]
        s = m.ranker.predict(m.features(u, *c), num_threads=1)
        out["two_stage"][u] = c[0][np.argsort(-s, kind="stable")]
    return out


def ann_recall(m, users, n=100):
    """HNSW vs exact inner-product search over the same catalogue: what does the approximation cost?"""
    import faiss
    iv = np.ascontiguousarray(m.S["item_vec"][m.S["indexed"]], dtype=np.float32)
    exact = faiss.IndexFlatIP(iv.shape[1])
    exact.add(iv)
    q = np.stack([m.query_vec(u) for u in users]).astype(np.float32)
    _, E = exact.search(q, n)
    _, A = m.index.search(q, n)
    return float(np.mean([len(set(a) & set(e)) / n for a, e in zip(A, E)]))


def save(path, arrays):
    os.makedirs(path, exist_ok=True)
    for k, v in arrays.items():
        np.save(f"{path}/{k}.npy", v)


def main():
    t0 = time.time()
    os.makedirs(REP, exist_ok=True)
    tx, cu, ar = load()
    T2 = int(tx.day.max()) - 6
    T1 = T2 - 7
    train = tx[(tx.day >= T1 - WEEKS * 7) & (tx.day < T1)]
    val, test = tx[(tx.day >= T1) & (tx.day < T2)], tx[tx.day >= T2]
    day = lambda d: str((EPOCH0 + pd.Timedelta(days=int(d))).date())
    print(f"{len(tx):,} transactions total | window: train {len(train):,} val {len(val):,} test {len(test):,} "
          f"| val week {day(T1)}, test week {day(T2)}..{day(T2 + 6)}", flush=True)

    print("[T1] two-tower on train", flush=True)
    W1, iid1 = fit_two_tower(train, cu, ar)
    m1 = Model(make_state(tx, T1, cu, ar, W1, iid1), W1)
    val_truth = truth_of(val)
    print("[T1] LightGBM lambdarank on val-week labels", flush=True)
    ranker = fit_ranker(m1, val_truth)
    print(f"  best iteration {ranker.best_iteration}", flush=True)

    print("[T2] two-tower refit on train+val", flush=True)
    W2, iid2 = fit_two_tower(tx[(tx.day >= T2 - WEEKS * 7) & (tx.day < T2)], cu, ar)
    S2 = make_state(tx, T2, cu, ar, W2, iid2)
    m2 = Model(S2, W2, ranker)
    test_truth = truth_of(test)
    te = time.time()
    lists = evaluate(m2, test_truth)
    print(f"[T2] evaluated {len(test_truth):,} test users in {time.time() - te:.0f}s", flush=True)
    per = {p: score_lists(v, test_truth) for p, v in lists.items()}

    # ---- error analysis ----
    users = list(test_truth)
    n_hist = pd.Series(S2["user_n"][users], index=users)
    bucket = pd.cut(n_hist, [-1, 0, 4, 19, 49, np.inf], labels=["0 (cold)", "1-4", "5-19", "20-49", "50+"])
    seg = {p: {str(b): {**dict(zip(["ndcg10", "ci95"], mean_ci(df.ndcg10[bucket == b]))), "n": int((bucket == b).sum())}
               for b in bucket.cat.categories} for p, df in per.items()}
    new_items = set(np.flatnonzero(np.isnan(S2["item_first_day"])).tolist())
    cold_pairs = [(u, i) for u, t in test_truth.items() for i in t if i in new_items]
    n_pairs = sum(len(t) for t in test_truth.values())
    cold = {p: {f"hit@{k}": float(np.mean([i in set(list(v[u])[:k]) for u, i in cold_pairs])) if cold_pairs else None
                for k in (10, 100)} for p, v in lists.items()}
    bias = {p: {"coverage@10": len({i for l in v.values() for i in list(l)[:10]}) / len(ar),
                "mean_log_sales@10": float(np.mean([np.log1p(S2["sales_all"][np.asarray(list(l)[:10], int)]).mean()
                                                    for l in v.values()]))} for p, v in lists.items()}
    ann_r = ann_recall(m2, np.random.default_rng(0).choice(users, min(1000, len(users)), replace=False))

    # ---- per-user table for the offline A/B (covariates strictly before the test week) ----
    pop_t1 = set(m1.popular(10).tolist())
    last4 = tx[(tx.day >= T2 - 28) & (tx.day < T2)].groupby("u").size()
    pd.DataFrame({
        "user_id": S2["user_key"][users].astype(str), "n_hist": n_hist.to_numpy(),
        "pre_log_n_window": np.log1p(n_hist.to_numpy()),
        "pre_n_4w": last4.reindex(users).fillna(0).to_numpy(),
        "pre_n_val": [len(val_truth.get(u, ())) for u in users],
        "pre_hits_ctrl": [len(pop_t1 & val_truth.get(u, set())) for u in users],
        **{f"{p}_{c}": per[p][c].to_numpy() for p in ("popular", "repurchase_pop", "two_stage")
           for c in ("ndcg10", "hits10")},
    }).to_csv(f"{REP}/per_user.csv", index=False)

    out = {"dataset": "H&M Personalized Fashion Recommendations (Kaggle)", "n_transactions_total": len(tx),
           "window_weeks": WEEKS, "n_interactions_used": len(train) + len(val) + len(test),
           "n_customers": len(cu), "n_articles": len(ar), "ann_catalogue": int(len(S2["indexed"])),
           "split": {"val_week": day(T1), "test_week": day(T2), "train": len(train), "val": len(val),
                     "test": len(test), "test_users": len(test_truth)},
           "test": {p: {c: round(float(v), 5) for c, v in df.mean().items()} for p, df in per.items()},
           "segments_ndcg10": seg,
           "new_items": {"share_of_test_purchases": len(cold_pairs) / n_pairs, "n_pairs": len(cold_pairs), "hit": cold},
           "popularity_bias": bias, "ann_recall@100_vs_exact": ann_r,
           "ranker": {"best_iter": ranker.best_iteration,
                      "gain": dict(zip(FEATURES, ranker.feature_importance("gain").round(1).tolist()))},
           "epochs": EPOCHS, "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{REP}/metrics.json", "w"), indent=2, default=float)
    save(f"{ART}/state", S2)
    save(f"{ART}/tower", W2)
    ranker.save_model(f"{ART}/ranker.txt")
    print(json.dumps(out["test"], indent=1))
    print(f"done in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
