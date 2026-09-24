"""Writes a small synthetic dataset with the H&M Kaggle schema, so CI can run the whole pipeline without Kaggle.

Customers prefer one product type and repurchase sometimes; a few articles launch in the final week and a few
customers buy for the first time in it, so the cold-start paths get exercised.
  python -m tests.fake_hm data/fake
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd


def write(out, n_cust=3000, n_art=600, days=140, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    cid = [hashlib.sha256(f"c{k}".encode()).hexdigest() for k in range(n_cust)]
    maybe = lambda vals, p_nan: np.where(rng.random(n_cust) < p_nan, None, rng.choice(vals, n_cust))
    pd.DataFrame({"customer_id": cid, "FN": maybe([1.0], .6), "Active": maybe([1.0], .6),
                  "club_member_status": maybe(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], .05),
                  "fashion_news_frequency": maybe(["NONE", "Regularly", "Monthly"], .05),
                  "age": np.where(rng.random(n_cust) < .02, np.nan, rng.integers(16, 80, n_cust)),
                  "postal_code": "x"}).to_csv(f"{out}/customers.csv", index=False)
    ptype = rng.integers(0, 20, n_art)
    colour = rng.integers(0, 10, n_art)
    pd.DataFrame({"article_id": 100000000 + np.arange(n_art), "product_code": np.arange(n_art),
                  "prod_name": [f"Item {k}" for k in range(n_art)], "product_type_no": ptype,
                  "product_group_name": np.array(list("ABCDE"))[ptype % 5],
                  "graphical_appearance_no": rng.integers(0, 5, n_art), "colour_group_code": colour,
                  "colour_group_name": [f"Colour {c}" for c in colour], "perceived_colour_master_id": colour % 4,
                  "department_no": ptype * 2 + rng.integers(0, 2, n_art), "index_group_no": ptype % 4,
                  "section_no": ptype % 10, "garment_group_no": ptype % 8}).to_csv(f"{out}/articles.csv", index=False)
    launch = np.where(rng.random(n_art) < .05, days - 7, rng.integers(0, days - 30, n_art))  # 5% launch in test week
    appeal = rng.pareto(1.5, n_art) + 0.1
    start = np.where(rng.random(n_cust) < .05, days - 7, 0)  # 5% of customers first appear in the test week
    pref = rng.integers(0, 20, n_cust)
    rows = []
    for d in range(days):
        live = launch <= d
        buyers = np.flatnonzero((rng.random(n_cust) < 0.03) & (start <= d))
        for c in buyers:
            w = appeal * live * np.where(ptype == pref[c], 8.0, 1.0)
            for a in rng.choice(n_art, rng.integers(1, 4), p=w / w.sum()):
                rows.append((d, cid[c], 100000000 + a, round(0.01 + 0.05 * rng.random(), 4), rng.integers(1, 3)))
    tx = pd.DataFrame(rows, columns=["day", "customer_id", "article_id", "price", "sales_channel_id"])
    tx.insert(0, "t_dat", (pd.Timestamp("2020-09-22") - pd.Timedelta(days=days - 1) + pd.to_timedelta(tx.pop("day"), "D")).dt.date)
    tx.to_csv(f"{out}/transactions_train.csv", index=False)
    return len(tx)


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else "data/fake"), "transactions")
