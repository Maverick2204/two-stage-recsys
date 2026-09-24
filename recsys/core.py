"""Numpy-only inference shared by offline eval and the API, so both run the exact same code path.

Time is in days. Per-user state keeps the user's last R purchases (strictly before the cutoff); the user tower,
the repurchase candidates and the cross features are all computed from that at request time.
"""
import os

import faiss
import lightgbm as lgb
import numpy as np

R = 20  # purchases of history kept per user
USER_CATS = ["age_b", "club", "news", "fn", "active"]
ITEM_CATS = ["product_type_no", "product_group_name", "graphical_appearance_no", "colour_group_code",
             "perceived_colour_master_id", "department_no", "index_group_no", "section_no", "garment_group_no"]
FEATURES = ["tt_score", "ann_rank", "src_repurchase", "src_ann", "src_pop",
            "log_sales_1w", "log_sales_4w", "log_sales_all", "item_price", "item_age_days", "item_days_since_last",
            "item_online_share", "log_user_n", "user_days_since_last", "user_age", "user_club", "user_news",
            "user_avg_price", "rep_count", "rep_days", "same_type_share", "same_dept_share", "price_ratio",
            "sim_recent_max", "sim_recent_mean"]
N_CAND, N_POP = 100, 30


def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def user_vec(W, hist, ucat):
    """User tower: mean embedding of past purchases (hist: [..., R], -1 = padding) + customer attributes.
    No per-user id table, so it scales to any number of customers and cold users still get a vector."""
    m = hist >= 0
    v = (W["u_hist"][np.where(m, hist, 0)] * m[..., None]).sum(-2) / np.maximum(m.sum(-1, keepdims=True), 1)
    for j, c in enumerate(USER_CATS):
        v = v + W[f"u_{c}"][ucat[..., j]]
    return l2norm(v)


def item_vec(W, iid_idx, icat):
    """Item tower: id (0 = 'unknown', trained by id dropout, used for never-sold articles) + article attributes."""
    v = W["i_id"][iid_idx]
    for j, c in enumerate(ITEM_CATS):
        v = v + W[f"i_{c}"][icat[..., j]]
    return l2norm(v)


def user_key(customer_id):
    """H&M customer ids are 64-hex hashes; the first 16 hex chars as uint64 are unique and 8x smaller to store."""
    return np.uint64(int(customer_id[:16], 16))


class Model:
    def __init__(self, S, W, ranker=None):
        self.S, self.W, self.ranker = S, W, ranker
        iv = np.ascontiguousarray(S["item_vec"][S["indexed"]], dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(iv.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction, self.index.hnsw.efSearch = 80, 128
        self.index.add(iv)

    @classmethod
    def load(cls, art_dir):
        arr = lambda d: {f[:-4]: np.load(f"{art_dir}/{d}/{f}", mmap_mode="r") for f in os.listdir(f"{art_dir}/{d}")}
        return cls(arr("state"), arr("tower"), lgb.Booster(model_file=f"{art_dir}/ranker.txt"))

    def user_index(self, customer_id):
        try:
            k = user_key(customer_id)
        except ValueError:
            return None
        keys = self.S["user_key"]
        i = int(np.searchsorted(keys, k))
        return i if i < len(keys) and keys[i] == k else None

    def history(self, u):
        h = self.S["user_hist"][u]
        return np.asarray(h[h >= 0])

    def query_vec(self, u):
        return user_vec(self.W, np.asarray(self.S["user_hist"][u]), np.asarray(self.S["user_cat"][u]))

    def ann(self, q, n):
        _, I = self.index.search(q[None].astype(np.float32), n)
        return np.asarray(self.S["indexed"])[I[0][I[0] >= 0]]

    def popular(self, n):
        return np.asarray(self.S["pop_top"][:n])

    def candidates(self, u, q, n=N_CAND):
        """Stage 1: union of repurchase (user's recent items), two-tower ANN, and last week's best sellers."""
        rep = list(dict.fromkeys(self.history(u).tolist()))
        ann = self.ann(q, n)
        pop = self.popular(N_POP)
        items = np.array(list(dict.fromkeys(rep + ann.tolist() + pop.tolist()))[:n], dtype=np.int64)
        src = np.column_stack([np.isin(items, rep), np.isin(items, ann), np.isin(items, pop)]).astype(np.float32)
        pos = {i: r for r, i in enumerate(ann.tolist())}
        rank = np.array([pos.get(i, np.nan) for i in items.tolist()], dtype=np.float32)
        return items, src, rank, self.S["item_vec"][items] @ q

    def features(self, u, items, src, rank, scores):
        S, t, n = self.S, float(self.S["cutoff"]), len(items)
        hist = self.history(u)
        match = items[:, None] == hist[None]
        hdays = np.asarray(S["user_hist_day"][u][:len(hist)])
        last_day = np.where(match, hdays[None], -np.inf).max(1) if len(hist) else np.full(n, -np.inf)
        icat = S["item_cat"]
        if len(hist):
            same_type = (icat[items, 0][:, None] == icat[hist, 0][None]).mean(1)
            same_dept = (icat[items, 5][:, None] == icat[hist, 5][None]).mean(1)
            sim = S["item_vec"][items] @ S["item_vec"][hist].T
            smax, smean = sim.max(1), sim.mean(1)
        else:
            same_type = same_dept = smax = smean = np.full(n, np.nan)
        uc = S["user_cat"][u]
        return np.column_stack([
            scores, rank, src,
            np.log1p(S["sales_1w"][items]), np.log1p(S["sales_4w"][items]), np.log1p(S["sales_all"][items]),
            S["item_price"][items], t - S["item_first_day"][items], t - S["item_last_day"][items],
            S["item_online_share"][items],
            np.full(n, np.log1p(S["user_n"][u])), np.full(n, t - S["user_last_day"][u]),
            np.full(n, S["user_age"][u]), np.full(n, uc[1]), np.full(n, uc[2]), np.full(n, S["user_avg_price"][u]),
            match.sum(1), np.where(np.isfinite(last_day), t - last_day, np.nan), same_type, same_dept,
            S["item_price"][items] / S["user_avg_price"][u], smax, smean,
        ]).astype(np.float32)

    def recommend(self, u, k=10, rank=True):
        """Two-stage recommendation for user row u (None = unknown customer -> last week's best sellers)."""
        if u is None:
            return self.popular(k)
        items, src, ann_rank, scores = self.candidates(u, self.query_vec(u))
        if rank:
            s = self.ranker.predict(self.features(u, items, src, ann_rank, scores), num_threads=1)
            items = items[np.argsort(-s, kind="stable")]
        return items[:k]
