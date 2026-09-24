import numpy as np


def ndcg_at_k(ranked, truth, k=10):
    """Binary-relevance NDCG@k. ranked: sequence of item ids, truth: set of relevant ids."""
    gains = [1.0 / np.log2(r + 2) for r, i in enumerate(ranked[:k]) if i in truth]
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(truth))))
    return sum(gains) / ideal if ideal else 0.0


def recall_at_k(ranked, truth, k):
    return len(set(ranked[:k]) & truth) / len(truth) if truth else 0.0


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    return x.mean(), z * x.std(ddof=1) / np.sqrt(len(x))


def cuped(y, X):
    """Regression CUPED: y - (X - mean X) @ theta, theta fit on pooled data (valid since X is pre-period)."""
    X = np.atleast_2d(np.asarray(X, float).T).T
    Xc = X - X.mean(0)
    theta = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)[0]
    return y - Xc @ theta


def diff_ci(y, treat, z=1.96):
    """Difference in means (treat - control) with Welch normal-approx CI."""
    a, b = y[~treat], y[treat]
    d = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - z * se, d + z * se, se
