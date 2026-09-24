import numpy as np

from recsys.metrics import cuped, diff_ci, ndcg_at_k, recall_at_k


def test_ndcg():
    assert ndcg_at_k([1, 2, 3], {1}) == 1.0
    assert ndcg_at_k([9, 1], {1}) == 1 / np.log2(3)
    assert ndcg_at_k([9, 8], {1}) == 0.0
    assert ndcg_at_k([1, 2], {1, 2, 3}) == (1 + 1 / np.log2(3)) / (1 + 1 / np.log2(3) + 0.5)


def test_recall():
    assert recall_at_k([1, 2, 3], {1, 4}, 2) == 0.5
    assert recall_at_k([1], set(), 10) == 0.0


def test_cuped_ci_coverage_and_variance_reduction():
    rng = np.random.default_rng(0)
    n, effect, covered, ratio = 2000, 0.1, 0, []
    for _ in range(300):
        x = rng.normal(size=n)
        t = rng.random(n) < 0.5
        y = 2 * x + effect * t + rng.normal(size=n)
        *_, se_raw = diff_ci(y, t)
        _, lo, hi, se = diff_ci(cuped(y, x), t)
        covered += lo < effect < hi
        ratio.append(se / se_raw)
    assert 0.91 <= covered / 300 <= 0.99  # nominal 95%
    assert np.mean(ratio) < 0.5  # corr(x, y) ~ 0.9 -> ~80% variance reduction
