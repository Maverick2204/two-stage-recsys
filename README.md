# Two-stage recommender: retrieval + ranking on H&M

**Live demo: https://hm-two-stage-recsys-latest.onrender.com/** (free tier: it sleeps after ~15 min idle, so the first load can take 30-60 s while it wakes up.)

**The problem.** A fashion retailer has about 105k live articles and 1.4M customers, and each customer sees 10 slots in a "Recommended for you" row. Many retailers fill that row with last week's best sellers. It's the same 10 items for everyone, it ignores what each customer already buys, and it can't surface the long tail. This project ranks articles per customer to predict what they will buy **next week**, and serves the result fast enough for a page render (p95 under 20 ms on the full model path).

**What I report.** Everything is evaluated on a strict *future* week that the models never see. Each stage is compared against popularity and a stronger heuristic baseline. There is an error analysis of where it fails, measured latency and throughput, and an offline A/B analysis with CUPED. That A/B analysis is clearly labelled as **offline replay, not production lift**.

| | NDCG@10 | Recall@10 | Hits@10 per user |
|---|---|---|---|
| Last week's best sellers (baseline) | 0.0145 | 0.0222 | 0.059 |
| Recent purchases + best sellers (heuristic) | 0.0345 | 0.0461 | 0.102 |
| **Two-stage (this project)** | **0.0455** | **0.0623** | **0.147** |
| Lift vs best sellers / vs heuristic | **+214% / +32%** | +181% / +35% | +151% / +45% |

Test week 2020-09-16 to 09-22: 68,984 customers and 240k purchases. Numbers come from [reports/metrics.json](reports/metrics.json).

---

## Architecture

```
                 ┌───────────────── stage 1: candidates (≤100) ─────────────────┐
request ──► user │ repurchase: customer's last 20 purchases                       │ ──► stage 2: LightGBM ──► top-k
  (id)    state  │ two-tower ANN: FAISS HNSW over 43k live articles               │     lambdarank, 25 features
                 │ best sellers: last 7 days, top 30                              │
                 └────────────────────────────────────────────────────────────────┘
FastAPI (4 uvicorn workers, state memory-mapped and shared) ◄──► Redis (response cache, TTL 10 min, fail-open)
```

- **Two-tower** ([recsys/train.py](recsys/train.py)): the user tower is the mean embedding of past purchases plus customer attributes (age band, club status, newsletter, etc.). It has **no per-user id table**, so it scales to any number of customers and brand-new customers still get a vector. The item tower is an article id embedding plus 9 article attributes, with 10% id-dropout so articles that have never sold get a usable attributes-only vector. It's trained with in-batch sampled softmax and a logQ popularity correction, where each training row sees only purchases from *earlier days*, so there's no leakage within a basket.
- **ANN**: FAISS HNSW (M=32, efSearch=128) over articles sold in the last 12 weeks plus never-sold (new) articles. **Recall@100 vs exact search: 0.942.**
- **Ranker**: LightGBM lambdarank. The strongest features are days since the customer last bought this article, last-week sales, max similarity to recent purchases, and customer recency.
- **One code path**: [recsys/core.py](recsys/core.py) is numpy-only and used by both the offline evaluation and the API, so offline and online can't drift apart.

## Evaluation protocol (split by time, not randomly)

| window | dates | used for |
|---|---|---|
| train | 26 weeks to 2020-09-08 (8.0M purchases) | two-tower |
| val | 2020-09-09 to 09-15 (255k) | ranker labels (candidates and features computed as of 09-09) |
| test | 2020-09-16 to 09-22 (240k) | final numbers only; the two-tower is refit on train+val and features are as of 09-16 |

Repeat purchases count (buying an article again is a real signal in fashion), so already-bought articles are *not* filtered out. All model choices were made on the val week. The test week was scored once.

### Per stage (test week)

| stage | NDCG@10 | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| two-tower ANN alone | 0.0236 | 0.0340 | 0.0799 | 0.1139 |
| candidate union (unranked) | 0.0331 | 0.0438 | 0.0915 | 0.1247 |
| + LightGBM ranker | **0.0455** | **0.0623** | **0.1089** | 0.1247 |

## Error analysis

**By customer history size** (NDCG@10 ± 95% CI):

| purchases in the 26-week window | customers | best sellers | heuristic | two-stage |
|---|---|---|---|---|
| 0 (cold) | 11,988 | **0.0134 ± .0013** | 0.0134 ± .0013 | 0.0103 ± .0010 |
| 1–4 | 10,292 | 0.0144 ± .0015 | 0.0418 ± .0032 | **0.0511 ± .0035** |
| 5–19 | 26,472 | 0.0147 ± .0010 | 0.0349 ± .0017 | **0.0504 ± .0020** |
| 20–49 | 15,219 | 0.0148 ± .0012 | 0.0386 ± .0023 | **0.0549 ± .0026** |
| 50+ | 5,013 | 0.0155 ± .0022 | 0.0559 ± .0044 | **0.0644 ± .0046** |

- **Cold customers (17% of the test week) are the one segment where the model loses.** For them the two-stage model is *worse* than best sellers. With no history, the user tower has only coarse attributes to go on, and the ranker's strongest features (repurchase, similarity) are empty. The fix is to route zero-history customers to best sellers, but that must be validated on the val week first, not chosen from these test numbers.
- **New articles are almost never recommended.** 3.9% of test-week purchases (8,334) were articles that had never sold before. Popularity-based policies can't surface them at all, and even the attributes-only item tower gets only 0.4% into its top 100. The item attributes (type, colour, department) don't carry enough signal. Image or text embeddings of the article would be the next step.
- **Stage 1 leaves recall on the table.** The 100-candidate union reaches recall@100 of 0.125, which is *below* the heuristic's own 100-item list (0.144), because only 30 best-seller slots make it into the pool. Widening the pool is the cheapest next gain.
- **Popularity bias**: best sellers show 10 articles to everyone. Two-stage shows **13.8% of the catalogue** in some customer's top 10.

## Offline A/B analysis with CUPED

> **This is an offline replay on logged purchases, not a measured production lift.** The logged customers never saw these recommendations, so this estimates ranking quality against what they bought anyway, not the causal effect of showing them.

Test-week customers are split 50/50 by a salted hash of their id, and each contributes only their own arm's NDCG@10. The CUPED covariates are all from *before* the test week: purchases in the window, purchases in the last 4 weeks, val-week purchases, and val-week hits of the control policy. Source: [recsys/ab.py](recsys/ab.py).

| treatment: two-stage vs | n (A / B) | raw lift [95% CI] | CUPED lift [95% CI] | variance reduction | paired check (all users) |
|---|---|---|---|---|---|
| best sellers | 34,532 / 34,452 | +220% [+207, +233] | +220% [+208, +233] | 1.4% | +214% [+206, +223] |
| recent purchases + best sellers | 34,532 / 34,452 | +35.2% [+28.7, +41.7] | +35.5% [+29.1, +41.9] | 3.6% | +31.9% [+29.7, +34.1] |

- In an **A/A check** (control vs itself, 1,000 random splits), the CI excluded zero 4.6% and 4.1% of the time, close to the nominal 5%.
- **CUPED helps little here, and that's a real finding.** Per-user NDCG@10 is sparse (most customers get 0), and pre-period behaviour barely predicts it: every covariate I tried, including repeat rate, recency and last-1/2-week activity, correlated at 0.2 or less, capping variance reduction around 5%. CUPED earns its keep on dense metrics like revenue per user or sessions, not on sparse per-user ranking scores.

## Serving performance

These were measured with [loadtest.py](loadtest.py): a closed loop with 8 client processes and distinct customers. The API ran in Docker with 4 uvicorn workers and Redis, and the load generator ran on the same 32-core machine.

| mode | concurrency | throughput | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|---|
| cache miss (full two-stage path) | 16 | **1,444 RPS** | 6.9 ms | **15.8 ms** | 21.3 ms | 0 |
| cache miss (full two-stage path) | 64 | 1,980 RPS | 20.1 ms | 58.2 ms | 68.9 ms | 0 |
| cache hit (Redis) | 64 | 5,208 RPS | 8.6 ms | 34.7 ms | 61.9 ms | 0 |

Memory: 4 workers serving 1.37M customers use **423 MB in total**, because the state is loaded as memory-mapped `.npy` files and shared between workers. A model call in-process takes under 1 ms, so most of the latency is HTTP and queueing. Plain `uvicorn` (the h11 parser) stalled about 45 ms per keep-alive request from a TCP delayed-ACK interaction inside the container, and `uvicorn[standard]` fixed it (0.4 ms).

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt
# Kaggle token in ~/.kaggle/kaggle.json and the competition rules accepted:
# https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
python -m recsys.train                 # ~30 min on 32 CPU cores; writes artifacts/ and reports/
python -m recsys.ab --control popular  # offline A/B replay -> reports/ab.json
docker compose up --build              # API on :8000 with Redis; demo page at /
python loadtest.py --flush-redis redis://localhost:6379/0
pytest -q                              # includes an end-to-end run on synthetic data
```

API: `GET /recommend/{customer_id}?k=10` accepts the 64-hex id or its first 16 hex characters. Unknown customers get best sellers. Other endpoints are `GET /users/sample?n=20`, `GET /health`, and `GET /` for the demo page.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the tests, trains on synthetic data with the H&M schema ([tests/fake_hm.py](tests/fake_hm.py)) because Kaggle data can't be pulled in CI, then builds the Docker image and hits the running container.

## Why not MovieLens?

I started on MovieLens-1M ([reports/movielens-1m/](reports/movielens-1m/)). On a global time split, 7-day popularity was almost unbeatable. Even EASE, a strong item-item model, lost to it, and a two-stage model tied it on NDCG@10 (0.286 vs 0.287). Most ML-1M ratings come from 2000 onboarding sessions, so the last 10% by time is a sparse two-year tail driven by what's trending. H&M has real weekly purchase behaviour with repeat buying, which is the setting a two-stage recommender is built for.

## Limitations and next steps
- Route zero-history customers to best sellers, validated on the val week.
- Widen the candidate pool (more best-seller and co-purchase slots). Stage 1 recall is the main ceiling right now.
- Add image or text article embeddings for the cold-item problem.
- Evaluate on several consecutive weeks (a rolling backtest), not a single test week.
- [x] Public demo (Render free tier, image on GHCR).

Data: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle, non-commercial use). Raw data and trained artifacts are not committed.
