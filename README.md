# Two-stage recommender: retrieval + ranking on H&M

> **Work in progress.** Everything below runs; the real H&M results, load-test numbers, and public demo are not in yet.

Two-tower retrieval (FAISS HNSW) plus repurchase and best-seller candidates, then a LightGBM lambdarank ranker.
It is split by week (train, then a val week, then a test week) and served with FastAPI and a Redis cache in Docker.

## Status
- [x] Pipeline ([recsys/train.py](recsys/train.py)): weekly time split, two-tower, candidate union, LightGBM ranker, and NDCG@10 / recall@k per stage against two baselines (last week's best sellers, and recent purchases plus best sellers)
- [x] Error analysis: results by history size (including cold users), new-article hit rate, popularity bias, and HNSW-vs-exact recall
- [x] API ([recsys/app.py](recsys/app.py)) with a cache that keeps serving if Redis is down, plus a demo page at `/`
- [x] Offline A/B replay with CUPED ([recsys/ab.py](recsys/ab.py)): Welch CIs, paired check, A/A false-positive check
- [x] Load test ([loadtest.py](loadtest.py)): multi-process, all-miss and mixed cache modes, p50/p95/p99 and RPS
- [x] CI: tests plus a Docker build and smoke test on synthetic data with the H&M schema ([tests/fake_hm.py](tests/fake_hm.py))
- [ ] Run on real H&M data (needs a Kaggle token in `~/.kaggle/kaggle.json` and the competition rules accepted)
- [ ] Load test, A/B and README write-up with real numbers
- [ ] Public demo (Hugging Face Spaces)

## Run
```bash
pip install -r requirements-train.txt
python -m recsys.train          # downloads H&M via the Kaggle CLI; writes artifacts/ and reports/
python -m recsys.ab             # offline A/B replay with CUPED -> reports/ab.json
docker compose up --build       # API on :8000 with Redis
python loadtest.py --flush-redis redis://localhost:6379/0
pytest -q                       # includes an end-to-end run on synthetic data
```

## Notes so far
- **MovieLens-1M was dropped** ([reports/movielens-1m/](reports/movielens-1m/)). On a global time split, 7-day popularity is nearly unbeatable. EASE loses to it, and the two-stage model tied it on NDCG@10 (0.286 vs 0.287) while getting +27% recall@10. Most of ML-1M's ratings come from 2000 onboarding sessions, so the last 10% by time is a sparse two-year tail driven by what's trending.
- **Plain `uvicorn` (h11) hit a roughly 45 ms delayed-ACK stall** on keep-alive connections in the container. `uvicorn[standard]` fixed it (0.4 ms).
