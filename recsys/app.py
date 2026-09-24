import json
import os

import numpy as np
import redis
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from recsys.core import Model

model = Model.load(os.environ.get("ART", "artifacts"))  # arrays are mmap'd, so uvicorn workers share one copy
cache = redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=0.05) if os.environ.get("REDIS_URL") else None
TTL = int(os.environ.get("CACHE_TTL", 600))
app = FastAPI(title="Two-stage recommender")


def cache_get(key):
    try:
        return cache.get(key) if cache else None
    except redis.RedisError:  # cache is an optimisation: if Redis is down, serve from the model
        return None


def cache_set(key, val):
    try:
        cache and cache.set(key, val, ex=TTL)
    except redis.RedisError:
        pass


@app.get("/health")
def health():
    return {"ok": True, "cache": cache is not None, "users": len(model.S["user_key"]), "items": len(model.S["item_ids"])}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=100)):
    key = f"rec:{user_id}:{k}"
    if hit := cache_get(key):
        return {**json.loads(hit), "cached": True}
    u = model.user_index(user_id)  # full 64-hex customer_id or its first 16 hex chars
    items = model.recommend(u, k)  # unknown customer -> last week's best sellers
    out = {"user_id": user_id, "strategy": "two_stage" if u is not None else "popular",
           "items": [{"article_id": f"{model.S['item_ids'][i]:010d}", "title": str(model.S["item_title"][i])} for i in items]}
    cache_set(key, json.dumps(out))
    return {**out, "cached": False}


@app.get("/users/sample")
def sample_users(n: int = Query(20, ge=1, le=10000), seed: int | None = None):
    """Random customers with purchase history, as 16-hex id prefixes (accepted by /recommend)."""
    ids = model.S["user_key"][np.flatnonzero(np.asarray(model.S["user_n"]) > 0)]
    return [f"{int(k):016x}" for k in np.random.default_rng(seed).choice(ids, min(n, len(ids)), replace=False)]


@app.get("/", response_class=HTMLResponse)
def demo():
    return open(os.path.join(os.path.dirname(__file__), "demo.html")).read()
