"""Closed-loop load test: C concurrent clients (spread over P processes) hit /recommend for D seconds.

  python loadtest.py --concurrency 64 --procs 8 --duration 30 [--flush-redis redis://localhost:6379/0]
--flush-redis empties the cache first and gives every request a distinct user, so each one is a miss (the full
retrieval + ranking path). Without it, users repeat and the cache hit rate is reported.
"""
import argparse
import asyncio
import json
import random
import time
from multiprocessing import Pool

import httpx
import numpy as np


def client(url, users, concurrency, stop, unique):
    async def go():
        lat, hits, errors = [], 0, 0
        pool = iter(users)
        async with httpx.AsyncClient(base_url=url, timeout=10, limits=httpx.Limits(max_connections=concurrency)) as c:
            async def worker():
                nonlocal hits, errors
                while time.time() < stop:
                    u = next(pool, None) if unique else random.choice(users)
                    if u is None:
                        return
                    t = time.perf_counter()
                    try:
                        r = await c.get(f"/recommend/{u}")
                        r.raise_for_status()
                        hits += r.json()["cached"]
                        lat.append(time.perf_counter() - t)
                    except httpx.HTTPError:
                        errors += 1
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        return lat, hits, errors
    return asyncio.run(go())


def main(a):
    users = httpx.get(f"{a.url}/users/sample", params={"n": a.users, "seed": 0}).json()
    if a.flush_redis:
        import redis
        redis.Redis.from_url(a.flush_redis).flushdb()
    random.shuffle(users)
    stop = time.time() + a.duration
    t0 = time.perf_counter()
    with Pool(a.procs) as p:
        res = p.starmap(client, [(a.url, users[i::a.procs], a.concurrency // a.procs, stop, bool(a.flush_redis))
                                 for i in range(a.procs)])
    wall = time.perf_counter() - t0
    ms = np.concatenate([r[0] for r in res]) * 1000
    hits, errors = sum(r[1] for r in res), sum(r[2] for r in res)
    out = {"mode": "all-miss" if a.flush_redis else "mixed", "concurrency": a.concurrency, "requests": len(ms),
           "errors": errors, "rps": round(len(ms) / wall, 1), "cache_hit_rate": round(hits / max(len(ms), 1), 3),
           **{f"p{q}_ms": round(float(np.percentile(ms, q)), 2) for q in (50, 95, 99)}}
    print(json.dumps(out))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--users", type=int, default=5000)
    p.add_argument("--flush-redis")
    p.add_argument("--out")
    main(p.parse_args())
