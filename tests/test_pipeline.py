"""End-to-end smoke test on synthetic H&M-schema data: train -> artifacts -> API -> offline A/B."""
import importlib
import os
import subprocess
import sys

from fastapi.testclient import TestClient

from tests.fake_hm import write


def test_end_to_end(tmp_path):
    write(tmp_path / "data")
    env = {**os.environ, "DATA": str(tmp_path / "data"), "ART": str(tmp_path / "art"), "REP": str(tmp_path / "rep"),
           "WEEKS": "16", "EPOCHS": "1"}
    subprocess.run([sys.executable, "-m", "recsys.train"], env=env, check=True)
    subprocess.run([sys.executable, "-m", "recsys.ab", "--per-user", f"{tmp_path}/rep/per_user.csv",
                    "--out", f"{tmp_path}/rep/ab.json"], env=env, check=True)

    os.environ["ART"] = env["ART"]
    os.environ.pop("REDIS_URL", None)
    import recsys.app
    c = TestClient(importlib.reload(recsys.app).app)
    assert c.get("/health").json()["ok"]
    assert "<form" in c.get("/").text
    users = c.get("/users/sample", params={"n": 5, "seed": 0}).json()
    for u in users:
        r = c.get(f"/recommend/{u}", params={"k": 12}).json()
        ids = [it["article_id"] for it in r["items"]]
        assert r["strategy"] == "two_stage" and len(ids) == 12 == len(set(ids))
    assert c.get("/recommend/not-a-customer").json()["strategy"] == "popular"
    assert c.get("/recommend/x", params={"k": 0}).status_code == 422
