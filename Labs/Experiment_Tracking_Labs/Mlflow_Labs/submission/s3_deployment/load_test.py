"""Load test: latency percentiles and throughput of the served model over real HTTP.

    python -m s3_deployment.load_test [--url http://127.0.0.1:5001]
    python -m s3_deployment.load_test --workers-compare      # starts its own servers on :5002, 1 vs 2 workers

Why percentiles, not averages: an average hides the slow tail, and the tail is what users feel and
what timeouts hit. p95/p99 are what an SLO is written against. Requests use DIFFERENT random rows,
so no cache or branch predictor sees the same payload twice.
Output: table + reports/load_test.md. Everything here is on ONE laptop (client and server share the
CPU), so read the numbers as relative comparisons, not capacity planning.
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from s2_registry.common import init
from s3_deployment.common import (
    MODEL_PORT, post, sample_features, split_payload, start_model_server, stop_model_server, url_for,
)
from s3_deployment.serve_local import DEFAULT_URI

REPORTS = config.ROOT / "reports"
# (batch size, number of timed requests): fewer requests for big batches so the test stays short
PLAN = [(1, 200), (10, 150), (100, 100), (1000, 40)]
CONCURRENCY = [1, 4]


def make_bodies(pool, batch: int, n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [json.dumps(split_payload(pool.iloc[rng.integers(0, len(pool), batch)])) for _ in range(n)]


def measure(url: str, bodies: list, concurrency: int) -> dict:
    for b in bodies[:5]:  # warm-up: first calls pay one-time costs
        post(url, b)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda b: post(url, b), bodies))
    wall = time.perf_counter() - t0
    lat = sorted(r[2] * 1000 for r in results if r[0] == 200)
    errors = sum(1 for r in results if r[0] != 200)
    pct = lambda q: float(np.percentile(lat, q)) if lat else float("nan")
    return {
        "p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99), "max_ms": max(lat) if lat else float("nan"),
        "req_per_s": len(results) / wall, "errors": errors,
    }


def run_suite(url: str, label: str, pool) -> list:
    rows = []
    for batch, n in PLAN:
        bodies = make_bodies(pool, batch, n)
        for conc in CONCURRENCY:
            m = measure(url, bodies, conc)
            rows.append({"server": label, "batch": batch, "concurrency": conc, **m, "rows_per_s": m["req_per_s"] * batch})
            print(f"  {label:10s} batch={batch:<5d} conc={conc}  p50={m['p50_ms']:7.1f}  p95={m['p95_ms']:7.1f}  "
                  f"p99={m['p99_ms']:7.1f} ms   {m['req_per_s']:7.1f} req/s  {rows[-1]['rows_per_s']:9.0f} rows/s  errors={m['errors']}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=url_for(MODEL_PORT))
    p.add_argument("--workers-compare", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pool = sample_features(5000, seed=3)
    rows = []
    if args.workers_compare:
        init()
        port = 5002
        for workers in (1, 2):
            info = start_model_server(DEFAULT_URI, port, workers)
            print(f"\nworkers={workers}: healthy in {info['seconds_to_healthy']:.1f}s")
            try:
                rows += run_suite(url_for(port), f"workers={workers}", pool)
            finally:
                stop_model_server(port)
    else:
        rows = run_suite(args.url, "server", pool)

    REPORTS.mkdir(exist_ok=True)
    md = ["# Load test\n", "| server | batch | conc | p50 ms | p95 ms | p99 ms | req/s | rows/s | errors |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['server']} | {r['batch']} | {r['concurrency']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['p99_ms']:.1f} | "
           f"{r['req_per_s']:.1f} | {r['rows_per_s']:.0f} | {r['errors']} |" for r in rows]
    (REPORTS / "load_test.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'load_test.md'}")


if __name__ == "__main__":
    main()
