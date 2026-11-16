"""Parity test: does the SERVED model answer exactly like the same model loaded in-process?

    python -m s3_deployment.parity_test [--url http://127.0.0.1:5001] [--version N] [--rows 2000]

Training-serving skew is the classic silent failure: the model that was evaluated is not the model
that answers requests (different library versions, different preprocessing, a payload that rounds
your inputs). This sends the same rows to the server and to the in-process model and compares.
Reused against the Docker container by docker_build.py.
"""
import argparse
import json
import sys

import mlflow
import numpy as np

import config
from s2_registry.common import alias_version, init
from s3_deployment.common import (
    MODEL_PORT, post, predictions, records_payload, sample_features, split_payload, url_for,
)

TOLERANCE = 1e-9


def run(url: str, version: str, n_rows: int = 2000) -> dict:
    X = sample_features(n_rows)
    local = np.asarray(mlflow.pyfunc.load_model(f"models:/{config.REGISTERED_MODEL_NAME}/{version}").predict(X))
    results = {}
    for fmt, payload in (("dataframe_split", split_payload(X)), ("dataframe_records", records_payload(X))):
        status, text, _ = post(url, json.dumps(payload))
        if status != 200:
            results[fmt] = {"ok": False, "detail": f"HTTP {status}: {text[:120]}"}
            continue
        remote = predictions(text)
        if remote.shape != local.shape:
            results[fmt] = {"ok": False, "detail": f"shape {remote.shape} != {local.shape}"}
            continue
        diff = np.abs(remote - local)
        results[fmt] = {
            "ok": bool(diff.max() <= TOLERANCE),
            "max_abs_diff": float(diff.max()),
            "exact_fraction": float((diff == 0).mean()),
            "detail": f"max |server - local| = {diff.max():.2e}, bit-identical on {(diff == 0).mean():.1%} of values",
        }
    return {"ok": all(r["ok"] for r in results.values()), "rows": len(X), "results": results}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=url_for(MODEL_PORT))
    p.add_argument("--version", help="registry version the server is serving (default: current champion)")
    p.add_argument("--rows", type=int, default=2000)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    client = init()
    version = args.version or alias_version(client, config.CHAMPION_ALIAS).version
    r = run(args.url, version, args.rows)
    print(f"parity vs in-process v{version} on {r['rows']} rows (tolerance {TOLERANCE:g})")
    for fmt, res in r["results"].items():
        print(f"  {fmt:18s} {'PASS' if res['ok'] else 'FAIL'}  {res['detail']}")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
