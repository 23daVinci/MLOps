"""Smoke test: the cheapest possible proof that a deployed model answers sensibly.

    python -m s3_deployment.smoke_test [--url http://127.0.0.1:5001]

Run it after EVERY deploy, before shifting traffic. It is also reused against the Docker container
(docker_build.py), so it takes a URL and nothing else.
"""
import argparse
import json
import sys

import numpy as np

from s3_deployment.common import MODEL_PORT, get, post, predictions, sample_features, split_payload, url_for


def run(url: str, n_rows: int = 5) -> dict:
    X = sample_features(n_rows)
    problems = []
    health = get(f"{url}/health")[0]
    if health != 200:
        problems.append(f"/health returned {health}")
    status, text, secs = post(url, json.dumps(split_payload(X)))
    out = None
    if status != 200:
        problems.append(f"/invocations returned {status}: {text[:150]}")
    else:
        out = predictions(text)
        if out.shape != (len(X), 2):
            problems.append(f"output shape {out.shape}, expected ({len(X)}, 2)")
        elif out.min() < 0 or out.max() > 1 or not np.allclose(out.sum(axis=1), 1.0, atol=1e-6):
            problems.append("outputs are not valid probabilities (range [0,1], rows sum to 1)")
    return {"ok": not problems, "problems": problems, "seconds": secs, "predictions": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=url_for(MODEL_PORT))
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    r = run(args.url)
    if r["ok"]:
        print(f"SMOKE TEST PASS  ({r['seconds'] * 1000:.0f} ms for 5 rows)")
        print("first row [P(<=50K), P(>50K)] =", np.round(r["predictions"][0], 4).tolist())
    else:
        print("SMOKE TEST FAIL")
        for prob in r["problems"]:
            print("  -", prob)
        sys.exit(1)


if __name__ == "__main__":
    main()
