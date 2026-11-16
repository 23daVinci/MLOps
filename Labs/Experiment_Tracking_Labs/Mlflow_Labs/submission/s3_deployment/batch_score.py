"""Batch scoring: apply a registered model to a large CSV, chunk by chunk, with row-level validation.

    python -m s3_deployment.batch_score make-sample --rows 20000 --bad 60
    python -m s3_deployment.batch_score run [--input data/batch_input.csv] [--chunk-size 5000]
                                            [--model-uri models:/adult-income-classifier@champion]
                                            [--max-quarantine-rate 0.05]

Replaces Lab 2's `mlflow.pyfunc.spark_udf` (steps 14-15) without Spark: same job (apply a registered
model to a big table), plus what a production batch job needs:
  * the alias is resolved ONCE at start and the resolved VERSION is stamped on every output row, so a
    promotion mid-run cannot mix two models in one file, and any prediction is traceable;
  * rows are validated BEFORE scoring (the model server does not check semantics: age=250 is scored);
    bad rows go to a quarantine file with the reason instead of crashing the job or being scored;
  * chunked reading keeps memory flat; an empty chunk is skipped (an empty batch crashes the model);
  * a quarantine-rate tripwire fails the job if too many rows are bad (upstream data is probably broken);
  * the run is tracked in MLflow ('adult-income-inference'): counts, throughput, score distribution.
"""
import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import config
import data
from s1_tracking.common import setup_mlflow
from s2_registry.common import init

NAME = config.REGISTERED_MODEL_NAME
DEFAULT_INPUT = config.DATA_DIR / "batch_input.csv"
FEATURES = [c for c in data.EXPECTED_COLUMNS if c != config.TARGET]
NUMERIC = list(data.NUMERIC_RANGES)
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


# ------------------------------------------------------------------ sample ---
def make_sample(rows: int, bad: int, out) -> None:
    """A realistic input file from validation rows (no labels), with `bad` rows deliberately corrupted."""
    _, _, _, val, _ = data.get_splits()
    X, _ = data.xy(val)
    X = X.sample(n=min(rows, len(X)), random_state=21).reset_index(drop=True)
    rng = np.random.default_rng(5)
    X = X.astype(object)
    corruptions = [("age", 250), ("age", "n/a"), ("hours-per-week", -3), ("capital-gain", 1e9), ("education-num", 99)]
    for i, idx in enumerate(rng.choice(len(X), size=min(bad, len(X)), replace=False)):
        col, val_ = corruptions[i % len(corruptions)]
        X.loc[idx, col] = val_
    X.loc[rng.integers(0, len(X)), "workclass"] = "Astronaut"  # unseen category: a warning, not an error
    out.parent.mkdir(exist_ok=True)
    X.to_csv(out, index=False)
    print(f"wrote {out}: {len(X)} rows, {bad} corrupted (+1 unseen category, plus natural missing values)")


# ----------------------------------------------------------------- validate ---
def validate_chunk(chunk: pd.DataFrame, known: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split a RAW (all-strings) chunk into (clean typed rows, quarantined rows with reasons, stats)."""
    reasons = pd.Series("", index=chunk.index, dtype=object)
    typed = chunk[FEATURES].copy()

    for col in CATEGORICAL:  # empty / '?' mean missing, same convention as training (data.clean)
        typed[col] = typed[col].replace({"": np.nan, "?": np.nan})
    for col in NUMERIC:
        raw = chunk[col]
        num = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        bad_parse = raw.ne("") & num.isna()  # non-empty but not a number, e.g. 'n/a'
        lo, hi = data.NUMERIC_RANGES[col]
        out_of_range = num.notna() & ((num < lo) | (num > hi))
        reasons[bad_parse] += f"{col} unparseable; "
        reasons[out_of_range] += f"{col} out of range [{lo}, {hi}]; "
        typed[col] = num.astype("float64")

    bad = reasons != ""
    unseen = sum(int((typed[c].notna() & ~typed[c].isin(known[c])).sum()) for c in CATEGORICAL)
    stats = {
        "unseen_category_cells": unseen,
        "rows_with_missing": int(typed.isna().any(axis=1).sum()),
    }
    quarantined = chunk[bad].copy()
    quarantined.insert(0, "reason", reasons[bad].str.rstrip("; "))
    return typed[~bad], quarantined, stats


# --------------------------------------------------------------------- run ---
def run(args) -> int:
    client = init()
    src = args.input
    if not src.exists():
        sys.exit(f"{src} not found; create one with: python -m s3_deployment.batch_score make-sample")
    head = pd.read_csv(src, nrows=0)
    missing = [c for c in FEATURES if c not in head.columns]
    if missing:  # a header problem poisons EVERY row: fail the whole job, don't quarantine 100%
        sys.exit(f"input is missing required columns {missing}; refusing to run")

    # ---- resolve the model ONCE and pin it for the whole job --------------------------------------
    uri = args.model_uri
    if "@" in uri:
        name, alias = uri.removeprefix("models:/").split("@")
        version = client.get_model_version_by_alias(name, alias).version
    else:
        name, version = uri.removeprefix("models:/").split("/")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    print(f"model: {uri} resolved to v{version} (pinned for this whole job)")

    known = {c: set(data.clean(data.load_raw())[c].dropna().unique()) for c in CATEGORICAL}  # categories seen in training data
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    out_path = args.output or config.DATA_DIR / f"predictions_v{version}_{digest}.csv"
    quar_path = out_path.with_name(out_path.stem + "_quarantine.csv")
    out_path.unlink(missing_ok=True)
    quar_path.unlink(missing_ok=True)

    setup_mlflow("adult-income-inference")
    scored_scores, totals, reason_counts = [], {"in": 0, "scored": 0, "quarantined": 0, "unseen": 0, "missing_rows": 0}, {}
    t0 = time.perf_counter()
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with mlflow.start_run(run_name=f"batch-score-v{version}") as run_:
        mlflow.set_tags({"job": "batch_score", "model_version": version, "model_uri_requested": uri})
        mlflow.log_params({"input": src.name, "input_sha256_12": digest, "chunk_size": args.chunk_size,
                           "max_quarantine_rate": args.max_quarantine_rate})
        offset = 0
        for chunk in pd.read_csv(src, chunksize=args.chunk_size, dtype=str, keep_default_na=False, na_values=[]):
            chunk.index = range(offset, offset + len(chunk))
            offset += len(chunk)
            good, quarantined, st = validate_chunk(chunk, known)
            totals["in"] += len(chunk)
            totals["unseen"] += st["unseen_category_cells"]
            totals["missing_rows"] += st["rows_with_missing"]
            if len(quarantined):
                quarantined.index.name = "row_id"
                quarantined.to_csv(quar_path, mode="a", header=not quar_path.exists())
                totals["quarantined"] += len(quarantined)
                for r in quarantined["reason"].str.split("; ").explode():
                    key = r.split(" ")[0] + "_" + ("unparseable" if "unparseable" in r else "out_of_range")
                    reason_counts[key] = reason_counts.get(key, 0) + 1
            if good.empty:  # an empty batch crashes the model (found in edge_cases): never send one
                continue
            p = np.asarray(model.predict(good))[:, 1]
            pd.DataFrame({
                "row_id": good.index, "p_gt_50k": p, "predicted_gt_50k": (p >= 0.5).astype(int),
                "model_version": version, "scored_at": scored_at,
            }).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            totals["scored"] += len(good)
            scored_scores.append(p)

        secs = time.perf_counter() - t0
        scores = np.concatenate(scored_scores) if scored_scores else np.array([])
        q_rate = totals["quarantined"] / max(totals["in"], 1)
        mlflow.log_metrics({
            "rows_in": totals["in"], "rows_scored": totals["scored"], "rows_quarantined": totals["quarantined"],
            "quarantine_rate": q_rate, "rows_with_missing_values": totals["missing_rows"],
            "unseen_category_cells": totals["unseen"], "seconds": secs, "rows_per_second": totals["in"] / secs,
            "mean_score": float(scores.mean()) if len(scores) else float("nan"),
            "predicted_positive_rate": float((scores >= 0.5).mean()) if len(scores) else float("nan"),
            **{f"quarantine__{k}": v for k, v in reason_counts.items()},
        })
        if len(scores):
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(scores, bins=40)
            ax.set_xlabel("P(income > 50K)"); ax.set_title(f"score distribution, v{version}, {len(scores)} rows")
            mlflow.log_figure(fig, "score_distribution.png"); plt.close(fig)
        if quar_path.exists():
            mlflow.log_artifact(str(quar_path), "quarantine")
        tripped = q_rate > args.max_quarantine_rate
        mlflow.set_tag("job_status", "failed_quality_tripwire" if tripped else "ok")

    print(f"\nrows in {totals['in']}  scored {totals['scored']}  quarantined {totals['quarantined']} ({q_rate:.2%})"
          f"  | {secs:.1f}s = {totals['in'] / secs:,.0f} rows/s")
    print(f"quarantine reasons: {reason_counts or 'none'}  | unseen-category cells (scored anyway): {totals['unseen']}"
          f"  | rows with missing values (model imputes): {totals['missing_rows']}")
    print(f"predictions -> {out_path}" + (f"\nquarantine  -> {quar_path}" if quar_path.exists() else ""))
    if tripped:
        print(f"\nTRIPWIRE: quarantine rate {q_rate:.2%} > {args.max_quarantine_rate:.2%}. Predictions were written, but "
              f"do NOT trust this input: something upstream is broken. Exiting with code 3.")
        return 3
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make-sample")
    s.add_argument("--rows", type=int, default=20000)
    s.add_argument("--bad", type=int, default=60)
    s.add_argument("--out", type=lambda x: __import__("pathlib").Path(x), default=DEFAULT_INPUT)
    r = sub.add_parser("run")
    r.add_argument("--input", type=lambda x: __import__("pathlib").Path(x), default=DEFAULT_INPUT)
    r.add_argument("--output", type=lambda x: __import__("pathlib").Path(x), default=None)
    r.add_argument("--model-uri", default=f"models:/{NAME}@{config.CHAMPION_ALIAS}")
    r.add_argument("--chunk-size", type=int, default=5000)
    r.add_argument("--max-quarantine-rate", type=float, default=0.05)
    args = p.parse_args()
    if args.cmd == "make-sample":
        make_sample(args.rows, args.bad, args.out)
    else:
        sys.exit(run(args))


if __name__ == "__main__":
    main()
