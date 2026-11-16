"""Tracking hygiene shared by every training script.

Why this exists: earlier runs recorded a git SHA for code that was not committed, so the SHA could
not reproduce the run. These helpers make that visible (git_dirty) and add the other things every
industry run should carry: a data profile, system metrics and training time.
"""
import subprocess
import sys
import time
from contextlib import contextmanager

import mlflow
import pandas as pd

import config


def setup_mlflow(experiment: str = config.EXPERIMENT_NAME) -> None:
    """Point at the tracking server, select the experiment, turn on system metrics."""
    # MLflow prints an emoji when a run ends; the default Windows console (cp1252) can't.
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_registry_uri(config.TRACKING_URI)
    mlflow.set_experiment(experiment)
    try:  # CPU / memory time series per run; needs psutil
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=config.ROOT, check=True
    ).stdout.strip()


def git_tags() -> dict:
    """MLflow already records the commit SHA. It does NOT say whether the working tree matched it.
    A dirty tree means the SHA alone cannot reproduce this run."""
    try:
        # only files under submission/ matter for reproducing this pipeline
        dirty_files = [l for l in _git("status", "--porcelain", "--", ".").splitlines() if l.strip()]
        return {
            "git_dirty": str(bool(dirty_files)).lower(),
            "git_dirty_files": str(len(dirty_files)),
        }
    except Exception:
        return {"git_dirty": "unknown"}


def profile(df: pd.DataFrame) -> dict:
    """Compact, loggable data profile: shape, label rate and per-column stats."""
    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype), "null_rate": round(float(s.isna().mean()), 4), "n_unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            info.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 4))
        else:
            info["top"] = s.value_counts().head(5).to_dict()
        cols[c] = info
    return {"n_rows": len(df), "columns": cols}


def log_run_context(data_version: str, train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Tags + data profile that every training run should carry."""
    mlflow.set_tags({"data_version": data_version, **git_tags()})
    mlflow.log_dict({"train": profile(train), "val": profile(val)}, "data/profile.json")


@contextmanager
def timed(metric_name: str = "train_seconds"):
    """Log wall-clock time of the enclosed block as a metric (cost is part of model quality)."""
    t0 = time.perf_counter()
    yield
    mlflow.log_metric(metric_name, round(time.perf_counter() - t0, 3))
