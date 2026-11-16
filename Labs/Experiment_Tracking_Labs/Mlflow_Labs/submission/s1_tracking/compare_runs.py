"""Leaderboard of every tracked model + sampler comparison, built purely from MLflow queries.

    python -m s1_tracking.compare_runs

Outputs (gitignored): reports/leaderboard.md, reports/sampler_comparison.png
Teaches: mlflow.search_runs filter/order syntax, parent/child run traversal, and why "compare
runs" must also compare cost (time, size) and overfitting gap, not just one metric.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

import config

REPORTS = config.ROOT / "reports"
COLS = {
    "tags.mlflow.runName": "run",
    "tags.model_family": "family",
    "tags.sampler": "sampler",
    "metrics.val_auc": "val_auc",
    "metrics.train_auc": "train_auc",
    "metrics.val_f1": "val_f1",
    "metrics.val_log_loss": "val_logloss",
    "metrics.train_seconds": "fit_s",
    "tags.git_dirty": "git_dirty",
    "tags.data_version": "data",
}


def top_level_runs() -> pd.DataFrame:
    """Parents + baseline only: trials are children and would bury the leaderboard."""
    df = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",  # MLflow filters support AND only: no OR / IN
        order_by=["metrics.val_auc DESC"],
    )
    df = df[df["tags.stage"].isin(["baseline", "candidate"])]  # so stage selection happens in pandas
    out = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    out["overfit_gap"] = (out["train_auc"] - out["val_auc"]).round(4)
    return out


def trial_scores(parent_run_id: str) -> list:
    kids = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["attributes.start_time ASC"],
    )
    return kids["metrics.val_auc"].tolist()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    REPORTS.mkdir(exist_ok=True)

    board = top_level_runs()
    fmt = board.copy()
    for c in ("val_auc", "train_auc", "val_f1", "val_logloss", "fit_s"):
        if c in fmt:
            fmt[c] = fmt[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    print(fmt.to_string(index=False))
    try:
        text = fmt.to_markdown(index=False)  # needs the optional `tabulate` package
    except ImportError:
        text = fmt.to_string(index=False)
    (REPORTS / "leaderboard.md").write_text(text)

    # ---- sampler comparison: same model, same budget, different search strategy ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, family in zip(axes, ["gradient_boosting", "neural_net"]):
        for sampler, color in [("random", "tab:gray"), ("tpe", "tab:blue")]:
            sub = board[(board["family"] == family) & (board["sampler"] == sampler)]
            if sub.empty:
                continue
            parent_id = mlflow.search_runs(
                experiment_names=[config.EXPERIMENT_NAME],
                filter_string=f"tags.model_family = '{family}' and tags.sampler = '{sampler}' and tags.stage = 'candidate'",
                order_by=["attributes.start_time DESC"], max_results=1,
            ).iloc[0]["run_id"]
            scores = trial_scores(parent_id)
            ax.plot(pd.Series(scores).cummax(), color=color, label=f"{sampler} (best so far)")
            ax.plot(scores, "o", color=color, alpha=0.3)
        ax.set_title(family); ax.set_xlabel("trial"); ax.set_ylabel("val_auc"); ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "sampler_comparison.png", dpi=120)
    print(f"\nwrote {REPORTS / 'leaderboard.md'} and sampler_comparison.png")


if __name__ == "__main__":
    main()
