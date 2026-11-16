"""Lab 1 parity: autologging vs manual logging, measured instead of asserted.

Lab 1's starter notebook shows `mlflow.autolog()` needs 1 call where manual logging needs ~6. The
useful question is what you LOSE. This trains the same baseline pipeline with sklearn autolog in a
separate experiment, then diffs what each approach recorded against our manual baseline run.

    python -m s1_tracking.autolog_demo

Output: reports/autolog_vs_manual.md (also printed). Uses its own experiment so the real
leaderboard stays clean.
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import config
import data
from features import build_preprocessor
from s1_tracking.common import setup_mlflow

DEMO_EXPERIMENT = "adult-income-autolog-demo"


def summarize(run, client, X_probe) -> dict:
    """What a run recorded, as comparable facts. Includes a REAL test of the served interface."""
    d = run.data
    metrics = {k: v for k, v in d.metrics.items() if not k.startswith("system/")}  # CPU/RAM noise
    arts = {a.path for a in client.list_artifacts(run.info.run_id)}
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    mine = [m for m in models if m.source_run_id == run.info.run_id]
    sig, served = None, "no model saved"
    if mine:
        uri = f"models:/{mine[0].model_id}"
        sig = mlflow.models.get_model_info(uri).signature is not None
        out = np.asarray(mlflow.pyfunc.load_model(uri).predict(X_probe))
        served = (f"labels {sorted(set(out.ravel().tolist()))}" if out.ndim == 1 and set(out.tolist()) <= {0, 1}
                  else f"array shape {out.shape}, values in [{out.min():.2f}, {out.max():.2f}]")
    return {
        "params logged": len(d.params),
        "metrics logged (excl. system)": len(metrics),
        "has a validation metric": any(k.startswith("val_") for k in metrics),
        "model saved": bool(mine),
        "model signature": sig,
        "dataset lineage (log_input)": len(run.inputs.dataset_inputs) > 0,
        "data_version tag": "data_version" in d.tags,
        "git_dirty tag": "git_dirty" in d.tags,
        "data profile artifact": "data" in arts,
        "plots (roc / confusion)": "val_plots" in arts,
        "what pyfunc.predict returns": served,
    }


def main() -> None:
    setup_mlflow(DEMO_EXPERIMENT)
    _, _, train, val, _ = data.get_splits()
    X_train, y_train = data.xy(train)
    X_val, y_val = data.xy(val)
    num, cat = data.feature_columns(X_train)

    # ONE line replaces the manual params/metrics/model calls. log_datasets/input examples are opt-in.
    mlflow.sklearn.autolog(log_input_examples=True, log_post_training_metrics=True)

    pipe = Pipeline([("prep", build_preprocessor(num, cat)), ("model", LogisticRegression(max_iter=1000, random_state=config.SEED))])
    with mlflow.start_run(run_name="baseline-logreg-autolog") as run:
        pipe.fit(X_train, y_train)
        # Post-training metrics: autolog captures metric-function calls on arrays it can trace.
        roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    mlflow.sklearn.autolog(disable=True)

    client = mlflow.MlflowClient()
    auto = client.get_run(run.info.run_id)
    manual = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string="tags.stage = 'baseline' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=1, output_format="list",
    )[0]

    probe = X_val.head(200)
    a, m = summarize(auto, client, probe), summarize(client.get_run(manual.info.run_id), client, probe)
    table = pd.DataFrame({"autolog (1 call)": a, "manual (train_baseline.py)": m})
    print(table.to_string())

    print("\nautolog metric names:", sorted(auto.data.metrics))
    print("autolog param names (first 8):", sorted(auto.data.params)[:8])

    out = config.ROOT / "reports"
    out.mkdir(exist_ok=True)
    try:
        text = table.to_markdown()
    except ImportError:
        text = table.to_string()
    (out / "autolog_vs_manual.md").write_text(text)
    print(f"\nwrote {out / 'autolog_vs_manual.md'}")


if __name__ == "__main__":
    main()
