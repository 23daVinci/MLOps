"""Registry step 1: turn tracked candidates into registered, documented model versions.

Registers the latest finished baseline run and the latest finished gbm-search run under ONE
registered model name. Idempotent: a run that already has a version is skipped.
No aliases are set here -- promotion is a separate, gated step (promote.py).
"""
import sys

import mlflow
from mlflow import MlflowClient

import config

MODEL_DESCRIPTION = """\
Predicts whether an adult's annual income exceeds $50K (UCI Adult Census Income).

Input: 13 raw features (age, workclass, education, ...). Preprocessing (imputation, scaling,
one-hot encoding) is part of the model pipeline -- send raw values.
Output: 2 columns [P(income <= 50K), P(income > 50K)].

Intended use: learning/demo pipeline. NOT for real credit, hiring or housing decisions.
Known limits: census data from 1994; contains sensitive attributes (sex, race, native-country);
class imbalance (~24% positive). Fairness slices are evaluated in the evaluation stage.
Split: stratified 60/20/20 train/val/test; versions are compared on validation, test is held out.
"""


def latest_finished_run(stage: str):
    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string=f"tags.stage = '{stage}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        sys.exit(f"No finished run with stage='{stage}'. Train it first.")
    return runs[0]


def logged_model_id(run_id: str) -> str:
    models = [
        m
        for m in mlflow.search_logged_models(
            experiment_ids=[mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id],
            output_format="list",
        )
        if m.source_run_id == run_id
    ]
    if not models:
        sys.exit(f"Run {run_id} has no logged model.")
    return models[0].model_id


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_registry_uri(config.TRACKING_URI)
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME

    # ---- the registered model: a name + model card ------------------------
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            name,
            description=MODEL_DESCRIPTION,
            tags={"task": "binary-classification", "dataset": "uci-adult", "owner": "mlops-course"},
        )
        print(f"created registered model '{name}'")

    already = {v.run_id for v in client.search_model_versions(f"name='{name}'")}

    # ---- one version per candidate ----------------------------------------
    for stage in ["baseline", "candidate"]:
        run = latest_finished_run(stage)
        if run.info.run_id in already:
            print(f"[{stage}] run {run.info.run_id[:8]} already registered, skipping")
            continue

        model_id = logged_model_id(run.info.run_id)
        mv = mlflow.register_model(f"models:/{model_id}", name)
        m = run.data.metrics
        client.update_model_version(
            name,
            mv.version,
            description=(
                f"{run.data.tags.get('model_family')} model from run '{run.data.tags['mlflow.runName']}'. "
                f"val AUC {m['val_auc']:.4f}, val F1 {m['val_f1']:.4f}."
            ),
        )
        for key, value in {
            "model_family": run.data.tags.get("model_family"),
            "data_version": run.data.tags.get("data_version"),
            "git_commit": run.data.tags.get("mlflow.source.git.commit", "unknown"),
            "val_auc": f"{m['val_auc']:.4f}",
            "validation_status": "pending",  # promote.py flips this after the gate
        }.items():
            client.set_model_version_tag(name, mv.version, key, value)
        print(f"[{stage}] registered version {mv.version} <- run {run.info.run_id[:8]} (val_auc {m['val_auc']:.4f})")


if __name__ == "__main__":
    main()
