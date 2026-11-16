"""Registry step 1: turn tracked candidates into registered, documented model versions.

    python -m s2_registry.register_models [--family neural_net] [--dry-run]

Policy (a registry is a catalog, not a dump of every run):
  * candidates = the best finished top-level run of each model FAMILY (linear / gradient_boosting /
    neural_net), best by val_auc, ties broken by most recent;
  * a family with no registered version is always registered;
  * a family that already has one is only registered again if the new run beats the best registered
    version of that family by MIN_REGISTER_GAIN -- otherwise it is skipped with the reason printed;
  * idempotent: a run that already has a version is never registered twice.
No aliases are set here. Promotion is a separate, gated step (promote.py).
"""
import argparse
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config
from s2_registry.common import init, model_size_bytes, run_metric

FAMILIES = ["linear", "gradient_boosting", "neural_net"]

MODEL_DESCRIPTION = """\
Predicts whether an adult's annual income exceeds $50K (UCI Adult Census Income).

Input: 13 raw features (age, workclass, education, ...). Preprocessing (imputation, scaling,
one-hot encoding / embeddings) is part of the model -- send raw values; missing values are allowed
for workclass, occupation and native-country.
Output: 2 columns [P(income <= 50K), P(income > 50K)].

Versions may be different algorithms (tag model_family): linear, gradient_boosting, neural_net.
Aliases: champion = serving, previous-champion = rollback target, challenger = under evaluation.

Intended use: learning/demo pipeline. NOT for real credit, hiring or housing decisions.
Known limits: census data from 1994; contains sensitive attributes (sex, race, native-country);
class imbalance (~24% positive).
Split: stratified 60/20/20 train/val/test; versions are compared on validation, test is held out
for the evaluation stage.
"""


def best_run_per_family() -> dict:
    """{family: best finished top-level run (a pandas row)}. Trials are children and are excluded
    by requiring a stage tag of baseline/candidate (filters can't express IN, so pandas does it)."""
    df = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_auc DESC", "attributes.start_time DESC"],
    )
    df = df[df["tags.stage"].isin(["baseline", "candidate"]) & df["tags.model_family"].notna()]
    return {fam: g.iloc[0] for fam, g in df.groupby("tags.model_family")}


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


def decide(client, family: str, run, versions: list) -> tuple[bool, str]:
    """Should this family's best run get a new version? Returns (register?, reason)."""
    if any(v.run_id == run.run_id for v in versions):
        return False, "this run is already registered"
    same_family = [v for v in versions if v.tags.get("model_family") == family]
    if not same_family:
        return True, "first version of this family"
    best_registered = max(run_metric(client, v) for v in same_family)
    gain = run["metrics.val_auc"] - best_registered
    if gain >= config.MIN_REGISTER_GAIN:
        return True, f"beats best registered {family} by {gain:+.4f} (>= {config.MIN_REGISTER_GAIN})"
    return False, f"gain over best registered {family} is {gain:+.4f} (< {config.MIN_REGISTER_GAIN}): not worth a new version"


def register(client, name: str, family: str, run) -> str:
    model_id = logged_model_id(run.run_id)
    mv = mlflow.register_model(f"models:/{model_id}", name)
    size = model_size_bytes(name, mv.version)
    auc, f1 = run["metrics.val_auc"], run.get("metrics.val_f1", float("nan"))
    client.update_model_version(
        name, mv.version,
        description=(
            f"{family} model from run '{run['tags.mlflow.runName']}'"
            f"{' (' + run['tags.sampler'] + ' search)' if pd.notna(run.get('tags.sampler')) else ''}. "
            f"val AUC {auc:.4f}, val F1 {f1:.4f}, {size / 1e6:.2f} MB."
        ),
    )
    tags = {
        "model_family": family,
        "sampler": run.get("tags.sampler"),
        "data_version": run.get("tags.data_version"),
        "git_commit": run.get("tags.mlflow.source.git.commit") or "unknown",
        # was the working tree clean when this was trained? 'true' means git alone cannot reproduce it
        "git_dirty": run.get("tags.git_dirty") if pd.notna(run.get("tags.git_dirty")) else "unknown",
        "val_auc": f"{auc:.4f}",
        "model_size_mb": f"{size / 1e6:.3f}",
        "validation_status": "pending",  # promote.py flips this after the gates
    }
    for key, value in tags.items():
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            client.set_model_version_tag(name, mv.version, key, str(value))
    return mv.version


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=FAMILIES, help="only consider this model family")
    p.add_argument("--dry-run", action="store_true", help="print decisions, register nothing")
    args = p.parse_args()

    client = init()
    name = config.REGISTERED_MODEL_NAME
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        if not args.dry_run:
            client.create_registered_model(
                name, description=MODEL_DESCRIPTION,
                tags={"task": "binary-classification", "dataset": "uci-adult", "owner": "mlops-course"},
            )
            print(f"created registered model '{name}'")
    versions = client.search_model_versions(f"name='{name}'")

    best = best_run_per_family()
    for family in [args.family] if args.family else FAMILIES:
        if family not in best:
            print(f"[{family:18s}] no finished run: train it first")
            continue
        run = best[family]
        ok, reason = decide(client, family, run, versions)
        label = f"{run['tags.mlflow.runName']} (val_auc {run['metrics.val_auc']:.4f})"
        if not ok:
            print(f"[{family:18s}] SKIP     {label}: {reason}")
        elif args.dry_run:
            print(f"[{family:18s}] WOULD REGISTER {label}: {reason}")
        else:
            v = register(client, name, family, run)
            print(f"[{family:18s}] REGISTERED v{v}  {label}: {reason}")


if __name__ == "__main__":
    main()
