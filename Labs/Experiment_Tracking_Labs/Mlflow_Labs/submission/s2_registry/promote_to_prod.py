"""Environment promotion: copy a validated version into the PRODUCTION registered model.

    python -m s2_registry.promote_to_prod [--version N] [--allow-unreproducible]

Pattern (MLflow's recommended way to model environments): one registered model per environment.
  adult-income-classifier        <- everything we train and evaluate ("staging")
  adult-income-classifier-prod   <- only versions explicitly approved to serve customers
Production access can then be locked down separately, and prod's version history is a clean audit
trail of what was ever approved. `copy_model_version` copies the model files, so prod no longer
depends on the staging artifacts.

Preconditions (refused otherwise): status 'passed', not deprecated, and the source run was not
trained from a known-dirty working tree (production must be reproducible from git).
"""
import argparse
import sys

import mlflow
import numpy as np

import config
import data
from s2_registry.common import alias_version, init

SRC, DST = config.REGISTERED_MODEL_NAME, config.PROD_REGISTERED_MODEL_NAME


def ensure_prod_model(client) -> None:
    try:
        client.get_registered_model(DST)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            DST,
            description=f"Production-approved versions of '{SRC}'. Versions here are COPIES of staging "
                        f"versions that passed all gates; see tags source_version / source_run_id.",
            tags={"environment": "production", "source_model": SRC},
        )
        print(f"created registered model '{DST}'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", help="staging version to promote (default: current champion)")
    p.add_argument("--allow-unreproducible", action="store_true",
                   help="allow a version trained from a known-dirty git tree (not recommended)")
    args = p.parse_args()

    client = init()
    mv = (client.get_model_version(SRC, args.version) if args.version
          else alias_version(client, config.CHAMPION_ALIAS))
    if mv is None:
        sys.exit("no champion in staging and no --version given")

    # ---- preconditions ----------------------------------------------------
    t = mv.tags
    if t.get("validation_status") != "passed":
        sys.exit(f"REFUSING: v{mv.version} status is {t.get('validation_status')!r}, not 'passed'")
    if t.get("deprecated") == "true":
        sys.exit(f"REFUSING: v{mv.version} is deprecated")
    dirty = t.get("git_dirty", "unknown")
    if dirty == "true" and not args.allow_unreproducible:
        sys.exit(f"REFUSING: v{mv.version} was trained from an uncommitted working tree; commit "
                 f"{t.get('git_commit', '?')[:7]} cannot reproduce it. Retrain from a clean tree.")
    if dirty == "unknown":
        print(f"WARNING: git state at training time of v{mv.version} is unknown (predates the git_dirty tag)")

    ensure_prod_model(client)

    # ---- copy (idempotent: one prod version per staging version) ----------------
    existing = [v for v in client.search_model_versions(f"name='{DST}'")
                if v.tags.get("source_version") == mv.version and v.tags.get("source_model") == SRC]
    if existing:
        prod_mv = existing[0]
        print(f"staging v{mv.version} is already prod v{prod_mv.version}; not copying again")
    else:
        prod_mv = client.copy_model_version(f"models:/{SRC}/{mv.version}", DST)
        for key, value in {
            "source_model": SRC, "source_version": mv.version, "source_run_id": mv.run_id,
            "model_family": t.get("model_family"), "val_auc": t.get("val_auc"),
            "data_version": t.get("data_version"), "git_commit": t.get("git_commit"), "git_dirty": dirty,
            "validation_status": "passed",
        }.items():
            if value is not None:
                client.set_model_version_tag(DST, prod_mv.version, key, str(value))
        print(f"copied staging v{mv.version} -> prod v{prod_mv.version}")

    # ---- alias: same previous-champion pattern as staging ---------------------------
    current = alias_version(client, config.CHAMPION_ALIAS, DST)
    if current is None or current.version != prod_mv.version:
        if current is not None:
            client.set_registered_model_alias(DST, config.PREVIOUS_ALIAS, current.version)
        client.set_registered_model_alias(DST, config.CHAMPION_ALIAS, prod_mv.version)
    print(f"-> {DST}@{config.CHAMPION_ALIAS} = v{prod_mv.version}")

    # ---- verify the COPY behaves exactly like the original ---------------------------
    _, _, _, val, _ = data.get_splits()
    X, _ = data.xy(val)
    X = X.head(500)
    a = np.asarray(mlflow.pyfunc.load_model(f"models:/{SRC}/{mv.version}").predict(X))
    b = np.asarray(mlflow.pyfunc.load_model(f"models:/{DST}@{config.CHAMPION_ALIAS}").predict(X))
    if not np.array_equal(a, b):
        sys.exit(f"FAIL: prod copy differs from staging (max abs diff {np.abs(a - b).max():.2e})")
    print(f"verified: prod copy predicts identically to staging v{mv.version} on {len(X)} rows")


if __name__ == "__main__":
    main()
