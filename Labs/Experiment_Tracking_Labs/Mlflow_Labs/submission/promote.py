"""Registry step 2: gated promotion and rollback via aliases.

    python promote.py <version>     # try to make <version> the champion
    python promote.py --rollback    # point champion back at previous-champion

A version becomes champion only if it passes two gates:
  1. CONTRACT: loaded through the registry (pyfunc) it returns valid probabilities and
     reproduces the AUC that was logged during tracking.
  2. PERFORMANCE: it beats the current champion's validation AUC by PROMOTION_MARGIN
     (the first-ever champion has nothing to beat and only needs gate 1).
The test split is NOT used here; it stays held out for the evaluation stage.
"""
import argparse
import sys
from datetime import datetime, timezone

import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

import config
import data

TOLERANCE = 1e-3


def get_alias_version(client, alias):
    try:
        return client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)
    except mlflow.exceptions.MlflowException:
        return None


def run_val_auc(client, mv) -> float:
    """Read the metric from the SOURCE RUN (system of record), not from a copied tag."""
    return client.get_run(mv.run_id).data.metrics["val_auc"]


def check_contract(mv, X_val, y_val) -> tuple[bool, str]:
    name = config.REGISTERED_MODEL_NAME
    model = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}")  # pinned, by version
    out = np.asarray(model.predict(X_val))
    if out.ndim != 2 or out.shape != (len(X_val), 2):
        return False, f"output shape {out.shape}, expected ({len(X_val)}, 2)"
    if out.min() < 0 or out.max() > 1 or not np.allclose(out.sum(axis=1), 1.0, atol=1e-6):
        return False, "outputs are not valid probabilities"
    auc = roc_auc_score(y_val, out[:, 1])
    logged = run_val_auc(MlflowClient(), mv)
    if abs(auc - logged) > TOLERANCE:
        return False, f"AUC after loading {auc:.4f} != logged {logged:.4f}"
    return True, f"valid probabilities, AUC {auc:.4f} reproduces logged {logged:.4f}"


def set_status(client, mv, status, reason):
    name = config.REGISTERED_MODEL_NAME
    client.set_model_version_tag(name, mv.version, "validation_status", status)
    client.set_model_version_tag(name, mv.version, "validation_note", reason[:250])


def promote(version: str):
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME
    mv = client.get_model_version(name, version)
    champion = get_alias_version(client, config.CHAMPION_ALIAS)

    _, _, _, val, _ = data.get_splits()
    X_val, y_val = data.xy(val)

    ok, msg = check_contract(mv, X_val, y_val)
    print(f"gate 1 (contract):    {'PASS' if ok else 'FAIL'} - {msg}")
    if not ok:
        set_status(client, mv, "failed", f"contract: {msg}")
        sys.exit(1)

    if champion is not None:
        if champion.version == mv.version:
            sys.exit(f"version {version} is already champion")
        cand, base = run_val_auc(client, mv), run_val_auc(client, champion)
        gain = cand - base
        ok = gain >= config.PROMOTION_MARGIN
        msg = f"candidate {cand:.4f} vs champion v{champion.version} {base:.4f} (gain {gain:+.4f}, need >= {config.PROMOTION_MARGIN})"
        print(f"gate 2 (performance): {'PASS' if ok else 'FAIL'} - {msg}")
        if not ok:
            # Don't clobber history: a version that already passed earlier was superseded,
            # not "failed". Only never-validated versions get marked rejected.
            if mv.tags.get("validation_status", "pending") == "pending":
                set_status(client, mv, "rejected", f"performance: {msg}")
            else:
                client.set_model_version_tag(name, mv.version, "last_promotion_attempt", f"rejected: {msg}"[:250])
            sys.exit(1)
        client.set_registered_model_alias(name, config.PREVIOUS_ALIAS, champion.version)
    else:
        print("gate 2 (performance): SKIP - no current champion (bootstrap)")

    client.set_registered_model_alias(name, config.CHAMPION_ALIAS, mv.version)
    set_status(client, mv, "passed", msg)
    client.set_model_version_tag(name, mv.version, "promoted_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"-> '{config.CHAMPION_ALIAS}' now points to version {mv.version}"
          + (f" (previous: v{champion.version})" if champion else ""))


def rollback():
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME
    current = get_alias_version(client, config.CHAMPION_ALIAS)
    previous = get_alias_version(client, config.PREVIOUS_ALIAS)
    if current is None or previous is None:
        sys.exit("Nothing to roll back to (need both a champion and a previous-champion).")
    client.set_registered_model_alias(name, config.CHAMPION_ALIAS, previous.version)
    client.set_registered_model_alias(name, config.PREVIOUS_ALIAS, current.version)  # allows undoing the rollback
    # A rollback is an event, not a validation result: record it separately and leave
    # validation_status (did it pass its gates?) untouched.
    client.set_model_version_tag(
        name, current.version, "last_rollback_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    print(f"rolled back: '{config.CHAMPION_ALIAS}' v{current.version} -> v{previous.version} "
          f"('{config.PREVIOUS_ALIAS}' now v{current.version})")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_registry_uri(config.TRACKING_URI)
    p = argparse.ArgumentParser()
    p.add_argument("version", nargs="?", help="model version to promote")
    p.add_argument("--rollback", action="store_true")
    args = p.parse_args()
    if args.rollback:
        rollback()
    elif args.version:
        promote(args.version)
    else:
        p.error("give a version or --rollback")


if __name__ == "__main__":
    main()
