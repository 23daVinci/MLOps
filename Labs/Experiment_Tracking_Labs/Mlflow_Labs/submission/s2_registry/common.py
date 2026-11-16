"""Registry helpers shared by register / promote / report / promote_to_prod."""
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.models import Model

import config


def init() -> MlflowClient:
    """Point MLflow at the server (tracking AND registry) and return a client."""
    sys.stdout.reconfigure(encoding="utf-8")  # MLflow prints emoji; cp1252 consoles can't
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_registry_uri(config.TRACKING_URI)
    return MlflowClient()


def alias_version(client, alias: str, name: str = config.REGISTERED_MODEL_NAME):
    """The ModelVersion an alias points to, or None if the alias does not exist."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None


def aliases_by_version(client, name: str = config.REGISTERED_MODEL_NAME) -> dict:
    """{version: [aliases]}. Read from the registered model: search_model_versions() omits aliases."""
    out: dict = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        out.setdefault(str(version), []).append(alias)
    return out


def source_run(client, mv):
    """The training run behind a version (the system of record for its metrics and git info)."""
    return client.get_run(mv.run_id) if mv.run_id else None


def run_metric(client, mv, key: str = "val_auc") -> float:
    """Read metrics from the SOURCE RUN, not from tags copied at registration time."""
    return client.get_run(mv.run_id).data.metrics[key]


def model_size_bytes(name: str, version) -> int:
    """Size recorded in the version's MLmodel file (ModelInfo.model_size_bytes is empty in 3.18)."""
    return Model.load(f"models:/{name}/{version}").model_size_bytes or 0


def set_status(client, mv, status: str, note: str, name: str = config.REGISTERED_MODEL_NAME) -> None:
    client.set_model_version_tag(name, mv.version, "validation_status", status)
    client.set_model_version_tag(name, mv.version, "validation_note", note[:250])


def record_failure(client, mv, status: str, note: str, name: str = config.REGISTERED_MODEL_NAME) -> None:
    """Record a failed check WITHOUT rewriting history.
    A never-validated version becomes `status` (failed / rejected). A version that already passed
    earlier keeps `passed`: losing a later contest makes it superseded, not invalid. In that case we
    only note the attempt in a separate tag. (Overloading validation_status for both meanings was a
    real bug in an earlier version of this code.)"""
    if mv.tags.get("validation_status", "pending") == "pending":
        set_status(client, mv, status, note, name)
    else:
        client.set_model_version_tag(name, mv.version, "last_promotion_attempt", f"{status}: {note}"[:250])
