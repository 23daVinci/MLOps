"""Course-notes parity: experiment administration (rename, tag, delete, restore, permanent delete).

Runs ONLY on a throwaway experiment it creates itself; it never touches `adult-income`.

    python -m s1_tracking.experiment_admin

Key idea: delete in MLflow is a SOFT delete (lifecycle_stage='deleted'): recoverable until `mlflow gc`
removes it for good. That is why deleting a run is safe to do and why disk usage does not drop.
"""
import subprocess
import sys

import mlflow
from mlflow import MlflowClient

import config
from s1_tracking.common import setup_mlflow

NAME, RENAMED = "admin-demo-tmp", "admin-demo-tmp-renamed"


def stage(client, exp_id):
    return client.get_experiment(exp_id).lifecycle_stage


def main() -> None:
    setup_mlflow(NAME)
    client = MlflowClient()
    exp = client.get_experiment_by_name(NAME)
    eid = exp.experiment_id
    with mlflow.start_run(run_name="demo-run"):
        mlflow.log_metric("x", 1.0)
    print(f"1. created experiment {NAME!r} (id {eid}) with 1 run")

    client.rename_experiment(eid, RENAMED)  # course notes: MlflowClient().rename_experiment
    print(f"2. renamed -> {client.get_experiment(eid).name!r}; id is stable ({eid}): names are labels, ids are identity")

    client.set_experiment_tag(eid, "owner", "mlops-course")
    print(f"3. experiment tag set: {client.get_experiment(eid).tags}")

    client.delete_experiment(eid)
    print(f"4. deleted -> lifecycle_stage={stage(client, eid)!r}; runs still on disk: "
          f"{len(client.search_runs([eid], run_view_type=mlflow.entities.ViewType.ALL))}")

    client.restore_experiment(eid)
    print(f"5. restored -> lifecycle_stage={stage(client, eid)!r}")

    client.delete_experiment(eid)
    # Permanent removal needs direct backend access (the server API has no hard-delete).
    res = subprocess.run(
        [sys.executable, "-m", "mlflow", "gc",
         "--backend-store-uri", f"sqlite:///{config.ROOT / 'mlflow.db'}",
         "--experiment-ids", eid],
        capture_output=True, text=True,
    )
    gone = True
    try:
        client.get_experiment(eid)
        gone = client.get_experiment(eid).lifecycle_stage != "deleted"
    except Exception:
        pass
    print(f"6. `mlflow gc --experiment-ids {eid}` exit={res.returncode}; "
          f"{'permanently removed' if res.returncode == 0 else 'FAILED: ' + res.stderr.strip()[-200:]}")
    if res.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
