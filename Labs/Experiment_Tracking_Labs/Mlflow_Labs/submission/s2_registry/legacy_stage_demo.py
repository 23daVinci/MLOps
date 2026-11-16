"""Lab 2 parity (steps 11-13): the registry STAGE API, run for real against MLflow 3.18.

    python -m s2_registry.legacy_stage_demo

Lab 2 promotes with `transition_model_version_stage(..., stage="Production")` and loads
`models:/name/production`. Stages are deprecated in favour of aliases. Rather than assert that, this
runs the lab's exact steps on a THROWAWAY model ('stage-api-demo-tmp', deleted at the end; it never
touches adult-income-classifier) and records what actually happens, then shows the alias equivalent.
"""
import warnings

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data
from s1_tracking.common import setup_mlflow
from s2_registry.common import init

DEMO_EXPERIMENT = "adult-income-registry-demo"
NAME = "stage-api-demo-tmp"


def tiny_model_versions(client, n: int) -> list:
    _, _, train, _, _ = data.get_splits()
    X, y = data.xy(train)
    X = X[data.feature_columns(X)[0]].head(2000)
    y = y.loc[X.index]
    versions = []
    for i in range(n):
        model = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=10 ** (i - 1), max_iter=300))]).fit(X, y)
        with mlflow.start_run(run_name=f"stage-demo-v{i + 1}"):
            info = mlflow.sklearn.log_model(
                model, name="model", signature=infer_signature(X, model.predict_proba(X)),
                pyfunc_predict_fn="predict_proba", skops_trusted_types=["numpy.dtype"],
            )
        versions.append(mlflow.register_model(f"models:/{info.model_id}", NAME).version)
    return versions, X


def stages(client) -> dict:
    return {f"v{v.version}": v.current_stage for v in client.search_model_versions(f"name='{NAME}'")}


def main() -> None:
    client = init()
    setup_mlflow(DEMO_EXPERIMENT)
    report = []

    def step(label, fn):
        """Run one lab step, capturing the outcome and any deprecation warning it emits."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = fn()
                outcome = f"OK -> {result}" if result is not None else "OK"
            except Exception as e:  # report honestly: the lab step may simply not work any more
                outcome = f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:110]}"
        dep = [str(w.message).splitlines()[0][:130] for w in caught if "deprecat" in str(w.message).lower()]
        report.append((label, outcome, dep[0] if dep else "-"))

    try:
        (v1, v2), X = tiny_model_versions(client, 2)
        print(f"registered two versions of throwaway model '{NAME}': v{v1}, v{v2}\n")

        # Lab 2 step 12
        step("transition v1 -> Staging", lambda: client.transition_model_version_stage(NAME, v1, "Staging").current_stage)
        step("transition v1 -> Production", lambda: client.transition_model_version_stage(NAME, v1, "Production").current_stage)
        # starter.ipynb: promoting v2 and archiving the old production version
        step("v2 -> Production, archive_existing_versions=True",
             lambda: client.transition_model_version_stage(NAME, v2, "Production", archive_existing_versions=True).current_stage)
        step("stages afterwards", lambda: stages(client))
        # Lab 2 step 13
        step("load models:/name/Production (lab step 13)",
             lambda: f"predict -> {np.asarray(mlflow.pyfunc.load_model(f'models:/{NAME}/Production').predict(X.head(3))).shape}")
        # the modern equivalent
        step("set_registered_model_alias(champion -> v2)", lambda: client.set_registered_model_alias(NAME, "champion", v2))
        step("load models:/name@champion (modern)",
             lambda: f"predict -> {np.asarray(mlflow.pyfunc.load_model(f'models:/{NAME}@champion').predict(X.head(3))).shape}")
    finally:
        try:
            client.delete_registered_model(NAME)
            print(f"cleaned up: deleted throwaway registered model '{NAME}'")
        except Exception as e:
            print(f"CLEANUP FAILED for '{NAME}': {e}  (delete it by hand)")

    print()
    for label, outcome, dep in report:
        print(f"* {label}\n    {outcome}\n    deprecation warning: {dep}")


if __name__ == "__main__":
    main()
