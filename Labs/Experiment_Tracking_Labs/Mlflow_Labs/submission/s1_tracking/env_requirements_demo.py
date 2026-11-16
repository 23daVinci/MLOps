"""Lab 1 parity (serving.py): control the pip requirements a model is saved with.

Lab 1 demonstrates five ways to specify a model's environment and asserts what lands in the saved
`requirements.txt` / `constraints.txt`. Ported to MLflow 3 + Windows:
  * Lab code used NamedTemporaryFile and then reopened it: that fails on Windows while the file is
    still open. We use delete=False and clean up ourselves.
  * log_model(artifact_path=...) is deprecated; we use name=.
  * we also show mlflow.pyfunc.get_model_dependencies (the Lab 1 'troubleshooting' tool).

    python -m s1_tracking.env_requirements_demo
"""
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data
from s1_tracking.common import setup_mlflow

DEMO_EXPERIMENT = "adult-income-env-demo"
SKLEARN_REQ = f"scikit-learn=={sklearn.__version__}"
NUMPY_REQ = f"numpy=={np.__version__}"


def lines(path: str) -> list:
    with open(path) as f:
        return [l.strip() for l in f.read().splitlines() if l.strip()]


def requirements_of(info, constraints: bool = False) -> list:
    """`models:/m-<id>/requirements.txt` does NOT work: MLflow 3 reads `m-<id>` as a registered-model
    NAME once a sub-path follows. Ask the logged model for its real artifact location instead."""
    name = "constraints.txt" if constraints else "requirements.txt"
    root = mlflow.get_logged_model(info.model_id).artifact_location
    return lines(download_artifacts(artifact_uri=f"{root}/{name}"))


def tmp_file(content: str, suffix: str) -> str:
    """Windows-safe temp file: closed before anyone else opens it."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def main() -> None:
    setup_mlflow(DEMO_EXPERIMENT)
    _, _, train, _, _ = data.get_splits()
    X, y = data.xy(train)
    X = X[data.feature_columns(X)[0]].head(3000)  # numeric columns only: tiny, fast model
    y = y.loc[X.index]
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=500))]).fit(X, y)
    sig = infer_signature(X, model.predict_proba(X))
    results = []

    def log(label, **kw):
        return mlflow.sklearn.log_model(
            model, name=label, signature=sig, skops_trusted_types=["numpy.dtype"], **kw
        )

    req_file = tmp_file(SKLEARN_REQ, ".requirements.txt")
    con_file = tmp_file(SKLEARN_REQ, ".constraints.txt")
    try:
        with mlflow.start_run(run_name="env-requirements-variants"):
            # 1. default: MLflow infers requirements from the flavor
            info = log("default")
            reqs = requirements_of(info)
            assert SKLEARN_REQ in reqs, reqs
            results.append(("default (inferred)", True, f"{len(reqs)} pins incl. {SKLEARN_REQ}"))

            # 2. pip_requirements REPLACES the inferred set
            info = log("pip_requirements", pip_requirements=[NUMPY_REQ])
            reqs = requirements_of(info)
            assert NUMPY_REQ in reqs and SKLEARN_REQ not in reqs, reqs
            results.append(("pip_requirements=[numpy]", True, f"replaced; sklearn pin absent: {reqs}"))

            # 3. extra_pip_requirements ADDS to the inferred set
            info = log("extra_pip_requirements", extra_pip_requirements=[NUMPY_REQ])
            reqs = requirements_of(info)
            assert {SKLEARN_REQ, NUMPY_REQ} <= set(reqs), reqs
            results.append(("extra_pip_requirements=[numpy]", True, "added on top of inferred set"))

            # 4. a requirements FILE path
            info = log("requirements_file_path", pip_requirements=req_file)
            reqs = requirements_of(info)
            assert SKLEARN_REQ in reqs, reqs
            results.append(("pip_requirements=<file path>", True, f"read from file: {reqs}"))

            # 5. list mixing a pin and an include of a file
            info = log("requirements_file_list", pip_requirements=[NUMPY_REQ, f"-r {req_file}"])
            reqs = requirements_of(info)
            assert {NUMPY_REQ, SKLEARN_REQ} <= set(reqs) or any(r.startswith("-r") for r in reqs), reqs
            results.append(("pip_requirements=[pin, '-r file']", True, f"{reqs}"))

            # 6. a constraints file (-c): limits versions WITHOUT adding packages
            info = log("constraints_file", pip_requirements=[NUMPY_REQ, f"-c {con_file}"])
            reqs, cons = requirements_of(info), requirements_of(info, constraints=True)
            assert "-c constraints.txt" in reqs and cons == [SKLEARN_REQ], (reqs, cons)
            results.append(("pip_requirements=[pin, '-c file']", True, f"requirements={reqs}, constraints={cons}"))

            # 7. the troubleshooting tool from Lab 1's serving notebook
            dep_path = mlflow.pyfunc.get_model_dependencies(info.model_uri)
            results.append(("get_model_dependencies(uri)", True, f"-> {os.path.basename(dep_path)}: {lines(dep_path)[:3]}"))
    finally:
        for p in (req_file, con_file):
            os.remove(p)

    print(pd.DataFrame(results, columns=["variant", "assert passed", "what MLflow saved"]).to_string(index=False))


if __name__ == "__main__":
    main()
