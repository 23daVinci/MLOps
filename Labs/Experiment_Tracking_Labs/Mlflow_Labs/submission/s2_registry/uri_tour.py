"""Lab 1 parity (loading models by path / runs: / models:) -- measured on MLflow 3.18.

    python -m s2_registry.uri_tour

Loads the SAME champion through every URI form MLflow accepts (plus two that used to work), and for
each one reports: does it load, and WHICH registered version's predictions does it actually give?
That second column is the point: some URIs are floating pointers, and a floating pointer that
resolves to a different version than you assumed is a classic production incident.
"""
import tempfile

import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.models import Model

import config
import data
from s2_registry.common import alias_version, init

NAME = config.REGISTERED_MODEL_NAME


def main() -> None:
    client = init()
    _, _, _, val, _ = data.get_splits()
    X = data.xy(val)[0].head(200)

    versions = sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version))
    out = {v.version: np.asarray(mlflow.pyfunc.load_model(f"models:/{NAME}/{v.version}").predict(X)) for v in versions}
    champion = alias_version(client, config.CHAMPION_ALIAS)
    run_id = champion.run_id
    info = Model.load(f"models:/{NAME}/{champion.version}")
    logged = mlflow.get_logged_model(info.model_id)

    def who(pred) -> str:
        """Which registered version(s) give exactly these predictions?"""
        hits = [f"v{v}" for v, ref in out.items() if ref.shape == pred.shape and np.array_equal(ref, pred)]
        return ",".join(hits) or "no registered version"

    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    local_dir = download_artifacts(artifact_uri=logged.artifact_location, dst_path=tmp.name)

    cases = [
        ("models:/NAME@champion",           f"models:/{NAME}@{config.CHAMPION_ALIAS}", "alias: floats with promotions. Serving default."),
        (f"models:/NAME/{champion.version}", f"models:/{NAME}/{champion.version}", "pinned version: never changes. Tests, audits, prod deploys."),
        ("models:/<model_id>",              f"models:/{info.model_id}", "a specific logged model (MLflow 3); skips the registry."),
        ("local directory path",            local_dir, "downloaded copy; works offline, no server needed."),
        ("mlflow-artifacts:/... location",  logged.artifact_location, "raw artifact-store location of the logged model."),
        ("runs:/<run_id>/model",            f"runs:/{run_id}/model", "Lab 1/2 style; still loads on 3.18. Points at a RUN, not a registry version: bypasses promotion."),
        ("models:/NAME/Production",         f"models:/{NAME}/Production", "Lab 2 stage URI: parses fine, but fails here only because no version holds that stage (we use aliases)."),
        ("models:/NAME/latest",             f"models:/{NAME}/latest", "'latest' = HIGHEST VERSION NUMBER, not the best model."),
    ]
    rows = []
    for label, uri, note in cases:
        try:
            pred = np.asarray(mlflow.pyfunc.load_model(uri).predict(X))
            rows.append((label, "loads", who(pred), note))
        except Exception as e:
            rows.append((label, "FAILS", "-", f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"))
    tmp.cleanup()

    df = pd.DataFrame(rows, columns=["URI form", "result", "predictions equal", "note / error"])
    pd.set_option("display.max_colwidth", 100, "display.width", 250)
    print(f"champion = v{champion.version}; highest version number = v{versions[-1].version}\n")
    print(df.to_string(index=False))

    latest = next(r for r in rows if r[0].endswith("/latest"))
    if latest[1] == "loads" and f"v{champion.version}" not in latest[2]:
        print(f"\n!! 'latest' resolved to {latest[2]}, NOT the champion v{champion.version}: never serve by 'latest'.")


if __name__ == "__main__":
    main()
