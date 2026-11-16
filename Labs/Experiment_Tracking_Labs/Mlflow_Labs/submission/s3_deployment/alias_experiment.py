"""Experiment: does a RUNNING server follow a registry alias change?  (Spoiler in the docstring of serve_local.)

    python -m s3_deployment.alias_experiment

Steps: serve @champion -> roll the alias back -> ask the running server who it is -> restart -> ask again.
Who answers is decided by comparing the server's output with every registered version's predictions
on the same rows, not by trusting any label. Changes the real registry temporarily; a `finally` block
restores both aliases and removes the audit tag the rollback leaves, so nothing leaks into history.
"""
import json
import sys

import mlflow
import numpy as np

import config
from s2_registry import promote
from s2_registry.common import alias_version, init
from s3_deployment.common import MODEL_PORT, post, predictions, sample_features, split_payload, start_model_server, stop_model_server, url_for
from s3_deployment.serve_local import DEFAULT_URI

NAME = config.REGISTERED_MODEL_NAME


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    client = init()
    X = sample_features(200, seed=7)
    body = json.dumps(split_payload(X))

    versions = [v.version for v in client.search_model_versions(f"name='{NAME}'")]
    refs = {v: np.asarray(mlflow.pyfunc.load_model(f"models:/{NAME}/{v}").predict(X)) for v in versions}

    def who(out) -> str:
        hits = [f"v{v}" for v, ref in refs.items() if ref.shape == out.shape and np.allclose(ref, out, atol=1e-9)]
        return ",".join(hits) or "no registered version"

    def ask_server() -> str:
        status, text, _ = post(url_for(MODEL_PORT), body)
        return who(predictions(text)) if status == 200 else f"HTTP {status}"

    def ask_registry_now() -> str:
        return who(np.asarray(mlflow.pyfunc.load_model(DEFAULT_URI).predict(X)))  # a NEW load re-resolves the alias

    champ0, prev0 = alias_version(client, config.CHAMPION_ALIAS), alias_version(client, config.PREVIOUS_ALIAS)
    if champ0 is None or prev0 is None:
        sys.exit("need both a champion and a previous-champion to run this experiment")
    demoted_tags_before = set(client.get_model_version(NAME, champ0.version).tags)
    print(f"start: champion=v{champ0.version}, previous-champion=v{prev0.version}\n")

    stop_model_server(MODEL_PORT)
    try:
        start_model_server(DEFAULT_URI, MODEL_PORT)
        print(f"1. server started on {DEFAULT_URI}")
        print(f"   server answers as:                {ask_server()}")

        promote.rollback()  # champion <-> previous-champion
        print(f"\n2. ROLLED BACK. alias 'champion' now -> v{alias_version(client, config.CHAMPION_ALIAS).version}")
        print(f"   a brand-new load of @champion gives: {ask_registry_now()}   (the registry moved)")
        server_after = ask_server()
        print(f"   the RUNNING server still answers as: {server_after}   <- it resolved the alias once, at startup")

        stop_model_server(MODEL_PORT)
        start_model_server(DEFAULT_URI, MODEL_PORT)
        restarted = ask_server()
        print(f"\n3. server RESTARTED. it now answers as: {restarted}")

        stale = server_after == f"v{champ0.version}"
        follows = restarted == f"v{prev0.version}"
        print("\nCONCLUSION:",
              "CONFIRMED. A running server is stale after a promotion/rollback until it restarts. "
              "A deploy pipeline must restart/roll the servers (or pin versions) as an explicit step."
              if stale and follows else f"UNEXPECTED (stale={stale}, follows_after_restart={follows}); inspect above")
    finally:
        stop_model_server(MODEL_PORT)
        # ---- restore the registry exactly as found ------------------------------------------
        client.set_registered_model_alias(NAME, config.CHAMPION_ALIAS, champ0.version)
        client.set_registered_model_alias(NAME, config.PREVIOUS_ALIAS, prev0.version)
        for tag in set(client.get_model_version(NAME, champ0.version).tags) - demoted_tags_before:
            client.delete_model_version_tag(NAME, champ0.version, tag)  # e.g. last_rollback_at from the experiment
        now = (alias_version(client, config.CHAMPION_ALIAS).version, alias_version(client, config.PREVIOUS_ALIAS).version)
        print(f"\nrestored: champion=v{now[0]}, previous-champion=v{now[1]} "
              f"({'matches the start state' if now == (champ0.version, prev0.version) else 'MISMATCH: fix by hand'})")


if __name__ == "__main__":
    main()
