"""Final evaluation: score every registered version on the TEST split, once, with confidence intervals.

    python -m s4_evaluation.final_eval [--force]

The test split has been held out of every earlier step (training, tuning, promotion all used validation),
so this is the first honest estimate of how each model does on unseen data. Touch it once: every extra
look lets you pick whatever flatters you. A `test_evaluated` tag on each version makes a second run
refuse unless you pass --force.

Confidence intervals: bootstrap. Resample the test rows with replacement 1000 times and recompute AUC each
time. The SAME resample is used for every model (paired), so "is v2 really better than v3?" is answered
by the interval of the AUC DIFFERENCE, which is much tighter than comparing two separate intervals.
"""
import argparse
import sys

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

import config
import data
from s2_registry.common import alias_version, init

NAME = config.REGISTERED_MODEL_NAME
N_BOOT = 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="evaluate again even though test was already used")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    client = init()

    versions = sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version))
    done = [v.version for v in versions if v.tags.get("test_evaluated") == "true"]
    if done and not args.force:
        sys.exit(f"test split already used for versions {done}. Looking again invites cherry-picking; use --force if you must.")

    _, _, _, _, test = data.get_splits()
    X, y = data.xy(test)
    y = y.to_numpy()

    scores = {v.version: np.asarray(mlflow.pyfunc.load_model(f"models:/{NAME}/{v.version}").predict(X))[:, 1] for v in versions}
    champion = alias_version(client, config.CHAMPION_ALIAS).version

    rng = np.random.default_rng(config.SEED)
    boots = {v: [] for v in scores}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))  # one resample, shared by all models
        if y[idx].min() == y[idx].max():
            continue
        for v, s in scores.items():
            boots[v].append(roc_auc_score(y[idx], s[idx]))
    boots = {v: np.array(b) for v, b in boots.items()}

    print(f"test rows: {len(y)}   bootstrap resamples: {N_BOOT}   champion: v{champion}\n")
    print(f"{'ver':4s} {'family':18s} {'val AUC':>8s} {'test AUC':>9s} {'95% CI':>17s}   vs champion (AUC diff, 95% CI)")
    for v in versions:
        auc = roc_auc_score(y, scores[v.version])
        lo, hi = np.percentile(boots[v.version], [2.5, 97.5])
        val_auc = client.get_run(v.run_id).data.metrics["val_auc"]
        if v.version == champion:
            diff_txt = "(champion)"
        else:
            d = boots[v.version] - boots[champion]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            verdict = "CLEARLY different" if dlo > 0 or dhi < 0 else "NOT distinguishable from zero"
            diff_txt = f"{auc - roc_auc_score(y, scores[champion]):+.4f} [{dlo:+.4f}, {dhi:+.4f}]  {verdict}"
        print(f"v{v.version:<3s} {v.tags.get('model_family', '?'):18s} {val_auc:8.4f} {auc:9.4f} [{lo:.4f}, {hi:.4f}]   {diff_txt}")
        client.set_model_version_tag(NAME, v.version, "test_auc", f"{auc:.4f}")
        client.set_model_version_tag(NAME, v.version, "test_evaluated", "true")

    print("\nTest AUCs are recorded as version tags (test_auc). Do not tune against them.")


if __name__ == "__main__":
    main()
