"""Shadow comparison: score the SAME traffic with champion and challenger, compare behaviour.

    python -m s3_deployment.shadow_compare [--rows 8000]

"Shadow" = the challenger sees real traffic but its answers are never shown to users. This is the
cheap, safe way to learn how a new model WOULD behave before it replaces the champion. Offline
accuracy (val AUC) says how well each model ranks; shadowing adds what AUC cannot:
  * disagreement rate: how many individual decisions would change on promotion?
  * score shift: does the average predicted probability move? (downstream thresholds assume a scale)
  * rank agreement: do they at least order people similarly?
It informs the promotion decision; it never promotes anything. Logged to experiment
'adult-income-shadow' so every comparison is on record. Traffic = validation rows (test stays held out).
"""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config
import data
from s1_tracking.common import setup_mlflow
from s2_registry.common import alias_version, init
from s2_registry.promote import measure_latency_p95_ms

NAME = config.REGISTERED_MODEL_NAME
THRESHOLD = 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=8000)
    args = p.parse_args()
    client = init()
    champ, chall = alias_version(client, config.CHAMPION_ALIAS), alias_version(client, config.CHALLENGER_ALIAS)
    if champ is None or chall is None:
        sys.exit("need both a champion and a challenger (python -m s2_registry.promote --challenge <v>)")

    _, _, _, val, _ = data.get_splits()
    val = val.sample(n=min(args.rows, len(val)), random_state=11)
    X, y = data.xy(val)
    models = {k: mlflow.pyfunc.load_model(f"models:/{NAME}/{v.version}") for k, v in (("champion", champ), ("challenger", chall))}
    p_a = np.asarray(models["champion"].predict(X))[:, 1]
    p_b = np.asarray(models["challenger"].predict(X))[:, 1]

    a_cls, b_cls = p_a >= THRESHOLD, p_b >= THRESHOLD
    diff = np.abs(p_a - p_b)
    stats = {
        "rows": len(X),
        "auc_champion": roc_auc_score(y, p_a), "auc_challenger": roc_auc_score(y, p_b),
        "disagreement_rate": float((a_cls != b_cls).mean()),
        "champion_yes_challenger_no": int((a_cls & ~b_cls).sum()),
        "champion_no_challenger_yes": int((~a_cls & b_cls).sum()),
        "mean_abs_prob_diff": float(diff.mean()),
        "p95_abs_prob_diff": float(np.percentile(diff, 95)),
        "mean_score_champion": float(p_a.mean()), "mean_score_challenger": float(p_b.mean()),
        "rank_correlation_spearman": float(spearmanr(p_a, p_b).statistic),
        "latency_p95_ms_champion": measure_latency_p95_ms(models["champion"], X),
        "latency_p95_ms_challenger": measure_latency_p95_ms(models["challenger"], X),
    }
    # the decisions that changed: who would be treated differently, and how confident was each model?
    flipped = a_cls != b_cls
    stats["flipped_mean_champion_conf"] = float(np.abs(p_a[flipped] - 0.5).mean() + 0.5) if flipped.any() else float("nan")

    setup_mlflow("adult-income-shadow")
    with mlflow.start_run(run_name=f"shadow-v{champ.version}-vs-v{chall.version}"):
        mlflow.set_tags({"champion_version": champ.version, "challenger_version": chall.version,
                         "champion_family": champ.tags.get("model_family", "?"), "challenger_family": chall.tags.get("model_family", "?")})
        mlflow.log_params({"rows": len(X), "threshold": THRESHOLD})
        mlflow.log_metrics({k: v for k, v in stats.items() if k != "rows" and not np.isnan(v)})
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(p_a, p_b, s=4, alpha=0.25)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.axhline(THRESHOLD, c="r", lw=0.5); ax.axvline(THRESHOLD, c="r", lw=0.5)
        ax.set_xlabel(f"champion v{champ.version} P(>50K)"); ax.set_ylabel(f"challenger v{chall.version} P(>50K)")
        ax.set_title("same traffic, two models (red = decision threshold)")
        mlflow.log_figure(fig, "shadow_scatter.png"); plt.close(fig)

    print(f"champion v{champ.version} ({champ.tags.get('model_family')})  vs  challenger v{chall.version} ({chall.tags.get('model_family')})  on {len(X)} rows\n")
    print(f"  AUC                      champion {stats['auc_champion']:.4f}   challenger {stats['auc_challenger']:.4f}   (gap {stats['auc_challenger'] - stats['auc_champion']:+.4f})")
    print(f"  decisions that would flip {stats['disagreement_rate']:.1%}  "
          f"(champion>50K/challenger<=50K: {stats['champion_yes_challenger_no']}, the reverse: {stats['champion_no_challenger_yes']})")
    print(f"  probability difference    mean {stats['mean_abs_prob_diff']:.3f}, p95 {stats['p95_abs_prob_diff']:.3f}")
    print(f"  mean predicted P(>50K)    champion {stats['mean_score_champion']:.3f}   challenger {stats['mean_score_challenger']:.3f}   (true rate {y.mean():.3f})")
    print(f"  rank correlation          {stats['rank_correlation_spearman']:.3f}")
    print(f"  p95 latency (in-process)  champion {stats['latency_p95_ms_champion']:.1f} ms   challenger {stats['latency_p95_ms_challenger']:.1f} ms")
    print("\nThis comparison informs a promotion decision; it changes no alias.")


if __name__ == "__main__":
    main()
