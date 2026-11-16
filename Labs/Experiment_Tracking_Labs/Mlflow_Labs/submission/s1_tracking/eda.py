"""Lab 2 parity (steps 4-7): exploratory data analysis, logged as MLflow artifacts.

    python -m s1_tracking.eda

Modernized from the lab: `sns.distplot` was removed from seaborn, so this uses matplotlib only.
One deliberate difference from the lab: EDA looks at the TRAIN split only. The lab explores the full
dataset, which means choices made from the plots (which features to drop, thresholds) have already
"seen" validation/test rows. Any decision you take from a plot is a form of fitting, so it must not
use data you later use to judge the model.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

import config
import data
from s1_tracking.common import setup_mlflow, log_run_context

DEMO_EXPERIMENT = "adult-income-eda"


def main() -> None:
    setup_mlflow(DEMO_EXPERIMENT)
    df, version, train, val, _ = data.get_splits()
    X, y = data.xy(train)
    num, cat = data.feature_columns(X)

    with mlflow.start_run(run_name="eda-train-split"):
        mlflow.set_tag("stage", "eda")
        log_run_context(version, train, val)

        # Step 4 (distribution of the quality variable) -> here: the target + a key numeric feature
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts = y.value_counts().sort_index()
        axes[0].bar(["<=50K", ">50K"], counts.values, color=["tab:gray", "tab:blue"])
        axes[0].set_title(f"target balance ({y.mean():.1%} positive)")
        axes[1].hist(X["age"], bins=40)
        axes[1].set_title("age distribution")
        mlflow.log_figure(fig, "eda/target_and_age.png"); plt.close(fig)
        mlflow.log_metric("positive_rate", float(y.mean()))

        # Step 6 (box plots of each feature split by class: which features separate the classes?)
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
        for ax, col in zip(axes.ravel(), num):
            ax.boxplot([X.loc[y == 0, col], X.loc[y == 1, col]], tick_labels=["<=50K", ">50K"], showfliers=False)
            ax.set_title(col)
        for ax in axes.ravel()[len(num):]:
            ax.axis("off")
        fig.suptitle("numeric features by class (outliers hidden)")
        mlflow.log_figure(fig, "eda/numeric_by_class.png"); plt.close(fig)

        # Categorical signal: positive rate per category (what a tree/embedding will pick up)
        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        for ax, col in zip(axes.ravel(), cat):
            rate = y.groupby(X[col].fillna("(missing)")).mean().sort_values()
            ax.barh(rate.index, rate.values)
            ax.set_title(f"P(>50K) by {col}", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(cat):]:
            ax.axis("off")
        fig.tight_layout()
        mlflow.log_figure(fig, "eda/categorical_positive_rate.png"); plt.close(fig)

        # Step 7 (missing data: isna().any()) -> a rate per column, plus whether missingness is informative
        null_rate = X.isna().mean()
        missing = null_rate[null_rate > 0].sort_values(ascending=False)
        report = {}
        for col in missing.index:
            m = X[col].isna()
            report[col] = {
                "null_rate": round(float(null_rate[col]), 4),
                "positive_rate_when_missing": round(float(y[m].mean()), 4),
                "positive_rate_when_present": round(float(y[~m].mean()), 4),
            }
        mlflow.log_dict(report, "eda/missingness.json")
        for col, r in report.items():
            mlflow.log_metric(f"null_rate__{col}", r["null_rate"])

        # Correlation among numeric features (redundancy check)
        corr = X[num].corr().to_numpy()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(num)), num, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(num)), num, fontsize=8)
        fig.colorbar(im)
        mlflow.log_figure(fig, "eda/numeric_correlation.png"); plt.close(fig)

    print(f"logged EDA for {len(train)} train rows (data_version {version})")
    print("missingness (informative if the two positive rates differ):")
    for col, r in report.items():
        print(f"  {col:16s} null {r['null_rate']:.1%}   P(>50K | missing)={r['positive_rate_when_missing']:.2f}   P(>50K | present)={r['positive_rate_when_present']:.2f}")


if __name__ == "__main__":
    main()
