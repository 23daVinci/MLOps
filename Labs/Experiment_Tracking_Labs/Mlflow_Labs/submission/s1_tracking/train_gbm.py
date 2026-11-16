"""Gradient boosting with a tracked hyperparameter search.

Structure in the MLflow UI:
    gbm-search-<sampler> (parent)  <- search config, best params, final model, plots
      |- trial-00 (child)          <- params + train/val metrics for ONE configuration
      |- trial-01 ...
Only the best configuration is refit and logged as a model; trials log metrics only,
so we don't store a dozen throwaway model folders.

    python -m s1_tracking.train_gbm --sampler tpe --trials 12
    python -m s1_tracking.train_gbm --sampler random --trials 12     # reference to compare against
"""
import argparse

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import config
import data
from features import build_preprocessor
from metrics import compute_metrics, make_plots
from s1_tracking.common import log_run_context, setup_mlflow, strict_signature, timed
from s1_tracking.search import SAMPLERS, run_search

SELECTION_METRIC = "val_auc"  # chosen on validation; test is never used here


def space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63]),
        "max_iter": trial.suggest_categorical("max_iter", [200, 400, 600]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10, 20, 50]),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
    }


def build_pipeline(params, numeric_cols, categorical_cols) -> Pipeline:
    return Pipeline(
        [
            # trees don't need scaling; sklearn's histogram GBM needs dense input
            ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False, dense=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    early_stopping=True,  # uses an internal slice of TRAIN, not our val set
                    random_state=config.SEED,
                    **params,
                ),
            ),
        ]
    )


def evaluate_pipeline(pipeline, X_train, y_train, X_val, y_val) -> dict:
    out = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        proba = pipeline.predict_proba(X)[:, 1]
        out.update({f"{name}_{k}": v for k, v in compute_metrics(y, proba).items()})
    return out


def baseline_val_auc() -> float | None:
    """Look up the baseline's val_auc so the final run reports its improvement."""
    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string="tags.stage = 'baseline' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return runs[0].data.metrics.get(SELECTION_METRIC) if runs else None


def main(sampler: str, n_trials: int):
    setup_mlflow()
    df, data_version, train, val, _test = data.get_splits()
    X_train, y_train = data.xy(train)
    X_val, y_val = data.xy(val)
    numeric_cols, categorical_cols = data.feature_columns(X_train)

    def fit_eval(params: dict) -> dict:
        with timed("trial_train_seconds"):
            pipe = build_pipeline(params, numeric_cols, categorical_cols).fit(X_train, y_train)
        out = evaluate_pipeline(pipe, X_train, y_train, X_val, y_val)
        out["n_iter_used"] = int(pipe.named_steps["model"].n_iter_)  # rounds early stopping kept
        return out

    with mlflow.start_run(run_name=f"gbm-search-{sampler}") as parent:
        mlflow.set_tags(
            {"stage": "candidate", "model_family": "gradient_boosting", "sampler": sampler}
        )
        log_run_context(data_version, train, val)
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "selection_metric": SELECTION_METRIC,
                "search_strategy": sampler,
                "seed": config.SEED,
                "n_train": len(train),
                "n_val": len(val),
                "numeric_cols": ",".join(numeric_cols),
                "categorical_cols": ",".join(categorical_cols),
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(train, name="adult-train", targets=config.TARGET), context="training"
        )
        mlflow.log_input(
            mlflow.data.from_pandas(val, name="adult-val", targets=config.TARGET), context="validation"
        )

        # ------------------------- the search -------------------------------
        result = run_search(space, fit_eval, n_trials, sampler=sampler, seed=config.SEED)

        # ---------------- refit the winner and log it as THE model -----------
        mlflow.log_params({f"best__{k}": v for k, v in result.best_params.items()})
        with timed():
            final = build_pipeline(result.best_params, numeric_cols, categorical_cols).fit(X_train, y_train)
        final_metrics = evaluate_pipeline(final, X_train, y_train, X_val, y_val)
        mlflow.log_metrics(final_metrics)

        base = baseline_val_auc()
        if base is not None:
            mlflow.log_metric("baseline_val_auc", base)
            mlflow.log_metric("val_auc_gain_vs_baseline", final_metrics["val_auc"] - base)

        val_proba = final.predict_proba(X_val)[:, 1]
        for filename, fig in make_plots(y_val, val_proba).items():
            mlflow.log_figure(fig, f"val_plots/{filename}")
            plt.close(fig)

        # Served (pyfunc) output = class probabilities, matching what we evaluated.
        # Default would be hard 0/1 labels. Declared as 2 columns [P(<=50K), P(>50K)].
        signature = strict_signature(X_train, final.predict_proba(X_train))
        mlflow.sklearn.log_model(
            final,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            pyfunc_predict_fn="predict_proba",
            # skops refuses unknown types by default. We trust exactly these, reviewed:
            #  - numpy.dtype: held by fitted imputers/encoders (harmless)
            #  - TreePredictor: the GBM's trees. skops warns a *maliciously crafted* file could
            #    hold out-of-range node indices; fine for artifacts we produced ourselves, but
            #    never load a model like this from an untrusted source.
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        print(f"\nparent run_id: {parent.info.run_id}")
        print(f"sampler:       {sampler}")
        print(f"best params:   {result.best_params}")
        print(f"val_auc:       {final_metrics['val_auc']:.4f}  (baseline: {base})")
        print(f"train_auc:     {final_metrics['train_auc']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", choices=SAMPLERS, default="tpe")
    p.add_argument("--trials", type=int, default=12)
    args = p.parse_args()
    main(args.sampler, args.trials)
