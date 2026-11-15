"""Gradient boosting with a tracked random hyperparameter search.

Structure in the MLflow UI:
    gbm-search (parent)          <- search config, best params, final model, plots
      |- trial-00 (child)        <- params + train/val metrics for ONE configuration
      |- trial-01 ...
Only the best configuration is refit and logged as a model; trials log metrics only,
so we don't store a dozen throwaway model folders.

Run the MLflow server first (see README), then:  python train_gbm.py
"""
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import config
import data
import evaluation

N_TRIALS = 12
SELECTION_METRIC = "val_auc"  # chosen on validation; test is never used here


def sample_params(rng: np.random.Generator) -> dict:
    """One random draw from the search space (seeded rng -> reproducible search)."""
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.3)))),
        "max_leaf_nodes": int(rng.choice([15, 31, 63])),
        "max_iter": int(rng.choice([200, 400, 600])),
        "min_samples_leaf": int(rng.choice([10, 20, 50])),
        "l2_regularization": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
    }


def build_pipeline(params, numeric_cols, categorical_cols) -> Pipeline:
    return Pipeline(
        [
            # trees don't need scaling
            (
                "prep",
                evaluation.build_preprocessor(
                    numeric_cols, categorical_cols, scale_numeric=False, dense=True
                ),
            ),
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
    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        proba = pipeline.predict_proba(X)[:, 1]
        metrics.update({f"{name}_{k}": v for k, v in evaluation.compute_metrics(y, proba).items()})
    return metrics


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df, data_version, train, val, _test = data.get_splits()
    X_train, y_train = data.xy(train)
    X_val, y_val = data.xy(val)
    numeric_cols, categorical_cols = data.feature_columns(X_train)
    rng = np.random.default_rng(config.SEED)

    with mlflow.start_run(run_name="gbm-search") as parent:
        mlflow.set_tags(
            {"stage": "candidate", "model_family": "gradient_boosting", "data_version": data_version}
        )
        mlflow.log_params(
            {
                "n_trials": N_TRIALS,
                "selection_metric": SELECTION_METRIC,
                "search_strategy": "random",
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
        best = {"score": -np.inf, "params": None}
        for i in range(N_TRIALS):
            params = sample_params(rng)
            with mlflow.start_run(run_name=f"trial-{i:02d}", nested=True):
                mlflow.set_tag("stage", "trial")
                mlflow.log_params(params)
                pipe = build_pipeline(params, numeric_cols, categorical_cols).fit(X_train, y_train)
                metrics = evaluate_pipeline(pipe, X_train, y_train, X_val, y_val)
                mlflow.log_metrics(metrics)
                # how many boosting rounds early stopping actually used
                mlflow.log_metric("n_iter_used", int(pipe.named_steps["model"].n_iter_))
            print(f"trial-{i:02d}  val_auc={metrics[SELECTION_METRIC]:.4f}  {params}")
            if metrics[SELECTION_METRIC] > best["score"]:
                best = {"score": metrics[SELECTION_METRIC], "params": params}

        # ---------------- refit the winner and log it as THE model -----------
        mlflow.log_params({f"best__{k}": v for k, v in best["params"].items()})
        final = build_pipeline(best["params"], numeric_cols, categorical_cols).fit(X_train, y_train)
        final_metrics = evaluate_pipeline(final, X_train, y_train, X_val, y_val)
        mlflow.log_metrics(final_metrics)

        base = baseline_val_auc()
        if base is not None:
            mlflow.log_metric("baseline_val_auc", base)
            mlflow.log_metric("val_auc_gain_vs_baseline", final_metrics["val_auc"] - base)

        val_proba = final.predict_proba(X_val)[:, 1]
        for filename, fig in evaluation.make_plots(y_val, val_proba).items():
            mlflow.log_figure(fig, f"val_plots/{filename}")
            plt.close(fig)

        # Served (pyfunc) output = class probabilities, matching what we evaluated.
        # Default would be hard 0/1 labels. Declared as 2 columns [P(<=50K), P(>50K)].
        signature = infer_signature(X_train, final.predict_proba(X_train))
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
        print(f"best params:   {best['params']}")
        print(f"val_auc:       {final_metrics['val_auc']:.4f}  (baseline: {base})")
        print(f"train_auc:     {final_metrics['train_auc']:.4f}")


if __name__ == "__main__":
    main()
