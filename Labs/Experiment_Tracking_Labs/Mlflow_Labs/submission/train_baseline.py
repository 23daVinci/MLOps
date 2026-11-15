"""Baseline run: logistic regression inside a preprocessing Pipeline.

Run the MLflow server first (see README), then:  python train_baseline.py
"""
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
import data
import evaluation

PARAMS = {"C": 1.0, "max_iter": 1000}


def main():
    # MLflow prints an emoji when a run ends; the default Windows console (cp1252) can't.
    sys.stdout.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)  # creates it on first use

    df, data_version, train, val, _test = data.get_splits()  # test stays untouched
    X_train, y_train = data.xy(train)
    X_val, y_val = data.xy(val)

    numeric_cols, categorical_cols = data.feature_columns(X_train)
    pipeline = Pipeline(
        [
            ("prep", evaluation.build_preprocessor(numeric_cols, categorical_cols)),
            ("model", LogisticRegression(random_state=config.SEED, **PARAMS)),
        ]
    )

    with mlflow.start_run(run_name="baseline-logreg") as run:
        # --- tags: searchable labels --------------------------------------
        mlflow.set_tags(
            {
                "stage": "baseline",
                "model_family": "linear",
                "data_version": data_version,
            }
        )

        # --- params: everything that determined the result ----------------
        mlflow.log_params(
            {
                **{f"model__{k}": v for k, v in PARAMS.items()},
                "seed": config.SEED,
                "val_size": config.VAL_SIZE,
                "test_size": config.TEST_SIZE,
                "n_train": len(train),
                "n_val": len(val),
                "n_features_raw": X_train.shape[1],
                "numeric_cols": ",".join(numeric_cols),
                "categorical_cols": ",".join(categorical_cols),
                "scale_numeric": True,
            }
        )

        # --- dataset lineage ----------------------------------------------
        mlflow.log_input(
            mlflow.data.from_pandas(train, name="adult-train", targets=config.TARGET),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(val, name="adult-val", targets=config.TARGET),
            context="validation",
        )

        # --- train ---------------------------------------------------------
        pipeline.fit(X_train, y_train)

        # --- metrics: train AND validation, so overfitting is visible -----
        train_proba = pipeline.predict_proba(X_train)[:, 1]
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        for split_name, y, proba in [("train", y_train, train_proba), ("val", y_val, val_proba)]:
            mlflow.log_metrics(
                {f"{split_name}_{k}": v for k, v in evaluation.compute_metrics(y, proba).items()}
            )

        # --- artifacts: plots explain what a single number hides ----------
        for filename, fig in evaluation.make_plots(y_val, val_proba).items():
            mlflow.log_figure(fig, f"val_plots/{filename}")
            plt.close(fig)

        # --- the model: signature + example make it servable and testable --
        # The served (pyfunc) interface must match what we evaluated: class probabilities.
        # Default pyfunc.predict would return hard 0/1 labels, so we point it at
        # predict_proba and declare the 2-column output [P(<=50K), P(>50K)] in the signature.
        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            pyfunc_predict_fn="predict_proba",
            # MLflow 3 saves sklearn models with skops (safe: no arbitrary pickle code).
            # Fitted imputers/encoders hold numpy dtypes, which skops asks us to trust.
            skops_trusted_types=["numpy.dtype"],
        )

        print(f"run_id:       {run.info.run_id}")
        print(f"data_version: {data_version}")
        print(f"val_auc:      {evaluation.compute_metrics(y_val, val_proba)['auc']:.4f}")


if __name__ == "__main__":
    main()
