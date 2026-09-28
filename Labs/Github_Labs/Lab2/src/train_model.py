# from sklearn.datasets import fetch_rcv1
import mlflow, datetime, os, pickle
# import sklearn
import numpy as np
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))


def load_and_split_data(test_size=0.2, val_size=0.25, random_state=0):
    """Split the breast cancer dataset into train/validation/test sets (60/20/20 by default)."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, random_state=0):
    """Fit a model per hyperparameter combination and return the one with the best validation F1."""
    best_params, best_score = None, -1
    results = []
    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
            model = RandomForestClassifier(random_state=random_state, **params)
            model.fit(X_train, y_train)
            val_f1 = f1_score(y_val, model.predict(X_val))
            results.append({**params, "val_f1": val_f1})
            if val_f1 > best_score:
                best_score, best_params = val_f1, params
    return best_params, best_score, results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp

    # Use the timestamp in your script for model versioning
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # Persist the splits so evaluate_model.py can score the final model on a true held-out test set
    save_pickle(X_train, 'data/X_train.pickle')
    save_pickle(y_train, 'data/y_train.pickle')
    save_pickle(X_val, 'data/X_val.pickle')
    save_pickle(y_val, 'data/y_val.pickle')
    save_pickle(X_test, 'data/X_test.pickle')
    save_pickle(y_test, 'data/y_test.pickle')

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):

        params = {
                    "dataset_name": dataset_name,
                    "number of dimensions": X_train.shape[1],
                    "train size": X_train.shape[0],
                    "validation size": X_val.shape[0],
                    "test size": X_test.shape[0]}

        mlflow.log_params(params)

        # Finetune hyperparameters using the train/validation split
        best_params, best_val_f1, _ = tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("Validation F1 Score", best_val_f1)

        # Refit the best hyperparameters on train + validation combined for the final model
        X_train_final = np.concatenate([X_train, X_val])
        y_train_final = np.concatenate([y_train, y_val])
        forest = RandomForestClassifier(random_state=0, **best_params)
        forest.fit(X_train_final, y_train_final)

        y_predict = forest.predict(X_train_final)
        mlflow.log_metrics({'Train Accuracy': accuracy_score(y_train_final, y_predict),
                            'Train F1 Score': f1_score(y_train_final, y_predict)})

        if not os.path.exists('models/'):
            # then create it.
            os.makedirs("models/")

        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        dump(forest, model_filename)
