import pickle, os, json, sys
from sklearn.metrics import accuracy_score, f1_score
import joblib
import argparse

sys.path.insert(0, os.path.abspath('..'))


def load_test_data(data_dir='data'):
    with open(os.path.join(data_dir, 'X_test.pickle'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_dir, 'y_test.pickle'), 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test


def evaluate(model, X_test, y_test):
    y_predict = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_predict),
        "F1_Score": f1_score(y_test, y_predict),
    }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except:
        raise ValueError('Failed to catching the latest model')

    try:
        # Score the model on the held-out test set produced by train_model.py
        X_test, y_test = load_test_data()
    except:
        raise ValueError('Failed to catching the data')

    metrics = evaluate(model, X_test, y_test)

    # Save metrics to a JSON file

    if not os.path.exists('metrics/'):
        # then create it.
        os.makedirs("metrics/")

    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
