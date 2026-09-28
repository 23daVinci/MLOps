import numpy as np
from sklearn.dummy import DummyClassifier

from src import evaluate_model


def test_evaluate_returns_expected_metric_keys():
    X_test = np.zeros((10, 30))
    y_test = np.array([0, 1] * 5)

    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(X_test, y_test)

    metrics = evaluate_model.evaluate(model, X_test, y_test)

    assert set(metrics.keys()) == {"Accuracy", "F1_Score"}
    assert metrics["Accuracy"] == 0.5


def test_evaluate_perfect_predictions():
    y_test = np.array([0, 1, 1, 0, 1])
    X_test = np.zeros((5, 30))

    class PerfectModel:
        def predict(self, X):
            return y_test

    metrics = evaluate_model.evaluate(PerfectModel(), X_test, y_test)

    assert metrics["Accuracy"] == 1.0
    assert metrics["F1_Score"] == 1.0


def test_load_test_data_roundtrip(tmp_path):
    X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_test = np.array([0, 1])

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    import pickle
    with open(data_dir / "X_test.pickle", "wb") as f:
        pickle.dump(X_test, f)
    with open(data_dir / "y_test.pickle", "wb") as f:
        pickle.dump(y_test, f)

    loaded_X, loaded_y = evaluate_model.load_test_data(data_dir=str(data_dir))

    assert np.array_equal(loaded_X, X_test)
    assert np.array_equal(loaded_y, y_test)
