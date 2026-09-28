import numpy as np
from src import train_model


def test_load_and_split_data_shapes():
    X_train, X_val, X_test, y_train, y_val, y_test = train_model.load_and_split_data()

    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == 569  # size of the breast cancer dataset
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 30
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_load_and_split_data_is_reproducible():
    split_a = train_model.load_and_split_data(random_state=0)
    split_b = train_model.load_and_split_data(random_state=0)
    for arr_a, arr_b in zip(split_a, split_b):
        assert np.array_equal(arr_a, arr_b)


def test_train_val_test_do_not_overlap():
    X_train, X_val, X_test, _, _, _ = train_model.load_and_split_data()

    train_rows = {tuple(row) for row in X_train}
    val_rows = {tuple(row) for row in X_val}
    test_rows = {tuple(row) for row in X_test}

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


def test_tune_hyperparameters_picks_best_validation_score():
    X_train, X_val, _, y_train, y_val, _ = train_model.load_and_split_data()
    param_grid = {"n_estimators": [10, 50], "max_depth": [None, 3]}

    best_params, best_score, results = train_model.tune_hyperparameters(
        X_train, y_train, X_val, y_val, param_grid
    )

    assert best_params["n_estimators"] in param_grid["n_estimators"]
    assert best_params["max_depth"] in param_grid["max_depth"]
    assert 0.0 <= best_score <= 1.0
    assert len(results) == len(param_grid["n_estimators"]) * len(param_grid["max_depth"])
    assert best_score == max(r["val_f1"] for r in results)


def test_save_pickle_roundtrip(tmp_path):
    obj = np.array([1, 2, 3])
    path = tmp_path / "nested" / "obj.pickle"

    train_model.save_pickle(obj, str(path))

    import pickle
    with open(path, 'rb') as f:
        loaded = pickle.load(f)

    assert np.array_equal(loaded, obj)
