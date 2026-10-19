# File: src/model_development.py
import os
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_PREPROCESSED_PATH = os.path.join(WORKING_DIR, "preprocessed.pkl")

# Candidate models tried by the training DAG. Names (not objects) travel through
# XCom, so each one is built by name inside the task that uses it.
CANDIDATES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "svc": lambda: SVC(),
    "knn": lambda: KNeighborsClassifier(),
}

def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "advertising.csv",
    )
    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split, scale, and save (X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop(
        ["Timestamp", "Clicked on Ad", "Ad Topic Line", "Country", "City"],
        axis=1,
    )
    y = df["Clicked on Ad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    num_columns = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]

    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder="passthrough",
    )

    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def resolve_input_path(**context) -> str:
    """
    Return the preprocessed-data path sent by the upstream DAG through dag_run.conf,
    or the default location when this DAG is triggered by hand.
    """
    conf = context["dag_run"].conf or {}
    return conf.get("preprocessed_path") or DEFAULT_PREPROCESSED_PATH

def evaluate_candidate(file_path: str, model_name: str) -> dict:
    """
    Score one candidate with 5-fold cross-validation on the TRAINING split only,
    so the test set stays untouched for the final report.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = CANDIDATES[model_name]()
    scores = cross_val_score(model, X_train, y_train, cv=5)
    result = {"model": model_name, "cv_accuracy": float(scores.mean())}
    print(f"{model_name}: cv_accuracy={result['cv_accuracy']:.4f}")
    return result

def select_and_save_best(file_path: str, results, filename: str) -> dict:
    """
    Pick the candidate with the best CV accuracy, refit it on the full training
    split, save it to MODEL_DIR/filename, and report its test accuracy.
    """
    results = list(results)  # mapped-task output arrives as a lazy sequence
    best = max(results, key=lambda r: r["cv_accuracy"])

    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = CANDIDATES[best["model"]]()
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    summary = {
        "model": best["model"],
        "cv_accuracy": best["cv_accuracy"],
        "test_accuracy": float(model.score(X_test, y_test)),
        "path": model_path,
        "all_results": results,
    }
    print(f"Best model: {summary['model']} (cv={summary['cv_accuracy']:.4f}, test={summary['test_accuracy']:.4f})")
    return summary

def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print score, and return first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    score = model.score(X_test, y_test)
    print(f"Model score on test data: {score}")

    pred = model.predict(X_test)
    return int(pred[0])
