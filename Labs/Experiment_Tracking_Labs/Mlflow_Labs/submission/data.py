"""Load, clean and split the Adult Census Income dataset."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import config


def load_raw() -> pd.DataFrame:
    """Download once, then reuse a local CSV cache."""
    cache = config.DATA_DIR / "adult.csv"
    if cache.exists():
        return pd.read_csv(cache)
    config.DATA_DIR.mkdir(exist_ok=True)
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df.to_csv(cache, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.TARGET] = (df["class"].astype(str).str.strip() == ">50K").astype(int)
    df = df.drop(columns=["class", *config.DROP_COLUMNS])
    # Numeric features as float64: MLflow signatures infer ints as non-nullable, so a
    # request with a missing value would fail schema enforcement at serving time.
    num_cols = df.drop(columns=[config.TARGET]).select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].astype("float64")
    # Normalise missing markers so imputers see real NaNs.
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        df[col] = df[col].replace("?", np.nan)
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the cleaned data: the 'data version' we log with each run.
    Same bytes in -> same hash out, so two runs with the same hash saw the same data."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:12]


def split(df: pd.DataFrame):
    """Stratified train/val/test. Stratify so each split keeps the ~24% positive rate."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=config.SEED
    )
    # val is a fraction of what remains, so rescale to get 20% of the total
    val_frac = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val[config.TARGET], random_state=config.SEED
    )
    return train, val, test


def xy(df: pd.DataFrame):
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def feature_columns(X: pd.DataFrame):
    """(numeric, categorical) column names, derived once from the training features."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def get_splits():
    """Convenience: cleaned data, its fingerprint and the three splits."""
    df = clean(load_raw())
    return df, dataset_fingerprint(df), *split(df)
