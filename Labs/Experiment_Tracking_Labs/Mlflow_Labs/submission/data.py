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


class DataValidationError(ValueError):
    pass


EXPECTED_COLUMNS = [
    "age", "workclass", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", config.TARGET,
]
# (min, max) allowed for numeric features; anything outside means the data changed under us.
NUMERIC_RANGES = {
    "age": (17, 100), "education-num": (1, 16), "capital-gain": (0, 100_000),
    "capital-loss": (0, 5_000), "hours-per-week": (1, 99),
}
MAX_NULL_RATE = 0.10             # categorical columns here have <6% missing
POSITIVE_RATE_RANGE = (0.15, 0.35)  # ~24% in the real data; far outside = wrong target/labels


def validate(df: pd.DataFrame) -> dict:
    """Schema + range + null-rate + label-balance checks on the CLEANED frame.
    Raises DataValidationError listing every failure; returns a report dict otherwise."""
    problems = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if not problems:
        if not set(df[config.TARGET].unique()) <= {0, 1}:
            problems.append("target is not binary 0/1")
        for col, (lo, hi) in NUMERIC_RANGES.items():
            cmin, cmax = df[col].min(), df[col].max()
            if cmin < lo or cmax > hi:
                problems.append(f"{col} range [{cmin}, {cmax}] outside allowed [{lo}, {hi}]")
        null_rate = df.isna().mean()
        for col, rate in null_rate.items():
            if rate > MAX_NULL_RATE:
                problems.append(f"{col} null rate {rate:.1%} > {MAX_NULL_RATE:.0%}")
        pos = df[config.TARGET].mean()
        if not POSITIVE_RATE_RANGE[0] <= pos <= POSITIVE_RATE_RANGE[1]:
            problems.append(f"positive rate {pos:.1%} outside {POSITIVE_RATE_RANGE}")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"n_rows": len(df), "positive_rate": float(df[config.TARGET].mean()), "checks_passed": True}


def get_splits():
    """Convenience: validated cleaned data, its fingerprint and the three splits."""
    df = clean(load_raw())
    validate(df)  # fail before any model trains on bad data
    return df, dataset_fingerprint(df), *split(df)
