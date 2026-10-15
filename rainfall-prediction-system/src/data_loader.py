"""Loading, cleaning and splitting the weather dataset."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .synthetic_data import generate_synthetic_weather

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw_data(path=None) -> Tuple[pd.DataFrame, str]:
    """Load the raw data. Returns ``(dataframe, source_description)``.

    Priority: explicit ``path`` > data/weatherAUS.csv (real) > synthetic data.
    """
    if path is not None:
        return pd.read_csv(path), f"custom file: {path}"

    if config.REAL_DATA_PATH.exists():
        log.info("Loading real dataset from %s", config.REAL_DATA_PATH)
        return pd.read_csv(config.REAL_DATA_PATH), "real dataset (weatherAUS.csv)"

    log.warning(
        "data/weatherAUS.csv not found - using SYNTHETIC data with the same schema. "
        "Download the real dataset (see data/README.md) for real-world results."
    )
    if config.SYNTHETIC_DATA_PATH.exists():
        df = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    else:
        df = generate_synthetic_weather()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.SYNTHETIC_DATA_PATH, index=False)
    return df, "synthetic dataset (same schema as weatherAUS)"


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe to do *before* the train/test split.

    Only operations that do not learn anything from the data are done here
    (dropping rows without a label, removing duplicates, nulling physically
    impossible values, parsing dates). Statistical steps such as imputation and
    scaling live in the sklearn Pipeline and are fitted on training data only,
    which prevents data leakage.
    """
    df = df.copy()

    missing_cols = [c for c in [config.DATE_COL, config.TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    df = df.dropna(subset=[config.DATE_COL, config.TARGET])
    df = df.drop_duplicates(subset=[config.DATE_COL, "Location"] if "Location" in df.columns else None)

    # Standardise labels to strictly Yes/No.
    for col in (config.TARGET, "RainToday"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()
            df.loc[~df[col].isin(["Yes", "No"]), col] = np.nan
    df = df.dropna(subset=[config.TARGET])

    # Physically impossible readings -> NaN (imputed later inside the pipeline).
    for col, (lo, hi, _, _) in config.NUMERIC_RANGES.items():
        if col in df.columns and col != "Month":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Seasonality feature.
    df["Month"] = df[config.DATE_COL].dt.month

    # Keep categories as plain object dtype (sklearn-friendly, NaN allowed).
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    df = df.sort_values([config.DATE_COL] + (["Location"] if "Location" in df.columns else []))
    return df.reset_index(drop=True)


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into feature matrix X and binary target y (1 = rain)."""
    features = [c for c in config.FEATURES if c in df.columns]
    X = df[features].copy()
    y = (df[config.TARGET] == "Yes").astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, mode: str = "chronological", test_size: float = config.TEST_SIZE):
    """Return ``X_train, X_test, y_train, y_test``.

    * ``chronological`` - train on the earliest (1-test_size) share of the
      calendar, test on the most recent days. Mirrors real forecasting, where
      you never get to see the future. **Used for model selection.**
    * ``stratified``    - random split that keeps the rain / no-rain ratio equal
      in both sets. Convenient, but optimistic for weather data because
      neighbouring days are strongly correlated.
    """
    X, y = get_xy(df)
    if mode == "chronological":
        cutoff = df[config.DATE_COL].quantile(1 - test_size, interpolation="nearest")
        train_mask = (df[config.DATE_COL] <= cutoff).to_numpy()
        return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
    if mode == "stratified":
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
        )
    raise ValueError("mode must be 'chronological' or 'stratified'")


def class_balance(y: pd.Series) -> dict:
    """Class counts, percentages and the imbalance ratio."""
    counts = y.value_counts().sort_index()
    n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
    return {
        "no_rain": n0,
        "rain": n1,
        "rain_pct": round(100 * n1 / max(n0 + n1, 1), 2),
        "imbalance_ratio": round(n0 / max(n1, 1), 2),
    }
