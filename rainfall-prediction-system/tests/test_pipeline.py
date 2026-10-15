import numpy as np
import pytest
from sklearn.base import clone

from src import config
from src.data_loader import class_balance, clean_data, get_xy, split_data
from src.evaluate import cross_validate_model, holdout_metrics, select_best_model
from src.models import MODEL_NAMES, build_all_models, build_model
from src.synthetic_data import generate_synthetic_weather


def test_synthetic_schema_and_missing_values():
    raw = generate_synthetic_weather(years=1, seed=1)
    for col in config.FEATURES + ["Date", config.TARGET]:
        if col != "Month":
            assert col in raw.columns
    assert raw["Sunshine"].isna().mean() > 0.2          # realistic missingness


def test_cleaning(small_df):
    assert small_df[config.TARGET].isin(["Yes", "No"]).all()
    assert small_df["Month"].between(1, 12).all()
    assert small_df["Date"].is_monotonic_increasing


def test_cleaning_nulls_impossible_values():
    raw = generate_synthetic_weather(years=1, seed=2)
    raw.loc[0, "Humidity3pm"] = 999
    cleaned = clean_data(raw)
    assert (cleaned["Humidity3pm"].dropna() <= 100).all()


def test_chronological_split_has_no_time_overlap(small_df):
    X_tr, X_te, y_tr, y_te = split_data(small_df, "chronological")
    assert small_df.loc[X_tr.index, "Date"].max() <= small_df.loc[X_te.index, "Date"].min()
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(small_df)


def test_stratified_split_preserves_class_ratio(small_df):
    _, _, y_tr, y_te = split_data(small_df, "stratified")
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_target_not_in_features(small_df):
    X, _ = get_xy(small_df)
    assert config.TARGET not in X.columns and "Date" not in X.columns


def test_preprocessing_fitted_on_train_only(small_df):
    """Imputer medians must equal the TRAIN medians, not the full-data medians."""
    X_tr, X_te, y_tr, _ = split_data(small_df, "chronological")
    pipe = build_model("Logistic Regression").fit(X_tr, y_tr)
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    expected = X_tr[config.NUMERIC_FEATURES].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)


def test_class_balance(small_df):
    _, y = get_xy(small_df)
    b = class_balance(y)
    assert b["rain"] + b["no_rain"] == len(y) and b["imbalance_ratio"] > 1


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains_and_predicts(name, small_df):
    # Subsample so the SVM / KNN tests stay quick.
    X_tr, X_te, y_tr, y_te = split_data(small_df, "chronological")
    X_tr, y_tr = X_tr.iloc[:1500], y_tr.iloc[:1500]
    X_te, y_te = X_te.iloc[:400], y_te.iloc[:400]
    pipe = build_all_models(y_tr)[name].fit(X_tr, y_tr)
    m = holdout_metrics(pipe, X_te, y_te)
    assert 0 <= m["accuracy"] <= 1 and m["roc_auc"] > 0.6
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y_te)


def test_cross_validation_returns_all_metrics(small_df):
    X_tr, _, y_tr, _ = split_data(small_df, "chronological")
    res = cross_validate_model(build_model("Logistic Regression"), X_tr, y_tr)
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert f"cv_{m}_mean" in res and f"cv_{m}_std" in res


def test_model_selection_uses_cv_not_test():
    import pandas as pd
    df = pd.DataFrame({
        "model": ["A", "B"], "cv_roc_auc_mean": [0.80, 0.85], "cv_f1_mean": [0.5, 0.5],
        "test_roc_auc": [0.99, 0.10],
    })
    assert select_best_model(df) == "B"
