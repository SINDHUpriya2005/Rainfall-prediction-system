"""Shared fixtures: a small, fast model so tests never depend on training artifacts."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import clean_data, split_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.predict import RainfallPredictor  # noqa: E402
from src.synthetic_data import generate_synthetic_weather  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return clean_data(generate_synthetic_weather(years=3, seed=7))


@pytest.fixture(scope="session")
def predictor(small_df, tmp_path_factory):
    X_tr, X_te, y_tr, y_te = split_data(small_df, "chronological")
    pipe = build_model("Logistic Regression").fit(X_tr, y_tr)
    folder = tmp_path_factory.mktemp("model")
    model_path, meta_path = folder / "m.joblib", folder / "meta.json"
    joblib.dump(pipe, model_path)
    ref_cols = ["Humidity3pm", "Cloud3pm", "Pressure3pm", "Sunshine", "WindGustSpeed"]
    tmp = X_tr.assign(_y=y_tr)
    meta = {
        "model_name": "Logistic Regression",
        "is_synthetic": True,
        "locations": sorted(small_df["Location"].unique()),
        "reference_means": {
            "rain": tmp[tmp._y == 1][ref_cols].mean().to_dict(),
            "dry": tmp[tmp._y == 0][ref_cols].mean().to_dict(),
        },
        "results": [], "stratified_split_results": [],
    }
    meta_path.write_text(json.dumps(meta))
    return RainfallPredictor(model_path, meta_path)


@pytest.fixture()
def client(predictor):
    from app.app import create_app
    app = create_app(predictor)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def valid_form():
    return {
        "Location": "Sydney", "MaxTemp": "24", "MinTemp": "15", "Humidity3pm": "70",
        "Pressure3pm": "1010", "Cloud3pm": "6", "Rainfall": "2.4", "Sunshine": "4",
        "WindGustDir": "SE",
    }
