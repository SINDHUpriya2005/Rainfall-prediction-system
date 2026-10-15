"""Central configuration: paths, feature lists, validation ranges, constants."""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_IMG_DIR = ROOT_DIR / "app" / "static" / "img"

# Real dataset (Kaggle "Rain in Australia", weatherAUS.csv). If this file is
# present it is used automatically, otherwise a synthetic dataset with the
# same schema is generated so the project always runs end to end.
REAL_DATA_PATH = DATA_DIR / "weatherAUS.csv"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_weather.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
RESULTS_PATH = REPORTS_DIR / "model_comparison.csv"

# --------------------------------------------------------------------------- #
# Reproducibility / evaluation settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "RainTomorrow"
DATE_COL = "Date"

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm",
    "Month",  # derived from Date: captures seasonality
]

CATEGORICAL_FEATURES = [
    "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

LOCATIONS = [
    "Sydney", "Melbourne", "Brisbane", "Perth",
    "Adelaide", "Canberra", "Darwin", "Hobart",
]

# --------------------------------------------------------------------------- #
# Input validation ranges used by the web app: (min, max, unit, label)
# --------------------------------------------------------------------------- #
NUMERIC_RANGES = {
    "MinTemp":       (-10.0, 40.0,   "°C",   "Minimum temperature"),
    "MaxTemp":       (-5.0,  52.0,   "°C",   "Maximum temperature"),
    "Rainfall":      (0.0,   500.0,  "mm",   "Rainfall today"),
    "Sunshine":      (0.0,   14.5,   "hours", "Sunshine"),
    "WindGustSpeed": (0.0,   150.0,  "km/h", "Wind gust speed"),
    "WindSpeed9am":  (0.0,   100.0,  "km/h", "Wind speed at 9am"),
    "WindSpeed3pm":  (0.0,   100.0,  "km/h", "Wind speed at 3pm"),
    "Humidity9am":   (0.0,   100.0,  "%",    "Humidity at 9am"),
    "Humidity3pm":   (0.0,   100.0,  "%",    "Humidity at 3pm"),
    "Pressure9am":   (950.0, 1060.0, "hPa",  "Pressure at 9am"),
    "Pressure3pm":   (950.0, 1060.0, "hPa",  "Pressure at 3pm"),
    "Cloud9am":      (0.0,   9.0,    "oktas", "Cloud cover at 9am"),
    "Cloud3pm":      (0.0,   9.0,    "oktas", "Cloud cover at 3pm"),
    "Temp9am":       (-10.0, 45.0,   "°C",   "Temperature at 9am"),
    "Temp3pm":       (-5.0,  50.0,   "°C",   "Temperature at 3pm"),
    "Month":         (1,     12,     "",     "Month"),
}

# The minimum set of fields a user must provide. Everything else is optional
# and is imputed by the model pipeline using training-set statistics.
REQUIRED_INPUTS = [
    "MaxTemp", "Humidity3pm", "Pressure3pm", "Cloud3pm", "Rainfall",
]
