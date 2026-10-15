"""Flask web application for next-day rainfall prediction.

Run locally:   python -m app.app          (or: flask --app app.app run)
Production:    gunicorn app.app:app
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting inside the server

# Allow `python app/app.py` as well as `python -m app.app`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src import config  # noqa: E402
from src import visualize as viz  # noqa: E402
from src.predict import RainfallPredictor  # noqa: E402
from src.validation import validate_inputs  # noqa: E402

log = logging.getLogger("rainfall-app")

# Form layout: (section title, [fields]) - drives the HTML form.
FORM_SECTIONS = [
    ("Temperature", ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]),
    ("Humidity, pressure & clouds", ["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"]),
    ("Rain & sunshine", ["Rainfall", "Sunshine"]),
    ("Wind speed", ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]),
]

EXAMPLES = {
    "rainy": {
        "Location": "Sydney", "MinTemp": 16.5, "MaxTemp": 21.0, "Temp9am": 18.0, "Temp3pm": 19.5,
        "Humidity9am": 90, "Humidity3pm": 85, "Pressure9am": 1006, "Pressure3pm": 1004,
        "Cloud9am": 7, "Cloud3pm": 8, "Rainfall": 12.4, "Sunshine": 1.5,
        "WindGustSpeed": 54, "WindSpeed9am": 22, "WindSpeed3pm": 30,
        "WindGustDir": "SE", "WindDir9am": "SE", "WindDir3pm": "SSE", "RainToday": "Yes",
    },
    "sunny": {
        "Location": "Perth", "MinTemp": 14.0, "MaxTemp": 29.5, "Temp9am": 20.0, "Temp3pm": 28.0,
        "Humidity9am": 48, "Humidity3pm": 28, "Pressure9am": 1022, "Pressure3pm": 1020,
        "Cloud9am": 1, "Cloud3pm": 1, "Rainfall": 0, "Sunshine": 11.5,
        "WindGustSpeed": 31, "WindSpeed9am": 11, "WindSpeed3pm": 17,
        "WindGustDir": "SW", "WindDir9am": "E", "WindDir3pm": "SW", "RainToday": "No",
    },
}

GUIDANCE = {
    "Low": {
        "Agriculture & irrigation": "Low rain risk: normal irrigation can go ahead; good conditions for spraying and harvesting.",
        "Water resources": "Little inflow expected; no change to reservoir or drainage operations.",
        "Disaster risk": "No rain-driven hazard signal from these inputs.",
    },
    "Moderate": {
        "Agriculture & irrigation": "Uncertain: consider delaying irrigation or spraying and re-check the forecast.",
        "Water resources": "Some inflow possible; keep storm drains clear and monitor gauges.",
        "Disaster risk": "Watch conditions; low-lying and flood-prone areas should stay alert.",
    },
    "High": {
        "Agriculture & irrigation": "Likely rain: hold irrigation and fertiliser/pesticide application; protect harvested crops.",
        "Water resources": "Expect inflow: plan reservoir and stormwater capacity.",
        "Disaster risk": "Elevated flood / runoff risk if heavy rain persists - consult official warnings.",
    },
}


def create_app(predictor: RainfallPredictor | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # inputs are tiny; reject oversized bodies
    app.config["JSON_SORT_KEYS"] = False

    app.extensions["predictor"] = predictor
    app.extensions["load_error"] = None
    if predictor is None:
        try:
            app.extensions["predictor"] = RainfallPredictor()
        except Exception as exc:  # model missing / incompatible: app still starts and explains why
            log.exception("Could not load model")
            app.extensions["load_error"] = str(exc)

    def get_predictor():
        return app.extensions["predictor"]

    def common_context():
        p = get_predictor()
        meta = p.metadata if p else {}
        return {
            "ranges": config.NUMERIC_RANGES,
            "required": config.REQUIRED_INPUTS,
            "sections": FORM_SECTIONS,
            "wind_dirs": config.WIND_DIRECTIONS,
            "locations": p.locations if p else [],
            "model_name": p.model_name if p else None,
            "is_synthetic": meta.get("is_synthetic", False),
            "load_error": app.extensions["load_error"],
        }

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", form={}, errors={}, result=None,
                               examples=EXAMPLES, **common_context())

    @app.post("/predict")
    def predict():
        p = get_predictor()
        ctx = common_context()
        if p is None:
            return render_template("index.html", form=request.form, errors={}, result=None,
                                   examples=EXAMPLES, **ctx), 503

        values, errors = validate_inputs(request.form, p.locations)
        if errors:
            return render_template("index.html", form=request.form, errors=errors, result=None,
                                   examples=EXAMPLES, **ctx), 400
        try:
            result = p.predict(values)
            ref = p.metadata.get("reference_means")
            result["gauge"] = viz.probability_gauge(result["probability_rain"])
            result["comparison"] = viz.input_comparison_chart(values, ref) if ref else ""
            result["guidance"] = GUIDANCE[result["risk_level"]]
        except Exception:
            log.exception("Prediction failed")
            return render_template("index.html", form=request.form,
                                   errors={"_form": "Sorry, the prediction failed. Please check your inputs and try again."},
                                   result=None, examples=EXAMPLES, **ctx), 500
        return render_template("index.html", form=request.form, errors={}, result=result,
                               examples=EXAMPLES, **ctx)

    @app.get("/models")
    def models_page():
        p = get_predictor()
        meta = p.metadata if p else {}
        return render_template("models.html", meta=meta, **common_context())

    # -------------------------------------------------------------------- API
    @app.post("/api/predict")
    def api_predict():
        p = get_predictor()
        if p is None:
            return jsonify(error="Model not available", detail=app.extensions["load_error"]), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object with weather parameters."), 400
        values, errors = validate_inputs(payload, p.locations)
        if errors:
            return jsonify(error="Invalid input", details=errors), 400
        return jsonify(p.predict(values))

    @app.get("/health")
    def health():
        p = get_predictor()
        return jsonify(status="ok" if p else "model_unavailable", model=p.model_name if p else None)

    # ----------------------------------------------------------------- errors
    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="That page doesn't exist.", **common_context()), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", code=413, message="Request too large.", **common_context()), 413

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, message="Something went wrong on our side.", **common_context()), 500

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
