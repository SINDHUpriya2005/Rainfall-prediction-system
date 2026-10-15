"""Load the trained pipeline and make next-day rainfall predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from . import config

THRESHOLD = 0.5


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Moderate"
    return "High"


class RainfallPredictor:
    """Thin wrapper around the saved sklearn Pipeline plus its metadata."""

    def __init__(self, model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {self.model_path}. Run `python -m src.train` first."
            )
        self.model = joblib.load(self.model_path)
        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @property
    def model_name(self) -> str:
        return self.metadata.get("model_name", type(self.model.named_steps["model"]).__name__)

    @property
    def locations(self) -> List[str]:
        return self.metadata.get("locations", [])

    def _to_frame(self, values: Mapping[str, Any]) -> pd.DataFrame:
        row = {}
        for col in config.FEATURES:
            v = values.get(col)
            row[col] = np.nan if v is None else v
        frame = pd.DataFrame([row], columns=config.FEATURES)
        for col in config.NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in config.CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype(object)
        return frame

    def predict(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        """``values`` must already be validated (see ``src.validation``)."""
        frame = self._to_frame(values)
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(frame)[0, 1])
        else:  # defensive fallback: squash the margin
            prob = float(1 / (1 + np.exp(-self.model.decision_function(frame)[0])))
        will_rain = prob >= THRESHOLD
        return {
            "prediction": "Rain expected tomorrow" if will_rain else "No rain expected tomorrow",
            "will_rain": bool(will_rain),
            "probability_rain": round(prob, 4),
            "probability_no_rain": round(1 - prob, 4),
            "risk_level": risk_level(prob),
            "threshold": THRESHOLD,
            "model": self.model_name,
            "imputed_fields": [c for c in config.FEATURES if values.get(c) is None],
        }
