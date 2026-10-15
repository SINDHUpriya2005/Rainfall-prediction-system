"""Input validation for user-supplied weather observations (web form and JSON API)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from . import config


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate_inputs(
    raw: Mapping[str, Any],
    locations: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Validate and clean raw input.

    Returns ``(clean_values, errors)``. ``clean_values`` contains every model
    feature (``None`` where the user left an optional field empty); ``errors``
    maps field name -> human readable message and is empty when input is valid.
    """
    clean: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    # ---- numeric fields --------------------------------------------------
    for field, (lo, hi, unit, label) in config.NUMERIC_RANGES.items():
        value = raw.get(field)

        if field == "Month":
            if _is_blank(value):
                clean[field] = date.today().month
                continue
            try:
                month = int(float(str(value).strip()))
                if month != float(str(value).strip()):
                    raise ValueError
            except (TypeError, ValueError):
                errors[field] = "Month must be a whole number from 1 to 12."
                continue
            if not 1 <= month <= 12:
                errors[field] = "Month must be between 1 and 12."
            else:
                clean[field] = month
            continue

        if _is_blank(value):
            if field in config.REQUIRED_INPUTS:
                errors[field] = f"{label} is required."
            clean[field] = None
            continue

        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            errors[field] = f"{label} must be a number."
            clean[field] = None
            continue

        if not math.isfinite(number):
            errors[field] = f"{label} must be a finite number."
            clean[field] = None
        elif not lo <= number <= hi:
            errors[field] = f"{label} must be between {lo:g} and {hi:g} {unit}.".replace(" .", ".")
            clean[field] = None
        else:
            clean[field] = number

    # ---- categorical fields ----------------------------------------------
    for field in ("WindGustDir", "WindDir9am", "WindDir3pm"):
        value = raw.get(field)
        if _is_blank(value):
            clean[field] = None
        elif str(value).strip().upper() in config.WIND_DIRECTIONS:
            clean[field] = str(value).strip().upper()
        else:
            errors[field] = f"{field} must be one of the 16 compass points (N, NNE, ... NNW)."
            clean[field] = None

    loc = raw.get("Location")
    if _is_blank(loc):
        clean["Location"] = None
    else:
        loc = str(loc).strip()
        allowed = list(locations) if locations else []
        match = next((a for a in allowed if a.lower() == loc.lower()), None)
        if allowed and match is None:
            errors["Location"] = "Unknown location. Choose one from the list."
            clean["Location"] = None
        else:
            clean["Location"] = match or loc

    rain_today = raw.get("RainToday")
    if _is_blank(rain_today):
        # Derive from today's rainfall (>= 1 mm is the standard definition).
        rainfall = clean.get("Rainfall")
        clean["RainToday"] = None if rainfall is None else ("Yes" if rainfall >= 1.0 else "No")
    elif str(rain_today).strip().lower() in ("yes", "no"):
        clean["RainToday"] = str(rain_today).strip().title()
    else:
        errors["RainToday"] = "Rain today must be Yes or No."
        clean["RainToday"] = None

    # ---- cross-field sanity checks ----------------------------------------
    tmin, tmax = clean.get("MinTemp"), clean.get("MaxTemp")
    if tmin is not None and tmax is not None and tmin > tmax and "MinTemp" not in errors and "MaxTemp" not in errors:
        errors["MinTemp"] = "Minimum temperature cannot be higher than maximum temperature."

    return clean, errors
