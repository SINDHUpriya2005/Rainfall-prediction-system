from src.validation import validate_inputs

LOCS = ["Sydney", "Perth"]
BASE = {"MaxTemp": "24", "Humidity3pm": "70", "Pressure3pm": "1010", "Cloud3pm": "6", "Rainfall": "0"}


def test_valid_minimal_input():
    clean, errors = validate_inputs(BASE, LOCS)
    assert errors == {}
    assert clean["MaxTemp"] == 24.0
    assert clean["Sunshine"] is None            # optional -> None
    assert clean["RainToday"] == "No"           # derived from Rainfall < 1 mm


def test_rain_today_derived_from_rainfall():
    clean, _ = validate_inputs({**BASE, "Rainfall": "3.2"}, LOCS)
    assert clean["RainToday"] == "Yes"


def test_required_fields_reported():
    _, errors = validate_inputs({}, LOCS)
    for f in ("MaxTemp", "Humidity3pm", "Pressure3pm", "Cloud3pm", "Rainfall"):
        assert f in errors


def test_non_numeric_and_out_of_range():
    _, errors = validate_inputs({**BASE, "Humidity3pm": "abc", "Pressure3pm": "5000"}, LOCS)
    assert "Humidity3pm" in errors and "Pressure3pm" in errors


def test_nan_and_inf_rejected():
    _, errors = validate_inputs({**BASE, "MaxTemp": "nan", "Cloud3pm": "inf"}, LOCS)
    assert "MaxTemp" in errors and "Cloud3pm" in errors


def test_min_temp_above_max_temp():
    _, errors = validate_inputs({**BASE, "MinTemp": "30"}, LOCS)
    assert "MinTemp" in errors


def test_bad_categories():
    _, errors = validate_inputs({**BASE, "WindGustDir": "UP", "Location": "Atlantis", "RainToday": "maybe"}, LOCS)
    assert {"WindGustDir", "Location", "RainToday"} <= set(errors)


def test_location_case_insensitive_and_wind_normalised():
    clean, errors = validate_inputs({**BASE, "Location": "sydney", "WindGustDir": "nne"}, LOCS)
    assert errors == {} and clean["Location"] == "Sydney" and clean["WindGustDir"] == "NNE"


def test_month_validation():
    assert "Month" in validate_inputs({**BASE, "Month": "13"}, LOCS)[1]
    assert "Month" in validate_inputs({**BASE, "Month": "2.5"}, LOCS)[1]
    assert validate_inputs({**BASE, "Month": "7"}, LOCS)[0]["Month"] == 7
