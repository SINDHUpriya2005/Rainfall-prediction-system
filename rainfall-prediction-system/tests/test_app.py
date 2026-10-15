import json


def test_home_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Will it rain tomorrow" in r.data


def test_models_page_renders(client):
    assert client.get("/models").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_unknown_page_404(client):
    assert client.get("/nope").status_code == 404


def test_form_prediction_shows_result_and_charts(client, valid_form):
    r = client.post("/predict", data=valid_form)
    assert r.status_code == 200
    assert b"Prediction for tomorrow" in r.data
    assert b"data:image/png;base64," in r.data      # gauge + comparison chart


def test_form_validation_errors(client, valid_form):
    bad = {**valid_form, "Humidity3pm": "250", "Cloud3pm": ""}
    r = client.post("/predict", data=bad)
    assert r.status_code == 400
    assert b"Humidity at 3pm must be between" in r.data
    assert b"Cloud cover at 3pm is required" in r.data
    assert b"Prediction for tomorrow" not in r.data


def test_api_prediction(client, valid_form):
    r = client.post("/api/predict", json=valid_form)
    body = r.get_json()
    assert r.status_code == 200
    assert 0 <= body["probability_rain"] <= 1
    assert abs(body["probability_rain"] + body["probability_no_rain"] - 1) < 1e-3
    assert body["risk_level"] in {"Low", "Moderate", "High"}


def test_api_rejects_bad_input(client):
    assert client.post("/api/predict", json={"MaxTemp": "hot"}).status_code == 400
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400


def test_wet_day_scores_higher_than_dry_day(client):
    wet = dict(MaxTemp=20, MinTemp=16, Humidity3pm=95, Pressure3pm=1002, Cloud3pm=8, Rainfall=15, Sunshine=0.5)
    dry = dict(MaxTemp=30, MinTemp=15, Humidity3pm=15, Pressure3pm=1024, Cloud3pm=0, Rainfall=0, Sunshine=12)
    p_wet = client.post("/api/predict", json=wet).get_json()["probability_rain"]
    p_dry = client.post("/api/predict", json=dry).get_json()["probability_rain"]
    assert p_wet > 0.6 > 0.3 > p_dry


def test_app_starts_without_model_and_reports_503(tmp_path, monkeypatch):
    from app import app as app_module
    monkeypatch.setattr(app_module, "RainfallPredictor",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no model")))
    a = app_module.create_app()
    c = a.test_client()
    assert c.get("/").status_code == 200
    assert c.post("/api/predict", json={}).status_code == 503
