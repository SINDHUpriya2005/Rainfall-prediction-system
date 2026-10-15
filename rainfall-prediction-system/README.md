# 🌧️ Rainfall Prediction System

A complete machine-learning project that predicts **whether it will rain tomorrow** from today's weather
observations (temperature, humidity, rainfall, wind, pressure, cloud cover, sunshine). It trains and compares six
classifiers, validates them properly (no data leakage, chronological hold-out, 5-fold stratified CV), picks the best,
and serves it through a Flask web app and a JSON API.

![CI](https://github.com/<your-username>/rainfall-prediction-system/actions/workflows/ci.yml/badge.svg)

> **Important - data note.** The repository runs out of the box on a **synthetic dataset** that has the same schema as
> the public Kaggle *Rain in Australia* (`weatherAUS.csv`) data. Numbers in this README come from that synthetic data and
> are **not real-world performance**. For real results, download the real file (see [`data/README.md`](data/README.md)),
> put it in `data/`, and run `python -m src.train`. The trained model committed in `models/` should be replaced by one
> trained on real data before you rely on its predictions.

---

## Features

* **Six models compared:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM (RBF), XGBoost
* **Leakage-safe preprocessing:** imputation, scaling and one-hot encoding live inside a scikit-learn `Pipeline`, so they are
  fitted on training rows only (and re-fitted inside every CV fold)
* **Realistic validation:** chronological train/test split (train on the past, test on the most recent 20%), plus a
  stratified random split for comparison, and **5-fold stratified cross-validation**
* **Class-imbalance handling:** analysis plus class weights (`balanced`, `scale_pos_weight` for XGBoost)
* **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves, feature importance
* **Model selection without test-set peeking:** best model = highest mean CV ROC-AUC on training data (F1 breaks ties)
* **Flask web app:** validated input form, prediction + probability, gauge and comparison charts, model-results page,
  JSON API (`/api/predict`), friendly error pages
* **Tested:** 35 pytest tests (validation, pipeline, leakage, splits, app, API) + GitHub Actions CI
* **Deploy-ready:** `Procfile`, `render.yaml`, `Dockerfile`, `gunicorn`

## Project structure

```
rainfall-prediction-system/
├── app/
│   ├── app.py                  # Flask application (pages + JSON API)
│   ├── templates/              # base.html, index.html, models.html, error.html
│   └── static/                 # css/style.css, img/ (charts shown on the results page)
├── src/
│   ├── config.py               # paths, feature lists, validation ranges
│   ├── synthetic_data.py       # weatherAUS-shaped data generator (fallback)
│   ├── data_loader.py          # load, clean, chronological/stratified split, class balance
│   ├── preprocessing.py        # leakage-safe ColumnTransformer
│   ├── models.py               # the six models
│   ├── evaluate.py             # CV, hold-out metrics, model selection
│   ├── visualize.py            # EDA, evaluation and web-app charts
│   ├── validation.py           # user-input validation
│   ├── predict.py              # RainfallPredictor (load model, predict)
│   └── train.py                # end-to-end training script
├── notebooks/rainfall_prediction.ipynb   # step-by-step, executed notebook
├── models/                     # best_model.joblib + metadata.json
├── reports/                    # model_comparison.csv and figures/
├── tests/                      # pytest suite
├── data/                       # put weatherAUS.csv here (see data/README.md)
├── requirements.txt  requirements-deploy.txt  Procfile  render.yaml  Dockerfile
└── .github/workflows/ci.yml
```

## Quick start

```bash
git clone https://github.com/<your-username>/rainfall-prediction-system.git
cd rainfall-prediction-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional but recommended) add the real dataset as data/weatherAUS.csv, then:
python -m src.train                # trains all 6 models, writes models/ and reports/

python -m app.app                  # open http://127.0.0.1:5000
```

Anaconda users: `conda create -n rain python=3.11 && conda activate rain && pip install -r requirements.txt`, then
`jupyter notebook notebooks/rainfall_prediction.ipynb`.

### Training options

```bash
python -m src.train --data path/to/file.csv         # use a specific CSV
python -m src.train --refit-full                    # refit the winner on ALL data before saving (for deployment)
python -m src.train --models "XGBoost,Random Forest"  # subset (quick experiments)
python -m src.train --no-stratified                 # skip the secondary stratified-split experiment
```

Full training takes roughly 4-5 minutes on a laptop; most of that is SVM cross-validation.

## Results (synthetic demo data)

Data: synthetic dataset (same schema as weatherAUS), 28,630 rows (2008-01-01 to 2017-12-29),
23.05% rainy days (imbalance ratio 3.34:1).
Chronological split: first 80% of dates for training/CV, last 20% as the untouched test period.

| Model | CV ROC-AUC (5-fold) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|--------------------|----------|-----------|--------|----|---------|
| Logistic Regression **(selected)** | 0.852 ± 0.004 | 0.781 | 0.535 | 0.745 | 0.623 | 0.851 |
| Decision Tree | 0.830 ± 0.005 | 0.795 | 0.564 | 0.682 | 0.618 | 0.832 |
| Random Forest | 0.849 ± 0.004 | 0.814 | 0.609 | 0.653 | 0.630 | 0.852 |
| KNN | 0.829 ± 0.006 | 0.814 | 0.662 | 0.474 | 0.552 | 0.832 |
| SVM | 0.831 ± 0.004 | 0.790 | 0.552 | 0.714 | 0.622 | 0.838 |
| XGBoost | 0.850 ± 0.003 | 0.783 | 0.538 | 0.735 | 0.621 | 0.850 |

*CV column = training period only (used for selection). Remaining columns = hold-out test period.*

**Reading the results honestly:**

* Several models are within noise of each other on ROC-AUC (0.85 vs 0.85), and the selected model won by a hair
  (the CV standard deviation is about 0.004). The synthetic data was generated from a mostly linear latent signal, so a
  linear model doing well is expected. **On the real weatherAUS data, tree ensembles (Random Forest / XGBoost) usually
  come out ahead - retrain and let the CV decide.**
* Accuracy alone is misleading with ~3:1 imbalance. KNN has the highest accuracy here but the **lowest recall** (it misses
  more than half of the rainy days). If missing rain is costly, prefer high recall / F1 or lower the decision threshold.
* The same models on a stratified random split:

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 0.780 | 0.604 | 0.846 |
| Decision Tree | 0.792 | 0.601 | 0.829 |
| Random Forest | 0.812 | 0.612 | 0.843 |
| KNN | 0.822 | 0.556 | 0.824 |
| SVM | 0.787 | 0.603 | 0.829 |
| XGBoost | 0.779 | 0.601 | 0.845 |

Charts (class balance, correlations, ROC curves, confusion matrices, feature importance) are in `reports/figures/` and
on the web app's **Model results** page.

## Methodology

| Concern | How it is handled |
|---------|-------------------|
| Missing values | `SimpleImputer` (median numeric / most-frequent categorical) inside the pipeline |
| Categorical features | One-hot encoding (`handle_unknown="ignore"`) so unseen locations don't crash |
| Scaling | `StandardScaler` only for Logistic Regression, KNN and SVM (tree models are scale-invariant) |
| Data leakage | Everything that learns from data is inside the `Pipeline`; `RainTomorrow` and `Date` are never features; a test asserts imputer statistics equal *training* medians |
| Time dependence | Primary evaluation is a **chronological** split; stratified split reported separately |
| Class imbalance | Class weights; metrics beyond accuracy; stratified CV folds |
| Model selection | Mean 5-fold stratified CV ROC-AUC on training data (not the test set) |
| Seasonality | `Month` derived from `Date` |

*Caveat:* the stratified CV shuffles days within the training period, which is standard but slightly optimistic for
time series. The chronological hold-out is the more trustworthy estimate.

## Web application

* **/** - form for today's observations. Required: max temperature, humidity (3pm), pressure (3pm), cloud (3pm), rainfall.
  Everything else is optional and imputed with training-data typical values (the result lists which fields were imputed).
  "Try an example" buttons pre-fill a rainy and a clear day.
* Output: rain / no-rain prediction, probability, risk level (Low < 30%, Moderate 30-60%, High > 60%), probability
  gauge, comparison against typical pre-rain and pre-dry conditions, and practical notes for agriculture, water resources
  and disaster-risk use.
* **/models** - metric table and charts for all six models.
* **Validation:** type, range, finite-number, category and cross-field checks (min temp <= max temp) with
  per-field messages; oversized requests rejected; error pages for 404/413/500; the app still starts and explains the
  problem if no trained model is present.

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "Location": "Sydney", "MaxTemp": 21, "MinTemp": 16.5, "Humidity3pm": 85,
  "Pressure3pm": 1004, "Cloud3pm": 8, "Rainfall": 12.4, "Sunshine": 1.5, "Month": 6
}'
```

Returns `prediction`, `will_rain`, `probability_rain`, `risk_level`, `model`, `imputed_fields` (HTTP 400 with `details`
on invalid input). `GET /health` is a liveness check.

## Testing

```bash
python -m pytest -q        # 35 tests, a few seconds
```

## Deploy

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: rainfall prediction system"
git branch -M main
git remote add origin https://github.com/<your-username>/rainfall-prediction-system.git
git push -u origin main
```

GitHub Actions (`.github/workflows/ci.yml`) then runs the tests on every push.

### 2. Host the web app (pick one)

* **Render (easiest):** New → Blueprint → select the repo. `render.yaml` configures everything.
* **Heroku / Railway / any Procfile host:** `Procfile` runs `gunicorn app.app:app`.
* **Docker:** `docker build -t rainfall . && docker run -p 8000:8000 rainfall`
* **Not GitHub Pages:** Pages only serves static files; this app needs a Python server.

The model file is committed (small), so the host does not need to retrain. If you retrain on real data, commit the new
`models/best_model.joblib` and `models/metadata.json`. Use the same scikit-learn version for training and serving
(the version is recorded in `metadata.json`).

## Real-world applications

* **Agriculture & irrigation:** decide whether to irrigate, spray or harvest the next day
* **Meteorological analysis:** quick baselines and feature-importance insights from station data
* **Water-resource planning:** anticipate inflow, reservoir and stormwater operations
* **Disaster-risk assessment:** flag high-probability rain days as an early signal (always defer to official warnings)

## Requirements

| | |
|-|-|
| Hardware | Intel i3 or better, 4 GB RAM (8 GB recommended), 2 GB free disk, internet for downloading data/packages |
| Software | Windows / Linux / macOS, Python 3.8+ (3.10+ recommended), Anaconda (optional), Jupyter, VS Code or PyCharm, a modern browser |

## Limitations

* A statistical model of one station-day's readings is **not** a numerical weather forecast; it ignores upstream
  weather systems, satellite and radar data. Use it as decision support, not as an official forecast.
* Predictions for locations or climates unlike the training data may be unreliable.
* Reported metrics from the bundled synthetic data are illustrative only.

## License

MIT - see [LICENSE](LICENSE).
