"""Synthetic weather data generator.

Produces a dataset with exactly the same schema as the Kaggle "Rain in
Australia" (weatherAUS.csv) dataset, so the whole pipeline runs out of the box
without any download. For real results, drop the real ``weatherAUS.csv`` into
``data/`` and re-run training - the loader prefers it automatically.

How the data is generated
-------------------------
For every location a latent, autocorrelated "weather state" ``z`` is simulated
day by day (seasonal component + AR(1) noise). Rain occurs on days where ``z``
exceeds a location-specific threshold. All observed variables (humidity,
cloud, pressure, sunshine, temperature, wind...) are noisy functions of ``z``,
so they carry realistic, partial information about whether it will rain the
next day. Missing values are injected at rates similar to the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# location -> (mean max temp, seasonal temp amplitude, base humidity, wetness
#              seasonal phase shift (days), preferred wind direction index)
_CLIMATE = {
    "Sydney":    (22.5, 5.5, 62, 60, 8),
    "Melbourne": (20.0, 6.5, 58, 200, 12),
    "Brisbane":  (26.5, 4.0, 60, 30, 6),
    "Perth":     (25.0, 7.0, 52, 190, 10),
    "Adelaide":  (22.5, 8.0, 50, 210, 11),
    "Canberra":  (19.5, 8.5, 52, 150, 13),
    "Darwin":    (32.0, 2.0, 68, 15, 7),
    "Hobart":    (17.5, 5.0, 56, 220, 13),
}


def _simulate_location(name: str, n_days: int, start: str, rng: np.random.Generator) -> pd.DataFrame:
    tmax_mean, tmax_amp, hum_base, wet_phase, wind_pref = _CLIMATE[name]
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()

    # Southern-hemisphere seasonality: warm in Jan (doy ~ 15), cold in Jul.
    temp_season = np.cos(2 * np.pi * (doy - 15) / 365.25)
    wet_season = np.cos(2 * np.pi * (doy - wet_phase) / 365.25)

    # Latent weather state with day-to-day persistence.
    z = np.zeros(n_days)
    eps = rng.normal(0, 0.62, n_days)
    for t in range(1, n_days):
        z[t] = 0.74 * z[t - 1] + eps[t]
    z = z + 0.35 * wet_season
    z = (z - z.mean()) / z.std()

    rain_threshold = np.quantile(z, 0.77)  # ~23% of days rainy
    rainy = z > rain_threshold

    n = n_days
    clip = np.clip

    humidity3 = clip(hum_base - 8 + 15 * z + rng.normal(0, 11, n), 5, 100)
    humidity9 = clip(hum_base + 12 + 10 * z + rng.normal(0, 11, n), 10, 100)
    cloud3 = clip(np.round(3.6 + 1.6 * z + rng.normal(0, 1.4, n)), 0, 8)
    cloud9 = clip(np.round(3.9 + 1.4 * z + rng.normal(0, 1.5, n)), 0, 8)
    pressure3 = clip(1016 - 4.6 * z + 2.0 * temp_season * -1 + rng.normal(0, 3.4, n), 985, 1042)
    pressure9 = clip(pressure3 + 1.2 + rng.normal(0, 1.6, n), 985, 1045)
    sunshine = clip(8.6 + 1.8 * temp_season - 2.3 * z + rng.normal(0, 2.3, n), 0, 14)
    max_temp = tmax_mean + tmax_amp * temp_season - 1.3 * z + rng.normal(0, 3.0, n)
    min_temp = max_temp - 10.5 + 1.2 * z + rng.normal(0, 2.4, n)
    temp9 = min_temp + 0.42 * (max_temp - min_temp) + rng.normal(0, 1.4, n)
    temp3 = max_temp - rng.uniform(0.4, 2.6, n)
    gust = clip(38 + 5.5 * z + rng.normal(0, 11, n), 9, 120)
    wind9 = clip(0.36 * gust + rng.normal(0, 4.5, n), 0, 70)
    wind3 = clip(0.46 * gust + rng.normal(0, 4.5, n), 0, 80)

    # Rainfall (mm): rainy days >= 1 mm, dry days mostly 0 with small traces.
    heavy = rng.exponential(5.0, n) * (1 + 0.35 * clip(z - rain_threshold, 0, None))
    rainfall = np.where(
        rainy,
        np.round(1.0 + heavy, 1),
        np.round(np.where(rng.random(n) < 0.18, rng.uniform(0.0, 0.9, n), 0.0), 1),
    )

    def wind_dir(spread: float, shift_with_z: float) -> np.ndarray:
        idx = np.round(wind_pref + shift_with_z * z + rng.normal(0, spread, n)).astype(int) % 16
        return np.array(config.WIND_DIRECTIONS)[idx]

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Location": name,
        "MinTemp": min_temp.round(1),
        "MaxTemp": max_temp.round(1),
        "Rainfall": rainfall,
        "Evaporation": clip(5 + 2.2 * temp_season - 1.5 * z + rng.normal(0, 1.8, n), 0, 20).round(1),
        "Sunshine": sunshine.round(1),
        "WindGustDir": wind_dir(2.6, 1.2),
        "WindGustSpeed": gust.round(0),
        "WindDir9am": wind_dir(3.0, 0.8),
        "WindDir3pm": wind_dir(2.8, 1.0),
        "WindSpeed9am": wind9.round(0),
        "WindSpeed3pm": wind3.round(0),
        "Humidity9am": humidity9.round(0),
        "Humidity3pm": humidity3.round(0),
        "Pressure9am": pressure9.round(1),
        "Pressure3pm": pressure3.round(1),
        "Cloud9am": cloud9,
        "Cloud3pm": cloud3,
        "Temp9am": temp9.round(1),
        "Temp3pm": temp3.round(1),
    })
    df["RainToday"] = np.where(df["Rainfall"] >= 1.0, "Yes", "No")
    df["RainTomorrow"] = df["RainToday"].shift(-1)  # last day of each series -> NaN
    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Blank out values at rates comparable to the real weatherAUS data."""
    rates = {
        "Sunshine": 0.38, "Evaporation": 0.42, "Cloud9am": 0.34, "Cloud3pm": 0.36,
        "Pressure9am": 0.10, "Pressure3pm": 0.10, "WindGustDir": 0.06,
        "WindGustSpeed": 0.06, "WindDir9am": 0.07, "WindDir3pm": 0.03,
        "Humidity9am": 0.02, "Humidity3pm": 0.03, "WindSpeed9am": 0.01,
        "WindSpeed3pm": 0.02, "Temp9am": 0.01, "Temp3pm": 0.02,
        "MinTemp": 0.01, "MaxTemp": 0.01, "Rainfall": 0.02,
    }
    df = df.copy()
    for col, rate in rates.items():
        df.loc[rng.random(len(df)) < rate, col] = np.nan
    # RainToday follows Rainfall; a few target labels are missing as well.
    df.loc[df["Rainfall"].isna(), "RainToday"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "RainTomorrow"] = np.nan
    return df


def generate_synthetic_weather(
    years: int = 10,
    start: str = "2008-01-01",
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a weatherAUS-shaped dataset (~29k rows for 10 years x 8 sites)."""
    rng = np.random.default_rng(seed)
    n_days = int(round(years * 365.25))
    frames = [_simulate_location(loc, n_days, start, rng) for loc in config.LOCATIONS]
    df = pd.concat(frames, ignore_index=True)
    return _inject_missing(df, rng)


if __name__ == "__main__":
    out = generate_synthetic_weather()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.SYNTHETIC_DATA_PATH, index=False)
    print(f"Wrote {len(out):,} rows to {config.SYNTHETIC_DATA_PATH}")
