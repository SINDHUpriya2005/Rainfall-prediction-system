"""End-to-end training script.

    python -m src.train                    # full run (synthetic data if no real CSV)
    python -m src.train --data path.csv    # use a specific CSV
    python -m src.train --refit-full       # refit the chosen model on ALL data for deployment
    python -m src.train --models "XGBoost,Random Forest"   # subset, handy for quick tests
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from . import config  # noqa: E402
from . import visualize as viz  # noqa: E402
from .data_loader import class_balance, clean_data, get_xy, load_raw_data, split_data  # noqa: E402
from .evaluate import compare_models, select_best_model  # noqa: E402
from .models import MODEL_NAMES, build_all_models, build_model  # noqa: E402

log = logging.getLogger("train")

REFERENCE_FEATURES = ["Humidity3pm", "Cloud3pm", "Pressure3pm", "Sunshine", "WindGustSpeed"]


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)}")


def run(data_path=None, refit_full: bool = False, model_subset=None, secondary_split: bool = True) -> dict:
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR, config.APP_IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    raw, source = load_raw_data(data_path)
    df = clean_data(raw)
    X_all, y_all = get_xy(df)
    balance = class_balance(y_all)
    log.info("Data source: %s | rows after cleaning: %d", source, len(df))
    log.info("Class balance: %s", balance)

    # ------------------------------------------------------------------- EDA
    viz.plot_class_balance(y_all, config.FIGURES_DIR / "class_balance.png")
    viz.plot_missing_values(df, config.FIGURES_DIR / "missing_values.png")
    viz.plot_correlation(df, config.FIGURES_DIR / "correlation.png")
    viz.plot_key_distributions(df, config.FIGURES_DIR / "distributions.png")
    viz.plot_monthly_rain(df, config.FIGURES_DIR / "monthly_rain.png")

    # --------------------------------- primary experiment: chronological split
    X_tr, X_te, y_tr, y_te = split_data(df, "chronological")
    log.info("Chronological split -> train %d rows (to %s), test %d rows (from %s)",
             len(X_tr), df.loc[X_tr.index, config.DATE_COL].max().date(),
             len(X_te), df.loc[X_te.index, config.DATE_COL].min().date())

    models = build_all_models(y_tr)
    if model_subset:
        models = {k: v for k, v in models.items() if k in model_subset}
    results, fitted = compare_models(models, X_tr, X_te, y_tr, y_te, do_cv=True)
    best_name = select_best_model(results)
    results["selected"] = results["model"] == best_name
    results.to_csv(config.RESULTS_PATH, index=False)
    log.info("Selected model (by 5-fold CV ROC-AUC on training data): %s", best_name)

    viz.plot_model_comparison(results, config.FIGURES_DIR / "model_comparison.png")
    viz.plot_roc_curves(fitted, X_te, y_te, config.FIGURES_DIR / "roc_curves.png")
    viz.plot_confusion_matrices(fitted, X_te, y_te, config.FIGURES_DIR / "confusion_matrices.png")
    viz.plot_single_confusion_matrix(
        fitted[best_name], X_te, y_te, f"{best_name} - confusion matrix",
        config.FIGURES_DIR / "best_confusion_matrix.png")
    viz.plot_feature_importance(
        fitted[best_name], title=f"{best_name} - top features",
        save_path=config.FIGURES_DIR / "feature_importance.png")

    # ------------------- secondary experiment: stratified random split (no CV)
    strat_results = None
    if secondary_split:
        Xs_tr, Xs_te, ys_tr, ys_te = split_data(df, "stratified")
        s_models = build_all_models(ys_tr)
        if model_subset:
            s_models = {k: v for k, v in s_models.items() if k in model_subset}
        strat_results, _ = compare_models(s_models, Xs_tr, Xs_te, ys_tr, ys_te, do_cv=False)
        strat_results.to_csv(config.REPORTS_DIR / "model_comparison_stratified_split.csv", index=False)

    # ------------------------------------------------- final model for the app
    if best_name == "SVM":  # need predict_proba for the web app -> Platt scaling
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        final = build_model("SVM", scale_pos_weight=spw, svm_probability=True)
    else:
        final = fitted[best_name]
    if refit_full:
        final.fit(X_all, y_all)
        log.info("Final model refitted on ALL %d rows (reported test metrics refer to the train-only fit).", len(X_all))
    elif best_name == "SVM":
        final.fit(X_tr, y_tr)
    joblib.dump(final, config.MODEL_PATH, compress=3)

    # ------------------------------------------------------------- metadata
    ref_src = X_tr.assign(_y=y_tr)
    reference = {
        "rain": ref_src[ref_src["_y"] == 1][REFERENCE_FEATURES].mean().round(2).to_dict(),
        "dry": ref_src[ref_src["_y"] == 0][REFERENCE_FEATURES].mean().round(2).to_dict(),
    }
    best_row = results[results["model"] == best_name].iloc[0].to_dict()
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "is_synthetic": "synthetic" in source,
        "n_rows": int(len(df)),
        "date_range": [str(df[config.DATE_COL].min().date()), str(df[config.DATE_COL].max().date())],
        "split": "chronological (earliest 80% train / latest 20% test)",
        "refit_on_full_data": bool(refit_full),
        "class_balance": balance,
        "features": list(X_all.columns),
        "locations": sorted(map(str, df["Location"].dropna().unique())) if "Location" in df else [],
        "wind_directions": config.WIND_DIRECTIONS,
        "reference_means": reference,
        "metrics": {k: v for k, v in best_row.items() if k.startswith(("cv_", "test_"))},
        "sklearn_version": sklearn.__version__,
        "results": results.drop(columns=["selected"]).round(4).to_dict(orient="records"),
        "stratified_split_results": (
            strat_results.round(4).to_dict(orient="records") if strat_results is not None else []
        ),
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_safe)

    # Copy the figures the web app displays.
    for name in ["model_comparison.png", "roc_curves.png", "best_confusion_matrix.png", "feature_importance.png"]:
        src = config.FIGURES_DIR / name
        if src.exists():
            (config.APP_IMG_DIR / name).write_bytes(src.read_bytes())

    _print_summary(results, strat_results, best_name, source)
    return metadata


def _print_summary(results, strat_results, best_name, source):
    cols = ["model", "cv_roc_auc_mean", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(f"DATA: {source}")
    print("CHRONOLOGICAL SPLIT (primary) - hold-out test metrics")
    print(results[cols].round(3).to_string(index=False))
    print(f"\n>>> BEST MODEL (5-fold stratified CV ROC-AUC): {best_name}")
    if strat_results is not None:
        print("\nSTRATIFIED RANDOM SPLIT (secondary, usually optimistic for time series)")
        print(strat_results[["model", "test_accuracy", "test_f1", "test_roc_auc"]].round(3).to_string(index=False))
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Train and compare rainfall prediction models.")
    parser.add_argument("--data", default=None, help="Path to a weatherAUS-style CSV (default: auto).")
    parser.add_argument("--refit-full", action="store_true", help="Refit the selected model on all data for deployment.")
    parser.add_argument("--models", default=None, help="Comma-separated subset of models to run.")
    parser.add_argument("--no-stratified", action="store_true", help="Skip the secondary stratified-split experiment.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    subset = [m.strip() for m in args.models.split(",")] if args.models else None
    if subset:
        unknown = [m for m in subset if m not in MODEL_NAMES]
        if unknown:
            parser.error(f"Unknown models {unknown}. Choose from {MODEL_NAMES}")
    run(args.data, args.refit_full, subset, secondary_split=not args.no_stratified)


if __name__ == "__main__":
    main()
