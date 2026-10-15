"""Evaluation helpers: cross-validation, hold-out metrics, model comparison."""

from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

log = logging.getLogger(__name__)

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def positive_scores(model, X) -> np.ndarray:
    """Score for the positive (rain) class: probability if available, else margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def holdout_metrics(model, X, y) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion-matrix cells."""
    y_pred = model.predict(X)
    scores = positive_scores(model, X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_validate_model(model, X, y, n_splits: int = config.CV_FOLDS) -> Dict[str, float]:
    """5-fold *stratified* CV. The whole Pipeline is refitted inside every fold,
    so imputers / scalers / encoders never see the validation fold (no leakage)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    res = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING, n_jobs=1, error_score="raise")
    out = {}
    for metric in CV_SCORING:
        vals = res[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(vals))
        out[f"cv_{metric}_std"] = float(np.std(vals))
    return out


def compare_models(models, X_train, X_test, y_train, y_test, do_cv: bool = True):
    """Fit every model on the training data and evaluate it.

    Returns ``(results_dataframe, fitted_models)``. Each row holds the optional
    cross-validation summary (training data only) plus hold-out test metrics.
    """
    rows, fitted = [], {}
    for name, pipe in models.items():
        row = {"model": name}
        t0 = time.time()
        if do_cv:
            log.info("[%s] running %d-fold stratified CV ...", name, config.CV_FOLDS)
            row.update(cross_validate_model(pipe, X_train, y_train))
        log.info("[%s] fitting on full training set ...", name)
        pipe.fit(X_train, y_train)
        row.update({f"test_{k}": v for k, v in holdout_metrics(pipe, X_test, y_test).items()})
        row["seconds"] = round(time.time() - t0, 1)
        log.info(
            "[%s] test acc=%.3f f1=%.3f auc=%.3f (%.1fs)",
            name, row["test_accuracy"], row["test_f1"], row["test_roc_auc"], row["seconds"],
        )
        rows.append(row)
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


def select_best_model(results: pd.DataFrame) -> str:
    """Pick the best model using *cross-validation on the training data only*
    (ROC-AUC first, F1 as tie-breaker) - never the test set, so the final test
    numbers remain an unbiased estimate."""
    ranked = results.sort_values(
        ["cv_roc_auc_mean", "cv_f1_mean"], ascending=False
    ).reset_index(drop=True)
    return str(ranked.loc[0, "model"])
