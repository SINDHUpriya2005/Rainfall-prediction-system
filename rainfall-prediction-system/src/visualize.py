"""Plotting helpers for EDA, model comparison and the web app.

NOTE: this module never forces a Matplotlib backend. Scripts / the Flask app
set ``Agg`` (headless) themselves; notebooks keep their inline backend.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc as auc_score, roc_curve

from . import config
from .evaluate import positive_scores

RAIN = "#2563eb"   # blue  - rain
DRY = "#f59e0b"    # amber - no rain
INK = "#1f2937"
sns.set_theme(style="whitegrid", context="notebook")


def _finish(fig, save_path: Optional[Path]):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def fig_to_base64(fig) -> str:
    """Encode a Matplotlib figure as a base64 PNG (for embedding in HTML)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------------- #
def plot_class_balance(y: pd.Series, save_path=None):
    counts = y.map({0: "No rain", 1: "Rain"}).value_counts().reindex(["No rain", "Rain"])
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(counts.index, counts.values, color=[DRY, RAIN])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100 * v / counts.sum():.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Class balance of RainTomorrow")
    ax.set_ylabel("Days")
    return _finish(fig, save_path)


def plot_missing_values(df: pd.DataFrame, save_path=None):
    cols = [c for c in config.FEATURES if c in df.columns]
    pct = (df[cols].isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(pct) + 1)))
    ax.barh(pct.index, pct.values, color=RAIN)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing data by feature")
    return _finish(fig, save_path)


def plot_correlation(df: pd.DataFrame, save_path=None):
    cols = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".1f", annot_kws={"size": 7},
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation of numeric features")
    return _finish(fig, save_path)


def plot_key_distributions(df: pd.DataFrame, save_path=None):
    feats = ["Humidity3pm", "Cloud3pm", "Pressure3pm", "Sunshine", "WindGustSpeed", "MaxTemp"]
    feats = [f for f in feats if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    label = df[config.TARGET].map({"No": "No rain tomorrow", "Yes": "Rain tomorrow"})
    for ax, f in zip(axes.ravel(), feats):
        sns.kdeplot(data=df.assign(_l=label), x=f, hue="_l", common_norm=False, fill=True, alpha=0.3,
                    palette={"No rain tomorrow": DRY, "Rain tomorrow": RAIN}, ax=ax, legend=ax is axes[0, 0])
        ax.set_title(f)
        ax.set_xlabel("")
    fig.suptitle("Key weather variables, split by next-day outcome", y=1.0)
    return _finish(fig, save_path)


def plot_monthly_rain(df: pd.DataFrame, save_path=None):
    monthly = (df[config.TARGET] == "Yes").groupby(df[config.DATE_COL].dt.month).mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(monthly.index, monthly.values, color=RAIN)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainy days (%)")
    ax.set_title("Share of rainy days by month")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #
def plot_model_comparison(results: pd.DataFrame, save_path=None):
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    nice = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    long = results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    long["metric"] = long["metric"].map(dict(zip(metrics, nice)))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.barplot(data=long, x="model", y="score", hue="metric", palette="viridis", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison on the hold-out test set")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    return _finish(fig, save_path)


def plot_roc_curves(fitted: Dict[str, object], X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, positive_scores(model, X_test))
        auc = auc_score(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig, save_path)


def plot_confusion_matrices(fitted: Dict[str, object], X_test, y_test, save_path=None):
    n = len(fitted)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    for ax, (name, model) in zip(np.atleast_1d(axes).ravel(), fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test), display_labels=["No rain", "Rain"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
        ax.grid(False)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    return _finish(fig, save_path)


def plot_single_confusion_matrix(model, X_test, y_test, title: str, save_path=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_test), display_labels=["No rain", "Rain"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.grid(False)
    return _finish(fig, save_path)


def get_feature_importance(pipe) -> Optional[pd.Series]:
    """Feature importance (or |coefficient|) with original feature names, if the model offers it."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_).ravel()
    else:
        return None
    return pd.Series(vals, index=names)


def plot_feature_importance(pipe, top_n: int = 15, title: str = "", save_path=None):
    imp = get_feature_importance(pipe)
    if imp is None:
        return None
    # Collapse one-hot columns back to their parent feature for readability.
    parents = {}
    for name, val in imp.items():
        base = next((c for c in config.CATEGORICAL_FEATURES if name.startswith(c + "_")), name)
        parents[base] = parents.get(base, 0.0) + float(val)
    top = pd.Series(parents).sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color=RAIN)
    ax.set_title(title or "Top feature importances")
    ax.set_xlabel("Importance")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Web-app visualisations (return base64 PNG strings)
# --------------------------------------------------------------------------- #
def probability_gauge(prob_rain: float) -> str:
    """Half-donut gauge showing the predicted chance of rain."""
    fig, ax = plt.subplots(figsize=(4.6, 2.9), subplot_kw={"aspect": "equal"})
    ax.pie([prob_rain, 1 - prob_rain, 1.0], colors=[RAIN, "#e5e7eb", "none"],
           startangle=180, counterclock=False, wedgeprops={"width": 0.34, "edgecolor": "white"})
    ax.text(0, -0.05, f"{prob_rain * 100:.0f}%", ha="center", va="center",
            fontsize=30, fontweight="bold", color=INK)
    ax.text(0, -0.40, "chance of rain tomorrow", ha="center", va="center", fontsize=10, color="#6b7280")
    ax.set_ylim(-0.6, 1.05)
    return fig_to_base64(fig)


def input_comparison_chart(user_values: Dict[str, float], reference: Dict[str, Dict[str, float]]) -> str:
    """Dot plot of the user's inputs against the average conditions seen the day
    *before* a rainy day and the day before a dry day (from the training data).

    Each feature gets its own axis in its own units, so nothing is distorted by
    normalisation: the further the dark marker sits toward the blue dot, the
    more the conditions resemble the run-up to rain.
    """
    feats = [f for f in ["Humidity3pm", "Cloud3pm", "Pressure3pm", "Sunshine", "WindGustSpeed"]
             if user_values.get(f) is not None and f in reference.get("rain", {}) and f in reference.get("dry", {})]
    if not feats:
        return ""

    fig, axes = plt.subplots(len(feats), 1, figsize=(6.6, 0.82 * len(feats) + 1.1), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        dry, rain, you = reference["dry"][f], reference["rain"][f], float(user_values[f])
        lo, hi = min(dry, rain, you), max(dry, rain, you)
        pad = max((hi - lo) * 0.18, 0.5)
        ax.hlines(0, lo - pad, hi + pad, color="#e5e7eb", lw=6, zorder=1)
        ax.hlines(0, min(dry, rain), max(dry, rain), color="#cbd5e1", lw=2, zorder=2)
        ax.scatter([dry], [0], s=90, color=DRY, zorder=3, label="Typical before a dry day")
        ax.scatter([rain], [0], s=90, color=RAIN, zorder=3, label="Typical before a rainy day")
        ax.scatter([you], [0], s=110, color=INK, marker="D", zorder=4, label="Your input")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_ylabel(f, rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Your conditions vs. typical conditions", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig_to_base64(fig)
