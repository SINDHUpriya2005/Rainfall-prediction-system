"""Model zoo: the six classifiers compared in this project.

Class imbalance (roughly 3-4 dry days for every rainy day) is handled with
class weights rather than resampling, so the evaluation sets stay untouched.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessor

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM",
    "XGBoost",
]


def build_model(name: str, scale_pos_weight: float = 1.0, svm_probability: bool = False) -> Pipeline:
    """Return an *unfitted* ``Pipeline(preprocessor -> classifier)`` for ``name``."""
    rs = config.RANDOM_STATE

    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs)
        scale = True
    elif name == "Decision Tree":
        clf = DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=40, class_weight="balanced", random_state=rs
        )
        scale = False
    elif name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=rs,
        )
        scale = False
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
        scale = True
    elif name == "SVM":
        # probability=True (Platt scaling) is slow, so it is only switched on when
        # the SVM is exported for the web app. Metrics use decision_function.
        clf = SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=svm_probability, cache_size=1000, random_state=rs,
        )
        scale = True
    elif name == "XGBoost":
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=2, random_state=rs, tree_method="hist",
        )
        scale = False
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([("preprocess", build_preprocessor(scale=scale)), ("model", clf)])


def build_all_models(y_train) -> Dict[str, Pipeline]:
    """Build every model; XGBoost's ``scale_pos_weight`` comes from the training labels."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)
    return {name: build_model(name, scale_pos_weight=spw) for name in MODEL_NAMES}
