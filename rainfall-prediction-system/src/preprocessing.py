"""Leakage-safe preprocessing built with scikit-learn.

Everything that *learns* from data (median/mode imputation, scaling, category
discovery) lives inside a ``ColumnTransformer`` that is part of the model
``Pipeline``. When the pipeline is fitted - whether on the full training set or
inside each cross-validation fold - those statistics come only from the
training rows, so nothing from validation/test data leaks in.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Imputation + (optional) scaling for numerics, imputation + one-hot for categoricals.

    ``scale=True`` for distance/margin based models (LogReg, KNN, SVM);
    tree-based models are scale-invariant so they skip it.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
