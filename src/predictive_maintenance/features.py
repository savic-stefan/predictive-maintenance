"""Leakage-safe feature selection and preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance.data import get_feature_columns


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only measurements available before failure and machine type."""
    columns = get_feature_columns(frame)
    return frame.loc[:, columns].copy()


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric measurements and one-hot encode categorical measurements."""
    features = feature_frame(frame)
    categorical = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
