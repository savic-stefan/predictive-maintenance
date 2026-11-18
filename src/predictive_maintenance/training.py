"""Reproducible model comparison on stratified train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import TARGET_COLUMN
from predictive_maintenance.features import build_preprocessor

RANDOM_STATE = 42


@dataclass
class TrainingResult:
    """Fitted validation candidates and selection details."""

    models: dict[str, Pipeline]
    validation_metrics: dict[str, dict[str, float]]
    selected_name: str
    selected_model: Pipeline
    selected_threshold: float
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_dataset(
    frame: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create 60/20/20 stratified train, validation, and test partitions."""
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN].astype(int)
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def _make_candidates(
    X_train: pd.DataFrame, random_state: int
) -> dict[str, Pipeline]:
    """Build pipelines so preprocessing is fitted on training data only."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    return float(thresholds[int(np.argmax(f1))])


def train_and_select(
    frame: pd.DataFrame, random_state: int = RANDOM_STATE
) -> TrainingResult:
    """Fit candidates, select by validation average precision, and tune threshold."""
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(frame, random_state=random_state)

    # Predicting the training-set failure rate gives a no-skill probability baseline.
    baseline_probability = np.full(len(y_validation), float(y_train.mean()))
    validation_metrics: dict[str, dict[str, float]] = {
        "majority_baseline": {
            "average_precision": float(
                average_precision_score(y_validation, baseline_probability)
            ),
            "roc_auc": 0.5,
        }
    }
    models = _make_candidates(X_train, random_state)

    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        validation_metrics[name] = {
            "average_precision": float(average_precision_score(y_validation, probabilities)),
            "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        }

    selected_name = max(
        models,
        key=lambda name: validation_metrics[name]["average_precision"],
    )
    selected_model = models[selected_name]
    selected_probabilities = selected_model.predict_proba(X_validation)[:, 1]
    threshold = _best_f1_threshold(y_validation, selected_probabilities)

    return TrainingResult(
        models=models,
        validation_metrics=validation_metrics,
        selected_name=selected_name,
        selected_model=selected_model,
        selected_threshold=threshold,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )


def fit_final_model(
    frame: pd.DataFrame, model_name: str, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Fit the selected model family on all labeled rows for later inference."""
    candidates = _make_candidates(frame.drop(columns=[TARGET_COLUMN]), random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model name: {model_name}")
    model = candidates[model_name]
    model.fit(frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN].astype(int))
    return model
