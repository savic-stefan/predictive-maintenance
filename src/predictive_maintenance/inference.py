"""Saving fitted models and producing local batch predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from predictive_maintenance.features import feature_frame

EXPECTED_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
OUTPUT_IDENTIFIER_COLUMNS = ("UDI", "Product ID")


def save_model_bundle(
    model: Any,
    path: str | Path,
    model_name: str,
    threshold: float,
) -> Path:
    """Persist the fitted preprocessing/model pipeline and its threshold."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "threshold": float(threshold),
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
    }
    joblib.dump(bundle, destination)
    return destination


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, dict) or not {"model", "threshold", "feature_columns"}.issubset(bundle):
        raise ValueError(f"Invalid model bundle: {source}")
    return bundle


def predict_frame(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Predict failures from a CSV-shaped frame without using labels or IDs."""
    missing = set(bundle["feature_columns"]).difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(sorted(missing))}")

    features = feature_frame(frame).loc[:, bundle["feature_columns"]]
    if features.isna().any().any():
        raise ValueError("Prediction features contain missing values")

    probabilities = bundle["model"].predict_proba(features)[:, 1]
    output_columns = [column for column in OUTPUT_IDENTIFIER_COLUMNS if column in frame.columns]
    predictions = frame.loc[:, output_columns].copy()
    predictions["failure_probability"] = probabilities
    predictions["predicted_failure"] = (probabilities >= bundle["threshold"]).astype(int)
    return predictions


def predict_csv(
    input_path: str | Path, model_path: str | Path, output_path: str | Path
) -> tuple[Path, pd.DataFrame]:
    """Read a local CSV, predict, and write a compact results CSV."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read prediction CSV: {exc}") from exc

    predictions = predict_frame(frame, load_model_bundle(model_path))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return destination, predictions
