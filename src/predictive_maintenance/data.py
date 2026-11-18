"""Loading and validating the AI4I 2020 source dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Machine failure"
REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}
FAILURE_MODE_COLUMNS = {"TWF", "HDF", "PWF", "OSF", "RNF"}
IDENTIFIER_COLUMNS = {"UDI", "Product ID"}


class DatasetValidationError(ValueError):
    """Raised when an input file does not match the expected data contract."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read and validate the raw AI4I CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")

    try:
        frame = pd.read_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DatasetValidationError(f"Could not read dataset CSV: {exc}") from exc

    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Check required fields, basic integrity, and the binary target."""
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(f"Dataset is missing required columns: {names}")
    if frame.empty:
        raise DatasetValidationError("Dataset contains no rows")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise DatasetValidationError("Dataset contains missing values in required columns")

    target_values = set(frame[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise DatasetValidationError(
            f"Target must contain only 0 and 1; found {sorted(target_values)}"
        )
    if target_values != {0, 1}:
        raise DatasetValidationError("Target must contain both normal and failure examples")


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return sensor and product-type fields available before a failure."""
    excluded = IDENTIFIER_COLUMNS | FAILURE_MODE_COLUMNS | {TARGET_COLUMN}
    return [column for column in frame.columns if column not in excluded]
