"""Small reproducible data-quality and class-balance report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from predictive_maintenance.data import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    get_feature_columns,
)


def build_data_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact quality, target, and numeric-feature summaries."""
    target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    numeric_features = frame[get_feature_columns(frame)].select_dtypes(include="number")
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_cell_count": int(frame.isna().sum().sum()),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "target_column": TARGET_COLUMN,
        "target_counts": {str(label): int(count) for label, count in target_counts.items()},
        "failure_rate": float(frame[TARGET_COLUMN].mean()),
        "feature_columns": get_feature_columns(frame),
        "excluded_columns": sorted(IDENTIFIER_COLUMNS | FAILURE_MODE_COLUMNS | {TARGET_COLUMN}),
        "numeric_feature_summary": numeric_features.describe().round(3).to_dict(),
    }


def write_data_summary(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the data summary as readable, deterministic JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_data_summary(frame), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination
