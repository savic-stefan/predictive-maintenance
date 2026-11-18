from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.data import (
    DatasetValidationError,
    get_feature_columns,
    load_dataset,
    validate_dataset,
)
from predictive_maintenance.eda import build_data_summary
from predictive_maintenance.features import build_preprocessor, feature_frame

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "ai4i2020.csv"


def test_real_dataset_loads_with_expected_shape_and_target():
    frame = load_dataset(DATASET)

    assert frame.shape == (10_000, 14)
    assert frame["Machine failure"].value_counts().to_dict() == {0: 9661, 1: 339}


def test_identifiers_and_failure_modes_are_not_features():
    frame = load_dataset(DATASET)

    features = get_feature_columns(frame)

    assert features == [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]


def test_missing_required_column_is_rejected():
    frame = pd.read_csv(DATASET).drop(columns=["Torque [Nm]"])

    with pytest.raises(DatasetValidationError, match=r"Torque \[Nm\]"):
        validate_dataset(frame)


def test_target_with_missing_class_is_rejected():
    frame = pd.read_csv(DATASET)
    frame["Machine failure"] = 0

    with pytest.raises(DatasetValidationError, match="both normal and failure"):
        validate_dataset(frame)


def test_preprocessing_uses_only_six_non_leaking_features():
    frame = load_dataset(DATASET)
    features = feature_frame(frame)
    transformed = build_preprocessor(frame).fit_transform(features)

    assert transformed.shape == (10_000, 8)
    assert "Machine failure" not in features
    assert "TWF" not in features


def test_data_summary_reports_rare_failure_rate_and_exclusions():
    summary = build_data_summary(load_dataset(DATASET))

    assert summary["row_count"] == 10_000
    assert summary["target_counts"] == {"0": 9661, "1": 339}
    assert summary["failure_rate"] == pytest.approx(0.0339)
    assert "Product ID" in summary["excluded_columns"]
