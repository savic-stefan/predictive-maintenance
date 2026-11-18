from pathlib import Path

import pytest

from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import evaluate_selected_model, write_evaluation_artifacts
from predictive_maintenance.inference import (
    predict_csv,
    predict_frame,
    save_model_bundle,
)
from predictive_maintenance.training import fit_final_model
from predictive_maintenance.training import split_dataset, train_and_select

DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"


def test_split_is_stratified_and_has_60_20_20_sizes():
    frame = load_dataset(DATASET)

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(frame)

    assert (len(X_train), len(X_validation), len(X_test)) == (6000, 2000, 2000)
    assert (int(y_train.sum()), int(y_validation.sum()), int(y_test.sum())) == (203, 68, 68)


def test_training_selects_a_real_model_and_finite_threshold():
    result = train_and_select(load_dataset(DATASET))

    assert result.selected_name in {"logistic_regression", "random_forest"}
    assert 0 < result.selected_threshold < 1
    assert set(result.validation_metrics) == {
        "majority_baseline",
        "logistic_regression",
        "random_forest",
    }
    assert all(
        0 <= metrics["average_precision"] <= 1
        for metrics in result.validation_metrics.values()
    )


def test_held_out_evaluation_and_reports_are_written(tmp_path):
    result = train_and_select(load_dataset(DATASET))

    report = evaluate_selected_model(result)
    report_path, written_report = write_evaluation_artifacts(result, tmp_path)

    assert report["split_sizes"] == {"train": 6000, "validation": 2000, "test": 2000}
    assert 0 <= report["test_metrics"]["average_precision"] <= 1
    assert report["test_metrics"]["confusion_matrix"]["tp"] > 0
    assert report_path.is_file()
    assert written_report["validation_permutation_importance"]
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "validation_model_comparison.png",
        "confusion_matrix.png",
        "precision_recall_curve.png",
        "permutation_importance.png",
    }


def test_saved_pipeline_predicts_from_feature_only_csv(tmp_path):
    frame = load_dataset(DATASET)
    model = fit_final_model(frame, "logistic_regression")
    model_path = save_model_bundle(model, tmp_path / "model.joblib", "logistic_regression", 0.5)
    input_path = tmp_path / "input.csv"
    frame.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]).head(5).to_csv(
        input_path, index=False
    )

    output_path, predictions = predict_csv(input_path, model_path, tmp_path / "predictions.csv")

    assert output_path.is_file()
    assert len(predictions) == 5
    assert {"failure_probability", "predicted_failure", "UDI"}.issubset(predictions.columns)
    assert predictions["failure_probability"].between(0, 1).all()
    assert "Machine failure" not in predictions


def test_predict_frame_rejects_missing_sensor_features(tmp_path):
    frame = load_dataset(DATASET)
    model = fit_final_model(frame, "logistic_regression")
    bundle_path = save_model_bundle(model, tmp_path / "model.joblib", "logistic_regression", 0.5)
    from predictive_maintenance.inference import load_model_bundle

    with pytest.raises(ValueError, match="missing columns"):
        predict_frame(frame.drop(columns=["Torque [Nm]"]), load_model_bundle(bundle_path))
