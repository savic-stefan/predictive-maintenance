"""Held-out evaluation and interpretation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predictive_maintenance.features import feature_frame
from predictive_maintenance.training import TrainingResult


def _classification_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_selected_model(result: TrainingResult) -> dict[str, Any]:
    """Evaluate the selected model once on the untouched test partition."""
    probabilities = result.selected_model.predict_proba(result.X_test)[:, 1]
    selected = _classification_metrics(
        result.y_test, probabilities, result.selected_threshold
    )
    baseline = _classification_metrics(
        result.y_test,
        np.full(len(result.y_test), float(result.y_train.mean())),
        0.5,
    )
    return {
        "selected_model": result.selected_name,
        "validation_threshold": result.selected_threshold,
        "validation_metrics": result.validation_metrics,
        "test_metrics": selected,
        "test_baseline": baseline,
        "split_sizes": {
            "train": int(len(result.y_train)),
            "validation": int(len(result.y_validation)),
            "test": int(len(result.y_test)),
        },
    }


def _save_model_comparison(result: TrainingResult, destination: Path) -> None:
    names = list(result.validation_metrics)
    values = [result.validation_metrics[name]["average_precision"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, values, color=["#9aa0a6", "#4c78a8", "#f58518"][: len(names)])
    axis.set_ylabel("Average precision (validation)")
    axis.set_title("Validation model comparison")
    axis.set_ylim(0, min(1.0, max(values) * 1.2 + 0.02))
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _save_test_plots(result: TrainingResult, metrics: dict[str, Any], output_dir: Path) -> None:
    probabilities = result.selected_model.predict_proba(result.X_test)[:, 1]
    predictions = (probabilities >= result.selected_threshold).astype(int)
    matrix = confusion_matrix(result.y_test, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual")
    axis.set_xticklabels(["Normal", "Failure"])
    axis.set_yticklabels(["Normal", "Failure"])
    axis.set_title("Test confusion matrix")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(result.y_test, probabilities)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(recall, precision, color="#4c78a8", label="Selected model")
    axis.axhline(float(result.y_test.mean()), color="#9aa0a6", linestyle="--", label="No-skill prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.set_title(f"Test precision-recall curve (AP={metrics['test_metrics']['average_precision']:.3f})")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(figure)


def _save_permutation_importance(result: TrainingResult, output_dir: Path) -> dict[str, float]:
    safe_validation = feature_frame(result.X_validation)
    importance = permutation_importance(
        result.selected_model,
        safe_validation,
        result.y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    values = pd.Series(importance.importances_mean, index=safe_validation.columns).sort_values()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    values.plot.barh(ax=axis, color="#54a24b")
    axis.set_xlabel("Decrease in validation average precision")
    axis.set_title("Permutation importance on validation data")
    figure.tight_layout()
    figure.savefig(output_dir / "permutation_importance.png", dpi=160)
    plt.close(figure)
    return {name: float(value) for name, value in values.sort_values(ascending=False).items()}


def write_evaluation_artifacts(
    result: TrainingResult, output_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    """Write metrics JSON and a compact plot set under the requested folder."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = evaluate_selected_model(result)
    _save_model_comparison(result, destination / "validation_model_comparison.png")
    _save_test_plots(result, report, destination)
    report["validation_permutation_importance"] = _save_permutation_importance(
        result, destination
    )

    report_path = destination / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, report
