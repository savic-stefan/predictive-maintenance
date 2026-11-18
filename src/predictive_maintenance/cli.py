"""Command-line interface for the local predictive-maintenance workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from predictive_maintenance.data import load_dataset
from predictive_maintenance.eda import write_data_summary
from predictive_maintenance.evaluation import write_evaluation_artifacts
from predictive_maintenance.inference import predict_csv, save_model_bundle
from predictive_maintenance.training import fit_final_model, train_and_select


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintenance-ml")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data", help="validate the AI4I source CSV")
    validate.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))

    analyze = commands.add_parser("analyze", help="write a data-quality summary")
    analyze.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    analyze.add_argument("--output", type=Path, default=Path("reports/generated/data_summary.json"))

    train = commands.add_parser("train", help="compare models, evaluate, and save a fitted model")
    train.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    train.add_argument("--model", type=Path, default=Path("artifacts/failure_model.joblib"))
    train.add_argument("--reports", type=Path, default=Path("reports/generated"))

    predict = commands.add_parser("predict", help="predict failures from a local CSV")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=Path("artifacts/failure_model.joblib"))
    predict.add_argument("--output", type=Path, default=Path("reports/generated/predictions.csv"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        frame = load_dataset(args.data)
        print(f"Valid dataset: {len(frame)} rows, {len(frame.columns)} columns")
        return 0

    if args.command == "analyze":
        frame = load_dataset(args.data)
        output = write_data_summary(frame, args.output)
        print(f"Data summary saved to {output}")
        return 0

    if args.command == "train":
        frame = load_dataset(args.data)
        result = train_and_select(frame)
        _, report = write_evaluation_artifacts(result, args.reports)
        final_model = fit_final_model(frame, result.selected_name)
        model_path = save_model_bundle(
            final_model,
            args.model,
            result.selected_name,
            result.selected_threshold,
        )
        print(json.dumps({
            "selected_model": report["selected_model"],
            "test_metrics": report["test_metrics"],
            "model_path": str(model_path),
        }, indent=2))
        return 0

    if args.command == "predict":
        output, predictions = predict_csv(args.input, args.model, args.output)
        print(f"Wrote {len(predictions)} predictions to {output}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
