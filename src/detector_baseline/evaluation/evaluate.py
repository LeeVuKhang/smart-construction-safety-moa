"""Evaluate a trained YOLO model on the configured validation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """Load a YAML experiment configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(config: dict, weights: Path) -> dict[str, float]:
    """Run YOLO validation and return the main detection metrics."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=config["dataset_yaml"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        conf=config["confidence_threshold"],
        project=config["output_dir"],
        name="evaluation",
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": calculate_f1(precision, recall),
        "map50": map50,
        "map50_95": map5095,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run evaluation and print metrics."""
    args = parse_args()
    config = load_config(args.config)
    results = evaluate(config, args.weights)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
