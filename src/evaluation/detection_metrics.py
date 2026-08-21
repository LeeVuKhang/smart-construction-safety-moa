"""Standard detection evaluation metrics for YOLO baselines."""

from __future__ import annotations

import argparse
import json
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


def metrics_from_yolo(metrics) -> dict[str, float]:
    """Extract standard metrics from an Ultralytics validation result."""
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": calculate_f1(precision, recall),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def evaluate(config: dict, weights: Path, output_dir: Path | None = None) -> dict[str, float]:
    """Run YOLO validation and return the main detection metrics."""
    model = YOLO(str(weights))
    validation = config.get("validation", {})
    output_dir = output_dir or Path("results/evaluation")
    metrics = model.val(
        data=config["dataset_yaml"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        conf=validation.get("conf", 0.25),
        iou=validation.get("iou", 0.7),
        project=str(output_dir),
        name=config.get("experiment_name", "evaluation"),
        exist_ok=True,
    )

    results = metrics_from_yolo(metrics)
    save_metrics(results, output_dir / f"{config.get('experiment_name', 'evaluation')}.json")
    return results


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON without fabricating missing values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/yolo11n.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    return parser.parse_args()


def main() -> None:
    """Run evaluation and print metrics."""
    args = parse_args()
    config = load_config(args.config)
    results = evaluate(config, args.weights, args.output_dir)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
