"""Run YOLO inference on an image, video, or directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """Load a YAML experiment configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict(config: dict, weights: Path, source: Path, save: bool = True) -> None:
    """Run inference with a trained YOLO model."""
    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        conf=config["validation"]["conf"],
        imgsz=config["imgsz"],
        project="results/evaluation",
        name="predictions",
        save=save,
        exist_ok=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/yolo11n.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--no-save", action="store_true", help="Do not save visual predictions.")
    return parser.parse_args()


def main() -> None:
    """Run prediction."""
    args = parse_args()
    config = load_config(args.config)
    predict(config, args.weights, args.source, save=not args.no_save)


if __name__ == "__main__":
    main()
