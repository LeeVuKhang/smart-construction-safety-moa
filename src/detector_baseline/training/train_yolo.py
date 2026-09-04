"""Train the YOLO11n baseline using an experiment YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """Load a YAML experiment configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train(config: dict) -> None:
    """Train YOLO using the configured baseline parameters."""
    model = YOLO(config["model"])
    model.train(
        data=config["dataset_yaml"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        patience=config["patience"],
        seed=config["seed"],
        project=config["output_dir"],
        name="train",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the YOLO11n baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return parser.parse_args()


def main() -> None:
    """Run training."""
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
