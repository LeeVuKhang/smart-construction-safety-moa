"""Train a YOLO11 baseline from a reproducible YAML configuration."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


REQUIRED_FIELDS = {
    "model",
    "dataset_yaml",
    "output_dir",
    "experiment_name",
    "device",
    "epochs",
    "batch",
    "imgsz",
    "patience",
    "seed",
    "workers",
    "optimizer",
    "validation",
    "classes",
}
EXPECTED_CLASSES = ["person", "helmet", "no_helmet"]


def load_config(config_path: Path) -> dict:
    """Load a YAML experiment configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_config(config: dict) -> None:
    """Validate required experiment fields and current detector classes."""
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    if config["classes"] != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes {EXPECTED_CLASSES}, got {config['classes']}")
    for field in ("conf", "iou"):
        if field not in config["validation"]:
            raise ValueError(f"Missing validation.{field} in config.")


def set_seed(seed: int) -> None:
    """Set deterministic seeds where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_arg(device: str):
    """Return a YOLO-compatible device argument."""
    return None if device == "auto" else device


def save_training_metadata(config: dict, experiment_dir: Path) -> None:
    """Persist config and basic environment metadata for reproducibility."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / "training_args.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "dataset_yaml": config["dataset_yaml"],
        "seed": config["seed"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "epochs": config["epochs"],
        "patience": config["patience"],
        "optimizer": config["optimizer"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    with (experiment_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_training_summary(experiment_dir: Path) -> None:
    """Save observed training summary when Ultralytics outputs are available."""
    results_csv = experiment_dir / "results.csv"
    summary = {
        "actual_stopping_epoch": None,
        "best_weights": str(experiment_dir / "weights" / "best.pt"),
        "last_weights": str(experiment_dir / "weights" / "last.pt"),
        "validation_metrics": {},
    }

    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            last_row = rows[-1]
            epoch = last_row.get("epoch")
            summary["actual_stopping_epoch"] = int(float(epoch)) if epoch else None
            summary["validation_metrics"] = {
                key.strip(): value
                for key, value in last_row.items()
                if key and "metrics/" in key
            }

    with (experiment_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def train(config: dict) -> Path:
    """Train YOLO using configured parameters and return the result directory."""
    validate_config(config)
    set_seed(int(config["seed"]))

    experiment_dir = Path(config["output_dir"]) / config["experiment_name"]
    save_training_metadata(config, experiment_dir)

    model = YOLO(config["model"])
    model.train(
        data=config["dataset_yaml"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        patience=config["patience"],
        seed=config["seed"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        project=config["output_dir"],
        name=config["experiment_name"],
        device=_device_arg(config["device"]),
        exist_ok=True,
    )
    save_training_summary(experiment_dir)
    return experiment_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a YOLO11 baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/yolo11n.yaml"))
    return parser.parse_args()


def main() -> None:
    """Run training."""
    args = parse_args()
    config = load_config(args.config)
    experiment_dir = train(config)
    print(f"Training outputs: {experiment_dir}")


if __name__ == "__main__":
    main()
