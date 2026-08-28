"""Train a YOLO11 baseline from a reproducible YAML configuration."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.data.audit_dataset import audit_dataset


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
    "augmentation",
    "validation",
    "classes",
    "dataset_version",
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
        "dataset_version": config["dataset_version"],
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
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "git_commit": git_commit(),
    }
    with (experiment_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def observed_training_summary(experiment_dir: Path) -> dict:
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
            rows = [
                {key.strip(): value for key, value in row.items()}
                for row in csv.DictReader(file)
            ]
        if rows:
            last_row = rows[-1]
            epoch = last_row.get("epoch")
            summary["actual_stopping_epoch"] = int(float(epoch)) if epoch else None
            summary["validation_metrics"] = {
                key.strip(): value
                for key, value in last_row.items()
                if key and "metrics/" in key
            }
    return summary


def save_training_summary(experiment_dir: Path) -> dict:
    """Persist observed training summary."""
    summary = observed_training_summary(experiment_dir)
    with (experiment_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def git_commit() -> str:
    """Return current git commit hash when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def save_experiment_manifest(
    config: dict,
    experiment_dir: Path,
    epochs_completed: int | None,
) -> None:
    """Save standardized experiment manifest."""
    manifest = {
        "model": config["model"],
        "dataset_version": config["dataset_version"],
        "dataset_yaml": config["dataset_yaml"],
        "seed": config["seed"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "epochs_requested": config["epochs"],
        "epochs_completed": epochs_completed,
        "patience": config["patience"],
        "optimizer": config["optimizer"],
        "device": config["device"],
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_weights": str(experiment_dir / "weights" / "best.pt"),
        "last_weights": str(experiment_dir / "weights" / "last.pt"),
    }
    with (experiment_dir / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def experiment_dir_for(config: dict) -> Path:
    """Return the absolute standardized experiment directory."""
    return (Path(config["output_dir"]).resolve() / config["experiment_name"])


def train(config: dict) -> Path:
    """Train YOLO using configured parameters and return the result directory."""
    validate_config(config)
    set_seed(int(config["seed"]))

    experiment_dir = experiment_dir_for(config)
    audit = audit_dataset(
        Path(config["dataset_yaml"]),
        Path("results/dataset_audit"),
        str(config["dataset_version"]),
    )
    if audit.critical_errors:
        raise RuntimeError(
            "Dataset audit failed. Fix critical errors before training: "
            + "; ".join(audit.critical_errors)
        )

    save_training_metadata(config, experiment_dir)
    save_experiment_manifest(config, experiment_dir, epochs_completed=0)

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
        project=str(Path(config["output_dir"]).resolve()),
        name=config["experiment_name"],
        device=_device_arg(config["device"]),
        exist_ok=True,
    )
    summary = save_training_summary(experiment_dir)
    actual_epoch = summary["actual_stopping_epoch"]
    save_experiment_manifest(
        config,
        experiment_dir,
        epochs_completed=actual_epoch + 1 if actual_epoch is not None else None,
    )
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
