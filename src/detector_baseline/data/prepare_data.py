"""Create a minimal YOLO dataset YAML for the current baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_CLASSES = ["helmet", "head"]


def write_dataset_yaml(dataset_dir: Path, output_path: Path, classes: list[str]) -> None:
    """Write a YOLO dataset YAML using paths relative to the dataset root."""
    dataset = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(classes)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dataset, file, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare a YOLO dataset YAML.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dataset.yaml"))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    """Run dataset YAML preparation."""
    args = parse_args()
    write_dataset_yaml(args.dataset_dir, args.output, args.classes)
    print(f"Wrote dataset YAML to {args.output}")


if __name__ == "__main__":
    main()
