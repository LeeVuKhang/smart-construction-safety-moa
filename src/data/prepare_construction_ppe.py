"""Prepare the selected Construction-PPE dataset for the baseline taxonomy."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml


DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
TARGET_CLASSES = ["person", "helmet", "no_helmet"]
SOURCE_TO_TARGET = {
    6: 0,  # Person -> person
    0: 1,  # helmet -> helmet
    7: 2,  # no_helmet -> no_helmet
}
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_zip(url: str, output_path: Path) -> None:
    """Download a dataset archive if it is not already present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    urllib.request.urlretrieve(url, output_path)


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract the dataset archive and return the source dataset root."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    candidates = [
        output_dir / "construction-ppe",
        output_dir / "Construction-PPE",
        output_dir,
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(f"Could not find YOLO images/labels folders under {output_dir}")


def iter_images(image_dir: Path) -> list[Path]:
    """Return image files in stable order."""
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def remap_label_file(source_label: Path, target_label: Path) -> tuple[int, int]:
    """Remap one YOLO label file and drop classes outside the baseline taxonomy."""
    kept = 0
    dropped = 0
    remapped_lines: list[str] = []

    if source_label.exists():
        for raw_line in source_label.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            source_class = int(float(parts[0]))
            if source_class not in SOURCE_TO_TARGET:
                dropped += 1
                continue
            parts[0] = str(SOURCE_TO_TARGET[source_class])
            remapped_lines.append(" ".join(parts))
            kept += 1

    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_label.write_text("\n".join(remapped_lines), encoding="utf-8")
    return kept, dropped


def prepare_dataset(source_root: Path, target_root: Path) -> dict:
    """Copy images and remap labels from Construction-PPE to the target taxonomy."""
    summary = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "target_classes": TARGET_CLASSES,
        "source_to_target": {
            "helmet": "helmet",
            "Person": "person",
            "no_helmet": "no_helmet",
        },
        "dropped_source_classes": [
            "gloves",
            "vest",
            "boots",
            "goggles",
            "none",
            "no_goggle",
            "no_gloves",
            "no_boots",
        ],
        "splits": {},
    }

    for split in SPLITS:
        source_images = source_root / "images" / split
        source_labels = source_root / "labels" / split
        target_images = target_root / "images" / split
        target_labels = target_root / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        kept_objects = 0
        dropped_objects = 0
        image_count = 0
        empty_labels = 0
        for image_path in iter_images(source_images):
            relative_image = image_path.relative_to(source_images)
            target_image = target_images / relative_image
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)

            source_label = source_labels / relative_image.with_suffix(".txt")
            target_label = target_labels / relative_image.with_suffix(".txt")
            kept, dropped = remap_label_file(source_label, target_label)
            kept_objects += kept
            dropped_objects += dropped
            empty_labels += int(kept == 0)
            image_count += 1

        summary["splits"][split] = {
            "images": image_count,
            "kept_target_objects": kept_objects,
            "dropped_non_target_objects": dropped_objects,
            "empty_target_labels": empty_labels,
        }

    dataset_yaml = {
        "path": str(target_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
    }
    with (target_root / "dataset.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(dataset_yaml, file, sort_keys=False)

    (target_root / "mapping_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Construction-PPE for the baseline taxonomy.")
    parser.add_argument("--source-root", type=Path, help="Existing extracted Construction-PPE root.")
    parser.add_argument("--download", action="store_true", help="Download the public Construction-PPE archive.")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/construction-ppe.zip"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--target-root", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    """Run dataset preparation."""
    args = parse_args()
    source_root = args.source_root
    if args.download:
        download_zip(args.url, args.archive)
        source_root = extract_zip(args.archive, args.extract_dir)
    if source_root is None:
        raise SystemExit("Provide --source-root or use --download.")

    summary = prepare_dataset(source_root, args.target_root)
    print(json.dumps(summary, indent=2))
    print(f"Wrote dataset YAML to {args.target_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
