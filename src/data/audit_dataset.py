"""Audit a YOLO-format dataset before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml


EXPECTED_CLASSES = ["person", "helmet", "no_helmet"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass
class DatasetAudit:
    """Structured dataset audit result."""

    dataset_yaml: str
    dataset_root: str
    classes: list[str]
    image_counts: dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    total_boxes: int = 0
    bbox_count_per_class: dict[str, int] = field(default_factory=dict)
    image_count_per_class: dict[str, int] = field(default_factory=dict)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    invalid_bboxes: list[dict] = field(default_factory=list)
    bbox_outside_image_boundaries: list[dict] = field(default_factory=list)
    unknown_class_ids: list[dict] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    duplicate_image_filenames: dict[str, list[str]] = field(default_factory=dict)
    duplicate_images: dict[str, list[str]] = field(default_factory=dict)
    cross_split_duplicates: list[dict] = field(default_factory=list)
    possible_source_leakage: list[dict] = field(default_factory=list)
    class_imbalance_ratio: float | None = None
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_dataset_yaml(path: Path) -> dict:
    """Load and normalize a YOLO dataset YAML."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    names = config.get("names")
    if isinstance(names, dict):
        classes = [names[index] for index in sorted(names)]
    else:
        classes = list(names or [])
    config["classes"] = classes
    return config


def dataset_root(dataset_yaml: Path, config: dict) -> Path:
    """Resolve dataset root relative to the YAML location when needed."""
    root = Path(config.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        yaml_relative = (dataset_yaml.parent / root).resolve()
        cwd_relative = root.resolve()
        root = yaml_relative if yaml_relative.exists() else cwd_relative
    return root


def split_image_dir(root: Path, split_value: str | list[str]) -> list[Path]:
    """Resolve image directories or manifest paths for one split."""
    values = split_value if isinstance(split_value, list) else [split_value]
    paths = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        paths.append(path)
    return paths


def collect_images(paths: list[Path]) -> list[Path]:
    """Collect image files from directories or text manifests."""
    images: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".txt":
            base = path.parent
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    image_path = Path(line.strip())
                    images.append(image_path if image_path.is_absolute() else base / image_path)
        elif path.is_dir():
            images.extend(
                file for file in path.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted({image.resolve() for image in images})


def label_path_for_image(root: Path, image_path: Path) -> Path:
    """Infer YOLO label path from an image path."""
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return root / "labels" / image_path.with_suffix(".txt").name


def image_shape(image_path: Path) -> tuple[int, int] | None:
    """Return image width and height, or None for corrupt images."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def parse_label_file(label_path: Path, classes: list[str], audit: DatasetAudit) -> list[int]:
    """Parse one YOLO label file and record label-level issues."""
    class_ids: list[int] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        audit.empty_label_files.append(str(label_path))
        return class_ids

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            audit.invalid_bboxes.append(
                {"label": str(label_path), "line": line_number, "reason": "expected 5 values"}
            )
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            audit.invalid_bboxes.append(
                {"label": str(label_path), "line": line_number, "reason": "non-numeric value"}
            )
            continue

        if class_id < 0 or class_id >= len(classes):
            audit.unknown_class_ids.append(
                {"label": str(label_path), "line": line_number, "class_id": class_id}
            )
            continue

        if width <= 0 or height <= 0:
            audit.invalid_bboxes.append(
                {"label": str(label_path), "line": line_number, "reason": "non-positive size"}
            )
            continue

        if not all(0 <= value <= 1 for value in [x_center, y_center, width, height]):
            audit.bbox_outside_image_boundaries.append(
                {"label": str(label_path), "line": line_number, "reason": "YOLO value outside [0,1]"}
            )
            continue

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        if min(x1, y1) < 0 or max(x2, y2) > 1:
            audit.bbox_outside_image_boundaries.append(
                {"label": str(label_path), "line": line_number, "reason": "box extends outside image"}
            )
            continue

        class_ids.append(class_id)
    return class_ids


def audit_dataset(
    dataset_yaml: Path,
    output_dir: Path = Path("results/dataset_audit"),
    dataset_version: str = "v1.0",
) -> DatasetAudit:
    """Audit dataset and write summary outputs."""
    config = load_dataset_yaml(dataset_yaml)
    root = dataset_root(dataset_yaml, config)
    classes = config["classes"]
    audit = DatasetAudit(str(dataset_yaml), str(root), classes)

    if classes != EXPECTED_CLASSES:
        audit.critical_errors.append(f"Expected classes {EXPECTED_CLASSES}, got {classes}")

    split_images: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if split_value is None:
            audit.critical_errors.append(f"Missing split '{split}' in dataset YAML.")
            split_images[split] = []
            continue
        images = collect_images(split_image_dir(root, split_value))
        split_images[split] = images
        audit.image_counts[split] = len(images)

    all_images = [image for images in split_images.values() for image in images]
    audit.total_images = len(all_images)
    write_split_manifests(root, split_images)
    write_dataset_version(root, dataset_version)

    filename_map: dict[str, list[str]] = defaultdict(list)
    hash_map: dict[str, list[str]] = defaultdict(list)
    labels_seen: set[Path] = set()
    bbox_counter: Counter[str] = Counter()
    image_class_counter: Counter[str] = Counter()
    hash_to_splits: dict[str, set[str]] = defaultdict(set)

    for split, images in split_images.items():
        for image_path in images:
            filename_map[image_path.name].append(str(image_path))
            shape = image_shape(image_path)
            if shape is None:
                audit.corrupt_images.append(str(image_path))
                continue

            digest = file_hash(image_path)
            hash_map[digest].append(str(image_path))
            hash_to_splits[digest].add(split)

            label_path = label_path_for_image(root, image_path)
            if not label_path.exists():
                audit.missing_labels.append(str(image_path))
                continue

            labels_seen.add(label_path.resolve())
            class_ids = parse_label_file(label_path, classes, audit)
            audit.total_boxes += len(class_ids)
            class_names_in_image = set()
            for class_id in class_ids:
                class_name = classes[class_id]
                bbox_counter[class_name] += 1
                class_names_in_image.add(class_name)
            for class_name in class_names_in_image:
                image_class_counter[class_name] += 1

    label_files = sorted((root / "labels").rglob("*.txt")) if (root / "labels").exists() else []
    audit.orphan_labels = [
        str(label)
        for label in label_files
        if label.resolve() not in labels_seen and not expected_image_for_label(label).exists()
    ]

    audit.bbox_count_per_class = {class_name: bbox_counter[class_name] for class_name in classes}
    audit.image_count_per_class = {
        class_name: image_class_counter[class_name] for class_name in classes
    }
    nonzero_counts = [count for count in audit.bbox_count_per_class.values() if count > 0]
    if nonzero_counts:
        audit.class_imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)

    audit.duplicate_image_filenames = {
        name: paths for name, paths in filename_map.items() if len(paths) > 1
    }
    audit.duplicate_images = {
        digest: paths for digest, paths in hash_map.items() if len(paths) > 1
    }
    audit.cross_split_duplicates = [
        {"hash": digest, "splits": sorted(splits), "images": hash_map[digest]}
        for digest, splits in hash_to_splits.items()
        if len(splits) > 1
    ]
    audit.possible_source_leakage = detect_possible_source_leakage(split_images)
    finalize_critical_errors(audit)
    write_outputs(audit, output_dir)
    return audit


def expected_image_for_label(label_path: Path) -> Path:
    """Infer the expected image path for an orphan-label check."""
    parts = list(label_path.parts)
    if "labels" in parts:
        parts[parts.index("labels")] = "images"
        base = Path(*parts)
        for extension in IMAGE_EXTENSIONS:
            candidate = base.with_suffix(extension)
            if candidate.exists():
                return candidate
    return Path("__missing__")


def file_hash(path: Path) -> str:
    """Calculate a SHA256 digest for duplicate-image detection."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split_manifests(root: Path, split_images: dict[str, list[Path]]) -> None:
    """Write fixed split manifests without moving images."""
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, images in split_images.items():
        lines = [relative_to_root(root, image) for image in images]
        (split_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_version(root: Path, dataset_version: str) -> None:
    """Write dataset version metadata for reproducibility."""
    version = {
        "dataset_version": dataset_version,
        "classes": EXPECTED_CLASSES,
        "seed": 42,
        "split": SPLIT_RATIOS,
    }
    (root / "dataset_version.json").write_text(json.dumps(version, indent=2), encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable path string relative to dataset root when possible."""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def detect_possible_source_leakage(split_images: dict[str, list[Path]]) -> list[dict]:
    """Flag likely source leakage using conservative filename grouping."""
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split, images in split_images.items():
        for image in images:
            key = source_key(image.stem)
            groups[key][split].append(str(image))

    return [
        {"source_key": key, "splits": sorted(split_map), "images": split_map}
        for key, split_map in groups.items()
        if len(split_map) > 1 and len(key) >= 4
    ]


def source_key(stem: str) -> str:
    """Create a simple source key by removing common frame/augmentation suffixes."""
    tokens = stem.replace("-", "_").split("_")
    while tokens and (tokens[-1].isdigit() or tokens[-1].lower() in {"aug", "copy"}):
        tokens.pop()
    return "_".join(tokens) if tokens else stem


def finalize_critical_errors(audit: DatasetAudit) -> None:
    """Promote serious audit findings to critical errors."""
    critical_fields = {
        "missing_labels": audit.missing_labels,
        "orphan_labels": audit.orphan_labels,
        "corrupt_images": audit.corrupt_images,
        "invalid_bboxes": audit.invalid_bboxes,
        "bbox_outside_image_boundaries": audit.bbox_outside_image_boundaries,
        "unknown_class_ids": audit.unknown_class_ids,
        "cross_split_duplicates": audit.cross_split_duplicates,
    }
    for name, values in critical_fields.items():
        if values:
            audit.critical_errors.append(f"{name}: {len(values)} issue(s)")
    if audit.total_images == 0:
        audit.critical_errors.append("No images found in configured train/val/test splits.")
    for class_name in EXPECTED_CLASSES:
        if audit.bbox_count_per_class.get(class_name, 0) == 0:
            audit.critical_errors.append(f"No bounding boxes found for class '{class_name}'.")
    if audit.possible_source_leakage:
        audit.warnings.append(
            "Possible source/video leakage detected from filename groups. "
            "Review before training if images come from videos."
        )


def write_outputs(audit: DatasetAudit, output_dir: Path) -> None:
    """Write JSON, CSV, and Markdown audit outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit.__dict__
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "bbox_count", "image_count"])
        writer.writeheader()
        for class_name in audit.classes:
            writer.writerow(
                {
                    "class": class_name,
                    "bbox_count": audit.bbox_count_per_class.get(class_name, 0),
                    "image_count": audit.image_count_per_class.get(class_name, 0),
                }
            )

    lines = [
        "# Dataset Audit Report",
        "",
        f"- Dataset YAML: `{audit.dataset_yaml}`",
        f"- Dataset root: `{audit.dataset_root}`",
        f"- Classes: {', '.join(audit.classes)}",
        f"- Total images: {audit.total_images}",
        f"- Train/Val/Test images: {audit.image_counts}",
        f"- Total boxes: {audit.total_boxes}",
        f"- Class imbalance ratio: {audit.class_imbalance_ratio}",
        "",
        "## Class Distribution",
        "",
        "| Class | Boxes | Images |",
        "| --- | ---: | ---: |",
    ]
    for class_name in audit.classes:
        lines.append(
            f"| {class_name} | {audit.bbox_count_per_class.get(class_name, 0)} | "
            f"{audit.image_count_per_class.get(class_name, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Critical Errors",
            "",
            *(f"- {error}" for error in audit.critical_errors),
            "",
            "## Warnings",
            "",
            *(f"- {warning}" for warning in audit.warnings),
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit a YOLO dataset before training.")
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/processed/dataset.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument("--dataset-version", default="v1.0")
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run dataset audit from the CLI."""
    args = parse_args()
    audit = audit_dataset(args.dataset_yaml, args.output_dir, args.dataset_version)
    print(f"Dataset audit written to {args.output_dir}")
    if audit.critical_errors:
        print("Critical dataset errors:")
        for error in audit.critical_errors:
            print(f"- {error}")
        if args.fail_on_critical:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
