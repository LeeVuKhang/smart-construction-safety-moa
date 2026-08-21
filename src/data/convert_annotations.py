"""Convert Pascal VOC XML annotations to YOLO label files."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_CLASSES = ["person", "helmet", "no_helmet"]


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """Convert a Pascal VOC bounding box to normalized YOLO format."""
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return x_center, y_center, width, height


def convert_xml_file(xml_path: Path, output_dir: Path, class_names: list[str]) -> Path:
    """Convert one Pascal VOC XML file to one YOLO label file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")

    image_width = float(size.findtext("width", default="0"))
    image_height = float(size.findtext("height", default="0"))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: list[str] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in class_names:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        yolo_box = voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height)
        class_id = class_names.index(class_name)
        values = " ".join(f"{value:.6f}" for value in yolo_box)
        lines.append(f"{class_id} {values}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{xml_path.stem}.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def convert_directory(xml_dir: Path, output_dir: Path, class_names: list[str]) -> list[Path]:
    """Convert all XML files in a directory."""
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {xml_dir}")
    return [convert_xml_file(xml_file, output_dir, class_names) for xml_file in xml_files]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO labels.")
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, help="Reserved for compatibility checks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    """Run annotation conversion."""
    args = parse_args()
    converted = convert_directory(args.xml_dir, args.output_dir, args.classes)
    print(f"Converted {len(converted)} annotation files to {args.output_dir}")


if __name__ == "__main__":
    main()
