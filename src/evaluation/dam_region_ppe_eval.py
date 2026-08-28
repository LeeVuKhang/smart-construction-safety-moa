"""Evaluate Describe Anything Model as a region-level PPE classifier.

DAM is not an object detector. This script supplies ground-truth YOLO regions to
a running DAM OpenAI-compatible server and evaluates the text response as a
region classification decision.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from src.data.audit_dataset import (
    collect_images,
    dataset_root,
    label_path_for_image,
    load_dataset_yaml,
    split_image_dir,
)


TARGET_CLASSES = ["person", "helmet", "no_helmet"]
VALID_PREDICTIONS = TARGET_CLASSES + ["other", "unknown"]
DEFAULT_PROMPT = """
You are evaluating construction-site PPE from a masked image region.
Answer with JSON only:
{"label": "person|helmet|no_helmet|other", "confidence": 0.0-1.0, "reason": "..."}
Use "person" only for a visible worker/person body region.
Use "helmet" only for a safety helmet region.
Use "no_helmet" only when the masked region clearly indicates a worker/head without a safety helmet.
Use "other" if the region is not one of these classes or is ambiguous.
""".strip()


def yolo_to_xyxy(values: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized YOLO box values to clipped pixel coordinates."""
    x_center, y_center, box_width, box_height = values
    x1 = int(round((x_center - box_width / 2) * width))
    y1 = int(round((y_center - box_height / 2) * height))
    x2 = int(round((x_center + box_width / 2) * width))
    y2 = int(round((y_center + box_height / 2) * height))
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


def encode_masked_region(image_path: Path, box: tuple[int, int, int, int]) -> str:
    """Return an RGBA data URL where alpha marks the queried region."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = box
    alpha[y1:y2, x1:x2] = 255
    rgba = np.dstack([rgb, alpha])
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError(f"Could not encode masked image: {image_path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def read_region_samples(dataset_yaml: Path, split: str, max_samples: int | None) -> list[dict]:
    """Collect YOLO regions from one dataset split."""
    config = load_dataset_yaml(dataset_yaml)
    root = dataset_root(dataset_yaml, config)
    classes = config["classes"]
    if classes != TARGET_CLASSES:
        raise ValueError(f"Expected classes {TARGET_CLASSES}, got {classes}")

    images = collect_images(split_image_dir(root, config[split]))
    samples: list[dict] = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = label_path_for_image(root, image_path)
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            class_id = int(float(parts[0]))
            box_values = [float(value) for value in parts[1:5]]
            samples.append(
                {
                    "sample_id": f"{split}:{image_path.name}:{line_number}",
                    "image": str(image_path),
                    "label": classes[class_id],
                    "class_id": class_id,
                    "box_xyxy": yolo_to_xyxy(box_values, width, height),
                }
            )
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples


def query_dam(
    sample: dict,
    server_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> tuple[str, dict]:
    """Query the DAM server and return raw text with parsed classification."""
    data_url = encode_masked_region(Path(sample["image"]), tuple(sample["box_xyxy"]))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 0.5,
        "use_cache": True,
        "num_beams": 1,
    }
    response = requests.post(f"{server_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    response.raise_for_status()
    raw_text = response.json()["choices"][0]["message"]["content"]
    return raw_text, parse_prediction(raw_text)


def parse_prediction(text: str) -> dict:
    """Parse a DAM response into a target label with conservative fallbacks."""
    parsed: dict = {"label": "unknown", "confidence": None, "reason": text}
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            candidate = json.loads(json_match.group(0))
            label = normalize_label(str(candidate.get("label", "")))
            if label in VALID_PREDICTIONS:
                parsed.update(candidate)
                parsed["label"] = label
                return parsed
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    if "no helmet" in lowered or "without a helmet" in lowered or "not wearing a helmet" in lowered:
        parsed["label"] = "no_helmet"
    elif "helmet" in lowered or "hard hat" in lowered or "hardhat" in lowered:
        parsed["label"] = "helmet"
    elif "person" in lowered or "worker" in lowered:
        parsed["label"] = "person"
    elif "other" in lowered:
        parsed["label"] = "other"
    return parsed


def normalize_label(label: str) -> str:
    """Normalize free-form label text."""
    value = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hard_hat": "helmet",
        "hardhat": "helmet",
        "nohelmet": "no_helmet",
        "without_helmet": "no_helmet",
        "not_wearing_helmet": "no_helmet",
        "worker": "person",
    }
    return aliases.get(value, value)


def calculate_metrics(rows: list[dict]) -> dict:
    """Calculate aggregate and per-class metrics from region predictions."""
    labels = TARGET_CLASSES
    confusion: dict[str, dict[str, int]] = {
        truth: {prediction: 0 for prediction in VALID_PREDICTIONS} for truth in labels
    }
    for row in rows:
        confusion[row["ground_truth"]][row["prediction"]] += 1

    per_class = {}
    correct = 0
    for label in labels:
        true_positive = confusion[label].get(label, 0)
        correct += true_positive
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(confusion[truth].get(label, 0) for truth in labels if truth != label)
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": sum(confusion[label].values()),
        }

    total = len(rows)
    return {
        "total_regions": total,
        "accuracy": safe_div(correct, total),
        "per_class": per_class,
        "confusion": confusion,
    }


def safe_div(numerator: float, denominator: float) -> float:
    """Return zero when a metric denominator is zero."""
    return numerator / denominator if denominator else 0.0


def write_outputs(rows: list[dict], metrics: dict, output_dir: Path, dry_run: bool) -> None:
    """Write DAM comparison outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "dam_region_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "dam_region_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# Describe Anything Region PPE Evaluation",
        "",
        "DAM is evaluated here as a region-level classifier by giving it ground-truth YOLO regions.",
        "This is not a detector mAP comparison because DAM does not generate construction PPE bounding boxes by itself.",
        "",
        f"- Dry run: {dry_run}",
        f"- Total regions: {metrics.get('total_regions', 0)}",
        f"- Accuracy: {metrics.get('accuracy', 0.0):.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for class_name, values in metrics.get("per_class", {}).items():
        lines.append(
            f"| {class_name} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |"
        )
    (output_dir / "dam_region_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DAM on YOLO region-level PPE labels.")
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/processed/dataset.yaml"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="describe_anything_model")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dam_comparison"))
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between DAM requests.")
    return parser.parse_args()


def main() -> None:
    """Run DAM region-level evaluation."""
    args = parse_args()
    samples = read_region_samples(args.dataset_yaml, args.split, args.max_samples)
    rows = []
    if args.dry_run:
        rows = [
            {
                "sample_id": sample["sample_id"],
                "image": sample["image"],
                "box_xyxy": " ".join(str(value) for value in sample["box_xyxy"]),
                "ground_truth": sample["label"],
                "prediction": "unknown",
                "confidence": "",
                "raw_response": "",
            }
            for sample in samples
        ]
    else:
        for sample in samples:
            raw_response, prediction = query_dam(
                sample,
                args.server_url,
                args.model,
                DEFAULT_PROMPT,
                args.timeout,
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "image": sample["image"],
                    "box_xyxy": " ".join(str(value) for value in sample["box_xyxy"]),
                    "ground_truth": sample["label"],
                    "prediction": prediction["label"],
                    "confidence": prediction.get("confidence", ""),
                    "raw_response": raw_response.replace("\n", " "),
                }
            )
            if args.sleep:
                time.sleep(args.sleep)

    metrics = calculate_metrics(rows)
    write_outputs(rows, metrics, args.output_dir, args.dry_run)
    print(f"Wrote DAM region PPE evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
