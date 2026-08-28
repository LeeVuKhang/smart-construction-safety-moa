"""Evaluate DAM for prompt-based background zone recognition.

This covers the VLM-style zone task:

image + prompt -> configured zone label

It is separate from deterministic polygon grounding, which assigns detected
people to zones from bounding boxes and camera polygons.
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


ZONE_OPTIONS = {
    "Z00": "general_area",
    "Z01": "active_work_area",
    "Z02": "restricted_area",
}
VALID_ZONE_TYPES = set(ZONE_OPTIONS.values()) | {"unknown"}
DEFAULT_PROMPT = """
You are identifying the background safety zone in one construction-site image.
Use only visual scene/background evidence, such as work activity, barriers,
machinery proximity, restricted signage, excavation, or general access areas.
Return exactly one compact JSON object with keys zone_type, confidence, reason.
Allowed zone_type values are: active_work_area, restricted_area, general_area, unknown.
Choose restricted_area for clearly hazardous or access-controlled background areas.
Choose active_work_area for normal construction work zones with visible work activity.
Choose general_area for ordinary non-restricted background or unclear site access areas.
Choose unknown if the image does not contain enough background evidence.
""".strip()


def encode_full_frame_region(image_path: Path) -> str:
    """Return an RGBA data URL with the full frame selected by alpha."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError(f"Could not encode image: {image_path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def normalize_zone_type(value: str) -> str:
    """Normalize a DAM zone label into the configured zone vocabulary."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "active": "active_work_area",
        "work_area": "active_work_area",
        "work_zone": "active_work_area",
        "construction_zone": "active_work_area",
        "restricted": "restricted_area",
        "hazard_zone": "restricted_area",
        "danger_zone": "restricted_area",
        "general": "general_area",
        "safe_zone": "general_area",
        "public_area": "general_area",
    }
    return aliases.get(normalized, normalized)


def zone_id_for_type(zone_type: str) -> str:
    """Map a zone type back to the project zone ID vocabulary."""
    for zone_id, configured_type in ZONE_OPTIONS.items():
        if configured_type == zone_type:
            return zone_id
    return "unknown"


def parse_zone_prediction(text: str) -> dict:
    """Parse DAM text into a conservative zone prediction."""
    parsed = {
        "zone_id": "unknown",
        "zone_type": "unknown",
        "confidence": None,
        "reason": text,
    }
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    fallback_text = text
    if json_match:
        try:
            candidate = json.loads(json_match.group(0))
            zone_type = normalize_zone_type(str(candidate.get("zone_type", "")))
            if zone_type in VALID_ZONE_TYPES:
                parsed.update(candidate)
                parsed["zone_type"] = zone_type
                parsed["zone_id"] = zone_id_for_type(zone_type)
                return parsed
            fallback_text = str(candidate.get("reason", ""))
        except json.JSONDecodeError:
            pass

    lowered = fallback_text.lower()
    mentions = []
    if "restricted" in lowered or "hazard" in lowered or "danger" in lowered:
        mentions.append("restricted_area")
    if "active work" in lowered or "work zone" in lowered or "construction work" in lowered:
        mentions.append("active_work_area")
    if "general" in lowered or "ordinary" in lowered or "public" in lowered:
        mentions.append("general_area")

    unique_mentions = sorted(set(mentions))
    if len(unique_mentions) == 1:
        parsed["zone_type"] = unique_mentions[0]
        parsed["zone_id"] = zone_id_for_type(unique_mentions[0])
    return parsed


def query_dam_zone(
    image_path: Path,
    server_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> tuple[str, dict]:
    """Query DAM and return raw response text plus parsed zone prediction."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_full_frame_region(image_path)}},
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
    return raw_text, parse_zone_prediction(raw_text)


def load_manifest(path: Path) -> list[dict]:
    """Load image-level zone examples from a JSON manifest."""
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON list.")
    return rows


def calculate_metrics(rows: list[dict]) -> dict:
    """Calculate image-level zone accuracy when expected labels are present."""
    labeled = [row for row in rows if row.get("expected_zone_type")]
    correct = [
        row
        for row in labeled
        if normalize_zone_type(row["expected_zone_type"]) == row["predicted_zone_type"]
    ]
    return {
        "total_images": len(rows),
        "labeled_images": len(labeled),
        "zone_accuracy": len(correct) / len(labeled) if labeled else None,
    }


def write_outputs(rows: list[dict], metrics: dict, output_dir: Path) -> None:
    """Write prompt-zone predictions and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dam_prompt_zone_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    if rows:
        with (output_dir / "dam_prompt_zone_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            fieldnames = list(rows[0])
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DAM prompt-based zone recognition.")
    parser.add_argument("--image", type=Path, help="Single image to classify.")
    parser.add_argument("--manifest", type=Path, help="JSON list with image_path and optional expected_zone_type.")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="describe_anything_model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dam_zone_prompt"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Run single-image or manifest-based prompt zone recognition."""
    args = parse_args()
    if bool(args.image) == bool(args.manifest):
        raise ValueError("Provide exactly one of --image or --manifest.")

    examples = [{"image_path": str(args.image), "expected_zone_type": ""}]
    if args.manifest:
        examples = load_manifest(args.manifest)

    rows = []
    for index, example in enumerate(examples, start=1):
        image_path = Path(example["image_path"])
        raw_response, prediction = query_dam_zone(
            image_path=image_path,
            server_url=args.server_url,
            model=args.model,
            prompt=args.prompt,
            timeout=args.timeout,
        )
        rows.append(
            {
                "sample_id": example.get("sample_id", f"image_{index:04d}"),
                "image_path": str(image_path),
                "expected_zone_type": example.get("expected_zone_type", ""),
                "predicted_zone_id": prediction["zone_id"],
                "predicted_zone_type": prediction["zone_type"],
                "confidence": prediction.get("confidence", ""),
                "raw_response": raw_response.replace("\n", " "),
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    metrics = calculate_metrics(rows)
    write_outputs(rows, metrics, args.output_dir)
    print(json.dumps({"metrics": metrics, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
