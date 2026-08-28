"""Evaluate DAM for prompt-based background zone recognition."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.agents.dam_zone_agent import DAMPromptZoneAgent, DAMZoneClientConfig, normalize_zone_type


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
    parser.add_argument("--zone-config", type=Path, default=Path("configs/zones/cam_01.yaml"))
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="describe_anything_model")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dam_zone_prompt"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Run single-image or manifest-based prompt zone recognition."""
    args = parse_args()
    if bool(args.image) == bool(args.manifest):
        raise ValueError("Provide exactly one of --image or --manifest.")

    client_config = DAMZoneClientConfig(
        server_url=args.server_url,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    agent = DAMPromptZoneAgent.from_yaml(args.zone_config, client_config=client_config)
    examples = [{"image_path": str(args.image), "expected_zone_type": ""}]
    if args.manifest:
        examples = load_manifest(args.manifest)

    rows = []
    for index, example in enumerate(examples, start=1):
        image_path = Path(example["image_path"])
        prediction = agent.recognize(image_path, extra_prompt=args.prompt)
        prediction_dict = asdict(prediction)
        rows.append(
            {
                "sample_id": example.get("sample_id", f"image_{index:04d}"),
                "image_path": str(image_path),
                "expected_zone_type": example.get("expected_zone_type", ""),
                "predicted_zone_id": prediction_dict["zone_id"],
                "predicted_zone_type": prediction_dict["zone_type"],
                "confidence": prediction_dict["confidence"] if prediction_dict["confidence"] is not None else "",
                "reason": prediction_dict["reason"].replace("\n", " "),
                "raw_response": prediction_dict["raw_response"].replace("\n", " "),
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    metrics = calculate_metrics(rows)
    write_outputs(rows, metrics, args.output_dir)
    print(json.dumps({"metrics": metrics, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
