"""Evaluation metrics for deterministic zone grounding."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.agents.zone_grounding_agent import ZoneGroundingAgent
from src.detection.schemas import Detection, ZoneAssignment


def zone_assignment_accuracy(assignments: list[ZoneAssignment], ground_truth: list[dict]) -> float:
    """Calculate correct zone assignments divided by evaluated persons."""
    expected = {(item["frame_id"], item["person_id"]): item["expected_zone"] for item in ground_truth}
    if not expected:
        return 0.0

    correct = 0
    for assignment in assignments:
        key = ("frame_001", assignment.person_id)
        if expected.get(key) == assignment.zone_id:
            correct += 1
    return correct / len(expected)


def default_zone_rate(assignments: list[ZoneAssignment]) -> float:
    """Calculate the fraction of assignments resolved to the default zone."""
    if not assignments:
        return 0.0
    default_count = sum(assignment.source == "default" for assignment in assignments)
    return default_count / len(assignments)


def evaluate_zone_assignments(
    assignments: list[ZoneAssignment],
    ground_truth: list[dict],
) -> dict[str, float]:
    """Return available zone grounding metrics."""
    return {
        "zone_assignment_accuracy": zone_assignment_accuracy(assignments, ground_truth),
        "default_zone_rate": default_zone_rate(assignments),
    }


def save_zone_results(
    assignments: list[ZoneAssignment],
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Save zone assignment rows and aggregate metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "zone_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    rows = [asdict(assignment) for assignment in assignments]
    with (output_dir / "zone_results.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["person_id", "zone_id", "zone_type", "source", "anchor_point"]
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(path: Path) -> list[dict]:
    """Load zone evaluation ground truth JSON."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_person_detections(path: Path) -> list[Detection]:
    """Load person detections from JSON for lightweight zone evaluation."""
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return [
        Detection(
            object_id=row["person_id"],
            class_name="person",
            confidence=float(row.get("confidence", 1.0)),
            bbox=tuple(float(value) for value in row["bbox"]),
        )
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate deterministic zone grounding.")
    parser.add_argument("--zone-config", type=Path, default=Path("configs/zones/cam_01.yaml"))
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/zone_eval"))
    return parser.parse_args()


def main() -> None:
    """Run zone evaluation from prepared person detections."""
    args = parse_args()
    agent = ZoneGroundingAgent.from_yaml(args.zone_config)
    assignments = agent.assign(load_person_detections(args.detections))
    metrics = evaluate_zone_assignments(assignments, load_ground_truth(args.ground_truth))
    save_zone_results(assignments, metrics, args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
