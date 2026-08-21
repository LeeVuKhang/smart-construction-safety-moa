"""Current baseline pipeline: YOLO detection plus deterministic zone grounding."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.agents.ppe_agent import PPEAgent
from src.agents.zone_grounding_agent import ZoneGroundingAgent
from src.detection.yolo_detector import YOLODetector


def run_pipeline(
    image_path: Path,
    weights: Path,
    zone_config: Path,
    confidence_threshold: float,
) -> dict:
    """Run the baseline perception and zone grounding flow for one input."""
    detector = YOLODetector(weights, confidence_threshold)
    detections = detector.predict(image_path)

    zone_agent = ZoneGroundingAgent.from_yaml(zone_config)
    ppe_agent = PPEAgent()

    return {
        "detections": detections,
        "ppe_evidence": ppe_agent.infer(detections),
        "zone_assignments": zone_agent.assign(detections),
    }


def load_confidence(config_path: Path) -> float:
    """Load validation confidence threshold from a model config."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return float(config["validation"]["conf"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the current baseline pipeline.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--zone-config", type=Path, default=Path("configs/zones/cam_01.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/models/yolo11n.yaml"))
    return parser.parse_args()


def main() -> None:
    """Run the baseline pipeline from the CLI."""
    args = parse_args()
    output = run_pipeline(
        image_path=args.image,
        weights=args.weights,
        zone_config=args.zone_config,
        confidence_threshold=load_confidence(args.config),
    )
    print(output)


if __name__ == "__main__":
    main()
