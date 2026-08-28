"""Current baseline pipeline with DAM prompt-based zone recognition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from src.agents.dam_zone_agent import DAMPromptZoneAgent, DAMZoneClientConfig
from src.agents.ppe_agent import PPEAgent
from src.agents.zone_grounding_agent import ZoneGroundingAgent
from src.detection.schemas import Detection, ZoneAssignment, ZoneRecognition
from src.detection.yolo_detector import YOLODetector
from src.geometry.anchor import bottom_center


def run_pipeline(
    image_path: Path,
    weights: Path,
    zone_config: Path,
    confidence_threshold: float,
    zone_mode: str = "dam_prompt",
    dam_server_url: str = "http://localhost:8000",
    dam_model: str = "describe_anything_model",
    dam_timeout: int = 180,
    zone_prompt: str = "",
) -> dict:
    """Run perception and the selected zone-recognition flow for one input."""
    detector = YOLODetector(weights, confidence_threshold)
    detections = detector.predict(image_path)

    ppe_agent = PPEAgent()
    output = {
        "detections": detections,
        "ppe_evidence": ppe_agent.infer(detections),
        "zone_mode": zone_mode,
    }

    if zone_mode in {"dam_prompt", "both"}:
        dam_zone_agent = DAMPromptZoneAgent.from_yaml(
            zone_config,
            client_config=DAMZoneClientConfig(
                server_url=dam_server_url,
                model=dam_model,
                timeout=dam_timeout,
            ),
        )
        zone_recognition = dam_zone_agent.recognize(image_path, extra_prompt=zone_prompt)
        output["background_zone"] = zone_recognition
        output["zone_assignments"] = assign_background_zone_to_people(detections, zone_recognition)

    if zone_mode in {"polygon", "both"}:
        polygon_zone_agent = ZoneGroundingAgent.from_yaml(zone_config)
        polygon_assignments = polygon_zone_agent.assign(detections)
        if zone_mode == "polygon":
            output["zone_assignments"] = polygon_assignments
        else:
            output["polygon_zone_assignments"] = polygon_assignments

    return output


def assign_background_zone_to_people(
    detections: list[Detection],
    zone_recognition: ZoneRecognition,
) -> list[ZoneAssignment]:
    """Apply frame-level DAM zone recognition to detected people."""
    people = [detection for detection in detections if detection.class_name == "person"]
    return [
        ZoneAssignment(
            person_id=person.object_id,
            zone_id=zone_recognition.zone_id,
            zone_type=zone_recognition.zone_type,
            source=zone_recognition.source,
            anchor_point=bottom_center(person.bbox),
        )
        for person in people
    ]


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
    parser.add_argument("--zone-mode", choices=["dam_prompt", "polygon", "both"], default="dam_prompt")
    parser.add_argument("--dam-server-url", default="http://localhost:8000")
    parser.add_argument("--dam-model", default="describe_anything_model")
    parser.add_argument("--dam-timeout", type=int, default=180)
    parser.add_argument("--zone-prompt", default="")
    return parser.parse_args()


def main() -> None:
    """Run the baseline pipeline from the CLI."""
    args = parse_args()
    output = run_pipeline(
        image_path=args.image,
        weights=args.weights,
        zone_config=args.zone_config,
        confidence_threshold=load_confidence(args.config),
        zone_mode=args.zone_mode,
        dam_server_url=args.dam_server_url,
        dam_model=args.dam_model,
        dam_timeout=args.dam_timeout,
        zone_prompt=args.zone_prompt,
    )
    print(yaml.safe_dump(to_builtin(output), sort_keys=False))


def to_builtin(value: Any) -> Any:
    """Convert dataclasses and tuples to YAML-friendly builtins."""
    if is_dataclass(value):
        return {key: to_builtin(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    return value


if __name__ == "__main__":
    main()
