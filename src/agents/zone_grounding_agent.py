"""Deterministic fixed-camera zone grounding for detected people."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.detection.schemas import Detection, Point, ZoneAssignment
from src.geometry.anchor import bottom_center
from src.geometry.polygon import point_in_polygon, validate_polygon


@dataclass(frozen=True)
class Zone:
    """Configured camera zone."""

    zone_id: str
    zone_type: str
    priority: int
    polygon: list[Point]


@dataclass(frozen=True)
class DefaultZone:
    """Fallback zone for people outside all configured polygons."""

    zone_id: str
    zone_type: str


class ZoneGroundingAgent:
    """Assign detected people to configured zones.

    This agent performs only spatial grounding. It does not evaluate PPE,
    helmet status, violations, alerts, behavior, or multi-agent reasoning.
    """

    def __init__(self, camera_id: str, default_zone: DefaultZone, zones: list[Zone]):
        self.camera_id = camera_id
        self.default_zone = default_zone
        self.zones = zones

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ZoneGroundingAgent":
        """Load a zone configuration file."""
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        default_config = config["default_zone"]
        default_zone = DefaultZone(
            zone_id=default_config["zone_id"],
            zone_type=default_config["zone_type"],
        )

        zones = []
        for zone_config in config.get("zones", []):
            polygon = validate_polygon(zone_config["polygon"])
            zones.append(
                Zone(
                    zone_id=zone_config["zone_id"],
                    zone_type=zone_config["zone_type"],
                    priority=int(zone_config.get("priority", 0)),
                    polygon=polygon,
                )
            )

        return cls(config["camera_id"], default_zone, zones)

    def assign_person(self, person: Detection) -> ZoneAssignment:
        """Assign one person detection to the highest-priority matching zone."""
        if person.class_name != "person":
            raise ValueError("Zone grounding only accepts person detections.")

        anchor_point = bottom_center(person.bbox)
        matching_zones = [
            zone for zone in self.zones if point_in_polygon(anchor_point, zone.polygon)
        ]

        if not matching_zones:
            return ZoneAssignment(
                person_id=person.object_id,
                zone_id=self.default_zone.zone_id,
                zone_type=self.default_zone.zone_type,
                source="default",
                anchor_point=anchor_point,
            )

        selected_zone = sorted(
            matching_zones,
            key=lambda zone: (zone.priority, zone.zone_id),
            reverse=True,
        )[0]
        return ZoneAssignment(
            person_id=person.object_id,
            zone_id=selected_zone.zone_id,
            zone_type=selected_zone.zone_type,
            source="polygon",
            anchor_point=anchor_point,
        )

    def assign(self, detections: list[Detection]) -> list[ZoneAssignment]:
        """Assign all person detections to zones."""
        people = [detection for detection in detections if detection.class_name == "person"]
        return [self.assign_person(person) for person in people]
