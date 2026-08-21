"""Lightweight detection schemas shared by baseline components."""

from __future__ import annotations

from dataclasses import dataclass


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """Single detector output in image coordinates."""

    object_id: str
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class PPEEvidence:
    """PPE evidence associated with one detected person."""

    person_id: str
    helmet_status: str


@dataclass(frozen=True)
class ZoneAssignment:
    """Deterministic zone assignment for one detected person."""

    person_id: str
    zone_id: str
    zone_type: str
    source: str
    anchor_point: Point
