"""Minimal PPE evidence interface independent of zone grounding."""

from __future__ import annotations

from src.detection.schemas import Detection, PPEEvidence


class PPEAgent:
    """Produce PPE evidence from detector outputs without using zone state."""

    def infer(self, detections: list[Detection]) -> list[PPEEvidence]:
        """Return person-level helmet evidence from nearby PPE detections."""
        people = [detection for detection in detections if detection.class_name == "person"]
        ppe_detections = [
            detection
            for detection in detections
            if detection.class_name in {"helmet", "no_helmet"}
        ]
        return [
            PPEEvidence(
                person_id=person.object_id,
                helmet_status=_helmet_status(person, ppe_detections),
            )
            for person in people
        ]


def _helmet_status(person: Detection, ppe_detections: list[Detection]) -> str:
    """Infer helmet status from PPE detections whose centers fall inside a person bbox."""
    related = [
        detection
        for detection in ppe_detections
        if _point_inside_bbox(_bbox_center(detection.bbox), person.bbox)
    ]
    if any(detection.class_name == "no_helmet" for detection in related):
        return "missing"
    if any(detection.class_name == "helmet" for detection in related):
        return "present"
    return "unknown"


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_inside_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2
