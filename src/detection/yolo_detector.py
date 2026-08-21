"""YOLO detector wrapper that returns normalized detection structures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from src.detection.schemas import Detection


class YOLODetector:
    """Thin Ultralytics wrapper for perception only."""

    def __init__(self, weights: str | Path, confidence_threshold: float = 0.25):
        self.model = YOLO(str(weights))
        self.confidence_threshold = confidence_threshold

    def predict(self, source: str | Path) -> list[Detection]:
        """Run inference and convert YOLO results to Detection records."""
        results = self.model.predict(
            source=str(source),
            conf=self.confidence_threshold,
            verbose=False,
        )
        return list(_detections_from_results(results))


def _detections_from_results(results: Iterable) -> Iterable[Detection]:
    """Yield Detection records from Ultralytics result objects."""
    object_index = 1
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            yield Detection(
                object_id=f"D{object_index:04d}",
                class_name=names[class_id],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            )
            object_index += 1
