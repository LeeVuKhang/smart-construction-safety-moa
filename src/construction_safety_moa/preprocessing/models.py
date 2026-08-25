from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from construction_safety_moa.contracts import EvidenceIssue

RGBArray = NDArray[np.uint8]


def _validated_size(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value


def _coordinates(values: Sequence[object], expected_length: int, name: str) -> list[float]:
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} coordinates")
    coordinates: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} coordinates must be finite numbers")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError(f"{name} coordinates must be finite numbers")
        coordinates.append(coordinate)
    return coordinates


@dataclass(frozen=True, slots=True)
class RawCameraFrame:
    """One encoded frame supplied by an external camera transport."""

    frame_id: str
    camera_id: str
    timestamp: str
    source_ref: str
    image_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Small, model-neutral set of deterministic preprocessing controls."""

    target_size: tuple[int, int] = (640, 640)
    padding_value: int = 114
    max_source_bytes: int = 25 * 1024 * 1024
    max_source_pixels: int = 40_000_000
    min_short_side: int = 360
    low_light_threshold: float = 40.0

    def __post_init__(self) -> None:
        _validated_size(self.target_size, "target_size")
        if (
            isinstance(self.padding_value, bool)
            or not isinstance(self.padding_value, int)
            or not 0 <= self.padding_value <= 255
        ):
            raise ValueError("padding_value must be an integer between 0 and 255")
        for name, value in (
            ("max_source_bytes", self.max_source_bytes),
            ("max_source_pixels", self.max_source_pixels),
            ("min_short_side", self.min_short_side),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.low_light_threshold, bool)
            or not isinstance(self.low_light_threshold, (int, float))
            or not math.isfinite(float(self.low_light_threshold))
            or not 0.0 <= float(self.low_light_threshold) <= 255.0
        ):
            raise ValueError("low_light_threshold must be between 0 and 255")


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Map geometry between original-image and centered-letterbox coordinates."""

    original_size: tuple[int, int]
    model_size: tuple[int, int]
    scale: float
    padding: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _validated_size(self.original_size, "original_size")
        _validated_size(self.model_size, "model_size")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(float(self.scale))
            or self.scale <= 0
        ):
            raise ValueError("scale must be a positive finite number")
        if (
            not isinstance(self.padding, tuple)
            or len(self.padding) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.padding
            )
        ):
            raise ValueError("padding must contain four non-negative integers")

    def bbox_to_model(self, bbox: Sequence[object]) -> list[float]:
        x1, y1, x2, y2 = self._validated_bbox(bbox)
        left, top, _, _ = self.padding
        return self._clamp_bbox(
            [
                x1 * self.scale + left,
                y1 * self.scale + top,
                x2 * self.scale + left,
                y2 * self.scale + top,
            ],
            self.model_size,
        )

    def bbox_to_original(self, bbox: Sequence[object]) -> list[float]:
        x1, y1, x2, y2 = self._validated_bbox(bbox)
        left, top, _, _ = self.padding
        return self._clamp_bbox(
            [
                (x1 - left) / self.scale,
                (y1 - top) / self.scale,
                (x2 - left) / self.scale,
                (y2 - top) / self.scale,
            ],
            self.original_size,
        )

    def polygon_to_model(self, polygon: Sequence[Sequence[object]]) -> list[list[float]]:
        points = self._validated_polygon(polygon)
        left, top, _, _ = self.padding
        width, height = self.model_size
        return [
            [
                min(max(x * self.scale + left, 0.0), float(width)),
                min(max(y * self.scale + top, 0.0), float(height)),
            ]
            for x, y in points
        ]

    def polygon_to_original(self, polygon: Sequence[Sequence[object]]) -> list[list[float]]:
        points = self._validated_polygon(polygon)
        left, top, _, _ = self.padding
        width, height = self.original_size
        return [
            [
                min(max((x - left) / self.scale, 0.0), float(width)),
                min(max((y - top) / self.scale, 0.0), float(height)),
            ]
            for x, y in points
        ]

    def _validated_bbox(self, bbox: Sequence[object]) -> list[float]:
        coordinates = _coordinates(bbox, 4, "bbox")
        x1, y1, x2, y2 = coordinates
        if x1 > x2 or y1 > y2:
            raise ValueError("bbox coordinates must be ordered as x1, y1, x2, y2")
        return coordinates

    def _validated_polygon(
        self,
        polygon: Sequence[Sequence[object]],
    ) -> list[list[float]]:
        if len(polygon) < 3:
            raise ValueError("polygon must contain at least three points")
        return [_coordinates(point, 2, "polygon point") for point in polygon]

    def _clamp_bbox(
        self,
        bbox: list[float],
        size: tuple[int, int],
    ) -> list[float]:
        width, height = size
        return [
            min(max(bbox[0], 0.0), float(width)),
            min(max(bbox[1], 0.0), float(height)),
            min(max(bbox[2], 0.0), float(width)),
            min(max(bbox[3], 0.0), float(height)),
        ]


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """Shared, read-only image input for visual agents."""

    frame_id: str
    camera_id: str
    timestamp: str
    source_ref: str
    source_sha256: str
    original_rgb: RGBArray = field(repr=False, compare=False)
    model_rgb: RGBArray = field(repr=False, compare=False)
    transform: ImageTransform
    quality_flags: tuple[EvidenceIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
