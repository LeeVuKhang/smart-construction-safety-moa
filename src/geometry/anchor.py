"""Anchor-point utilities for assigning detected people to zones."""

from __future__ import annotations

from src.detection.schemas import BBox, Point


def bottom_center(bbox: BBox) -> Point:
    """Return the person's approximate ground-plane point.

    The bottom-center of a person bounding box is used because it approximates
    where the person touches the ground. This is more suitable for fixed-camera
    zone membership than the bounding-box center.
    """
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)
