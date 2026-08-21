"""Polygon validation and point-in-polygon checks for fixed zones."""

from __future__ import annotations

from collections.abc import Sequence

from src.detection.schemas import Point

PolygonPoints = Sequence[Sequence[float]]


def validate_polygon(polygon: PolygonPoints) -> list[Point]:
    """Validate and normalize polygon points.

    A valid zone polygon must contain at least three coordinate pairs and must
    not self-intersect. Boundary points are considered inside the polygon.
    """
    points = [_normalize_point(point) for point in polygon]
    if len(points) < 3:
        raise ValueError("A zone polygon must contain at least three points.")

    try:
        from shapely.geometry import Polygon
    except ImportError:
        _validate_without_shapely(points)
        return points

    shape = Polygon(points)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError("A zone polygon must be non-self-intersecting and have area.")
    return points


def point_in_polygon(point: Point, polygon: PolygonPoints) -> bool:
    """Return True when a point is inside or on a polygon boundary."""
    points = validate_polygon(polygon)

    try:
        from shapely.geometry import Point as ShapelyPoint
        from shapely.geometry import Polygon
    except ImportError:
        return _point_in_polygon_fallback(point, points)

    shape = Polygon(points)
    return bool(shape.covers(ShapelyPoint(point)))


def _normalize_point(point: Sequence[float]) -> Point:
    if len(point) != 2:
        raise ValueError("Each polygon point must contain exactly two coordinates.")
    return float(point[0]), float(point[1])


def _validate_without_shapely(points: list[Point]) -> None:
    if len(set(points)) < 3:
        raise ValueError("A zone polygon must contain at least three unique points.")
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    if area == 0:
        raise ValueError("A zone polygon must have non-zero area.")


def _point_in_polygon_fallback(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]

    for current_x, current_y in polygon:
        if _point_on_segment(point, (previous_x, previous_y), (current_x, current_y)):
            return True
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            slope_x = (previous_x - current_x) * (y - current_y)
            slope_x = slope_x / (previous_y - current_y) + current_x
            if x < slope_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y

    return inside


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
