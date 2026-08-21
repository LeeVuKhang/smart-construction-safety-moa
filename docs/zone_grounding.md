# Zone Grounding

Zone Grounding is deterministic because camera zones are fixed polygons configured manually for each camera. No separate AI model is needed when the camera viewpoint and zone layout are known.

## Fixed Camera Assumption

Each camera has its own zone configuration. One camera may contain multiple zones. Polygons may overlap.

## Person Anchor

The Zone Grounding Agent uses the bottom-center point of a person bounding box:

```text
anchor_x = (x1 + x2) / 2
anchor_y = y2
```

This approximates the person's ground-plane location and is more suitable for zone membership than the bounding-box center.

## Point-In-Polygon Rule

The project uses one consistent boundary rule:

```text
a point on the polygon boundary is considered inside the zone
```

Malformed polygons raise clear errors.

## Overlapping Zones

If a person belongs to multiple zones, all matching zones are collected and sorted by priority. The highest-priority zone wins.

Example:

```text
restricted_area priority = 100
active_work_area priority = 30
```

If a person belongs to both, `restricted_area` is selected.

## Default Zone

If a person is outside all configured polygons, the assignment uses `default_zone`.

## Responsibility Boundary

Zone Grounding Agent does:

- assign detected people to zones
- return zone ID, zone type, source, and anchor point

Zone Grounding Agent does not:

- evaluate helmet status
- check PPE rules
- calculate violation severity
- trigger alerts
- perform behavior analysis
- perform MoA reasoning
