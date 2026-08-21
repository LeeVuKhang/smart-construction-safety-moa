# Current Baseline Pipeline

The current baseline combines YOLO11 perception with deterministic zone grounding.

```text
Input Image
    |
    v
YOLO Detector
    |
    v
Structured Detections
    |
    +--> PPE Agent
    |
    +--> Person Detections
             |
             v
       Zone Grounding Agent
             |
             v
       Region Grounding
```

The detector predicts `person`, `helmet`, and `no_helmet`. The Zone Grounding Agent only assigns person detections to configured camera zones.

Future evidence joining, rule evaluation, context analysis, behavior analysis, and MoA reasoning are intentionally outside the current implementation.
