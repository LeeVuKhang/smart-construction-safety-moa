# Zone Grounding Comparison

This document covers the zone component only.

## Task Boundary

Zone grounding answers one spatial question:

```text
person bbox + fixed camera zone polygons -> zone_id
```

It does not classify PPE, detect helmets, detect people, or generate free-form scene descriptions.

## Baseline Method

The implemented Zone Grounding Agent is deterministic:

1. Take the bottom-center anchor of a detected person box.
2. Test the anchor against configured polygons.
3. If multiple zones match, select the highest-priority zone.
4. If no zone matches, assign the configured default zone.

## Evaluation Fixture

The fixture under `data/zone_eval/` uses `configs/zones/cam_01.yaml` and covers:

- normal active-work-area assignment
- restricted-area assignment
- overlapping polygons with priority selection
- boundary points counted as inside
- default-zone fallback outside configured polygons

Run:

```bash
python3 -m src.evaluation.zone_metrics \
  --zone-config configs/zones/cam_01.yaml \
  --detections data/zone_eval/detections.json \
  --ground-truth data/zone_eval/ground_truth.json \
  --output-dir results/zone_eval
```

## Result

| Method | Input | Output | Zone assignment accuracy | Default zone rate |
| --- | --- | --- | ---: | ---: |
| Deterministic polygon grounding | person bbox + camera polygons | configured zone ID | 1.0000 | 0.2500 |
| DAM-3B | image/masked region -> text | free-form localized description | not directly applicable | not directly applicable |

The evidence for the zone module is therefore not that DAM is worse at PPE classification. The evidence is that the current zone problem is a fixed-camera spatial grounding problem with exact configured polygons, while DAM is a detailed localized captioning model and does not natively output configured zone IDs from polygon membership.

## Correct Conclusion

For the zone component, the deterministic polygon method is the correct baseline because it is exact, reproducible, fast, and directly tied to the camera's configured safety zones. DAM can be discussed as a possible future semantic context component, but it is not a replacement for deterministic zone grounding unless an additional prompt-to-zone parser and camera calibration protocol are defined and evaluated.
