# Describe Anything Region PPE Evaluation

DAM is evaluated here as a region-level classifier by giving it ground-truth YOLO regions.
This is not a detector mAP comparison because DAM does not generate construction PPE bounding boxes by itself.

- Dry run: False
- Total regions: 100
- Runtime: 2:17.24
- Accuracy: 0.1900

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| person | 0.4318 | 0.4043 | 0.4176 | 47 |
| helmet | 0.0000 | 0.0000 | 0.0000 | 42 |
| no_helmet | 0.0000 | 0.0000 | 0.0000 | 11 |

## Prediction Distribution

| Prediction | Count |
| --- | ---: |
| unknown | 51 |
| person | 44 |
| no_helmet | 4 |
| helmet | 1 |
