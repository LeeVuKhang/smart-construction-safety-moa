# Baseline Pipeline

The current repository implements only the first YOLO11n baseline for construction PPE detection. The baseline focuses on `helmet` and `head` detection.

```text
Input Images / Video Frames
        |
        v
Data Preparation
        |
        v
YOLO11n Baseline
        |
        v
Bounding Boxes + Classes + Confidence Scores
        |
        v
Evaluation Metrics
```

## Current Baseline Scope

1. Prepare local image datasets in YOLO format.
2. Train YOLO11n using configurable experiment settings.
3. Run inference on images or directories.
4. Evaluate detection performance with precision, recall, F1-score, mAP@0.5, and mAP@0.5:0.95.

## Future Modules

The following modules are research directions only and are not implemented in the current baseline:

```text
Context Analysis
Behavior Analysis
Full PPE Detection
Multi-Agent / MoA
Automatic Safety Reporting
```

Future work should build on the baseline only after the data format, evaluation protocol, and initial YOLO11n results are stable.
