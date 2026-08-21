#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluation.detection_metrics \
  --config configs/models/yolo11n.yaml \
  --weights results/training/yolo11n_baseline/weights/best.pt

python -m src.evaluation.detection_metrics \
  --config configs/models/yolo11s.yaml \
  --weights results/training/yolo11s_baseline/weights/best.pt

python -m src.evaluation.detection_metrics \
  --config configs/models/yolo11m.yaml \
  --weights results/training/yolo11m_baseline/weights/best.pt
