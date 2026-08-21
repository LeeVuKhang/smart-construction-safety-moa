#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_yolo --config configs/models/yolo11n.yaml
python -m src.training.train_yolo --config configs/models/yolo11s.yaml
python -m src.training.train_yolo --config configs/models/yolo11m.yaml
