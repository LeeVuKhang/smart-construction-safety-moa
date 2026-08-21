#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m src.training.train_yolo --config configs/models/yolo11n.yaml
"${PYTHON_BIN}" -m src.training.train_yolo --config configs/models/yolo11s.yaml
"${PYTHON_BIN}" -m src.training.train_yolo --config configs/models/yolo11m.yaml
