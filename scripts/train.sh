#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_yolo --config configs/baseline.yaml
