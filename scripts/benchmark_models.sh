#!/usr/bin/env bash
set -euo pipefail

python -m src.training.benchmark_models \
  --configs configs/models/yolo11n.yaml configs/models/yolo11s.yaml configs/models/yolo11m.yaml \
  --output-dir results/benchmarks
