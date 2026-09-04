#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-results/train/weights/best.pt}"

python -m src.evaluation.evaluate --config configs/baseline.yaml --weights "$WEIGHTS"
