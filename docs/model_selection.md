# Model Selection

YOLO11n, YOLO11s, and YOLO11m are compared because they represent different speed and capacity trade-offs within the same detector family.

## Fair Comparison Protocol

All compared runs must use:

- identical dataset YAML
- identical train/validation split
- identical seed
- identical image size
- identical batch size
- identical optimizer setting
- identical validation confidence and IoU thresholds
- identical class list

The benchmark script checks these fields before comparing results.

## Metrics

The comparison records:

- model name
- number of parameters
- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95
- inference latency per image
- model size on disk

## Selection Principle

The final model should not be selected only by highest mAP. A practical recommendation should consider accuracy, latency, and model size together.

Run:

```bash
python -m src.training.benchmark_models \
  --configs configs/models/yolo11n.yaml configs/models/yolo11s.yaml configs/models/yolo11m.yaml
```

Outputs:

```text
results/benchmarks/model_comparison.csv
results/benchmarks/model_comparison.md
```
