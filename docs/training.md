# Reproducible YOLO Training

The training workflow uses Ultralytics YOLO11 with experiment parameters stored in YAML files under `configs/models/`.

## Dataset Assumptions

The dataset must be in YOLO format and must use exactly these class names:

```text
person
helmet
no_helmet
```

The dataset YAML path is configured through `dataset_yaml`. Do not use absolute machine-specific paths.

## Configuration

Each model config stores the model checkpoint, dataset YAML, output directory, experiment name, seed, image size, batch size, epoch count, patience, workers, optimizer, and validation thresholds.

Training output is saved under:

```text
results/training/<experiment_name>/
```

The script also saves `training_args.yaml` and `metadata.json` for reproducibility.

## Commands

Train YOLO11n:

```bash
python -m src.training.train_yolo --config configs/models/yolo11n.yaml
```

Train YOLO11s:

```bash
python -m src.training.train_yolo --config configs/models/yolo11s.yaml
```

Train YOLO11m:

```bash
python -m src.training.train_yolo --config configs/models/yolo11m.yaml
```

Train all configured models:

```bash
bash scripts/train_all.sh
```
