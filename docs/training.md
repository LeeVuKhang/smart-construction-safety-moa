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

## Dataset Audit

Before training, run:

```bash
python -m src.data.audit_dataset --fail-on-critical
```

The audit reports image counts, label counts, class distribution, missing labels, orphan labels, corrupt images, invalid boxes, unknown class IDs, empty labels, duplicates, cross-split duplicates, and class imbalance.

Outputs:

```text
results/dataset_audit/dataset_summary.json
results/dataset_audit/class_distribution.csv
results/dataset_audit/audit_report.md
```

Training calls the same audit and stops if critical errors are present.

## Fixed Split and Leakage Prevention

The initial benchmark uses one fixed dataset split for all models:

```text
train: 70%
validation: 20%
test: 10%
seed: 42
```

If `images/train`, `images/val`, and `images/test` already exist, the existing split is preserved. The audit writes split manifests under `data/processed/splits/`.

The audit detects exact duplicate images across splits and flags possible source leakage from filename groups. If source or video metadata is available, use a source-aware split before training so sequential frames from the same source do not appear in both train and test.

Do not create augmented duplicates before splitting. Split first, then let Ultralytics apply training augmentation only.

## Configuration

Each model config stores the model checkpoint, dataset YAML, output directory, experiment name, seed, image size, batch size, epoch count, patience, workers, optimizer, and validation thresholds.

Training output is saved under:

```text
results/training/<experiment_name>/
```

The script also saves `training_args.yaml` and `metadata.json` for reproducibility.
It also saves `experiment_manifest.json` and `training_summary.json`.

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
