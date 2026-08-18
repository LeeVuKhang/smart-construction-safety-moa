# Data Directory

Datasets should be stored locally and should not be committed to Git.

## Expected Layout

Raw datasets can be placed under:

```text
data/raw/
```

Processed YOLO-format datasets can be placed under:

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

The YOLO dataset YAML should define paths and classes, for example:

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: head
```

## Annotation Conversion

Some source annotations may use Pascal VOC XML. Convert annotations to YOLO text labels before training.

Use:

```bash
python -m src.data.convert_annotations --xml-dir data/raw/annotations --image-dir data/raw/images --output-dir data/processed/labels/train
```

Keep conversion and dataset preparation separate from training so experiments remain reproducible.
