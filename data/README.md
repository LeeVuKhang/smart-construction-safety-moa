# Data Directory

Datasets are stored locally and are not committed to Git.

## Detection Dataset

Raw datasets can be placed under:

```text
data/raw/
```

Processed YOLO-format datasets should use:

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

Example YOLO dataset YAML:

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
  2: no_helmet
```

## Annotation Conversion

Some source annotations may use Pascal VOC XML. Convert annotations to YOLO labels before training:

```bash
python -m src.data.convert_annotations \
  --xml-dir data/raw/annotations \
  --output-dir data/processed/labels/train
```

Keep annotation conversion separate from training.

## Zone Evaluation Data

Zone Grounding does not require a training dataset. It can be evaluated with manually prepared person detections and expected zones:

```text
data/zone_eval/
├── images/
├── zones/
│   └── cam_01.yaml
├── detections.json
└── ground_truth.json
```

Example ground truth:

```json
[
  {
    "frame_id": "frame_001",
    "person_id": "P01",
    "expected_zone": "Z01"
  }
]
```
