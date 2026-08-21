# Dataset Selection

This document records the dataset choice for the first construction-site safety baseline. The detector taxonomy is fixed:

```yaml
names:
  0: person
  1: helmet
  2: no_helmet
```

The baseline should start with one primary dataset. Merging datasets is deferred because mixed sources can hide duplicate images, incompatible object definitions, and inconsistent annotation policies.

## Sources Reviewed

| Dataset | Source | Original purpose | Size | Classes | Format | License/access | Split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hard Hat Workers | https://public.roboflow.com/object-detection/hard-hat-workers | Workplace hard-hat detection | 7,041 images; object count not available from public page | `head`, `helmet`, `person` | Roboflow exports; object detection | Public Domain | 75/25 train-test |
| Safety Helmet Detection / Hard Hat Detection | https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection and https://datasetninja.com/safety-helmet-detection | Safety/surveillance helmet detection | 5,000 images; Dataset Ninja reports 25,502 objects | `helmet`, `head`, `person` | Pascal VOC bounding boxes; PNG images reported by public mirrors | CC0 1.0 | No predefined train/val/test split |
| SH17 | https://github.com/ahmadmughees/SH17dataset and https://zenodo.org/records/12659325 | PPE detection in industrial/manufacturing environments | 8,099 images; 75,994 instances | 17 classes including `Person`, `Head`, `Helmet`, `Safety-vest`, tools, body parts, and PPE | YOLO-style labels published with images | Dataset record is restricted; images are sourced from Pexels with usage terms | No project-ready fixed split found in public metadata |
| Ultralytics Construction-PPE | https://docs.ultralytics.com/datasets/detect/construction-ppe | Construction-site PPE detection | 1,416 images; object count must be computed after download | `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots` | Ultralytics YOLO | Public download; Ultralytics dataset config is under the Ultralytics license notice | 1,132 train / 143 val / 141 test |
| Safety Helmet Extended Labels Dataset (SHELD) | https://data.mendeley.com/datasets/9rcv8mm682/3 | Repair and extend labels for the Kaggle Safety Helmet Detection dataset | 5,000 images; 75,578 labels | `helmet`, `head with helmet`, `person with helmet`, `head`, `person no helmet`, `face` | Extended bounding-box labels | CC BY 4.0 | No predefined project-ready split found |
| Color Helmet and Vest (CHV) | https://github.com/ZijianWang-ZW/PPE_detection | Real construction-site PPE detection for person, vest, and helmet colors | 1,330 images | `person`, `vest`, and helmet color classes | Detection annotations | Open for free use via linked drives; formal license not clear from repository page | Split details not clear from repository page |

## Class Mapping Analysis

| Source class | Target class | Decision |
| --- | --- | --- |
| `Person` / `person` | `person` | Accept if the source box is a full visible person instance. |
| `helmet` / helmet color classes | `helmet` | Accept for generic helmet detection. Helmet color is dropped for the baseline. |
| `no_helmet` | `no_helmet` | Accept only when the source annotation represents a missing-helmet violation object consistently enough for the detector. Visual audit is still required after download. |
| `head` | Not mapped | A head box can indicate a visible head without a helmet, but it is not semantically identical to a full-body `no_helmet` person instance. Mapping `head -> no_helmet` would mix head-sized boxes with person-sized boxes and create inconsistent bounding-box semantics. |
| `person no helmet` | Not mapped by default | This can represent a no-helmet person, but using it as both `person` and `no_helmet` duplicates one source object into two target objects with identical boxes. It needs a separate scientific justification before use. |
| `head with helmet` | Not mapped by default | This is a head state, not a generic helmet object. |
| Other PPE classes | Dropped | Boots, gloves, goggles, vest, tools, face, ears, hands, and similar classes are outside the first detector taxonomy. |

The critical scientific issue in Hard Hat Workers, Safety Helmet Detection, and SH17 is the `head` class. It may be useful for a future head-level helmet-status model, but it is not a clean replacement for target `no_helmet` in this repository because the baseline taxonomy expects consistent object definitions across YOLO11n/s/m.

## Scoring Rubric

Weights:

| Criterion | Weight |
| --- | ---: |
| Domain relevance | 25% |
| Label/taxonomy compatibility | 25% |
| Annotation quality | 15% |
| Dataset size | 15% |
| Class balance | 10% |
| License/accessibility | 5% |
| Split/reproducibility | 5% |

Scores are from 1 to 5. Weighted totals are out of 5.

| Dataset | Domain | Taxonomy | Quality | Size | Balance | License | Split | Weighted total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hard Hat Workers | 4 | 2 | 3 | 4 | 2 | 5 | 3 | 3.15 | Rejected for baseline due to `head -> no_helmet` semantic mismatch. |
| Safety Helmet Detection / Hard Hat Detection | 3 | 2 | 3 | 3 | 2 | 5 | 1 | 2.75 | Rejected for baseline due to same `head` mismatch and no official split. |
| SH17 | 4 | 2 | 4 | 5 | 2 | 2 | 2 | 3.20 | Rejected for first baseline; valuable PPE dataset but too broad and lacks clean `no_helmet`. |
| Ultralytics Construction-PPE | 5 | 5 | 3 | 2 | 3 | 4 | 5 | 4.05 | Selected as primary baseline dataset. |
| SHELD | 3 | 3 | 4 | 3 | 3 | 4 | 1 | 3.20 | Not selected; extended labels are useful, but `person no helmet` needs a harmonization study. |
| CHV | 5 | 2 | 4 | 2 | 2 | 2 | 2 | 3.05 | Not selected; strong construction domain but no direct `no_helmet` class. |

The selected dataset is not the largest dataset. It is selected because it is the only reviewed candidate with construction-site domain relevance, YOLO-format annotations, predefined train/val/test splits, and direct source classes for all three target classes without using `head` as a proxy.

## Selected Dataset

Primary dataset: Ultralytics Construction-PPE.

Selection rationale:

- It is explicitly construction-site PPE data.
- It has direct source classes for `person`, `helmet`, and `no_helmet`.
- It provides an official fixed split, which improves reproducibility.
- It avoids the main semantic error of mapping `head` to `no_helmet`.
- It is small enough for fast baseline iteration, which is acceptable for the first model-selection round.

Known risks:

- Public metadata does not provide target-class object counts; this must be measured after download.
- The exact bounding-box policy for `no_helmet` must be visually checked after preparation.
- Other PPE classes are dropped, so images containing only dropped classes become background or empty-label images.
- Source labels may contain boxes extending slightly outside image boundaries. The preparation script clips target boxes to valid YOLO image bounds and records the count in `mapping_summary.json`; this is treated as data sanitation, not a semantic relabeling.

## Dataset Preparation Plan

Use the official split from Construction-PPE rather than resplitting 70/20/10. The official split is preferred here because it is already provided by the dataset source and prevents accidental leakage from a naive random split.

Preparation command:

```bash
python -m src.data.prepare_construction_ppe --download
```

This creates:

```text
data/processed/dataset.yaml
data/processed/mapping_summary.json
data/processed/images/train
data/processed/images/val
data/processed/images/test
data/processed/labels/train
data/processed/labels/val
data/processed/labels/test
```

The preparation step records:

```text
kept_target_objects
dropped_non_target_objects
clipped_target_boxes
invalid_target_boxes_after_clip
empty_target_labels
```

Final mapping:

| Source dataset | Source class | Target class |
| --- | --- | --- |
| Ultralytics Construction-PPE | `Person` | `person` |
| Ultralytics Construction-PPE | `helmet` | `helmet` |
| Ultralytics Construction-PPE | `no_helmet` | `no_helmet` |

Dropped source classes:

```text
gloves, vest, boots, goggles, none, no_goggle, no_gloves, no_boots
```

## Audit And Training Gate

Run the audit before training:

```bash
python -m src.data.audit_dataset --fail-on-critical
```

The audit must generate:

```text
results/dataset_audit/dataset_summary.json
results/dataset_audit/class_distribution.csv
results/dataset_audit/audit_report.md
```

Training must not start if the audit reports critical errors.

After the dataset is selected, prepared, audited, and frozen, the three candidate models use identical conditions:

```yaml
epochs: 100
batch: 16
imgsz: 640
patience: 20
seed: 42
workers: 4
optimizer: auto
device: auto
validation:
  conf: 0.25
  iou: 0.7
```

Training commands:

```bash
bash scripts/train_all.sh
python -m src.training.benchmark_models --configs configs/models/yolo11n.yaml configs/models/yolo11s.yaml configs/models/yolo11m.yaml
```

Only after real benchmark rows exist should `configs/models/selected_model.yaml` be created.

## Final Status

- Candidate comparison: complete.
- Primary dataset selected: Ultralytics Construction-PPE.
- Class mapping documented: complete.
- Preparation script: `src/data/prepare_construction_ppe.py`.
- Dataset audit: pending until the dataset archive is downloaded and prepared in `data/processed`.
- YOLO11n/s/m training: pending a clean audit.
- Benchmark table and selected detector: pending completed training and benchmarking.
