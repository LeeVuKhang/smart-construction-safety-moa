# Final Baseline Report

## 1. Candidate Dataset Comparison

The evaluated candidates are documented in `docs/dataset_selection.md` and `results/dataset_selection/dataset_comparison.csv`.

Reviewed datasets:

- Hard Hat Workers
- Safety Helmet Detection / Hard Hat Detection
- SH17
- Ultralytics Construction-PPE
- Safety Helmet Extended Labels Dataset (SHELD)
- Color Helmet and Vest (CHV)

## 2. Dataset Scoring

The scoring rubric used:

| Criterion | Weight |
| --- | ---: |
| Domain relevance | 25% |
| Label/taxonomy compatibility | 25% |
| Annotation quality | 15% |
| Dataset size | 15% |
| Class balance | 10% |
| License/accessibility | 5% |
| Split/reproducibility | 5% |

Final weighted scores:

| Dataset | Weighted score | Decision |
| --- | ---: | --- |
| Hard Hat Workers | 3.15 | Rejected |
| Safety Helmet Detection / Hard Hat Detection | 2.75 | Rejected |
| SH17 | 3.20 | Rejected for first baseline |
| Ultralytics Construction-PPE | 4.05 | Selected |
| SHELD | 3.20 | Not selected |
| CHV | 3.05 | Not selected |

## 3. Selected Dataset

Selected primary dataset: Ultralytics Construction-PPE.

## 4. Scientific Reason For Selection

Ultralytics Construction-PPE is selected because it is construction-site relevant, directly provides `Person`, `helmet`, and `no_helmet` source classes, uses YOLO-format annotations, and includes an official train/val/test split. It avoids the critical semantic problem found in several larger datasets where `head` would have to be incorrectly mapped to `no_helmet`.

## 5. Rejected Datasets

- Hard Hat Workers: rejected because `head -> no_helmet` would mix head-level boxes with the target detector taxonomy.
- Safety Helmet Detection / Hard Hat Detection: rejected for the same `head` mismatch and no official train/val/test split.
- SH17: rejected for the first baseline because it is broad PPE/body-part data without a clean `no_helmet` class.
- SHELD: not selected because `person no helmet` requires a separate harmonization study before duplicating or remapping boxes.
- CHV: not selected because it has strong construction relevance but no direct `no_helmet` class.

## 6. Final Class Mapping

| Source class | Target class |
| --- | --- |
| `Person` | `person` |
| `helmet` | `helmet` |
| `no_helmet` | `no_helmet` |

Dropped classes:

```text
gloves, vest, boots, goggles, none, no_goggle, no_gloves, no_boots
```

## 7. Dataset Audit Summary

Audit output:

```text
results/dataset_audit/dataset_summary.json
results/dataset_audit/class_distribution.csv
results/dataset_audit/audit_report.md
```

Audit summary after preparation:

| Item | Value |
| --- | ---: |
| Total images | 1,416 |
| Train images | 1,132 |
| Val images | 143 |
| Test images | 141 |
| Total target boxes | 4,464 |
| `person` boxes | 2,245 |
| `helmet` boxes | 1,734 |
| `no_helmet` boxes | 485 |
| Class imbalance ratio | 4.63 |
| Critical errors | 0 |

The preparation step clipped source boxes that extended outside image boundaries and recorded the counts in `data/processed/mapping_summary.json`.

## 8. Fixed Split Strategy

The official Construction-PPE split is preserved:

```text
train: 1,132 images
val: 143 images
test: 141 images
seed: 42
```

The official split is used instead of a new random 70/20/10 split to avoid accidental leakage from naive resplitting.

## 9. YOLO11n/s/m Training Status

All models were trained on `quyhv-server` with the same dataset, taxonomy, seed, image size, batch size, optimizer policy, patience, workers, and validation thresholds.

| Model | Status | Epoch status | Best weights |
| --- | --- | --- | --- |
| YOLO11n | Complete | Early stopped after no improvement; best epoch reported by Ultralytics: 76 | `results/training/yolo11n_v1/weights/best.pt` |
| YOLO11s | Complete | Early stopped after no improvement; best epoch reported by Ultralytics: 52 | `results/training/yolo11s_v1/weights/best.pt` |
| YOLO11m | Complete | Completed 100 epochs | `results/training/yolo11m_v1/weights/best.pt` |

## 10. Benchmark Table

Benchmark output:

```text
results/benchmarks/model_comparison.csv
results/benchmarks/model_comparison.md
results/benchmarks/per_class_metrics.csv
```

| Model | Params | Size | Precision | Recall | F1 | mAP50 | mAP50-95 | Latency | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 2,582,737 | 5.23 MB | 0.7167 | 0.7875 | 0.7505 | 0.7353 | 0.3557 | 10.20 ms | 97.99 |
| YOLO11s | 9,413,961 | 18.30 MB | 0.7643 | 0.6548 | 0.7053 | 0.6163 | 0.3143 | 9.99 ms | 100.08 |
| YOLO11m | 20,032,345 | 38.65 MB | 0.7476 | 0.6907 | 0.7181 | 0.6488 | 0.3279 | 22.42 ms | 44.59 |

Per-class `no_helmet` recall:

| Model | `no_helmet` recall | `no_helmet` mAP50-95 |
| --- | ---: | ---: |
| YOLO11n | 0.6222 | 0.1526 |
| YOLO11s | 0.2889 | 0.0842 |
| YOLO11m | 0.4000 | 0.1187 |

## 11. Selected Detector

Selected detector: YOLO11n.

Justification:

- Highest primary metric: mAP50-95 = 0.3557.
- Highest aggregate recall: 0.7875.
- Highest `no_helmet` recall: 0.6222.
- Smallest model: 5.23 MB.
- Efficient latency: 10.20 ms/image on the measured Tesla T4.

YOLO11s is marginally faster in this latency run, but its accuracy and especially `no_helmet` recall are substantially worse. YOLO11m is larger and slower while not improving the primary metric.

## 12. Unresolved Issues

- `no_helmet` remains the smallest class with 485 target boxes and only 45 validation instances.
- Source labels required boundary clipping during preparation.
- Ultralytics removed one duplicate label from `image187.jpg` during training.
- Construction-PPE is a small baseline dataset; broader validation on additional construction scenes is still needed before safety-critical use.
- Benchmark latency was measured on one Tesla T4 and may differ on deployment hardware.

## 13. Reproduction Commands

```bash
cd /data/quyhv/data/Construction-Safety-Monitoring
git pull --ff-only
python3 -m src.data.prepare_construction_ppe --download
python3 -m src.data.audit_dataset --fail-on-critical
bash scripts/train_all.sh
python3 -m src.training.benchmark_models \
  --configs configs/models/yolo11n.yaml configs/models/yolo11s.yaml configs/models/yolo11m.yaml \
  --output-dir results/benchmarks
```

The selected detector config is written only after benchmarking:

```text
configs/models/selected_model.yaml
```
