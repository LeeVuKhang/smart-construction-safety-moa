# Construction Safety Monitoring

This repository supports a university AI research project on construction-site safety monitoring. The current implementation is a reproducible YOLO11 detection baseline with a separate deterministic Zone Grounding Agent.

## Research Motivation

Construction sites contain dynamic hazards, changing work areas, and different PPE requirements across regions. A clear perception baseline and deterministic region grounding are needed before adding rule-based safety reasoning or future multi-agent analysis.

## Current Research Problem

The detector uses exactly three classes:

- `person`
- `helmet`
- `no_helmet`

The detector performs perception only. PPE evidence, zone grounding, and future rule evaluation remain separate components.

## Current Research Pipeline

```text
Input Image / Video Frame
        |
        v
YOLO11 Detector
        |
        v
Structured Detections
        |
 ┌─────────────────┬─────────────────────┐
 v                 v
PPE Agent      Zone Grounding Agent
 v                 v
PPE Evidence    Region Grounding
```

Future modules are not implemented yet:

```text
Evidence Joiner
Rule Agent
Context Agent
Behavior Agent
Multi-Agent / MoA
Automatic Safety Reporting
```

## Current Baseline

YOLO11n is the starting baseline. YOLO11s and YOLO11m are configured for fair model-selection experiments using the same dataset, split, seed, image size, augmentation policy, and evaluation protocol.

No experimental results or model recommendation are reported until real training and benchmarking are completed.

## Metrics

Detection metrics:

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95

Zone grounding metric:

- Zone Assignment Accuracy

## Repository Structure

```text
Construction-Safety-Monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── models/
│   │   ├── yolo11n.yaml
│   │   ├── yolo11s.yaml
│   │   └── yolo11m.yaml
│   ├── zones/
│   │   └── cam_01.yaml
│   └── rules/
│       └── ppe_rules.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── baseline_yolo11.ipynb
├── src/
│   ├── detection/
│   ├── training/
│   ├── evaluation/
│   ├── agents/
│   ├── geometry/
│   ├── rules/
│   └── pipeline/
├── scripts/
├── results/
├── docs/
└── tests/
```

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Setup

Datasets are not committed to Git. Place YOLO-format datasets under `data/processed/` and update the `dataset_yaml` field in each file under `configs/models/`.

See `data/README.md` for the expected dataset and zone-evaluation layouts.

Run the dataset audit before training:

```bash
python -m src.data.audit_dataset --fail-on-critical
```

## Training

Train one model:

```bash
python -m src.training.train_yolo --config configs/models/yolo11n.yaml
```

Train all configured models:

```bash
bash scripts/train_all.sh
```

## Evaluation

```bash
python -m src.evaluation.detection_metrics \
  --config configs/models/yolo11n.yaml \
  --weights results/training/yolo11n_v1/weights/best.pt
```

## Model Benchmark

```bash
python -m src.training.benchmark_models \
  --configs configs/models/yolo11n.yaml configs/models/yolo11s.yaml configs/models/yolo11m.yaml
```

Benchmarking requires trained weights and a real dataset. It saves:

```text
results/benchmarks/model_comparison.csv
results/benchmarks/model_comparison.md
results/benchmarks/per_class_metrics.csv
```

## Zone Grounding

Zones are fixed polygons configured manually for each camera under `configs/zones/`. A person is assigned using the bottom-center anchor of the person bounding box. If multiple polygons contain the anchor, the highest-priority zone wins. If no polygon contains the anchor, the person is assigned to `default_zone`.

Zone Grounding does not check PPE, evaluate rules, trigger alerts, or perform behavior/MoA reasoning.

The zone comparison and fixture evaluation are documented in `docs/zone_grounding_comparison.md`.
