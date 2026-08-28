# Describe Anything Comparison

Describe Anything Model (DAM) is reviewed as an external comparison model from `NVlabs/describe-anything`.

Sources:

- GitHub: https://github.com/NVlabs/describe-anything
- Project page: https://describe-anything.github.io/
- Paper: https://arxiv.org/abs/2504.16072
- Model: https://huggingface.co/nvidia/DAM-3B

## Model Role

DAM is a detailed localized captioning model. It accepts an image or video region specified by points, boxes, scribbles, or masks and generates text describing that region. It is not a closed-set object detector and does not output construction PPE bounding boxes in the same form as YOLO.

Therefore, DAM must not be compared directly against YOLO11n/s/m using detector metrics such as mAP unless an additional localization and label-extraction pipeline is defined.

This file is a PPE semantic-classification appendix. It is not the zone-grounding comparison. The zone-specific comparison is documented in `docs/zone_grounding_comparison.md`.

## Fair Comparison Protocol

The first scientifically defensible comparison is a region-level PPE classification test:

1. Use the same audited Construction-PPE validation split.
2. Use ground-truth YOLO boxes as region prompts for DAM.
3. Ask DAM to classify the masked region as one of:

```text
person
helmet
no_helmet
other
```

4. Parse the returned text conservatively.
5. Report accuracy, per-class precision, recall, F1, and confusion matrix.

This tests whether DAM can recognize PPE semantics when localization is already supplied. It does not test detection ability.

## Why This Is Not A YOLO Replacement Benchmark

YOLO11 models are evaluated end-to-end as detectors:

```text
image -> boxes + class labels + confidence
```

DAM is evaluated as:

```text
image + region mask -> text description -> parsed class label
```

The second pipeline receives additional region information, so its metrics must be reported separately from detector mAP.

## Evaluation Command

Start the DAM server from the NVlabs repository:

```bash
python dam_server.py \
  --model-path nvidia/DAM-3B \
  --conv-mode v1 \
  --prompt-mode focal_prompt \
  --temperature 0.2 \
  --top_p 0.9 \
  --num_beams 1 \
  --max_new_tokens 512 \
  --workers 1
```

Then run:

```bash
python3 -m src.evaluation.dam_region_ppe_eval \
  --dataset-yaml data/processed/dataset.yaml \
  --split val \
  --server-url http://localhost:8000 \
  --max-samples 100 \
  --output-dir results/dam_comparison
```

For a dataset/protocol check without calling DAM:

```bash
python3 -m src.evaluation.dam_region_ppe_eval \
  --dataset-yaml data/processed/dataset.yaml \
  --split val \
  --dry-run \
  --max-samples 100
```

## Output Files

```text
results/dam_comparison/dam_region_predictions.csv
results/dam_comparison/dam_region_metrics.json
results/dam_comparison/dam_region_report.md
```

## Current Baseline For Comparison

The selected detector is YOLO11n:

| Model | mAP50-95 | Recall | no_helmet Recall | Latency | Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.3557 | 0.7875 | 0.6222 | 8.49 ms | 5.23 MB |

DAM comparison results should be reported under a separate "Region-level semantic PPE classification" section, not merged into the YOLO mAP table.

## Region-Level DAM Result

Run environment:

- Server: `quyhv-server`
- DAM source repository commit: `153ad3d`
- DAM model: `nvidia/DAM-3B`
- Evaluation split: validation
- Evaluated regions: first 100 ground-truth YOLO boxes
- Runtime: 2:17.24 for 100 region requests
- DAM server GPU memory during evaluation: approximately 7.5 GB on one Tesla T4

Aggregate result:

| Model | Evaluation type | Regions | Accuracy | Runtime |
| --- | --- | ---: | ---: | ---: |
| YOLO11n | End-to-end object detection | full validation split | mAP50-95 0.3557 | 8.49 ms/image |
| DAM-3B | Ground-truth-region semantic classification | 100 regions | 0.1900 | 2:17.24 total |

DAM per-class result:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `person` | 0.4318 | 0.4043 | 0.4176 | 47 |
| `helmet` | 0.0000 | 0.0000 | 0.0000 | 42 |
| `no_helmet` | 0.0000 | 0.0000 | 0.0000 | 11 |

Prediction distribution:

| Prediction | Count |
| --- | ---: |
| `unknown` | 51 |
| `person` | 44 |
| `no_helmet` | 4 |
| `helmet` | 1 |

Confusion matrix:

| Ground truth | person | helmet | no_helmet | other | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `person` | 19 | 1 | 0 | 0 | 27 |
| `helmet` | 14 | 0 | 4 | 0 | 24 |
| `no_helmet` | 11 | 0 | 0 | 0 | 0 |

Interpretation: DAM-3B did not work well as a closed-set PPE region classifier under this protocol. It often returned descriptive or coordinate-like outputs instead of a stable class label, so the evaluator uses a conservative parser and records unsupported responses as `unknown`. This result is evidence only for the PPE semantic-labeling path, not for the zone-grounding component.

## Status

- Comparison protocol: prepared.
- Evaluation client: `src/evaluation/dam_region_ppe_eval.py`.
- Actual DAM metrics: completed for the first 100 validation regions.
