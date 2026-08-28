# Prompt-Based Zone Recognition

This is the zone task where the input is one image plus a prompt, and the output is a background zone label.

## Task

```text
image + prompt -> zone_type / zone_id
```

The current configured zone vocabulary is:

| Zone ID | Zone type |
| --- | --- |
| `Z00` | `general_area` |
| `Z01` | `active_work_area` |
| `Z02` | `restricted_area` |

## DAM Baseline

Describe Anything Model can be used for this prompt-based zone task because it accepts an image region and returns text. For background-zone recognition, the full image is sent as the selected region and the prompt asks for one configured zone label.

Run DAM server:

```bash
cd /data/quyhv/data/describe-anything
CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/quyhv/data/hf_cache \
  /data/quyhv/data/dam_env/bin/python dam_server.py \
  --model-path nvidia/DAM-3B \
  --conv-mode v1 \
  --prompt-mode focal_prompt \
  --temperature 0.2 \
  --top_p 0.9 \
  --num_beams 1 \
  --max_new_tokens 512 \
  --workers 1
```

Classify one image:

```bash
python3 -m src.evaluation.dam_prompt_zone_eval \
  --image data/processed/images/val/image1010.jpg \
  --server-url http://localhost:8000 \
  --output-dir results/dam_zone_prompt
```

Evaluate a labeled manifest:

```bash
python3 -m src.evaluation.dam_prompt_zone_eval \
  --manifest data/zone_eval/prompt_zone_manifest.json \
  --server-url http://localhost:8000 \
  --output-dir results/dam_zone_prompt
```

Manifest format:

```json
[
  {
    "sample_id": "frame_001",
    "image_path": "data/zone_eval/images/frame_001.jpg",
    "expected_zone_type": "restricted_area"
  }
]
```

## Relation To Polygon Zone Grounding

Prompt-based zone recognition and polygon zone grounding are different tasks:

| Task | Input | Output | Best baseline |
| --- | --- | --- | --- |
| Prompt-based background zone recognition | image + prompt | semantic zone label | VLM/DAM-style baseline |
| Fixed-camera person zone grounding | person bbox + configured polygons | exact zone ID | deterministic polygon method |

The previous DAM PPE classification experiment is not the right evidence for this zone task. For this task, evidence must come from labeled images with expected background zone labels.
