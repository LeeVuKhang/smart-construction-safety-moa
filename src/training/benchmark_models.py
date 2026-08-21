"""Benchmark YOLO11n, YOLO11s, and YOLO11m with a fair protocol."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.evaluation.detection_metrics import calculate_f1, metrics_from_yolo
from src.training.train_yolo import load_config, train, validate_config


FAIRNESS_FIELDS = [
    "dataset_yaml",
    "epochs",
    "batch",
    "imgsz",
    "patience",
    "seed",
    "workers",
    "optimizer",
    "augmentation",
    "validation",
    "classes",
]


def validate_fair_comparison(configs: list[dict]) -> None:
    """Ensure model configs differ only where fair model comparison allows."""
    for config in configs:
        validate_config(config)

    reference = configs[0]
    for config in configs[1:]:
        mismatches = [
            field for field in FAIRNESS_FIELDS if config[field] != reference[field]
        ]
        if mismatches:
            raise ValueError(
                "Model configs are not comparable. "
                f"Mismatched fields for {config['experiment_name']}: {mismatches}"
            )


def benchmark_configs(
    config_paths: list[Path],
    output_dir: Path,
    train_missing: bool = False,
    warmup: int = 20,
    repeats: int = 100,
) -> list[dict]:
    """Validate trained models and save benchmark summary files."""
    configs = [load_config(path) for path in config_paths]
    validate_fair_comparison(configs)

    rows = []
    for config in configs:
        experiment_dir = Path(config["output_dir"]) / config["experiment_name"]
        weights = experiment_dir / "weights" / "best.pt"

        if train_missing and not weights.exists():
            train(config)

        if not weights.exists():
            raise FileNotFoundError(
                f"Missing weights for {config['experiment_name']}: {weights}. "
                "Run training first or pass --train-missing."
            )

        row, per_class = benchmark_model(config, weights, warmup=warmup, repeats=repeats)
        rows.append(row)
        write_per_class_rows(per_class, output_dir / "per_class_metrics.csv", append=bool(rows[:-1]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "model_comparison.csv")
    write_markdown(rows, output_dir / "model_comparison.md")
    write_hardware_info(output_dir / "hardware.json")
    write_selected_model(select_model(rows), output_dir)
    return rows


def benchmark_model(
    config: dict,
    weights: Path,
    warmup: int = 20,
    repeats: int = 100,
) -> tuple[dict, list[dict]]:
    """Evaluate one trained model and collect size/speed metadata."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=config["dataset_yaml"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        conf=config["validation"]["conf"],
        iou=config["validation"]["iou"],
        project="results/evaluation",
        name=f"{config['experiment_name']}_benchmark",
        exist_ok=True,
    )

    standard_metrics = metrics_from_yolo(metrics)
    latency = measure_latency(model, config["imgsz"], warmup=warmup, repeats=repeats)
    fps = 1000 / latency if latency > 0 else 0.0
    row = {
        "model": config["model"],
        "experiment_name": config["experiment_name"],
        "params": count_parameters(model),
        "size_mb": weights.stat().st_size / (1024 * 1024),
        "precision": standard_metrics["precision"],
        "recall": standard_metrics["recall"],
        "f1": calculate_f1(
            standard_metrics["precision"],
            standard_metrics["recall"],
        ),
        "map50": standard_metrics["map50"],
        "map50_95": standard_metrics["map50_95"],
        "latency_ms": latency,
        "fps": fps,
        "weights": str(weights),
        "recommendation_note": recommendation_note(standard_metrics, latency, fps, weights),
    }
    return row, per_class_metrics(config, metrics)


def measure_latency(model: YOLO, imgsz: int, warmup: int = 20, repeats: int = 100) -> float:
    """Measure mean inference latency excluding model-loading time."""
    import numpy as np

    image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, verbose=False)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(image, imgsz=imgsz, verbose=False)
    return ((time.perf_counter() - start) / repeats) * 1000


def count_parameters(model: YOLO) -> int | None:
    """Return model parameter count when available."""
    torch_model = getattr(model, "model", None)
    if torch_model is None:
        return None
    return sum(parameter.numel() for parameter in torch_model.parameters())


def recommendation_note(
    metrics: dict[str, float],
    latency_ms: float,
    fps: float,
    weights: Path,
) -> str:
    """Create a measured trade-off note without hardcoding a winner."""
    size_mb = weights.stat().st_size / (1024 * 1024)
    return (
        f"mAP50-95={metrics['map50_95']:.4f}, "
        f"latency={latency_ms:.2f} ms/image, FPS={fps:.2f}, size={size_mb:.2f} MB"
    )


def per_class_metrics(config: dict, metrics) -> list[dict]:
    """Extract per-class precision, recall, mAP50, and mAP50-95 when available."""
    rows = []
    box = metrics.box
    precision = metric_list(getattr(box, "p", []))
    recall = metric_list(getattr(box, "r", []))
    ap50 = metric_list(getattr(box, "ap50", []))
    ap = getattr(box, "ap", None)
    if ap is not None and getattr(ap, "ndim", 1) == 2:
        map50_95 = [float(row.mean()) for row in ap]
    else:
        map50_95 = metric_list(getattr(box, "maps", []))

    for index, class_name in enumerate(config["classes"]):
        rows.append(
            {
                "model": config["model"],
                "experiment_name": config["experiment_name"],
                "class": class_name,
                "precision": value_at(precision, index),
                "recall": value_at(recall, index),
                "map50": value_at(ap50, index),
                "map50_95": value_at(map50_95, index),
            }
        )
    return rows


def metric_list(values) -> list:
    """Convert optional scalar/array metric values to a Python list."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (float, int)):
        return [values]
    return list(values)


def value_at(values: list, index: int) -> float | None:
    """Return a float value from a metric list when available."""
    if index >= len(values):
        return None
    return float(values[index])


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Save benchmark rows as CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_rows(rows: list[dict], output_path: Path, append: bool) -> None:
    """Write or append per-class metric rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    """Save benchmark rows as a Markdown table."""
    headers = [
        "Model",
        "Params",
        "Size",
        "Precision",
        "Recall",
        "F1",
        "mAP50",
        "mAP50-95",
        "Latency",
        "FPS",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {params} | {size_mb:.2f} MB | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {map50:.4f} | {map50_95:.4f} | {latency_ms:.2f} ms | "
            "{fps:.2f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Scientific Interpretation")
    lines.append("")
    lines.append(interpretation(rows))
    lines.append("")
    lines.append("## Trade-off Notes")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['model']}: {row['recommendation_note']}")
    recommended = select_model(rows)
    lines.append("")
    lines.append(
        "Recommendation: `{}` is selected by the documented rule: prioritize "
        "mAP50-95, inspect recall, and prefer the smaller/faster model when "
        "accuracy gains are marginal relative to latency and size.".format(
            recommended["model"]
        )
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(rows: list[dict]) -> str:
    """Create concise benchmark interpretation from measured results."""
    best_map = max(rows, key=lambda row: row["map50_95"])
    best_recall = max(rows, key=lambda row: row["recall"])
    fastest = min(rows, key=lambda row: row["latency_ms"])
    return (
        f"`{best_map['model']}` has the highest mAP50-95. "
        f"`{best_recall['model']}` has the best aggregate recall. "
        f"`{fastest['model']}` is fastest under the measured hardware condition. "
        "Inspect `per_class_metrics.csv`, especially `no_helmet` recall, before "
        "using the detector in safety-monitoring experiments. Class imbalance effects "
        "should be interpreted together with the dataset audit report."
    )


def select_model(rows: list[dict]) -> dict:
    """Select a detector using a documented quality-first trade-off rule."""
    ranked = sorted(rows, key=lambda row: row["map50_95"], reverse=True)
    best = ranked[0]
    best_map = best["map50_95"]
    candidates = [row for row in rows if best_map - row["map50_95"] <= 0.01]
    candidates = sorted(candidates, key=lambda row: (-row["recall"], row["latency_ms"], row["size_mb"]))
    return candidates[0]


def write_selected_model(selected: dict, output_dir: Path) -> None:
    """Write selected model config only after real benchmark rows exist."""
    model_name = Path(selected["model"]).stem
    config = {
        "model_name": model_name,
        "weights": selected["weights"],
        "selection_basis": {
            "primary_metric": "mAP50-95",
            "secondary_metric": "recall",
            "efficiency_metric": "latency",
            "rule": "prefer smaller/faster model when mAP50-95 gain is within 0.01",
        },
    }
    path = Path("configs/models/selected_model.yaml")
    with path.open("w", encoding="utf-8") as file:
        import yaml

        yaml.safe_dump(config, file, sort_keys=False)


def write_hardware_info(output_path: Path) -> None:
    """Save hardware/runtime information for latency interpretation."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark YOLO11 model variants.")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            Path("configs/models/yolo11n.yaml"),
            Path("configs/models/yolo11s.yaml"),
            Path("configs/models/yolo11m.yaml"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Run model benchmark."""
    args = parse_args()
    rows = benchmark_configs(args.configs, args.output_dir, args.train_missing, args.warmup, args.repeats)
    print(f"Wrote benchmark results for {len(rows)} models to {args.output_dir}")


if __name__ == "__main__":
    main()
