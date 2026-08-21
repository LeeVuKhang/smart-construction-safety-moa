"""Benchmark YOLO11n, YOLO11s, and YOLO11m with a fair protocol."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import yaml
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

        row = benchmark_model(config, weights)
        rows.append(row)

    add_recommendation_scores(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "model_comparison.csv")
    write_markdown(rows, output_dir / "model_comparison.md")
    return rows


def benchmark_model(config: dict, weights: Path) -> dict:
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
    latency = measure_latency(model, config["imgsz"])
    return {
        "model": config["model"],
        "experiment_name": config["experiment_name"],
        "params": count_parameters(model),
        "precision": standard_metrics["precision"],
        "recall": standard_metrics["recall"],
        "f1": calculate_f1(
            standard_metrics["precision"],
            standard_metrics["recall"],
        ),
        "map50": standard_metrics["map50"],
        "map50_95": standard_metrics["map50_95"],
        "latency_ms": latency,
        "model_size_mb": weights.stat().st_size / (1024 * 1024),
        "recommendation_note": recommendation_note(standard_metrics, latency, weights),
    }


def measure_latency(model: YOLO, imgsz: int, repeats: int = 5) -> float:
    """Measure approximate CPU/GPU inference latency using a synthetic image."""
    import numpy as np

    image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
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


def recommendation_note(metrics: dict[str, float], latency_ms: float, weights: Path) -> str:
    """Create a measured trade-off note without hardcoding a winner."""
    size_mb = weights.stat().st_size / (1024 * 1024)
    return (
        f"mAP50-95={metrics['map50_95']:.4f}, "
        f"latency={latency_ms:.2f} ms/image, size={size_mb:.2f} MB"
    )


def add_recommendation_scores(rows: list[dict]) -> None:
    """Add a simple measured trade-off score to each benchmark row."""
    max_map = max(row["map50_95"] for row in rows) or 1.0
    min_latency = min(row["latency_ms"] for row in rows) or 1.0
    min_size = min(row["model_size_mb"] for row in rows) or 1.0

    for row in rows:
        accuracy_score = row["map50_95"] / max_map
        latency_score = min_latency / row["latency_ms"]
        size_score = min_size / row["model_size_mb"]
        row["recommendation_score"] = (
            0.5 * accuracy_score + 0.25 * latency_score + 0.25 * size_score
        )


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Save benchmark rows as CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    """Save benchmark rows as a Markdown table."""
    headers = ["Model", "Params", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "Latency"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {params} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {map50:.4f} | {map50_95:.4f} | {latency_ms:.2f} ms |".format(
                **row
            )
        )
    lines.append("")
    lines.append("Trade-off notes:")
    for row in rows:
        lines.append(f"- {row['model']}: {row['recommendation_note']}")
    recommended = max(rows, key=lambda row: row["recommendation_score"])
    lines.append("")
    lines.append(
        "Recommendation: `{}` has the highest measured trade-off score "
        "using 50% mAP50-95, 25% latency, and 25% model size. "
        "Review the table before adopting it for a specific deployment constraint.".format(
            recommended["model"]
        )
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    return parser.parse_args()


def main() -> None:
    """Run model benchmark."""
    args = parse_args()
    rows = benchmark_configs(args.configs, args.output_dir, args.train_missing)
    print(f"Wrote benchmark results for {len(rows)} models to {args.output_dir}")


if __name__ == "__main__":
    main()
