import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    train_test_split = None


CLASS_MAP = {
    "helmet": 0,
    "head": 1,
}

CLASS_NAMES = {
    0: "with_helmet",
    1: "without_helmet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline YOLO11 PPE detection pipeline for helmet / head detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/hard_hat_detection",
        help="Folder containing the raw dataset with images/ and annotations/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ppe_yolo",
        help="Folder where the YOLO dataset will be prepared",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="YOLO weights to start from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU id or CPU. Use 'cpu' if no GPU is available",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs_train",
        help="Directory for training output",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ppe_yolo11n",
        help="Run name in the project folder",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help=(
            "Attempt to download the hard-hat dataset from Kaggle "
            "when the local dataset is missing"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the output dataset folder even if it already exists",
    )
    return parser.parse_args()


def download_kaggle_dataset(dataset_root: Path):
    dataset_root = dataset_root.resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess

        zip_path = dataset_root.parent / "hard-hat-detection.zip"
        if zip_path.exists():
            zip_path.unlink()

        zip_target = str(dataset_root.parent)
        print(f"[INFO] Downloading dataset from Kaggle into {zip_target}...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "andrewmvd/hard-hat-detection",
                "-p",
                zip_target,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if zip_path.exists():
            shutil.unpack_archive(str(zip_path), str(dataset_root.parent))
            extracted_dir = dataset_root.parent / "hard_hat_detection"
            if extracted_dir.exists():
                return extracted_dir
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Kaggle download failed: {exc}")
        print(
            "[WARN] Please install Kaggle CLI and configure credentials, "
            "or provide your own dataset path."
        )

    raise FileNotFoundError(
        "Dataset not found. Please add a dataset with images/ and annotations/ "
        "under the dataset-root path "
        "or use --download-kaggle after configuring Kaggle API access."
    )


def prepare_dataset(raw_dataset: Path, output_dir: Path):
    if not raw_dataset.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_dataset}")

    images_dir = raw_dataset / "images"
    annotations_dir = raw_dataset / "annotations"
    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images/' and 'annotations/' folders inside {raw_dataset}."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )
    if not image_files:
        raise FileNotFoundError(f"No image files were found in {images_dir}.")

    if train_test_split is None:
        raise ImportError("scikit-learn is required to split the dataset.")

    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    splits = {"train": train_files, "val": val_files, "test": test_files}

    def convert_box(img_w, img_h, xmin, ymin, xmax, ymax):
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        return x_center, y_center, width, height

    converted_images = 0
    converted_objects = 0

    for split, files in splits.items():
        for image_path in files:
            annotation_path = annotations_dir / f"{image_path.stem}.xml"
            if not annotation_path.exists():
                continue

            if Image is None:
                raise ImportError("Pillow is required to read image dimensions.")

            image = Image.open(image_path)
            img_w, img_h = image.size
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            label_lines = []

            for obj in root.findall("object"):
                class_name = obj.find("name")
                if class_name is None:
                    continue
                class_name = class_name.text
                if class_name not in CLASS_MAP:
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                x_center, y_center, width, height = convert_box(
                    img_w, img_h, xmin, ymin, xmax, ymax
                )
                label_lines.append(
                    f"{CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

            if not label_lines:
                continue

            shutil.copy(image_path, output_dir / "images" / split / image_path.name)
            label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            converted_images += 1
            converted_objects += len(label_lines)

    data_yaml = f"""path: {str(output_dir.resolve())}
train: images/train
val: images/val
test: images/test

names:
  0: with_helmet
  1: without_helmet
"""
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    counter = Counter()
    for txt_file in (output_dir / "labels").rglob("*.txt"):
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                class_id = line.strip().split()[0]
                counter[class_id] += 1

    print(f"[INFO] Converted images: {converted_images}")
    print(f"[INFO] Converted objects: {converted_objects}")
    print(f"[INFO] Class distribution: {dict(counter)}")
    print(f"[INFO] Data config saved to: {output_dir / 'data.yaml'}")


def train_model(
    output_dir: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    device: str,
    project: str,
    run_name: str,
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ultralytics is required. Install dependencies with pip install -r requirements.txt."
        ) from exc

    model = YOLO(weights)
    model.train(
        data=str(output_dir / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=run_name,
        seed=42,
    )

    metrics = model.val(data=str(output_dir / "data.yaml"), split="test")
    print("[INFO] Validation metrics:")
    print(metrics)

    best_model = Path(project) / run_name / "weights" / "best.pt"
    if best_model.exists():
        print(f"[INFO] Best model saved at: {best_model}")
    return best_model


def run_inference(model_path: Path, output_dir: Path, conf: float = 0.25):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ultralytics is required. Install dependencies with pip install -r requirements.txt."
        ) from exc

    model = YOLO(str(model_path))
    test_source = output_dir / "images" / "test"
    if not test_source.exists():
        raise FileNotFoundError(f"Test folder not found: {test_source}")

    model.predict(
        source=str(test_source),
        conf=conf,
        save=True,
        project=str(output_dir.parent),
        name=f"{output_dir.name}_predict",
    )
    print(
        "[INFO] Inference finished. Results are under: "
        f"{output_dir.parent / f'{output_dir.name}_predict'}"
    )


def main():
    args = parse_args()
    raw_dataset = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not raw_dataset.exists() and args.download_kaggle:
        raw_dataset = download_kaggle_dataset(raw_dataset)

    if not raw_dataset.exists():
        print(
            "[ERR] Dataset not found. Provide the correct --dataset-root "
            "or enable --download-kaggle.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"[INFO] Preparing dataset from: {raw_dataset}")
    prepare_dataset(raw_dataset, output_dir)

    print(f"[INFO] Training with YOLO weights: {args.weights}")
    model_path = train_model(
        output_dir=output_dir,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        project=args.project,
        run_name=args.run_name,
    )

    run_inference(model_path, output_dir)


if __name__ == "__main__":
    main()
