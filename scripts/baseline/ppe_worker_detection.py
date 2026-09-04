"""PPE worker detection: locates whole-body worker bounding boxes in an image
and cross-references PPE (helmet) compliance for each worker.

Designed for single images now; the frame-based API (`process_frame`) is
already the right shape to plug into video processing later (call it per
decoded frame instead of per image file).

Zone assignment is a separate module (see worker_zone_detection.py) and is
intentionally not part of this script.
"""

import argparse
import json
from pathlib import Path

import cv2

DEFAULT_PERSON_WEIGHTS = "yolo11n.pt"
PERSON_CLASS_ID = 0  # COCO class id for "person"

# Class ids from the trained helmet model (results/yolo11_helmet/weights/best.pt)
PPE_CLASS_NAMES = {0: "with_helmet", 1: "without_helmet"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect workers (whole body) and report PPE compliance per worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True, help="Image file or folder of images")
    parser.add_argument(
        "--person-weights",
        type=str,
        default=DEFAULT_PERSON_WEIGHTS,
        help="YOLO weights for person detection",
    )
    parser.add_argument(
        "--ppe-weights",
        type=str,
        default=None,
        help="YOLO weights for helmet/PPE detection (optional)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs_ppe_worker",
        help="Folder to save annotated images and report",
    )
    return parser.parse_args()


def box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match_ppe_to_person(person_box, ppe_detections):
    """Pick the PPE detection whose center falls inside the person's box and
    is closest to the top of the box (helmets sit near the head)."""
    x1, y1, x2, y2 = person_box
    best = None
    best_dist = None
    for det in ppe_detections:
        cx, cy = box_center(det["box"])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            dist = cy - y1
            if best_dist is None or dist < best_dist:
                best = det
                best_dist = dist
    return best


def extract_detections(result, class_filter=None):
    detections = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if class_filter is not None and cls_id not in class_filter:
            continue
        xyxy = tuple(box.xyxy[0].tolist())
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "conf": float(box.conf[0]),
                "box": xyxy,
            }
        )
    return detections


def process_frame(image, person_model, ppe_model, conf):
    """Run person + optional PPE detection on a single image (numpy array or path)
    and return each worker's whole-body box with PPE status attached.
    """
    person_result = person_model.predict(source=image, conf=conf, verbose=False)[0]
    person_detections = extract_detections(person_result, class_filter={PERSON_CLASS_ID})

    ppe_detections = []
    if ppe_model is not None:
        ppe_result = ppe_model.predict(source=image, conf=conf, verbose=False)[0]
        ppe_detections = extract_detections(ppe_result)

    workers = []
    for person in person_detections:
        ppe_match = match_ppe_to_person(person["box"], ppe_detections) if ppe_detections else None
        workers.append(
            {
                "box": person["box"],
                "conf": person["conf"],
                "ppe_status": ppe_match["class_name"] if ppe_match else "unknown",
            }
        )

    return workers


def summarize_workers(workers):
    summary = {"total": 0, "with_helmet": 0, "without_helmet": 0, "unknown": 0}
    for worker in workers:
        summary["total"] += 1
        status = worker["ppe_status"]
        if status in ("with_helmet", "without_helmet"):
            summary[status] += 1
        else:
            summary["unknown"] += 1
    return summary


def draw_overlay(image, workers):
    for worker in workers:
        x1, y1, x2, y2 = [int(v) for v in worker["box"]]
        color = (
            (0, 255, 0)
            if worker["ppe_status"] == "with_helmet"
            else (0, 0, 255)
            if worker["ppe_status"] == "without_helmet"
            else (0, 165, 255)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            worker["ppe_status"],
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


def collect_image_paths(source):
    source_path = Path(source)
    if source_path.is_dir():
        exts = {".png", ".jpg", ".jpeg"}
        return sorted(p for p in source_path.iterdir() if p.suffix.lower() in exts)
    return [source_path]


def main():
    args = parse_args()
    from ultralytics import YOLO

    person_model = YOLO(args.person_weights)
    ppe_model = YOLO(args.ppe_weights) if args.ppe_weights else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(args.source)
    report = {}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        workers = process_frame(image, person_model, ppe_model, args.conf)
        summary = summarize_workers(workers)
        report[image_path.name] = {"workers": workers, "summary": summary}

        annotated = draw_overlay(image.copy(), workers)
        cv2.imwrite(str(output_dir / image_path.name), annotated)

        print(
            f"[INFO] {image_path.name}: {summary['total']} worker(s), "
            f"{summary['with_helmet']} with helmet, {summary['without_helmet']} without helmet, "
            f"{summary['unknown']} unknown"
        )

    report_path = output_dir / "ppe_worker_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved to: {report_path}")
    print(f"[INFO] Annotated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
