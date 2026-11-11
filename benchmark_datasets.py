import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter_ns

import cv2
import numpy as np

from coin_detection import detect_circles, input_images, load_config, read_image
from demo_data import evaluate_synthetic


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MAX_SIDE = 640
REPEATS = 3
CIRCLE_LABEL = re.compile(r"EUR_\d+ \((\d+),\s*(\d+)\) (\d+)")


def read_circle_labels(path):
    circles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("SUM "):
            continue
        match = CIRCLE_LABEL.fullmatch(line)
        if not match:
            raise ValueError(f"Unrecognized label in {path}: {line}")
        x, y, radius = map(int, match.groups())
        circles.append({"id": len(circles) + 1, "x": x, "y": y, "r": radius})
    if not circles:
        raise ValueError(f"No circle annotations in {path}")
    return circles


def real_benchmark_inputs():
    train = DATA / "real/eurocoin-vision/data/train"
    rectangles = {}
    for line in (train / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        rectangles[record["file_name"]] = len(record["objects"]["bbox"])
    paths = input_images(train)
    if {path.name for path in paths} != rectangles.keys():
        raise ValueError("Eurocoin Vision images do not match the rectangle annotations")
    for path in paths:
        yield "eurocoin-vision", "photos", path, {"rectangle_count": rectangles[path.name]}

    recognition = DATA / "real/eurocoin-recognition"
    for split, scene, pattern in (("dataset", "single_coin", "*/*.jpg"), ("test", "multi_coin", "*.jpg")):
        images = recognition / split / "images"
        paths = sorted(images.glob(pattern))
        if not paths:
            raise ValueError(f"No EuroCoinRecognition images in {images}")
        for path in paths:
            label_path = recognition / split / "labels" / path.relative_to(images).with_suffix(".txt")
            yield "eurocoin-recognition", scene, path, {"circles": read_circle_labels(label_path)}

    yield "skimage-coins", "official_sample", DATA / "coins.png", {}


def generalization_inputs():
    manifest = json.loads((DATA / "generalization.json").read_text(encoding="utf-8"))
    missing = [item["image"] for item in manifest if not (DATA / item["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing generalization images: {', '.join(missing)}. See data/README.md.")
    for item in manifest:
        path = DATA / item["image"]
        annotation = {"reference": item["reference"], "capture_group": item.get("capture_group")}
        if "circles" in item:
            annotation["circles"] = item["circles"]
        if "rectangle_count" in item:
            annotation["rectangle_count"] = item["rectangle_count"]
        yield item["source"], item["scene"], path, annotation


def benchmark_inputs(*, synthetic_only: bool = False):
    if not synthetic_only:
        yield from real_benchmark_inputs()

    demos = json.loads((DATA / "synthetic_truth.json").read_text(encoding="utf-8"))
    for name in ("synthetic.png", "synthetic_noisy.png"):
        yield "synthetic-demo", "demo", DATA / name, {"circles": demos[name]}


def scale_circle(circle, scale):
    # INTER_AREA with fx/fy scales source pixel centers about the image edge.
    return {
        **circle, "x": (circle["x"] + 0.5) * scale - 0.5,
        "y": (circle["y"] + 0.5) * scale - 0.5, "r": circle["r"] * scale,
    }


def measure_image(source, scene, path, annotation, config):
    start = perf_counter_ns()
    image = read_image(path)
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(width, height))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    read_resize_ms = (perf_counter_ns() - start) / 1e6

    timings = []
    for _ in range(REPEATS):
        start = perf_counter_ns()
        _, _, _, circles = detect_circles(image, config)
        timings.append((perf_counter_ns() - start) / 1e6)
    detections = [
        {"id": index, "x": round(x, 3), "y": round(y, 3), "r": round(radius, 3)}
        for index, (x, y, radius) in enumerate(circles, 1)
    ]
    truth = [scale_circle(circle, scale) for circle in annotation.get("circles", [])]
    evaluation = evaluate_synthetic(detections, truth) if "circles" in annotation else None
    result = {
        "source": source, "scene": scene, "image": path.relative_to(ROOT).as_posix(),
        "original_width": width, "original_height": height,
        "input_width": image.shape[1], "input_height": image.shape[0], "scale": scale,
        "read_resize_ms": round(read_resize_ms, 3), "detect_ms": round(median(timings), 3),
        "rectangle_count": annotation.get("rectangle_count"), "count": len(detections),
        "reference": annotation.get("reference"),
        "capture_group": annotation.get("capture_group"),
        "circles": detections, "evaluation": evaluation,
    }
    if evaluation is not None:
        for match in evaluation["matches"]:
            match["center_error_original_px"] = round(match["center_error_px"] / scale, 3)
            match["radius_error_original_px"] = round(match["radius_error_px"] / scale, 3)
    return result


def summarize(rows):
    measured = [row["evaluation"] for row in rows if row["evaluation"] is not None]
    matches = [match for item in measured for match in item["matches"]]
    return {
        "images": len(rows), "detected_circles": sum(row["count"] for row in rows),
        "evaluated_images": len(measured),
        "reference_types": sorted({row["reference"] for row in rows if row["reference"] is not None}),
        "median_detect_ms": round(median(row["detect_ms"] for row in rows), 3),
        "truth_circles": sum(item["truth_count"] for item in measured) if measured else None,
        "matched": len(matches) if measured else None,
        "missed": sum(len(item["missed_truth_ids"]) for item in measured) if measured else None,
        "false_positives": sum(len(item["false_positive_ids"]) for item in measured) if measured else None,
        "mean_center_error_px": round(mean(match["center_error_px"] for match in matches), 3) if matches else None,
        "mean_radius_error_px": round(mean(match["radius_error_px"] for match in matches), 3) if matches else None,
    }


def write_results(output, rows, config):
    output.mkdir(parents=True, exist_ok=True)
    sources = dict.fromkeys(row["source"] for row in rows)
    report = {
        "protocol": {"max_side_px": MAX_SIDE, "detect_repeats": REPEATS, "opencv_threads": 1,
                     "opencv_version": cv2.__version__, "timing_unit": "ms"},
        "config": config,
        "sources": {source: summarize([row for row in rows if row["source"] == source]) for source in sources},
        "scenes": {
            source: {
                scene: summarize([row for row in rows if row["source"] == source and row["scene"] == scene])
                for scene in dict.fromkeys(row["scene"] for row in rows if row["source"] == source)
            }
            for source in sources
        },
        "images": rows,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = ("truth_count", "matched_count", "mean_center_error_px", "mean_radius_error_px")
    fields = [key for key in rows[0] if key not in ("circles", "evaluation")]
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fields, *metrics, "missed", "false_positives"])
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields}
            if (item := row["evaluation"]) is not None:
                flat.update({key: item[key] for key in metrics}, missed=len(item["missed_truth_ids"]),
                            false_positives=len(item["false_positive_ids"]))
            writer.writerow(flat)
    print(f"Benchmarked {len(rows)} images in {len(sources)} datasets. Results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run the existing OpenCV baseline on the coin test datasets")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--synthetic-only", action="store_true", help="Run only the two synthetic demo images")
    group.add_argument("--generalization-only", action="store_true", help="Run the fixed generalization subset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dataset-benchmark")
    args = parser.parse_args()
    config = load_config(ROOT / "config.toml")
    cv2.setNumThreads(1)
    detect_circles(np.zeros((MAX_SIDE, 480, 3), dtype=np.uint8), config)

    inputs = (generalization_inputs() if args.generalization_only
              else benchmark_inputs(synthetic_only=args.synthetic_only))
    rows = [measure_image(source, scene, path, annotation, config)
            for source, scene, path, annotation in inputs]
    write_results(args.output, rows, config)


if __name__ == "__main__":
    main()
