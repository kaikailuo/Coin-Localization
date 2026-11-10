import json
from math import hypot
from pathlib import Path
from statistics import mean

import cv2
import numpy as np


def generate_examples(data_dir: Path):
    from skimage import data

    data_dir.mkdir(parents=True, exist_ok=True)
    truth = [
        {"id": 1, "x": 72, "y": 66, "r": 24},
        {"id": 2, "x": 192, "y": 68, "r": 32},
        {"id": 3, "x": 315, "y": 75, "r": 22},
        {"id": 4, "x": 112, "y": 215, "r": 29},
        {"id": 5, "x": 275, "y": 210, "r": 34},
    ]
    clean = np.full((303, 384), 44, dtype=np.uint8)
    for circle, intensity in zip(truth, (230, 190, 215, 205, 245)):
        cv2.circle(clean, (circle["x"], circle["y"]), circle["r"], intensity, -1)
    rng = np.random.default_rng(2026)
    noisy = np.rint(np.clip(clean + rng.normal(0, 20, clean.shape), 0, 255)).astype(np.uint8)

    examples = {"coins.png": data.coins(), "synthetic.png": clean, "synthetic_noisy.png": noisy}
    for name, image in examples.items():
        encoded, png = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError(f"无法编码 {name}")
        (data_dir / name).write_bytes(png.tobytes())
    truth_by_image = {"synthetic.png": truth, "synthetic_noisy.png": truth}
    (data_dir / "synthetic_truth.json").write_text(
        json.dumps(truth_by_image, indent=2) + "\n", encoding="utf-8"
    )
    return [data_dir / name for name in examples], truth_by_image


def evaluate_synthetic(detected, truth):
    tolerance = 5
    candidates = []
    for expected in truth:
        for found in detected:
            center_error = hypot(found["x"] - expected["x"], found["y"] - expected["y"])
            radius_error = abs(found["r"] - expected["r"])
            if center_error <= tolerance and radius_error <= tolerance:
                candidates.append(
                    (center_error + radius_error, expected["id"], found["id"], center_error, radius_error)
                )

    matched_truth, matched_detections = set(), set()
    matches = []
    for _, truth_id, detection_id, center_error, radius_error in sorted(candidates):
        if truth_id in matched_truth or detection_id in matched_detections:
            continue
        matched_truth.add(truth_id)
        matched_detections.add(detection_id)
        matches.append({
            "truth_id": truth_id,
            "detection_id": detection_id,
            "center_error_px": round(center_error, 3),
            "radius_error_px": round(radius_error, 3),
        })

    center_errors = [match["center_error_px"] for match in matches]
    radius_errors = [match["radius_error_px"] for match in matches]
    return {
        "center_tolerance_px": tolerance,
        "radius_tolerance_px": tolerance,
        "truth_count": len(truth),
        "matched_count": len(matches),
        "false_positive_ids": sorted({circle["id"] for circle in detected} - matched_detections),
        "missed_truth_ids": sorted({circle["id"] for circle in truth} - matched_truth),
        "mean_center_error_px": round(mean(center_errors), 3) if matches else None,
        "max_center_error_px": max(center_errors, default=None),
        "mean_radius_error_px": round(mean(radius_errors), 3) if matches else None,
        "max_radius_error_px": max(radius_errors, default=None),
        "matches": sorted(matches, key=lambda match: match["truth_id"]),
    }
