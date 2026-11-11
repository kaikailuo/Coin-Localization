"""Explainable Canny + gradient-guided circle Hough, alongside OpenCV baseline.

Educational implementation: cv2.Sobel and basic morphology are allowed, but the
Canny NMS/hysteresis and Hough voting are implemented here rather than calling
cv2.Canny or cv2.HoughCircles in the custom path.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass
class CannyTrace:
    gray: np.ndarray
    smooth: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray
    nms: np.ndarray
    strong: np.ndarray
    weak: np.ndarray
    edges: np.ndarray


def explain_canny(image: np.ndarray, *, sigma=1.2, kernel_size=5, low=50, high=120) -> CannyTrace:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Expected a grayscale or BGR image")
    if gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if kernel_size < 1 or kernel_size % 2 != 1 or sigma < 0 or not 0 <= low < high:
        raise ValueError("Invalid Gaussian kernel or Canny thresholds")
    smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.rad2deg(np.arctan2(gy, gx)), 180.0)

    # Four direction bins: 0, 45, 90, 135 degrees; image y increases downwards.
    bins = (np.floor((angle + 22.5) / 45).astype(np.uint8)) % 4
    padded = np.pad(magnitude, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    neighbors = (
        (padded[1:-1, :-2], padded[1:-1, 2:]),
        (padded[:-2, :-2], padded[2:, 2:]),
        (padded[:-2, 1:-1], padded[2:, 1:-1]),
        (padded[:-2, 2:], padded[2:, :-2]),
    )
    nms = np.zeros_like(magnitude)
    for direction, (negative, positive) in enumerate(neighbors):
        # One strict inequality breaks equal-height plateaus into thin edges.
        keep = (bins == direction) & (center > negative) & (center >= positive)
        nms[keep] = magnitude[keep]
    nms[[0, -1], :] = 0
    nms[:, [0, -1]] = 0

    strong = nms >= high
    weak = (nms >= low) & ~strong
    # Hysteresis: retain any 8-connected candidate component containing a strong pixel.
    count, labels = cv2.connectedComponents((strong | weak).astype(np.uint8), connectivity=8)
    touches_strong = np.zeros(count, dtype=bool)
    touches_strong[np.unique(labels[strong])] = True
    touches_strong[0] = False
    edges = (touches_strong[labels].astype(np.uint8) * 255)
    return CannyTrace(gray, smooth, gx, gy, magnitude, nms, strong, weak, edges)


def hough_circle_trace(trace: CannyTrace, *, min_radius=12, max_radius=37,
                       min_dist=32, max_circles=30, min_relative_vote=0.25):
    """Gradient directed 3D Hough voting: one bin per (r, cy, cx).

    Returns candidate circles and their actual center/radius vote maps. This is
    a transparent teaching method, not a claim of OpenCV HoughCircles parity.
    """
    if not (1 <= min_radius <= max_radius and min_dist > 0 and max_circles > 0):
        raise ValueError("Invalid Hough search limits")
    h, w = trace.gray.shape
    ys, xs = np.nonzero(trace.edges)
    if not len(xs):
        return [], np.zeros((h, w), dtype=np.float32), np.zeros(max_radius-min_radius+1)
    strength = trace.magnitude[ys, xs]
    good = strength > 1e-6
    xs, ys, strength = xs[good], ys[good], strength[good]
    ux, uy = trace.gx[ys, xs] / strength, trace.gy[ys, xs] / strength
    radii = np.arange(min_radius, max_radius + 1)
    # Up to O(number of radii * image pixels) for an inspectable student demo.
    volume = np.zeros((len(radii), h, w), dtype=np.uint16)
    for i, radius in enumerate(radii):
        for sign in (-1, +1):
            cx = np.rint(xs + sign * radius * ux).astype(np.int32)
            cy = np.rint(ys + sign * radius * uy).astype(np.int32)
            valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            np.add.at(volume[i], (cy[valid], cx[valid]), 1)

    # Summing radius slices is a center-only projection of the 3D accumulator.
    center_votes = volume.astype(np.uint32).sum(axis=0).astype(np.float32)
    smooth_votes = cv2.GaussianBlur(center_votes, (5, 5), 1.0)
    max_votes = float(smooth_votes.max())
    if max_votes == 0:
        return [], center_votes, np.zeros(len(radii))
    local = cv2.dilate(smooth_votes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    threshold = max(3.0, max_votes * min_relative_vote)
    cy, cx = np.nonzero((smooth_votes >= local - 1e-6) & (smooth_votes >= threshold))
    order = np.argsort(smooth_votes[cy, cx])[::-1]
    circles = []
    primary_profile = np.zeros(len(radii), dtype=np.float64)
    for index in order:
        x, y = int(cx[index]), int(cy[index])
        if any((x - c["x"])**2 + (y - c["y"])**2 < min_dist**2 for c in circles):
            continue
        # Integrate a 5x5 neighborhood because centers vote in adjacent bins.
        near = volume[:, max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
        profile = near.sum(axis=(1, 2)).astype(np.float64)
        best = int(profile.argmax())
        circles.append({"x": x, "y": y, "r": int(radii[best]),
                        "center_vote": round(float(smooth_votes[y, x]), 2),
                        "radius_vote": int(profile[best])})
        if len(circles) == 1:
            primary_profile = profile
        if len(circles) >= max_circles:
            break
    return circles, center_votes, primary_profile


def save_figures(image: np.ndarray, trace: CannyTrace, circles: list,
                 votes: np.ndarray, profile: np.ndarray, output: Path, reference: list):
    output.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    panels = [(rgb, "01 Original", None), (trace.gray, "02 Grayscale", "gray"),
              (trace.smooth, "03 Gaussian", "gray"),
              (cv2.absdiff(trace.gray, trace.smooth), "Removed high-frequency detail", "inferno")]
    for ax, (arr, title, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=0 if cmap == "gray" else None,
                  vmax=255 if cmap == "gray" else None)
        ax.set_title(title); ax.axis("off")
    fig.savefig(output / "01_preprocessing.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    strength_map = np.zeros((*trace.gray.shape, 3), dtype=np.uint8)
    strength_map[trace.weak] = (255, 170, 0)
    strength_map[trace.strong] = (255, 0, 0)
    panels = [(trace.gx, "Sobel Gx", "coolwarm"), (trace.gy, "Sobel Gy", "coolwarm"),
              (trace.magnitude, "Gradient magnitude", "magma"),
              (trace.nms, "Non-maximum suppression", "magma"),
              (strength_map, "Double threshold: strong(red)/weak(orange)", None),
              (trace.edges, "8-connected hysteresis", "gray")]
    for ax, (arr, title, cmap) in zip(axes.flat, panels):
        ax.imshow(arr, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.savefig(output / "02_canny_internals.png", dpi=160); plt.close(fig)

    def overlay(detections, color):
        result = rgb.copy()
        for c in detections:
            cv2.circle(result, (round(c["x"]), round(c["y"])), round(c["r"]), color, 1)
            cv2.circle(result, (round(c["x"]), round(c["y"])), 2, color, -1)
        return result

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    plots = [(trace.edges, "Input edges to custom Hough", "gray"),
             (votes, "Center votes (sum over radii)", "hot"),
             (overlay(circles, (70, 255, 70)), "Custom Hough candidates", None),
             (overlay(reference, (255, 210, 50)), "OpenCV reference (separate path)", None)]
    for ax, (arr, title, cmap) in zip(axes.flat[:4], plots):
        ax.imshow(arr, cmap=cmap); ax.set_title(title); ax.axis("off")
    if circles:
        ax = axes.flat[4]
        ax.plot(np.arange(len(profile)) + int(circles[0]["r"]) - int(profile.argmax()), profile)
        ax.axvline(circles[0]["r"], linestyle="--", color="orange")
        ax.set_title("Radius votes near strongest center"); ax.set_xlabel("r (pixels)")
        ax.set_ylabel("votes (5x5 neighborhood)")
    else:
        axes.flat[4].text(0.1, 0.5, "No custom circle candidates")
    # A local scanline through the best center demonstrates boundary gradients.
    row = int(circles[0]["y"]) if circles else trace.gray.shape[0] // 2
    ax = axes.flat[5]
    if circles:
        center_x, radius = circles[0]["x"], circles[0]["r"]
        left = max(0, center_x - radius - 12)
        right = min(trace.gray.shape[1], center_x + radius + 13)
    else:
        left, right = 0, trace.gray.shape[1]
    xx = np.arange(left, right)
    ax.plot(xx, trace.gray[row, left:right], label="grayscale")
    ax.plot(xx, trace.smooth[row, left:right], label="Gaussian")
    ax.plot(xx, trace.magnitude[row, left:right], label="gradient magnitude", alpha=0.7)
    if circles:
        for boundary in (center_x - radius, center_x + radius):
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Local edge profile at y={row}"); ax.set_xlabel("x (pixels)")
    ax.legend(fontsize=7)
    fig.savefig(output / "03_hough_and_profile.png", dpi=160); plt.close(fig)


def main():
    from coin_detection import detect_circles, load_config, read_image
    parser = argparse.ArgumentParser(description="Explainable Canny + Hough circle experiment")
    parser.add_argument("image", type=Path, help="One PNG/JPEG/WebP image")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/explainable"))
    parser.add_argument("--max-circles", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config)
    image = read_image(args.image)
    g, c, h = config["gaussian"], config["canny"], config["hough"]
    trace = explain_canny(image, sigma=g["sigma"], kernel_size=g["kernel_size"],
                          low=c["threshold1"], high=c["threshold2"])
    circles, votes, profile = hough_circle_trace(
        trace, min_radius=max(1, h["minRadius"]),
        max_radius=h["maxRadius"] or min(image.shape[:2]) // 2,
        min_dist=h["minDist"], max_circles=args.max_circles)
    _, _, _, reference_raw = detect_circles(image, config)
    reference = [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in reference_raw]
    save_figures(image, trace, circles, votes, profile, args.output, reference)
    report = {
        "image": str(args.image), "config": config,
        "explanation": "Custom votes use custom hysteresis edges; OpenCV Hough uses its own internal Canny.",
        "canny": {"gradient_nonzero": int(np.count_nonzero(trace.magnitude)),
                  "nms_nonzero": int(np.count_nonzero(trace.nms)),
                  "strong": int(trace.strong.sum()), "weak": int(trace.weak.sum()),
                  "retained_edges": int(np.count_nonzero(trace.edges))},
        "hough": {"max_center_votes": float(votes.max()), "custom_circles": circles,
                  "opencv_reference_circles": reference},
    }
    (args.output / "trace.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Custom {len(circles)} circles, OpenCV {len(reference)} circles; output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
