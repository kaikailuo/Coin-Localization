import argparse
import json
import math
from pathlib import Path
import sys
import tomllib

import cv2
import matplotlib
import numpy as np

from demo_data import evaluate_synthetic, generate_examples

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
STAGE_NAMES = (
    ("01_original", "Original"),
    ("02_grayscale", "Grayscale"),
    ("03_gaussian", "Gaussian blur"),
    ("04_canny", "Canny edges"),
    ("05_hough", "Hough circles"),
    ("06_localization", "Final localization"),
)
CONFIG_FIELDS = {
    "gaussian": {"kernel_size": int, "sigma": float},
    "canny": {"threshold1": float, "threshold2": float},
    "hough": {
        "dp": float,
        "minDist": float,
        "param1": float,
        "param2": float,
        "minRadius": int,
        "maxRadius": int,
    },
}


def load_config(path):
    with path.open("rb") as file:
        config = tomllib.load(file)

    if config.keys() != CONFIG_FIELDS.keys():
        raise ValueError("配置文件须包含 gaussian、canny、hough 三组参数")
    for name, fields in CONFIG_FIELDS.items():
        group = config[name]
        if not isinstance(group, dict) or group.keys() != fields.keys():
            raise ValueError(f"[{name}] 参数须为 {', '.join(fields)}")
        for key, kind in fields.items():
            value = group[key]
            allowed = (int,) if kind is int else (int, float)
            if type(value) not in allowed or not math.isfinite(value):
                raise ValueError(f"[{name}].{key} 须为有限的{('整数' if kind is int else '数值')}")

    gaussian, canny, hough = (config[name] for name in CONFIG_FIELDS)
    if gaussian["kernel_size"] < 1 or gaussian["kernel_size"] % 2 == 0:
        raise ValueError("kernel_size 须为正奇数")
    if gaussian["sigma"] < 0:
        raise ValueError("sigma 须大于或等于 0")
    if not 0 <= canny["threshold1"] < canny["threshold2"]:
        raise ValueError("Canny 须满足 0 <= threshold1 < threshold2")
    if hough["dp"] < 1 or min(hough["minDist"], hough["param1"], hough["param2"]) <= 0:
        raise ValueError("Hough 须满足 dp >= 1，且 minDist、param1、param2 均大于 0")
    min_radius, max_radius = hough["minRadius"], hough["maxRadius"]
    if min_radius < 0 or (max_radius != 0 and max_radius <= min_radius):
        raise ValueError("minRadius 须大于或等于 0，maxRadius 须为 0 或大于 minRadius")
    return config


def read_image(path):
    buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if buffer.size:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is not None:
            return image
    raise ValueError("无法解码图片")


def detect_circles(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = config["gaussian"]
    blurred = cv2.GaussianBlur(
        gray, (gaussian["kernel_size"], gaussian["kernel_size"]), gaussian["sigma"]
    )
    edges = cv2.Canny(blurred, **config["canny"])
    # HOUGH_GRADIENT 在滤波后的灰度图上计算自己的梯度和边缘。
    candidates = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **config["hough"])
    circles = [] if candidates is None else candidates[0].tolist()
    circles.sort(key=lambda circle: (circle[1], circle[0]))
    return gray, blurred, edges, circles


def draw_circles(image, circles):
    hough_view = image.copy()
    final_view = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, radius in circles:
        center = (round(x), round(y))
        cv2.circle(hough_view, center, round(radius), (0, 210, 255), 1, cv2.LINE_AA)
        cv2.circle(final_view, center, round(radius), (40, 240, 80), 1, cv2.LINE_AA)
        cv2.circle(final_view, center, 2, (45, 80, 255), -1, cv2.LINE_AA)

    height, width = image.shape[:2]
    for number, (x, y, radius) in enumerate(circles, start=1):
        for label, offset in ((f"#{number}", -5), (f"r={radius:.1f}", 12)):
            (label_width, label_height), baseline = cv2.getTextSize(label, font, 0.33, 1)
            left = max(0, min(round(x - label_width / 2), width - label_width - 1))
            bottom = max(label_height, min(round(y) + offset, height - baseline - 1))
            origin = (left, bottom)
            cv2.putText(final_view, label, origin, font, 0.33, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(final_view, label, origin, font, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return hough_view, final_view


def save_stages(output_dir, stages, count):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.3), layout="constrained")
    try:
        for axis, (filename, title), stage in zip(axes.flat, STAGE_NAMES, stages):
            encoded, png = cv2.imencode(".png", stage)
            if not encoded:
                raise ValueError(f"无法编码 {filename}")
            (output_dir / f"{filename}.png").write_bytes(png.tobytes())
            if stage.ndim == 2:
                axis.imshow(stage, cmap="gray", vmin=0, vmax=255)
            else:
                axis.imshow(cv2.cvtColor(stage, cv2.COLOR_BGR2RGB))
            axis.set_title(title)
            axis.set_axis_off()
        figure.suptitle(f"Coin detection | {count} circles")
        figure.savefig(output_dir / "overview.png", dpi=160)
    finally:
        plt.close(figure)


def process_image(path, output_dir, config):
    image = read_image(path)
    gray, blurred, edges, circles = detect_circles(image, config)
    hough_view, final_view = draw_circles(image, circles)
    count = len(circles)
    save_stages(output_dir / path.name, (image, gray, blurred, edges, hough_view, final_view), count)
    return {
        "image": path.name,
        "width": image.shape[1],
        "height": image.shape[0],
        "count": count,
        "circles": [
            {"id": number, "x": round(x, 2), "y": round(y, 2), "r": round(radius, 2)}
            for number, (x, y, radius) in enumerate(circles, start=1)
        ],
    }


def input_images(path):
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        paths = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: (item.name.casefold(), item.name),
        )
        if paths:
            return paths
        raise ValueError(f"目录中没有 JPG、JPEG、PNG 或 WebP 图片：{path}")
    raise ValueError(f"请提供存在的 JPG、JPEG、PNG 或 WebP 图片，或包含图片的目录：{path}")


def main():
    parser = argparse.ArgumentParser(description="用 OpenCV 定位钱币并保存六个处理阶段")
    parser.add_argument("input", type=Path, nargs="?", help="图片或目录，省略时运行三个示例")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml", help="TOML 配置文件")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs", help="输出目录")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.input is None:
            paths, truth = generate_examples(PROJECT_DIR / "data")
        else:
            paths, truth = input_images(args.input), {}
        args.output.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"错误：{error}\n")

    results, failures = [], []
    for path in paths:
        try:
            result = process_image(path, args.output, config)
        except (OSError, ValueError, cv2.error) as error:
            print(f"{path.name}：处理失败（{error}）", file=sys.stderr)
            failures.append({"image": path.name, "error": str(error)})
            continue

        if path.name in truth:
            result["synthetic_evaluation"] = evaluate_synthetic(result["circles"], truth[path.name])
        results.append(result)
        if result["count"]:
            print(f"{path.name}：检测到 {result['count']} 个圆")
        else:
            print(f"{path.name}：未检测到圆")

    report = {"config": config, "images": results, "errors": failures}
    result_file = args.output / "detections.json"
    try:
        result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        parser.exit(1, f"无法保存检测结果：{error}\n")
    print(f"检测结果：{result_file.resolve()}")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
