import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))

import numpy as np


def export_ncnn(weights: str, imgsz, half: bool = False) -> str:
    from ultralytics import YOLO

    result = YOLO(weights).export(format="ncnn", imgsz=imgsz, half=half)
    exported = str(result)
    if os.path.isfile(exported):
        exported = os.path.dirname(exported)
    if not os.path.isdir(exported):
        exported = os.path.splitext(weights)[0] + "_ncnn_model"
    return exported


def load_frame(image_path, width: int, height: int) -> np.ndarray:
    import cv2

    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            sys.exit(f"Error: could not read image: {image_path}")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    print("[note] no --image given: timing uses random noise, so detection counts are meaningless.")
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)


def time_model(model_path: str, frame: np.ndarray, imgsz, runs: int, warmup: int):
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect") if os.path.isdir(model_path) else YOLO(model_path)
    result = None
    for _ in range(warmup):
        result = model.predict(frame, imgsz=imgsz, verbose=False)
    samples_ms = []
    for _ in range(runs):
        started = time.perf_counter()
        result = model.predict(frame, imgsz=imgsz, verbose=False)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    boxes = result[0].boxes
    count = 0 if boxes is None else len(boxes)
    top = float(boxes.conf.max()) if count else 0.0
    samples_ms.sort()
    p95 = samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.95))]
    return statistics.median(samples_ms), p95, count, top


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt to NCNN and benchmark it.")
    parser.add_argument("--weights", required=True, help="trained .pt file")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640],
                        help="one value for square (plate model), two for [height, width] "
                             "(vehicle model - match config.yaml model.vehicle_imgsz)")
    parser.add_argument("--half", action="store_true", help="export FP16 instead of FP32")
    parser.add_argument("--benchmark", action="store_true", help="time the .pt vs the export")
    parser.add_argument("--image", default=None, help="sample image for --benchmark (recommended)")
    parser.add_argument("--input-size", default=None,
                        help="WxH of the image handed to the model in --benchmark. Default: derived from --imgsz")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"Error: weights not found: {args.weights}")

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else list(args.imgsz[:2])

    exported = export_ncnn(args.weights, imgsz, half=args.half)
    print(f"\n[done] NCNN model folder: {exported}")
    print(f"       set the weights path in modes/vehicle/config.yaml (model.vehicle_weights / model.plate_weights) or modes/person/config.yaml (weights) to this folder "
          f"(imgsz {imgsz} - a static export like this one only accepts exactly this size).")

    if not args.benchmark:
        return

    if args.input_size:
        width, height = (int(part) for part in args.input_size.lower().split("x"))
    elif isinstance(imgsz, list):
        height, width = imgsz
    else:
        width, height = imgsz, max(1, round(imgsz * 9 / 16))
    frame = load_frame(args.image, width, height)
    print(f"\n[benchmark] input {width}x{height}, imgsz={imgsz}, {args.runs} runs after {args.warmup} warm-up")
    rows = []
    for label, path in (("PyTorch .pt", args.weights), ("NCNN", exported)):
        median, p95, count, top = time_model(path, frame, imgsz, args.runs, args.warmup)
        rows.append((label, median, p95, count, top))
        print(f"  {label:<12} median {median:7.1f} ms   p95 {p95:7.1f} ms   detections {count}   top conf {top:.3f}")
    speedup = rows[0][1] / rows[1][1] if rows[1][1] else float("nan")
    print(f"\n  speed-up: {speedup:.2f}x (median).  Detections/top-conf should match closely; "
          f"a large gap means something is wrong with the export.")
    print("\n  Run this same command on the Pi itself - ARM vs x86 CPU timing does not transfer.")


if __name__ == "__main__":
    main()
