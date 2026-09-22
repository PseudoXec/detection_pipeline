"""
export_openvino.py
-------------------
Convert a trained YOLO .pt into an OpenVINO model folder, then (optionally)
time the .pt against the export on the SAME input the live pipeline feeds it.
No retraining: the export reuses your trained weights as-is.

SETUP (once)
    pip install openvino

USAGE
    # vehicle model - export at the width the pipeline downscales the ROI crop to
    # (CONFIG["stream_detect_width"], default 480), then compare speed:
    python export_openvino.py --weights models\\vehicle.pt --imgsz 480 --benchmark --image sample.jpg

    # plate model - the pipeline runs it at CONFIG["imgsz"] (default 640):
    python export_openvino.py --weights models\\platenum_closeup.pt --imgsz 640

It prints the export folder (e.g. models\\vehicle_openvino_model). Point
CONFIG["vehicle_weights"] / CONFIG["plate_weights"] in main.py at that FOLDER.

WHY --imgsz MATTERS
    A default (static) export is compiled for one input size. The pipeline must
    then run the model at exactly that size: vehicle -> stream_detect_width,
    plate -> imgsz. A mismatch shows up immediately as a "model warm-up failed"
    warning at startup. If you expect to change sizes often, add --dynamic
    (accepts any size; check with --benchmark that it isn't slower for you).
"""
import argparse
import os
import statistics
import sys
import time

# Same thread budget main.py gives detection, so the timing comparison matches
# real conditions. Must be set before torch loads.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))

import numpy as np


def export_openvino(weights: str, imgsz: int, dynamic: bool = False, half: bool = False) -> str:
    """Runs ultralytics' OpenVINO export and returns the exported folder path."""
    from ultralytics import YOLO

    result = YOLO(weights).export(format="openvino", imgsz=imgsz, dynamic=dynamic, half=half)
    exported = str(result)
    if os.path.isfile(exported):
        exported = os.path.dirname(exported)
    if not os.path.isdir(exported):
        exported = os.path.splitext(weights)[0] + "_openvino_model"
    return exported


def load_frame(image_path, width: int, height: int) -> np.ndarray:
    """A real image resized to the pipeline's input size, or random noise if none given."""
    import cv2

    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            sys.exit(f"Error: could not read image: {image_path}")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    print("[note] no --image given: timing uses random noise, so detection counts are meaningless.")
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)


def time_model(model_path: str, frame: np.ndarray, imgsz: int, runs: int, warmup: int):
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
    parser = argparse.ArgumentParser(description="Export a YOLO .pt to OpenVINO and benchmark it.")
    parser.add_argument("--weights", required=True, help="trained .pt file")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="size to export for (vehicle: CONFIG['stream_detect_width'], plate: CONFIG['imgsz'])")
    parser.add_argument("--dynamic", action="store_true", help="export with dynamic input shapes")
    parser.add_argument("--benchmark", action="store_true", help="time the .pt vs the export")
    parser.add_argument("--image", default=None, help="sample image for --benchmark (recommended)")
    parser.add_argument("--input-size", default=None,
                        help="WxH of the image handed to the model in --benchmark. Default: --imgsz wide, 16:9")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"Error: weights not found: {args.weights}")

    exported = export_openvino(args.weights, args.imgsz, dynamic=args.dynamic)
    print(f"\n[done] OpenVINO model folder: {exported}")
    print(f"       set CONFIG weights to this folder (imgsz {args.imgsz}"
          f"{', dynamic' if args.dynamic else ', static - run the pipeline at exactly this size'}).")

    if not args.benchmark:
        return

    if args.input_size:
        width, height = (int(part) for part in args.input_size.lower().split("x"))
    else:
        width, height = args.imgsz, max(1, round(args.imgsz * 9 / 16))
    frame = load_frame(args.image, width, height)
    print(f"\n[benchmark] input {width}x{height}, imgsz={args.imgsz}, {args.runs} runs after {args.warmup} warm-up")
    rows = []
    for label, path in (("PyTorch .pt", args.weights), ("OpenVINO", exported)):
        median, p95, count, top = time_model(path, frame, args.imgsz, args.runs, args.warmup)
        rows.append((label, median, p95, count, top))
        print(f"  {label:<12} median {median:7.1f} ms   p95 {p95:7.1f} ms   detections {count}   top conf {top:.3f}")
    speedup = rows[0][1] / rows[1][1] if rows[1][1] else float("nan")
    print(f"\n  speed-up: {speedup:.2f}x (median).  Detections/top-conf should match closely; "
          f"a large gap means something is wrong with the export.")


if __name__ == "__main__":
    main()
