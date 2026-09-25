"""
export_ncnn.py
---------------
Convert a trained YOLO .pt into an NCNN model folder, then (optionally) time
the .pt against the export on the SAME input the live pipeline feeds it.
No retraining: the export reuses your trained weights as-is.

Why NCNN on a Raspberry Pi 5
-----------------------------
NCNN runs noticeably faster than the plain .pt/.onnx checkpoints on the
Pi 5's ARM CPU (Ultralytics' own Pi 5 benchmarks agree). This script
converts a trained .pt checkpoint into that NCNN export.

SETUP (once)
    pip install ncnn "pnnx==20260526"
    (ultralytics pulls these automatically on first NCNN export too, but
    installing them ahead of time avoids a slow, network-dependent first run)

USAGE
    # vehicle model - export at the size the pipeline actually feeds it
    # (config.yaml model.vehicle_imgsz, e.g. 512x896 [height, width]):
    python export_ncnn.py --weights models/vehicle.pt --imgsz 512 896 --benchmark --image sample.jpg

    # plate model - config.yaml model.plate_imgsz (default 640, square):
    python export_ncnn.py --weights models/platenum_closeup.pt --imgsz 640

It prints the export folder (e.g. models/vehicle_ncnn_model). Point
config.yaml's model.vehicle_weights / model.plate_weights at that FOLDER.

FP16 (--half)
    Halves the model's on-disk/in-memory size and is usually a modest speed
    win on ARM. Accuracy loss is normally negligible for a detection model -
    still worth a quick before/after check with --benchmark on a few of your
    own images.

INT8 - NOT done by this script, on purpose
--------------------------------------------
Ultralytics' NCNN exporter (via PNNX) currently only wires up FP32/FP16 for
NCNN - there is no single-flag "give me a calibrated INT8 NCNN model" here,
unlike some of its other export formats. Real INT8 quantization for NCNN is a
separate, manual step using NCNN's OWN calibration tools (`ncnn2table` +
`ncnn2int8`, built from the NCNN C++ source - see
https://github.com/Tencent/ncnn/wiki/quantized-int8-inference), and it needs
50-200+ REPRESENTATIVE images of your actual camera's scenes (your vehicles,
your lighting, your ROI) to build a calibration table that doesn't wreck
accuracy. That's a meaningfully bigger effort than this export, and worth
doing only after you've confirmed FP32/FP16 NCNN isn't already fast enough -
say so if/when you want to go there and it can be scoped properly with your
own sample images rather than guessed at here.
"""
import argparse
import os
import statistics
import sys
import time

# Same thread budget the pipeline gives detection, so the timing comparison
# matches real conditions. Must be set before torch loads.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))

import numpy as np


def export_ncnn(weights: str, imgsz, half: bool = False) -> str:
    """Runs ultralytics' NCNN export and returns the exported folder path."""
    from ultralytics import YOLO

    result = YOLO(weights).export(format="ncnn", imgsz=imgsz, half=half)
    exported = str(result)
    if os.path.isfile(exported):
        exported = os.path.dirname(exported)
    if not os.path.isdir(exported):
        exported = os.path.splitext(weights)[0] + "_ncnn_model"
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
    parser.add_argument("--half", action="store_true", help="export FP16 instead of FP32 (see module docstring)")
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
    print(f"       set config.yaml's model.vehicle_weights / model.plate_weights to this folder "
          f"(imgsz {imgsz} - a static export like this one only accepts exactly this size).")

    if not args.benchmark:
        return

    if args.input_size:
        width, height = (int(part) for part in args.input_size.lower().split("x"))
    elif isinstance(imgsz, list):
        height, width = imgsz            # [h, w] as config.yaml expects
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
