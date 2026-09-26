"""
debug_plate.py
--------------
Answers ONE question: "why do my vehicles come out with no plate?"

It runs the plate model on a saved vehicle crop (the JPEGs the pipeline keeps
under output/vehicle_detection/) exactly the way the live loop does, and prints
either the detections or the exact error - so you can tell apart

    * the model CRASHES on the crop        (plate_detect_ms is NULL in the DB), from
    * the model runs but finds nothing     (plate_detect_ms would hold a number).

USAGE (from the project root)
    python debug_plate.py output/vehicle_detection/<some_crop>.jpg
    python debug_plate.py <crop>.jpg --conf 0.10       # also try a lower threshold
"""
import argparse
import sys
import time

import cv2

from config.config import PipelineConfig
from detection import detector, image_ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the plate model on one saved vehicle crop")
    parser.add_argument("image", help="a vehicle crop JPEG")
    parser.add_argument("--config", default=None)
    parser.add_argument("--conf", type=float, default=0.10, help="extra, lower threshold to try (default 0.10)")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"could not read {args.image}")
    print(f"crop: {image.shape[1]}x{image.shape[0]} px")

    print(f"loading {config.model.plate_weights} (plate_imgsz={config.model.plate_imgsz}) ...")
    model = detector.load_model(config.model.plate_weights, config.model.device)
    print("model class names:", getattr(model, "names", "?"))

    variants = [("raw crop", image), ("enhanced crop (what the pipeline feeds it)", image_ops.sharpen_and_denoise(image))]
    for label, picture in variants:
        for conf in (config.model.plate_conf_threshold, args.conf):
            try:
                started = time.perf_counter()
                result = detector.detect_batch(model, [picture], conf, config.model.vehicle_iou_threshold,
                                               config.model.plate_imgsz)[0]
                took = (time.perf_counter() - started) * 1000
                best = max((p["confidence"] for p in result), default=None)
                print(f"[{label}] conf>={conf:.2f}: {len(result)} plate(s), best={best}, {took:.0f} ms")
            except Exception as error:
                print(f"[{label}] conf>={conf:.2f}: MODEL CALL FAILED -> {type(error).__name__}: {error}")
                if error.__cause__:
                    print(f"      caused by: {type(error.__cause__).__name__}: {error.__cause__}")

    print("\nHow to read this:")
    print("  FAILED lines      -> the call itself is broken (paste the message to me).")
    print("  0 plates, no error-> the model runs but does not see a plate in this crop (see the lower-conf line).")
    print("  plates found here but not live -> the live crop differs (size, timing, enhancement).")


if __name__ == "__main__":
    main()
