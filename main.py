"""
main.py
-------
Full 3-stage pipeline:

    1. VEHICLE DETECTION  -> your vehicle .pt model finds every vehicle in
                              the photo (car/truck/bus/motorcycle/...),
                              image-processes the frame first (CLAHE +
                              denoise + resize) for better detection, then
                              crops each vehicle with an enlarged bounding
                              box (padding) so nothing gets clipped.
    2. PLATE DETECTION    -> local_model_infer's plate model runs on EACH
                              vehicle crop and crops the plate out of it.
    3. OCR                -> ocr_cropped_plates reads each plate crop and
                              saves a renamed copy with the recognized text.

FOLDER LAYOUT (per run) - one flat folder per stage, per day
--------------------------------------------------------------
<out_dir>/<dd-mm-yy>/
    vehicle_detection/     <- every cropped vehicle from every image today
        20260916_143012_125_car.png
        20260916_143012_125_truck.png
    plate_detection/       <- every cropped plate from every vehicle today
        20260916_143012_125_car_crop.png
    ocr/                   <- every OCR'd plate, renamed with its text
        20260916_143012_125_car_ABC1234.png
    pipeline_log.csv       <- one row per plate result, ties it all together

SETUP
-----
    pip install ultralytics paddleocr paddlepaddle opencv-python numpy

Requires local_model_infer.py and ocr_cropped_plates.py in the same folder
(reused for detection, cropping and OCR - nothing is duplicated).
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import cv2

import local_model_infer as detect
import ocr_cropped_plates as ocr

CONFIG = {
    # --- Source: set exactly ONE of these, leave the other as None ---
    "image": None,                                  # e.g. r"C:\cars\photo.jpg"
    "folder": r"C:\Users\User\Documents\detection_pipeline\data",

    # --- Models ---
    "vehicle_weights": r"C:\Users\User\Documents\detection_pipeline\models\vehicle_yolov11n.94mAP\weights\vehicle.pt",   # your trained vehicle .pt
    "plate_weights": r"C:\Users\User\Documents\detection_pipeline\models\plate_yolov11n.95mAP\weights\platenum.pt",       # your trained plate .pt
    "device": None,                                       # None = auto (GPU if available)

    # --- Vehicle detection stage ---
    "vehicle_classes": None,            # e.g. "car,truck,bus,motorcycle"; None = keep all classes
    "vehicle_conf_threshold": 0.35,
    "vehicle_iou_threshold": 0.45,
    "vehicle_crop_padding": 0.25,       # extra margin around each vehicle box (25%) - preserve the full vehicle
    "vehicle_crop_min_height": 200,     # zoom small vehicle crops up to at least this height (px)

    # --- Plate detection stage (runs on each vehicle crop) ---
    "plate_conf_threshold": 0.40,       # small plates often score below 0.35 inside a vehicle crop
    "plate_iou_threshold": detect.DEFAULT_IOU_THRESHOLD,
    "plate_crop_padding": 0.15,
    "plate_crop_min_height": 120,

    # --- Shared image-processing (applied before BOTH detection passes) ---
    "no_preprocess": False,             # True = skip CLAHE/denoise/resize before detection
    "imgsz": None,                      # None = model's own default
    "plate_imgsz": 1280,                # give the small plate model more pixels to work with
    "max_dim": detect.DEFAULT_MAX_DIM,
    "min_dim": detect.DEFAULT_MIN_DIM,

    # --- OCR stage ---
    "model_tier": "server",             # "server" (accurate, default) | "mobile-en" (fast) | "default"
    "single_pass": False,               # True = skip the raw/light/full best-of comparison
    "min_confidence": 0.80,             # below this, result is flagged "check"/LOWCONF

    # --- Output ---
    "out_dir": None,                    # base folder; a dd-mm-yy folder is created inside it
}
# ==========================================================================


def make_date_dir(base_out_dir: str) -> str:
    """Creates (if needed) and returns <base_out_dir>/<dd-mm-yy>."""
    date_str = datetime.now().strftime("%d-%m-%y")
    path = os.path.join(base_out_dir, date_str)
    os.makedirs(path, exist_ok=True)
    return path


def default_base_out_dir(image: Optional[str], folder: Optional[str]) -> str:
    """Same convention as local_model_infer.default_out_dir: a sibling
    'vehicle_pipeline_output' folder next to the source."""
    if folder:
        sample_folder = os.path.abspath(folder.rstrip("/\\"))
    else:
        sample_folder = os.path.abspath(os.path.dirname(image) or ".")
    return os.path.join(sample_folder, "vehicle_pipeline_output")


def source_timestamp(image_path: str) -> str:
    """Returns a filename-safe timestamp for a source image.

    File inputs use their modification time. An RTSP capture should pass its
    frame capture time instead when the live-feed adapter is added.
    """
    try:
        captured_at = datetime.fromtimestamp(os.path.getmtime(image_path))
    except OSError:
        captured_at = datetime.now()
    return captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def unique_stem(directory: str, stem: str, extension: str = ".png") -> str:
    """Returns a non-colliding filename stem inside directory."""
    candidate = stem
    counter = 2
    while os.path.exists(os.path.join(directory, f"{candidate}{extension}")):
        candidate = f"{stem}_{counter}"
        counter += 1
    return candidate


def crop_with_padding(
    image, pred: Dict[str, Any], padding_ratio: float, min_crop_height: int,
):
    """Crops a single detection out of `image` with an enlarged (padded)
    bounding box, upscaling if the result is too small. Returns a BGR
    numpy array, or None if the box is degenerate."""
    h, w = image.shape[:2]
    cx, cy, bw, bh = pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")
    if None in (cx, cy, bw, bh):
        return None

    pad_w, pad_h = bw * padding_ratio, bh * padding_ratio
    x1 = max(int(cx - bw / 2 - pad_w), 0)
    y1 = max(int(cy - bh / 2 - pad_h), 0)
    x2 = min(int(cx + bw / 2 + pad_w), w)
    y2 = min(int(cy + bh / 2 + pad_h), h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if min_crop_height and ch < min_crop_height:
        scale = min_crop_height / ch
        crop = cv2.resize(crop, (int(cw * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)
    return crop


def rescale_predictions(
    predictions: List[Dict[str, Any]], source_image, target_image,
) -> List[Dict[str, Any]]:
    """Maps pixel-coordinate detections from source_image to target_image."""
    source_height, source_width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    mapped = []
    for prediction in predictions:
        mapped_prediction = dict(prediction)
        mapped_prediction["x"] = prediction["x"] * scale_x
        mapped_prediction["y"] = prediction["y"] * scale_y
        mapped_prediction["width"] = prediction["width"] * scale_x
        mapped_prediction["height"] = prediction["height"] * scale_y
        mapped.append(mapped_prediction)
    return mapped


def run_ocr_on_plate_crops(
    ocr_engine, plate_paths: List[str], ocr_dir: str, args,
) -> List[Dict[str, Any]]:
    """OCRs every plate crop and records the result without saving a copy.

    The existing OCR filename is still generated and stored in the CSV so
    each result keeps the same naming convention and traceability.
    """
    results = []

    for plate_path in plate_paths:
        plate_image = cv2.imread(plate_path)
        if plate_image is None:
            print(f"    [warn] unreadable plate crop: {plate_path}")
            continue

        text, conf = ocr.read_plate_text(ocr_engine, plate_image)
        winning_variant = "raw"

        low_confidence = conf < args.min_confidence
        status = "check" if (low_confidence or not text) else "read"

        plate_stem = os.path.splitext(os.path.basename(plate_path))[0]
        sanitized_text = ocr.sanitize_for_filename(text)
        vehicle_stem = plate_stem.removesuffix("_crop")
        base_name = f"{vehicle_stem}_{sanitized_text}"
        dest_path = ocr.unique_destination(ocr_dir, base_name, ".png")

        detail = text if text else "no text found"
        print(f"    plate -> {status} ({detail}, conf={conf:.2f})")

        results.append({
            "plate_crop": plate_path,
            "ocr_file": dest_path,
            "text": text,
            "confidence": conf,
            "variant": winning_variant,
            "status": status,
        })

    return results


def process_single_image(
    image_path: str,
    args,
    vehicle_model,
    plate_model,
    ocr_engine,
    date_dir: str,
    vehicle_dir: str,
    plate_dir: str,
    ocr_dir: str,
    work_dir: str,
    agg_rows: List[Dict[str, Any]],
) -> int:
    """Runs the full 3-stage pipeline on one source photo. Returns the
    number of vehicles that were cropped."""
    filename = os.path.basename(image_path)
    # Stage 1a: detect vehicles directly on the original photo.
    detection_input_path = image_path
    vehicle_classes = (
        set(c.strip() for c in args.vehicle_classes.split(",") if c.strip())
        if args.vehicle_classes else None
    )

    try:
        vehicle_preds = detect.run_local_detection(
            vehicle_model, detection_input_path, args.vehicle_conf_threshold,
            args.vehicle_iou_threshold, args.imgsz, vehicle_classes,
        )
    except (FileNotFoundError, detect.LocalInferenceError) as e:
        print(f"{filename} -> vehicle detection error: {e}")
        return 0

    if not vehicle_preds:
        print(f"{filename} -> no vehicles detected")
        return 0

    full_img = cv2.imread(detection_input_path)
    if full_img is None:
        print(f"{filename} -> could not reload image for cropping")
        return 0

    print(f"{filename} -> {len(vehicle_preds)} vehicle(s) detected")
    timestamp = source_timestamp(image_path)
    vehicle_count = 0

    for i, pred in enumerate(vehicle_preds, start=1):
        cls = str(pred.get("class") or "vehicle")
        safe_cls = "".join(c if c.isalnum() else "_" for c in cls)
        v_conf = pred.get("confidence", 0.0)
        # Stage 1b: crop the vehicle with an enlarged bounding box.
        vehicle_crop = crop_with_padding(
            full_img, pred, args.vehicle_crop_padding, args.vehicle_crop_min_height,
        )
        if vehicle_crop is None:
            continue

        # All vehicle crops for the whole day land flat in vehicle_dir - the
        # filename itself (source image + vehicle index + class + conf) is
        # what keeps each one traceable, not a per-vehicle folder.
        vehicle_name = unique_stem(vehicle_dir, f"{timestamp}_{safe_cls}")
        vehicle_crop_path = os.path.join(vehicle_dir, f"{vehicle_name}.png")
        cv2.imwrite(vehicle_crop_path, vehicle_crop)
        vehicle_count += 1

        base_row = {
            "source_image": filename,
            "vehicle_id": vehicle_name,
            "vehicle_class": cls,
            "vehicle_confidence": f"{v_conf:.4f}",
            "vehicle_crop": os.path.relpath(vehicle_crop_path, date_dir),
            "plate_crop": "",
            "plate_text": "",
            "plate_confidence": "",
            "ocr_status": "",
            "ocr_file": "",
        }

        # Stage 2: detect the plate directly on the original vehicle crop.
        plate_input_path = vehicle_crop_path
        try:
            plate_preds = detect.run_local_detection(
                plate_model, plate_input_path, args.plate_conf_threshold,
                args.plate_iou_threshold, args.plate_imgsz, None,
            )
        except (FileNotFoundError, detect.LocalInferenceError) as e:
            print(f"  {vehicle_name} -> plate detection error: {e}")
            base_row["ocr_status"] = "plate_detection_error"
            agg_rows.append(base_row)
            continue

        if not plate_preds:
            print(f"  {vehicle_name} -> no plate detected")
            base_row["ocr_status"] = "no_plate"
            agg_rows.append(base_row)
            continue

        # Detection may use a resized/denoised image, but save the plate from
        # the original vehicle crop so preprocessing does not soften it.
        detection_img = cv2.imread(plate_input_path)
        vehicle_img = cv2.imread(vehicle_crop_path)
        if detection_img is None or vehicle_img is None:
            print(f"  {vehicle_name} -> could not reload plate crop source")
            base_row["ocr_status"] = "plate_crop_source_error"
            agg_rows.append(base_row)
            continue
        plate_preds = rescale_predictions(plate_preds, detection_img, vehicle_img)

        # Plate crops for the whole day also land flat in plate_dir, prefixed
        # with vehicle_name so each one still traces back to its vehicle/image.
        plate_paths = detect.crop_and_save_detections(
            vehicle_crop_path, plate_preds, plate_dir, vehicle_name,
            args.plate_crop_padding, args.plate_crop_min_height,
            output_extension=".png", simple_name=True,
        )
        if not plate_paths:
            print(f"  {vehicle_name} -> plate found but not croppable")
            base_row["ocr_status"] = "uncroppable_plate"
            agg_rows.append(base_row)
            continue

        print(f"  {vehicle_name} -> {len(plate_paths)} plate(s) cropped, running OCR...")

        # Stage 3: OCR each plate crop.
        ocr_results = run_ocr_on_plate_crops(ocr_engine, plate_paths, ocr_dir, args)
        if not ocr_results:
            base_row["ocr_status"] = "unreadable"
            agg_rows.append(base_row)
            continue

        for r in ocr_results:
            row = dict(base_row)
            row["plate_crop"] = os.path.relpath(r["plate_crop"], date_dir)
            row["plate_text"] = r["text"]
            row["plate_confidence"] = f"{r['confidence']:.4f}"
            row["ocr_status"] = r["status"]
            row["ocr_file"] = os.path.relpath(r["ocr_file"], date_dir)
            agg_rows.append(row)

    return vehicle_count


def write_aggregate_log(date_dir: str, agg_rows: List[Dict[str, Any]]) -> str:
    output_path = os.path.join(date_dir, "pipeline_log.csv")
    fieldnames = [
        "source_image", "vehicle_id", "vehicle_class", "ocr_status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: vehicle detection+crop -> plate detection+crop -> OCR. "
                     "Defaults come from the CONFIG dict at the top of this file."
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", default=CONFIG["image"])
    source_group.add_argument("--folder", default=CONFIG["folder"])

    parser.add_argument("--vehicle-weights", default=CONFIG["vehicle_weights"])
    parser.add_argument("--plate-weights", default=CONFIG["plate_weights"])
    parser.add_argument("--device", default=CONFIG["device"])

    parser.add_argument("--vehicle-classes", default=CONFIG["vehicle_classes"],
                         help="Comma-separated vehicle class names to keep, e.g. 'car,truck,bus'. "
                              "Default: keep every class the vehicle model detects.")
    parser.add_argument("--vehicle-conf-threshold", type=float, default=CONFIG["vehicle_conf_threshold"])
    parser.add_argument("--vehicle-iou-threshold", type=float, default=CONFIG["vehicle_iou_threshold"])
    parser.add_argument("--vehicle-crop-padding", type=float, default=CONFIG["vehicle_crop_padding"],
                         help="Extra margin added around each vehicle box before cropping "
                              f"(default {CONFIG['vehicle_crop_padding']} = "
                              f"{CONFIG['vehicle_crop_padding']*100:.0f}%%).")
    parser.add_argument("--vehicle-crop-min-height", type=int, default=CONFIG["vehicle_crop_min_height"])

    parser.add_argument("--plate-conf-threshold", type=float, default=CONFIG["plate_conf_threshold"])
    parser.add_argument("--plate-iou-threshold", type=float, default=CONFIG["plate_iou_threshold"])
    parser.add_argument("--plate-crop-padding", type=float, default=CONFIG["plate_crop_padding"])
    parser.add_argument("--plate-crop-min-height", type=int, default=CONFIG["plate_crop_min_height"])

    parser.add_argument("--no-preprocess", action="store_true", default=CONFIG["no_preprocess"],
                         help="Skip CLAHE/denoise/resize before both detection passes.")
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    parser.add_argument("--plate-imgsz", type=int, default=CONFIG["plate_imgsz"])
    parser.add_argument("--max-dim", type=int, default=CONFIG["max_dim"])
    parser.add_argument("--min-dim", type=int, default=CONFIG["min_dim"])

    parser.add_argument("--model-tier", choices=sorted(ocr.MODEL_TIERS), default=CONFIG["model_tier"])
    parser.add_argument("--single-pass", action="store_true", default=CONFIG["single_pass"])
    parser.add_argument("--min-confidence", type=float, default=CONFIG["min_confidence"])

    parser.add_argument("--out-dir", default=CONFIG["out_dir"],
                         help="Base output folder. A dd-mm-yy folder is created inside it for "
                              "each run's results. Defaults to a 'vehicle_pipeline_output' "
                              "folder next to the source.")

    args = parser.parse_args()

    if args.image and args.folder:
        sys.exit("Error: set only one of --image / --folder (or CONFIG['image'] / CONFIG['folder']).")
    if not args.image and not args.folder:
        sys.exit("Error: no source set. Edit CONFIG['folder'] or CONFIG['image'] at the top of this "
                  "file, or pass --image / --folder on the command line.")
    if args.folder and not os.path.isdir(args.folder):
        sys.exit(f"Error: folder not found: {args.folder}")
    if args.image and not os.path.isfile(args.image):
        sys.exit(f"Error: image not found: {args.image}")
    if not os.path.isfile(args.vehicle_weights):
        sys.exit(f"Error: vehicle weights not found: {args.vehicle_weights}")
    if not os.path.isfile(args.plate_weights):
        sys.exit(f"Error: plate weights not found: {args.plate_weights}")

    base_out_dir = args.out_dir or default_base_out_dir(args.image, args.folder)
    date_dir = make_date_dir(base_out_dir)
    vehicle_dir = os.path.join(date_dir, "vehicle_detection")
    plate_dir = os.path.join(date_dir, "plate_detection")
    ocr_dir = os.path.join(date_dir, "ocr")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(plate_dir, exist_ok=True)
    print(f"[info] Output folder for this run: {date_dir}")

    print("[info] Loading vehicle detection model...")
    vehicle_model = detect.build_model(args.vehicle_weights, args.device)
    print("[info] Loading plate detection model...")
    plate_model = detect.build_model(args.plate_weights, args.device)
    print(f"[info] Loading PaddleOCR engine (model tier: {args.model_tier})...")
    ocr_engine = ocr.build_ocr_engine(args.model_tier)

    if args.folder:
        image_paths = detect.list_images_in_folder(args.folder)
        if not image_paths:
            sys.exit(f"Error: no images ({sorted(detect.IMAGE_EXTENSIONS)}) found in: {args.folder}")
    else:
        image_paths = [args.image]

    work_dir = tempfile.mkdtemp(prefix="vehicle_plate_preprocess_")
    agg_rows: List[Dict[str, Any]] = []
    total_vehicles = 0
    try:
        for image_path in image_paths:
            total_vehicles += process_single_image(
                image_path, args, vehicle_model, plate_model, ocr_engine,
                date_dir, vehicle_dir, plate_dir, ocr_dir, work_dir, agg_rows,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    log_path = write_aggregate_log(date_dir, agg_rows)
    plates_read = sum(1 for r in agg_rows if r["ocr_status"] == "read")

    print(f"\n=== Pipeline complete ===")
    print(f"Images processed : {len(image_paths)}")
    print(f"Vehicles cropped : {total_vehicles}")
    print(f"Plates read      : {plates_read}/{len(agg_rows)}")
    print(f"Log              : {log_path}")


if __name__ == "__main__":
    main()
