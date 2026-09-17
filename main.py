"""
main.py
-------
Full raw-frame detection pipeline:

    1. VEHICLE + PLATE DETECTION -> both logical passes run on the same raw
                                    frame, independently of any crop.
    2. PLATE ASSOCIATION         -> plate boxes are matched to vehicle boxes.
    3. OCR + REVIEW OUTPUT       -> plates are cropped from the raw frame for
                                    OCR; vehicle crops are thumbnails only.

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
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import local_model_infer as detect
import ocr_cropped_plates as ocr

CONFIG = {
    # --- Source: set exactly ONE of these, leave the other as None ---
    "image": None,                                  # e.g. r"C:\cars\photo.jpg"
    "folder": r"C:\Users\User\Documents\detection_pipeline\data",

    # --- Models ---
    "vehicle_weights": r"C:\Users\User\Documents\detection_pipeline\models\vehicle.pt",   # your trained vehicle .pt
    "plate_weights": r"C:\Users\User\Documents\detection_pipeline\models\platenum.pt",   # your trained license-plate .pt
    "device": None,                                       # None = auto (GPU if available)

    # --- Vehicle detection stage ---
    "vehicle_classes": None,            # e.g. "car,truck,bus,motorcycle"; None = keep all classes
    "vehicle_conf_threshold": 0.35,
    "vehicle_iou_threshold": 0.45,
    "vehicle_crop_padding": 0.30,       # extra margin around each vehicle box (30%) - thumbnail only
    "vehicle_crop_min_height": 200,     # zoom small vehicle crops up to at least this height (px)

    # --- Plate association stage (uses plate detections from vehicle.pt) ---
    "plate_conf_threshold": 0.40,
    "plate_crop_padding": 8,             # pixels added on each side of the raw-frame plate box
    "plate_crop_min_height": 120,
    "plate_containment_threshold": 0.8,
    "plate_min_iou": 0.1,

    # --- Shared detector settings (both logical passes use the raw frame) ---
    "no_preprocess": False,             # True = skip CLAHE/denoise/resize before detection
    "imgsz": None,                      # None = model's own default
    "max_dim": detect.DEFAULT_MAX_DIM,
    "min_dim": detect.DEFAULT_MIN_DIM,

    # --- OCR stage ---
    "model_tier": "server",             # "server" (accurate, default) | "mobile-en" (fast) | "default"
    "single_pass": False,               # True = skip the raw/light/full best-of comparison
    "min_confidence": 0.80,             # below this, result is flagged "check"/LOWCONF

    # --- QA / debug output ---
    "SAVE_QA_IMAGES": True,             # Save annotated plate crops for manual review

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


def crop_vehicle(
    raw_frame: np.ndarray,
    vehicle_box: Dict[str, Any],
    pad_ratio: float = 0.3,
    min_crop_height: int = 0,
) -> Optional[np.ndarray]:
    """Crop a vehicle from the raw frame for thumbnail/review output only."""
    image = raw_frame
    h, w = image.shape[:2]
    cx, cy, bw, bh = (
        vehicle_box.get("x"), vehicle_box.get("y"),
        vehicle_box.get("width"), vehicle_box.get("height"),
    )
    if None in (cx, cy, bw, bh):
        return None

    pad_w, pad_h = bw * pad_ratio, bh * pad_ratio
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


def is_plate_prediction(prediction: Dict[str, Any]) -> bool:
    """Returns whether a vehicle.pt prediction represents a plate."""
    label = str(prediction.get("class") or "").lower()
    normalized = "".join(character for character in label if character.isalnum())
    if normalized in {"lp", "licenseplate", "licenceplate", "numberplate", "platenumber"}:
        return True
    return any(marker in normalized for marker in ("plate", "license", "licence", "registration"))


def _box_edges(box: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return a center-based detection box as left, top, right, bottom."""
    return (
        box["x"] - box["width"] / 2,
        box["y"] - box["height"] / 2,
        box["x"] + box["width"] / 2,
        box["y"] + box["height"] / 2,
    )


def get_centroid(box: Dict[str, Any]) -> Tuple[float, float]:
    """Return the center point of a center-based detection box."""
    return float(box["x"]), float(box["y"])


def compute_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Return intersection-over-union for two center-based detection boxes."""
    ax1, ay1, ax2, ay2 = _box_edges(box_a)
    bx1, by1, bx2, by2 = _box_edges(box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def compute_containment_ratio(inner_box: Dict[str, Any], outer_box: Dict[str, Any]) -> float:
    """Return the fraction of `inner_box` area covered by `outer_box`."""
    ix1, iy1, ix2, iy2 = _box_edges(inner_box)
    ox1, oy1, ox2, oy2 = _box_edges(outer_box)
    intersection = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / inner_area if inner_area else 0.0


def match_plates_to_vehicles(
    vehicle_detections: List[Dict[str, Any]],
    plate_detections: List[Dict[str, Any]],
    containment_thresh: float = 0.8,
    min_iou: float = 0.1,
) -> List[Dict[str, Any]]:
    """Attach every plate to its best vehicle, retaining orphan plates.

    Containment is the primary signal. IoU is used only when no vehicle
    contains the plate sufficiently, and all qualifying matches are kept.
    """
    matches: List[Dict[str, Any]] = []
    vehicle_plate_counts: Dict[int, List[int]] = {}

    for plate_index, plate in enumerate(plate_detections, start=1):
        candidates = []
        fallback_candidates = []
        for vehicle_index, vehicle in enumerate(vehicle_detections, start=1):
            containment = compute_containment_ratio(plate, vehicle)
            iou = compute_iou(plate, vehicle)
            candidate = (containment, iou, vehicle_index)
            if containment >= containment_thresh:
                candidates.append(candidate)
            elif iou >= min_iou:
                fallback_candidates.append(candidate)

        if candidates:
            containment, iou, vehicle_index = max(candidates, key=lambda item: (item[0], item[1]))
            match_method = "containment"
        elif fallback_candidates:
            containment, iou, vehicle_index = max(fallback_candidates, key=lambda item: (item[1], item[0]))
            match_method = "iou"
        else:
            print(f"[warn] plate {plate_index} has no vehicle match; retaining as orphan")
            matches.append({
                "plate_index": plate_index,
                "plate_box": plate,
                "vehicle_index": None,
                "vehicle_box": None,
                "track_id": None,
                "containment_ratio": 0.0,
                "iou": 0.0,
                "match_method": "orphan",
            })
            continue

        vehicle = vehicle_detections[vehicle_index - 1]
        vehicle_plate_counts.setdefault(vehicle_index, []).append(plate_index)
        matches.append({
            "plate_index": plate_index,
            "plate_box": plate,
            "vehicle_index": vehicle_index,
            "vehicle_box": vehicle,
            "track_id": vehicle.get("track_id"),
            "containment_ratio": containment,
            "iou": iou,
            "match_method": match_method,
        })

    for vehicle_index, plate_indices in vehicle_plate_counts.items():
        if len(plate_indices) > 1:
            track_id = vehicle_detections[vehicle_index - 1].get("track_id")
            print(f"[warn] track_id={track_id} matched to multiple plates: {plate_indices}")
    return matches


def crop_plate(
    raw_frame: np.ndarray,
    plate_box: Dict[str, Any],
    pad_pixels: int = 8,
    min_crop_height: int = 0,
) -> Optional[np.ndarray]:
    """Crop a plate directly from the raw frame with a border for OCR."""
    height, width = raw_frame.shape[:2]
    x1, y1, x2, y2 = _box_edges(plate_box)
    x1 = max(int(x1) - pad_pixels, 0)
    y1 = max(int(y1) - pad_pixels, 0)
    x2 = min(int(x2) + pad_pixels, width)
    y2 = min(int(y2) + pad_pixels, height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = raw_frame[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape[:2]
    if min_crop_height and crop_height < min_crop_height:
        scale = min_crop_height / crop_height
        crop = cv2.resize(
            crop, (int(crop_width * scale), min_crop_height), interpolation=cv2.INTER_CUBIC,
        )
    return crop


def draw_ocr_on_plate(
    plate_crop: np.ndarray,
    ocr_text: str,
    confidence: Optional[float] = None,
    output_path: Optional[str] = None,
    font_scale: float = 0.8,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay OCR text beneath a plate crop for QA/debug review.

    The crop itself is kept intact and a padded canvas is created below it so
    the text does not occlude the plate. When confidence is supplied, it is
    appended as a secondary value alongside the cleaned OCR text.
    """
    if plate_crop is None:
        return plate_crop

    image = np.array(plate_crop, copy=True)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("plate_crop must be a BGR/OpenCV image array")

    text = str(ocr_text or "UNRECOGNIZED")
    if confidence is not None:
        text = f"{text} ({confidence:.2f})"

    height, width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad_height = max(40, text_size[1] + 20)
    canvas_height = height + pad_height
    canvas = np.full((canvas_height, width, 3), 230, dtype=np.uint8)
    canvas[:height, :, :] = image

    text_x = 10
    text_y = height + text_size[1] + 10
    cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, canvas)

    return canvas


def run_ocr_on_plate_crops(
    ocr_engine: Any,
    plate_records: List[Dict[str, Any]],
    ocr_dir: str,
    args: Any,
) -> List[Dict[str, Any]]:
    """OCR raw-frame plate crops and return one result per plate record."""
    results = []

    for record in plate_records:
        plate_path = record["plate_path"]
        plate_image = cv2.imread(plate_path)
        if plate_image is None:
            print(f"    [warn] unreadable plate crop: {plate_path}")
            continue

        text, conf = ocr.read_plate_text(ocr_engine, plate_image)
        winning_variant = "raw"

        cleaned_text = ocr.clean_plate_text(text)
        low_confidence = conf < args.min_confidence
        recognized = cleaned_text != "UNRECOGNIZED"
        status = "check" if (low_confidence or not recognized) else "read"

        plate_stem = os.path.splitext(os.path.basename(plate_path))[0]
        sanitized_text = ocr.sanitize_for_filename(cleaned_text) if recognized else "unrecognized"
        base_name = f"{plate_stem}_{status}_{sanitized_text}"
        dest_path = ocr.unique_destination(ocr_dir, base_name, ".png")

        detail = cleaned_text if recognized else "no text found"
        print(f"    plate -> {status} ({detail}, conf={conf:.2f})")

        raw_text = text if text else "NO_TEXT"
        result_record = {
            "base_row": record["base_row"],
            "plate_index": record["plate_index"],
            "match_method": record["match_method"],
            "plate_crop": plate_path,
            "ocr_file": dest_path,
            "raw_text": raw_text,
            "text": cleaned_text,
            "confidence": conf,
            "variant": winning_variant,
            "status": status,
        }
        if getattr(args, "save_qa_images", False):
            base_row = result_record.get("base_row", {})
            track_id = base_row.get("track_id") or f"orphan_{record['plate_index']}"
            frame_id = os.path.splitext(os.path.basename(base_row.get("source_image", "frame")))[0]
            qa_name = f"{track_id}_{frame_id}_{record['plate_index']}"
            qa_path = os.path.join(getattr(args, "qa_dir", os.path.join(os.path.dirname(ocr_dir), "qa_review")), f"{qa_name}.jpg")
            draw_ocr_on_plate(plate_image, cleaned_text, conf, qa_path)
        results.append(result_record)

    return results


def process_single_image(
    image_path: str,
    args: Any,
    vehicle_model: Any,
    plate_model: Any,
    ocr_engine: Any,
    date_dir: str,
    vehicle_dir: str,
    plate_dir: str,
    ocr_dir: str,
    work_dir: str,
    agg_rows: List[Dict[str, Any]],
) -> int:
    """Run vehicle and plate detection independently on one raw source frame."""
    filename = os.path.basename(image_path)
    timestamp = source_timestamp(image_path)
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
        plate_preds = detect.run_local_detection(
            plate_model, detection_input_path, args.plate_conf_threshold,
            args.vehicle_iou_threshold, args.imgsz, None,
        )
    except (FileNotFoundError, detect.LocalInferenceError) as e:
        print(f"{filename} -> vehicle detection error: {e}")
        return 0

    vehicle_preds = [
        pred for pred in vehicle_preds
        if not is_plate_prediction(pred)
    ]
    plate_preds = [
        pred for pred in plate_preds
        if is_plate_prediction(pred)
    ]
    for vehicle_index, vehicle in enumerate(vehicle_preds, start=1):
        vehicle["track_id"] = f"{timestamp}_v{vehicle_index}"

    full_img = cv2.imread(detection_input_path)
    if full_img is None:
        print(f"{filename} -> could not reload raw frame")
        return 0

    matches = match_plates_to_vehicles(
        vehicle_preds,
        plate_preds,
        args.plate_containment_threshold,
        args.plate_min_iou,
    )
    print(f"{filename} -> {len(vehicle_preds)} vehicle(s), {len(plate_preds)} plate(s) detected")

    vehicle_rows: Dict[str, Dict[str, Any]] = {}
    vehicle_match_counts: Dict[str, int] = {}
    vehicle_count = 0
    for pred in vehicle_preds:
        track_id = pred["track_id"]
        cls = str(pred.get("class") or "vehicle")
        safe_cls = "".join(c if c.isalnum() else "_" for c in cls)
        v_conf = pred.get("confidence", 0.0)
        vehicle_crop = crop_vehicle(
            full_img, pred, args.vehicle_crop_padding, args.vehicle_crop_min_height,
        )
        vehicle_crop_path = ""
        if vehicle_crop is not None:
            vehicle_name = unique_stem(vehicle_dir, f"{track_id}_{safe_cls}")
            vehicle_crop_path = os.path.join(vehicle_dir, f"{vehicle_name}.png")
            cv2.imwrite(vehicle_crop_path, vehicle_crop)
            vehicle_count += 1
        else:
            vehicle_name = track_id
            print(f"  {track_id} -> vehicle thumbnail could not be cropped")

        vehicle_rows[track_id] = {
            "source_image": filename,
            "track_id": track_id,
            "vehicle_id": track_id,
            "vehicle_class": cls,
            "vehicle_confidence": f"{v_conf:.4f}",
            "vehicle_crop": os.path.relpath(vehicle_crop_path, date_dir) if vehicle_crop_path else "",
        }
        vehicle_match_counts[track_id] = 0

    plate_records: List[Dict[str, Any]] = []
    immediate_rows: List[Dict[str, Any]] = []
    for match in matches:
        track_id = match["track_id"]
        if track_id is None:
            base_row = {
                "source_image": filename,
                "track_id": "",
                "vehicle_id": "",
                "vehicle_class": "",
                "vehicle_confidence": "",
                "vehicle_crop": "",
            }
        else:
            base_row = vehicle_rows[track_id]
            vehicle_match_counts[track_id] += 1

        plate_image = crop_plate(
            full_img, match["plate_box"], args.plate_crop_padding, args.plate_crop_min_height,
        )
        plate_prefix = track_id or f"orphan_{timestamp}"
        plate_name = f"{plate_prefix}_plate_{match['plate_index']}"
        plate_path = os.path.join(plate_dir, f"{unique_stem(plate_dir, plate_name)}.png")
        if plate_image is None or not cv2.imwrite(plate_path, plate_image):
            row = dict(base_row)
            row.update({
                "plate_index": match["plate_index"],
                "match_method": match["match_method"],
                "containment_ratio": f"{match['containment_ratio']:.4f}",
                "iou": f"{match['iou']:.4f}",
                "plate_crop": "",
                "plate_text": "",
                "plate_confidence": "",
                "ocr_status": "uncroppable_plate",
                "ocr_file": "",
            })
            immediate_rows.append(row)
            continue

        plate_records.append({
            "base_row": base_row,
            "plate_index": match["plate_index"],
            "match_method": match["match_method"],
            "plate_path": plate_path,
            "containment_ratio": match["containment_ratio"],
            "iou": match["iou"],
        })

    for track_id, base_row in vehicle_rows.items():
        if vehicle_match_counts[track_id] == 0:
            row = dict(base_row)
            row.update({
                "plate_index": "",
                "match_method": "",
                "containment_ratio": "",
                "iou": "",
                "plate_crop": "",
                "plate_text": "",
                "plate_confidence": "",
                "ocr_status": "no_plate",
                "ocr_file": "",
            })
            immediate_rows.append(row)

    agg_rows.extend(immediate_rows)
    print(f"  {len(plate_records)} plate crop(s) saved from raw frame, running OCR...")
    ocr_results = run_ocr_on_plate_crops(ocr_engine, plate_records, ocr_dir, args)
    for result in ocr_results:
        row = dict(result["base_row"])
        record = next(
            record for record in plate_records
            if record["plate_index"] == result["plate_index"]
            and record["plate_path"] == result["plate_crop"]
        )
        row.update({
            "plate_index": result["plate_index"],
            "match_method": result["match_method"],
            "containment_ratio": f"{record['containment_ratio']:.4f}",
            "iou": f"{record['iou']:.4f}",
            "plate_crop": os.path.relpath(result["plate_crop"], date_dir),
            "raw_ocr_text": result["raw_text"],
            "plate_text": result["text"],
            "plate_confidence": f"{result['confidence']:.4f}",
            "ocr_status": result["status"],
            "ocr_file": os.path.relpath(result["ocr_file"], date_dir),
        })
        agg_rows.append(row)

    return vehicle_count


def write_aggregate_log(date_dir: str, agg_rows: List[Dict[str, Any]]) -> str:
    output_path = os.path.join(date_dir, "pipeline_log.csv")
    os.makedirs(date_dir, exist_ok=True)
    fieldnames = [
        "source_image",
        "track_id",
        "vehicle_id",
        "vehicle_class",
        "raw_ocr_text",
        "plate_text",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: raw-frame vehicle and plate detection -> matching -> OCR. "
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
    parser.add_argument("--plate-crop-padding", type=int, default=CONFIG["plate_crop_padding"],
                         help="Pixels added around each raw-frame plate box before OCR.")
    parser.add_argument("--plate-crop-min-height", type=int, default=CONFIG["plate_crop_min_height"])
    parser.add_argument("--plate-containment-threshold", type=float,
                         default=CONFIG["plate_containment_threshold"],
                         help="Minimum plate-area containment ratio for a primary match.")
    parser.add_argument("--plate-min-iou", type=float, default=CONFIG["plate_min_iou"],
                         help="Minimum IoU for a fallback plate-to-vehicle match.")

    parser.add_argument("--no-preprocess", action="store_true", default=CONFIG["no_preprocess"],
                         help="Skip CLAHE/denoise/resize before both detection passes.")
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    parser.add_argument("--max-dim", type=int, default=CONFIG["max_dim"])
    parser.add_argument("--min-dim", type=int, default=CONFIG["min_dim"])

    parser.add_argument("--model-tier", choices=sorted(ocr.MODEL_TIERS), default=CONFIG["model_tier"])
    parser.add_argument("--single-pass", action="store_true", default=CONFIG["single_pass"])
    parser.add_argument("--min-confidence", type=float, default=CONFIG["min_confidence"])
    parser.add_argument("--save-qa-images", action="store_true", default=CONFIG["SAVE_QA_IMAGES"],
                         help="Save annotated QA review crops showing the cleaned OCR text overlay.")

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
    qa_dir = os.path.join(date_dir, "qa_review")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(plate_dir, exist_ok=True)
    if args.save_qa_images:
        os.makedirs(qa_dir, exist_ok=True)
    args.qa_dir = qa_dir
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
