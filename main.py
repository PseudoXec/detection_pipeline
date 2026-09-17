import argparse
import csv
import math
import os
import queue
import shutil
import sys
import tempfile
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import local_model_infer as detect
import ocr_cropped_plates as ocr

CONFIG = {
    # --- Source: set exactly ONE of these, leave the other as None ---
    "image": None,                                  # e.g. r"C:\cars\photo.jpg"
    "folder": None,                                 # e.g. r"C:\cars\sample_images"
    "rtsp_url": r"rtsp://admin:Victoria2313*@192.168.100.229:554/stream1",                          
    "stream_frame_skip": 1,                         # detect every received frame at the slower stream rate
    "stream_width": 1280,                           # requested RTSP capture width
    "stream_height": 720,                          # requested RTSP capture height
    "stream_fps": 0,                                # slower feed gives tracking more time per frame
    "stream_buffer_size": 1,                        # keep only the newest buffered frame
    "stream_reconnect_delay": 5.0,                  # seconds between reconnect attempts
    "stream_max_frames": 0,                         # 0 = run until stopped; useful for testing
    "stream_preview": True,                         # show live detection window for RTSP input
    "stream_draw_boxes": False,                     # draw vehicle/plate boxes in the live RTSP preview
    "stream_preview_width": 1280,                   # preview window width
    "stream_preview_height": 720,                   # preview window height
    "stream_preview_delay_ms": 100,                 # slow preview to roughly 10 display updates/sec
    "stream_vehicle_conf_threshold": 0.25,          # RTSP-only threshold for more vehicle boxes
    "stream_track_iou_threshold": 0.3,              # minimum IoU to keep a vehicle track
    "stream_track_max_center_distance": 2.0,         # max center movement in vehicle-widths
    "stream_track_max_missing": 10,                 # processed frames before a track expires
    "stream_tracking_cooldown_frames": 2,           # skip redundant tracking passes while a stable track is still alive
    "stream_ocr_max_attempts": 2,                   # OCR attempts per tracked vehicle before giving up
    "stream_ocr_confidence_threshold": 0.85,         # final accepted OCR confidence
    "stream_plate_padding_ratio": 0.08,             # vehicle crop padding for live crops

    # --- Models ---
    "vehicle_weights": r"C:\Users\User\Documents\detection_pipeline\models\vehicle.pt",   # your trained vehicle .pt
    "plate_weights": r"C:\Users\User\Documents\detection_pipeline\models\platenum_closeup.pt",   # your trained license-plate .pt
    "device": None,                                       # None = auto (GPU if available)

    # --- Vehicle detection stage ---
    "vehicle_classes": None,            # e.g. "car,truck,bus,motorcycle"; None = keep all classes
    "vehicle_conf_threshold": 0.35,
    "vehicle_iou_threshold": 0.45,
    "vehicle_crop_padding": 0.08,       # extra margin around each vehicle box (8%)
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
    "model_tier": "mobile-en",          # lighter RTSP default: faster OCR for stable live tracks
    "single_pass": False,               # True = skip the raw/light/full best-of comparison
    "min_confidence": 0.80,             # below this, result is flagged "check"/LOWCONF
    "skip_ocr": False,                   # Run OCR for plates when a crop is available
    "detection_only": False,             # Keep crop + OCR enabled by default

    # --- QA / debug output ---
    "SAVE_QA_IMAGES": False,            # Disabled: do not save OCR-overlaid review images

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


def default_base_out_dir(image: Optional[str], folder: Optional[str], rtsp_url: Optional[str] = None) -> str:
    """Same convention as local_model_infer.default_out_dir: a sibling
    'vehicle_pipeline_output' folder next to the source."""
    if folder:
        sample_folder = os.path.abspath(folder.rstrip("/\\"))
    elif image:
        sample_folder = os.path.abspath(os.path.dirname(image) or ".")
    else:
        sample_folder = os.getcwd()
    return os.path.join(sample_folder, "vehicle_pipeline_output")


def source_timestamp(image_path: str) -> str:
    """Returns a filename-safe timestamp for a source image.

    File inputs use their modification time. An RTSP capture should pass its`
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

def update_stream_tracks(
    vehicle_predictions: List[Dict[str, Any]],
    tracker: Dict[str, Any],
    iou_threshold: float = 0.3,
    max_center_distance: float = 2.0,
    max_missing: int = 10,
) -> None:
    """Assign stable IDs to vehicle boxes across sampled stream frames."""
    tracks = tracker.setdefault("tracks", {})
    used_track_ids: Set[str] = set()
    next_track_number = int(tracker.get("next_track_number", 1))

    for prediction in sorted(vehicle_predictions, key=lambda item: item.get("confidence", 0.0), reverse=True):
        best_track_id = None
        best_iou = iou_threshold
        best_center_distance = max_center_distance
        for track_id, track in tracks.items():
            if track_id in used_track_ids:
                continue
            overlap = compute_iou(prediction, track["box"])
            previous_box = track["box"]
            center_distance = math.hypot(
                prediction["x"] - previous_box["x"],
                prediction["y"] - previous_box["y"],
            ) / max(previous_box["width"], previous_box["height"], 1.0)
            if overlap >= best_iou or (
                overlap > 0.0 and center_distance <= best_center_distance
            ):
                if overlap < best_iou and center_distance > best_center_distance:
                    continue
                best_iou = overlap
                best_center_distance = center_distance
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = f"stream_v{next_track_number}"
            next_track_number += 1
            tracks[best_track_id] = {"box": prediction, "missing": 0}
        else:
            tracks[best_track_id]["box"] = prediction
            tracks[best_track_id]["missing"] = 0
        prediction["track_id"] = best_track_id
        used_track_ids.add(best_track_id)

    for track_id in list(tracks):
        if track_id not in used_track_ids:
            tracks[track_id]["missing"] += 1
            if tracks[track_id]["missing"] > max_missing:
                del tracks[track_id]

    tracker["next_track_number"] = next_track_number


def should_reprocess_stream_tracks(
    vehicle_predictions: List[Dict[str, Any]],
    stream_track_state: Dict[str, Any],
) -> bool:
    """Only rerun expensive crop/OCR work while a track is new or incomplete."""
    active_track_ids = {pred.get("track_id") for pred in vehicle_predictions if pred.get("track_id")}

    if not stream_track_state and not active_track_ids:
        return True

    if not vehicle_predictions:
        return False

    if not active_track_ids:
        return False

    for track_id in active_track_ids:
        state = stream_track_state.get(track_id)
        if state is None:
            return True
        if state.get("finalized"):
            continue
        if state.get("ocr_pending"):
            return True
        if not state.get("ocr_done", False):
            return True
    return False


def should_trigger_tracking_for_frame(
    frame_number: int,
    last_tracking_frame: int,
    stream_track_state: Dict[str, Any],
    cooldown_frames: int = 2,
) -> bool:
    """Avoid redundant detections when all current tracks are already stable."""
    if cooldown_frames < 0:
        cooldown_frames = 0
    if frame_number - last_tracking_frame < cooldown_frames:
        return False

    if not stream_track_state:
        return True

    for state in stream_track_state.values():
        if state.get("finalized"):
            continue
        if state.get("ocr_pending"):
            return True
        if not state.get("ocr_done", False):
            return True
    return False


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


def draw_detection_preview(
    frame: np.ndarray,
    vehicle_predictions: List[Dict[str, Any]],
    plate_predictions: List[Dict[str, Any]],
    draw_boxes: bool = False,
) -> np.ndarray:
    """Optionally draw vehicle and plate boxes for the live RTSP preview."""
    if not draw_boxes:
        return np.array(frame, copy=True)

    preview = np.array(frame, copy=True)
    for prediction, color, label in [
        *[(prediction, (0, 200, 0), "vehicle") for prediction in vehicle_predictions],
        *[(prediction, (0, 165, 255), "plate") for prediction in plate_predictions],
    ]:
        x1, y1, x2, y2 = _box_edges(prediction)
        top_left = (max(0, int(x1)), max(0, int(y1)))
        bottom_right = (min(preview.shape[1] - 1, int(x2)), min(preview.shape[0] - 1, int(y2)))
        confidence = prediction.get("confidence")
        label_text = label if confidence is None else f"{label} {float(confidence):.2f}"
        cv2.rectangle(preview, top_left, bottom_right, color, 2)
        cv2.putText(preview, label_text, (top_left[0], max(18, top_left[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return preview


def finalize_stream_track(
    track_id: str,
    state: Dict[str, Any],
    args: Any,
    date_dir: str,
    agg_rows: List[Dict[str, Any]],
    force: bool = False,
) -> None:
    """Write one final CSV row when a stream track is accepted or expires."""
    if state["finalized"] or state.get("ocr_pending"):
        return
    if not force and not state["ocr_done"]:
        return
    state["finalized"] = True
    if not state["ocr_done"]:
        state["ocr_status"] = "low_confidence"
    row = dict(state["base_row"])
    row.update({
        "plate_index": 1,
        "match_method": "vehicle_crop",
        "containment_ratio": "1.0000",
        "iou": "1.0000",
        "plate_crop": os.path.relpath(state["plate_path"], date_dir) if state["plate_path"] else "",
        "raw_ocr_text": state["raw_ocr_text"],
        "plate_text": state["plate_text"],
        "plate_confidence": f"{state['best_ocr_conf']:.4f}",
        "ocr_status": state["ocr_status"],
        "ocr_file": "",
    })
    agg_rows.append(row)


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
    """Detect vehicles, then detect plates inside each vehicle crop and OCR them."""
    filename = os.path.basename(image_path)
    timestamp = source_timestamp(image_path)
    detection_input_path = image_path
    vehicle_classes = (
        set(c.strip() for c in args.vehicle_classes.split(",") if c.strip())
        if args.vehicle_classes else None
    )

    try:
        tracked_vehicle_preds = getattr(args, "_stream_vehicle_predictions", None)
        if tracked_vehicle_preds is None:
            vehicle_preds = detect.run_local_detection(
                vehicle_model, detection_input_path, args.vehicle_conf_threshold,
                args.vehicle_iou_threshold, args.imgsz, vehicle_classes,
            )
        else:
            vehicle_preds = [dict(prediction) for prediction in tracked_vehicle_preds]
    except (FileNotFoundError, detect.LocalInferenceError) as e:
        print(f"{filename} -> vehicle detection error: {e}")
        return 0

    vehicle_preds = [
        pred for pred in vehicle_preds
        if not is_plate_prediction(pred)
    ]
    stream_tracker = getattr(args, "_stream_tracker", None)
    if tracked_vehicle_preds is not None and any("track_id" in pred for pred in vehicle_preds):
        pass
    elif stream_tracker is not None:
        update_stream_tracks(
            vehicle_preds,
            stream_tracker,
            getattr(args, "stream_track_iou_threshold", 0.3),
            getattr(args, "stream_track_max_center_distance", 2.0),
            getattr(args, "stream_track_max_missing", 10),
        )
    else:
        for vehicle_index, vehicle in enumerate(vehicle_preds, start=1):
            vehicle["track_id"] = f"{timestamp}_v{vehicle_index}"

    full_img = cv2.imread(detection_input_path)
    if full_img is None:
        print(f"{filename} -> could not reload raw frame")
        return 0

    stream_track_state = getattr(args, "_stream_track_state", None)
    stream_frame_number = int(getattr(args, "_stream_current_frame", 0))

    preview_state = getattr(args, "_stream_preview_state", None)
    if preview_state is not None:
        with preview_state["lock"]:
            preview_state["vehicle_predictions"] = [dict(prediction) for prediction in vehicle_preds]
            preview_state["plate_predictions"] = []

    print(f"{filename} -> {len(vehicle_preds)} vehicle(s) detected")

    if getattr(args, "detection_only", False):
        for vehicle in vehicle_preds:
            agg_rows.append({
                "source_image": filename,
                "track_id": vehicle["track_id"],
                "vehicle_id": vehicle["track_id"],
                "vehicle_class": str(vehicle.get("class") or "vehicle"),
                "raw_ocr_text": "",
                "plate_text": "",
                "ocr_status": "detection_only",
            })
        return 0

    saved_vehicle_crops = getattr(args, "_saved_vehicle_crops", {})
    saved_plate_crops = getattr(args, "_saved_plate_crops", {})
    args._saved_vehicle_crops = saved_vehicle_crops
    args._saved_plate_crops = saved_plate_crops

    vehicle_rows: Dict[str, Dict[str, Any]] = {}
    vehicle_match_counts: Dict[str, int] = {}
    vehicle_crop_images: Dict[str, np.ndarray] = {}
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
        if track_id in saved_vehicle_crops:
            vehicle_crop_path = saved_vehicle_crops[track_id]
            vehicle_crop_images[track_id] = cv2.imread(vehicle_crop_path)
        elif vehicle_crop is not None:
            vehicle_name = unique_stem(vehicle_dir, f"{track_id}_{safe_cls}")
            vehicle_crop_path = os.path.join(vehicle_dir, f"{vehicle_name}.png")
            cv2.imwrite(vehicle_crop_path, vehicle_crop)
            saved_vehicle_crops[track_id] = vehicle_crop_path
            vehicle_crop_images[track_id] = vehicle_crop
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
    plate_predictions_for_preview: List[Dict[str, Any]] = []
    for vehicle_index, vehicle in enumerate(vehicle_preds, start=1):
        if stream_track_state is not None:
            continue
        track_id = vehicle["track_id"]
        if track_id is not None and track_id in saved_plate_crops:
            print(f"  {track_id} -> plate crop already saved; skipping duplicate")
            continue

        if track_id not in vehicle_crop_images or vehicle_crop_images[track_id] is None:
            continue
        vehicle_crop = vehicle_crop_images[track_id]
        vehicle_crop_path = saved_vehicle_crops[track_id]
        try:
            vehicle_plate_preds = detect.run_local_detection(
                plate_model, vehicle_crop_path, args.plate_conf_threshold,
                args.vehicle_iou_threshold, args.imgsz, None,
            )
        except (FileNotFoundError, detect.LocalInferenceError) as error:
            print(f"  {track_id} -> plate detection error: {error}")
            vehicle_plate_preds = []
        if not vehicle_plate_preds:
            continue

        best_plate = max(vehicle_plate_preds, key=lambda item: item.get("confidence", 0.0))
        plate_image = crop_plate(
            vehicle_crop, best_plate, args.plate_crop_padding, args.plate_crop_min_height,
        )
        plate_name = f"{track_id}_plate_1"
        plate_path = os.path.join(plate_dir, f"{unique_stem(plate_dir, plate_name)}.png")
        if plate_image is None or not cv2.imwrite(plate_path, plate_image):
            continue

        saved_plate_crops[track_id] = plate_path
        vehicle_match_counts[track_id] += 1

        plate_records.append({
            "base_row": vehicle_rows[track_id],
            "plate_index": 1,
            "match_method": "vehicle_crop",
            "plate_path": plate_path,
            "containment_ratio": 1.0,
            "iou": 1.0,
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

    if stream_track_state is not None:
        max_attempts = max(1, int(getattr(args, "stream_ocr_max_attempts", 3)))
        confidence_threshold = float(getattr(args, "stream_ocr_confidence_threshold", 0.85))
        for vehicle in vehicle_preds:
            track_id = vehicle["track_id"]
            state = stream_track_state.setdefault(track_id, {
                "best_plate_conf": 0.0,
                "best_ocr_conf": 0.0,
                "plate_text": "UNRECOGNIZED",
                "ocr_done": False,
                "last_seen_frame": stream_frame_number,
                "ocr_attempts": 0,
                "base_row": vehicle_rows[track_id],
                "plate_path": "",
                "plate_image": None,
                "raw_ocr_text": "NO_TEXT",
                "ocr_status": "low_confidence",
                "ocr_pending": False,
                "finalized": False,
            })
            state["last_seen_frame"] = stream_frame_number
            state["base_row"] = vehicle_rows[track_id]
            if state["ocr_done"] or state["ocr_attempts"] >= max_attempts:
                continue

            vehicle_crop_path = saved_vehicle_crops.get(track_id, "")
            if not vehicle_crop_path:
                continue
            try:
                vehicle_plate_preds = detect.run_local_detection(
                    plate_model, vehicle_crop_path, args.plate_conf_threshold,
                    args.vehicle_iou_threshold, args.imgsz, None,
                )
            except (FileNotFoundError, detect.LocalInferenceError) as error:
                print(f"  {track_id} -> plate detection error: {error}")
                continue
            if not vehicle_plate_preds:
                continue

            best_plate = max(vehicle_plate_preds, key=lambda item: item.get("confidence", 0.0))
            plate_conf = float(best_plate.get("confidence", 0.0))
            if state["ocr_attempts"] and plate_conf <= state["best_plate_conf"]:
                continue
            state["ocr_attempts"] += 1
            state["best_plate_conf"] = plate_conf
            vehicle_crop = vehicle_crop_images.get(track_id)
            if vehicle_crop is None:
                vehicle_crop = cv2.imread(vehicle_crop_path)
            plate_image = crop_plate(
                vehicle_crop, best_plate, args.plate_crop_padding, args.plate_crop_min_height,
            ) if vehicle_crop is not None else None
            if plate_image is None:
                continue

            plate_path = state["plate_path"] or os.path.join(plate_dir, f"{track_id}_plate_1.png")
            if not cv2.imwrite(plate_path, plate_image):
                continue
            state["plate_path"] = plate_path
            state["plate_image"] = plate_image
            ocr_queue = getattr(args, "_stream_ocr_queue", None)
            if ocr_queue is None:
                text, ocr_conf = ocr.read_plate_text(ocr_engine, plate_image)
                cleaned_text = ocr.clean_plate_text(text)
                if ocr_conf >= state["best_ocr_conf"]:
                    state["best_ocr_conf"] = float(ocr_conf)
                    state["plate_text"] = cleaned_text
                    state["raw_ocr_text"] = text if text else "NO_TEXT"
                if cleaned_text != "UNRECOGNIZED" and ocr_conf >= confidence_threshold:
                    state["ocr_done"] = True
                    state["ocr_status"] = "read"
            else:
                try:
                    state["ocr_pending"] = True
                    ocr_queue.put_nowait((track_id, plate_image.copy()))
                except queue.Full:
                    state["ocr_pending"] = False

        for track_id, state in stream_track_state.items():
            if state["finalized"] or state.get("ocr_pending"):
                continue
            if state["ocr_done"] or stream_frame_number - state["last_seen_frame"] > getattr(args, "stream_track_max_missing", 10):
                finalize_stream_track(track_id, state, args, date_dir, agg_rows, force=not state["ocr_done"])
        return vehicle_count

    agg_rows.extend(immediate_rows)
    if getattr(args, "skip_ocr", False):
        for record in plate_records:
            row = dict(record["base_row"])
            row.update({
                "plate_index": record["plate_index"],
                "match_method": record["match_method"],
                "containment_ratio": f"{record['containment_ratio']:.4f}",
                "iou": f"{record['iou']:.4f}",
                "plate_crop": os.path.relpath(record["plate_path"], date_dir),
                "raw_ocr_text": "",
                "plate_text": "",
                "plate_confidence": "",
                "ocr_status": "ocr_disabled",
                "ocr_file": "",
            })
            agg_rows.append(row)
        print(f"  {len(plate_records)} plate crop(s) saved; OCR disabled")
        return vehicle_count

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

    if preview_state is not None:
        with preview_state["lock"]:
            preview_state["vehicle_predictions"] = [dict(prediction) for prediction in vehicle_preds]
            preview_state["plate_predictions"] = plate_predictions_for_preview

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


def process_rtsp_stream(
    rtsp_url: str,
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
    capture_factory: Any = cv2.VideoCapture,
) -> int:
    """Read the newest RTSP frame and send selected frames through the pipeline."""
    frame_skip = max(1, int(args.stream_frame_skip))
    reconnect_delay = max(0.0, float(args.stream_reconnect_delay))
    max_frames = max(0, int(args.stream_max_frames))
    capture = capture_factory(rtsp_url)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    latest_frame: Dict[str, Any] = {"value": None, "number": 0}
    frame_lock = threading.Lock()
    stop_reader = threading.Event()
    reader_failed = threading.Event()
    stop_preview = threading.Event()

    def read_latest_frames() -> None:
        while not stop_reader.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[stream] Frame read failed; stopping capture reader")
                reader_failed.set()
                stop_reader.set()
                return
            with frame_lock:
                latest_frame["value"] = frame
                latest_frame["number"] += 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, max(0, int(getattr(args, "stream_width", 1280))))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max(0, int(getattr(args, "stream_height", 720))))
    stream_fps = getattr(args, "stream_fps", None)
    if stream_fps:
        capture.set(cv2.CAP_PROP_FPS, max(1, int(stream_fps)))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(getattr(args, "stream_buffer_size", 1))))
    reader = threading.Thread(target=read_latest_frames, daemon=True)
    reader.start()
    args._stream_tracker = {"tracks": {}, "next_track_number": 1}
    args._stream_track_state = {}
    preview_state = {
        "lock": threading.Lock(),
        "vehicle_predictions": [],
        "plate_predictions": [],
    }
    args._stream_preview_state = preview_state
    inference_queue: queue.Queue = queue.Queue(maxsize=1)
    ocr_queue: queue.Queue = queue.Queue(maxsize=1)
    worker_stop = threading.Event()
    worker_state = {"processed_frames": 0, "total_vehicles": 0}

    if getattr(args, "show_stream_preview", False):
        cv2.namedWindow("RTSP detection preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "RTSP detection preview",
            max(320, int(getattr(args, "stream_preview_width", 1280))),
            max(240, int(getattr(args, "stream_preview_height", 720))),
        )

    def run_inference_worker() -> None:
        while not worker_stop.is_set():
            try:
                frame_path, frame_number, inference_frame = inference_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                args._stream_current_frame = frame_number
                args._stream_vehicle_predictions = detect.run_local_tracking(
                    vehicle_model,
                    inference_frame,
                    getattr(args, "stream_vehicle_conf_threshold", 0.25),
                    getattr(args, "vehicle_iou_threshold", 0.45),
                    getattr(args, "imgsz", None),
                    set(c.strip() for c in args.vehicle_classes.split(",") if c.strip())
                    if getattr(args, "vehicle_classes", None) else None,
                )
                with preview_state["lock"]:
                    preview_state["vehicle_predictions"] = [
                        dict(prediction) for prediction in args._stream_vehicle_predictions
                        if not is_plate_prediction(prediction)
                    ]

                stream_track_state = getattr(args, "_stream_track_state", {})
                tracked_predictions = [dict(prediction) for prediction in args._stream_vehicle_predictions if not is_plate_prediction(prediction)]
                if should_reprocess_stream_tracks(tracked_predictions, stream_track_state):
                    worker_state["total_vehicles"] += process_single_image(
                        frame_path, args, vehicle_model, plate_model, ocr_engine,
                        date_dir, vehicle_dir, plate_dir, ocr_dir, work_dir, agg_rows,
                    )
                    worker_state["processed_frames"] += 1
                    print(f"[stream] Processed frame {worker_state['processed_frames']} (captured {frame_number})")
                else:
                    worker_state["processed_frames"] += 1
                    print(f"[stream] Tracking continued without crop/OCR rerun on frame {frame_number}")
            except (FileNotFoundError, detect.LocalInferenceError) as error:
                print(f"[stream] Inference error on frame {frame_number}: {error}")
            finally:
                if hasattr(args, "_stream_vehicle_predictions"):
                    del args._stream_vehicle_predictions
                inference_queue.task_done()

    def run_ocr_worker() -> None:
        while not worker_stop.is_set() or not ocr_queue.empty():
            try:
                track_id, plate_image = ocr_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                state = args._stream_track_state.get(track_id)
                if state is None or state["finalized"]:
                    continue
                text, ocr_conf = ocr.read_plate_text(ocr_engine, plate_image)
                cleaned_text = ocr.clean_plate_text(text)
                state["ocr_pending"] = False
                if ocr_conf >= state["best_ocr_conf"]:
                    state["best_ocr_conf"] = float(ocr_conf)
                    state["plate_text"] = cleaned_text
                    state["raw_ocr_text"] = text if text else "NO_TEXT"
                if cleaned_text != "UNRECOGNIZED" and ocr_conf >= getattr(args, "stream_ocr_confidence_threshold", 0.85):
                    state["ocr_done"] = True
                    state["ocr_status"] = "read"
                    finalize_stream_track(track_id, state, args, date_dir, agg_rows)
            finally:
                ocr_queue.task_done()

    inference_worker = threading.Thread(target=run_inference_worker, daemon=True)
    ocr_worker = threading.Thread(target=run_ocr_worker, daemon=True)
    inference_worker.start()
    ocr_worker.start()
    last_submitted_frame = 0
    last_tracking_frame = 0
    tracking_cooldown_frames = max(0, int(getattr(args, "stream_tracking_cooldown_frames", 2)))

    try:
        while max_frames == 0 or worker_state["processed_frames"] < max_frames:
            if stop_preview.is_set():
                break
            with frame_lock:
                frame = latest_frame["value"]
                frame_number = latest_frame["number"]

            if reader_failed.is_set() and (frame is None or frame_number == last_submitted_frame):
                break

            if frame is not None and getattr(args, "show_stream_preview", False):
                with preview_state["lock"]:
                    vehicle_predictions = preview_state["vehicle_predictions"]
                    plate_predictions = preview_state["plate_predictions"]
                preview = draw_detection_preview(
                    frame,
                    vehicle_predictions,
                    plate_predictions,
                    draw_boxes=bool(getattr(args, "stream_draw_boxes", False)),
                )
                cv2.imshow("RTSP detection preview", preview)
                if cv2.waitKey(max(1, int(getattr(args, "stream_preview_delay_ms", 100)))) & 0xFF == ord("q"):
                    stop_preview.set()
                    stop_reader.set()
                    break

            if frame is None or frame_number == last_submitted_frame:
                time.sleep(0.01)
                continue
            if (frame_number - 1) % frame_skip != 0:
                last_submitted_frame = frame_number
                continue
            if not inference_queue.empty():
                time.sleep(0.005)
                continue
            if not should_trigger_tracking_for_frame(
                frame_number,
                last_tracking_frame,
                getattr(args, "_stream_track_state", {}),
                tracking_cooldown_frames,
            ):
                last_submitted_frame = frame_number
                continue

            frame_name = datetime.now().strftime("stream_%Y%m%d_%H%M%S_%f.jpg")
            frame_path = os.path.join(work_dir, frame_name)
            if not cv2.imwrite(frame_path, frame):
                print(f"[stream] Could not stage frame {frame_number}")
                last_submitted_frame = frame_number
                continue
            args._stream_current_frame = frame_number
            inference_queue.put((frame_path, frame_number, frame.copy()))
            last_submitted_frame = frame_number
            last_tracking_frame = frame_number
    except KeyboardInterrupt:
        print("\n[stream] Stopped by user")
    finally:
        stop_preview.set()
        inference_queue.join()
        ocr_queue.join()
        for track_id, state in args._stream_track_state.items():
            finalize_stream_track(track_id, state, args, date_dir, agg_rows, force=True)
        worker_stop.set()
        inference_worker.join(timeout=2.0)
        ocr_worker.join(timeout=2.0)
        if hasattr(args, "_stream_tracker"):
            del args._stream_tracker
        if hasattr(args, "_stream_track_state"):
            del args._stream_track_state
        if hasattr(args, "_stream_preview_state"):
            del args._stream_preview_state
        stop_preview.set()
        stop_reader.set()
        capture.release()
        if getattr(args, "show_stream_preview", False):
            cv2.destroyAllWindows()

    return worker_state["total_vehicles"]


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: raw-frame vehicle and plate detection -> matching -> OCR. "
                     "Defaults come from the CONFIG dict at the top of this file."
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image")
    source_group.add_argument("--folder")
    source_group.add_argument("--rtsp-url",
                              help="RTSP camera URL, e.g. rtsp://user:password@camera/stream")

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
    parser.add_argument("--skip-ocr", action="store_true", default=CONFIG["skip_ocr"],
                         help="Disable OCR and save only vehicle/plate detections and crops.")
    parser.add_argument("--detection-only", action="store_true", default=CONFIG["detection_only"],
                         help="Run vehicle detection only; disable all cropping, plate detection, and OCR.")
    parser.add_argument("--stream-frame-skip", type=int, default=CONFIG["stream_frame_skip"],
                         help="Process one frame every N captured frames for RTSP input.")
    parser.add_argument("--stream-width", type=int, default=CONFIG["stream_width"],
                         help="Requested RTSP frame width; the camera may ignore this value.")
    parser.add_argument("--stream-height", type=int, default=CONFIG["stream_height"],
                         help="Requested RTSP frame height; the camera may ignore this value.")
    parser.add_argument("--stream-fps", type=int, default=CONFIG["stream_fps"],
                         help="Requested RTSP FPS; the camera may ignore this value.")
    parser.add_argument("--stream-buffer-size", type=int, default=CONFIG["stream_buffer_size"],
                         help="Capture buffer size; 1 minimizes live-feed delay.")
    parser.add_argument("--stream-preview-delay-ms", type=int,
                         default=CONFIG["stream_preview_delay_ms"],
                         help="Preview delay in milliseconds; larger values slow the display.")
    parser.add_argument("--stream-draw-boxes", action="store_true", default=CONFIG["stream_draw_boxes"],
                         help="Overlay detected vehicle and plate boxes on the RTSP preview image.")
    parser.add_argument("--no-stream-draw-boxes", action="store_false", dest="stream_draw_boxes",
                         default=CONFIG["stream_draw_boxes"],
                         help="Disable the vehicle/plate box overlay on the RTSP preview image.")
    parser.add_argument("--stream-track-iou-threshold", type=float,
                         default=CONFIG["stream_track_iou_threshold"],
                         help="Minimum box overlap used to keep a vehicle's RTSP track.")
    parser.add_argument("--stream-track-max-center-distance", type=float,
                         default=CONFIG["stream_track_max_center_distance"],
                         help="Maximum center movement in previous vehicle-widths.")
    parser.add_argument("--stream-track-max-missing", type=int,
                         default=CONFIG["stream_track_max_missing"],
                         help="Processed RTSP frames a track may disappear before expiring.")
    parser.add_argument("--stream-tracking-cooldown-frames", type=int,
                         default=CONFIG["stream_tracking_cooldown_frames"],
                         help="Skip redundant RTSP tracking passes for a few frames after a stable track is live.")
    parser.add_argument("--stream-reconnect-delay", type=float,
                         default=CONFIG["stream_reconnect_delay"],
                         help="Seconds to wait before reconnecting after an RTSP failure.")
    parser.add_argument("--stream-max-frames", type=int, default=CONFIG["stream_max_frames"],
                         help="Stop after this many processed RTSP frames; 0 runs until stopped.")
    parser.add_argument("--stream-vehicle-conf-threshold", type=float,
                         default=CONFIG["stream_vehicle_conf_threshold"],
                         help="RTSP-only vehicle confidence threshold; lower values show more boxes.")
    parser.add_argument("--stream-ocr-max-attempts", type=int,
                         default=CONFIG["stream_ocr_max_attempts"],
                         help="Maximum plate/OCR attempts for each RTSP track.")
    parser.add_argument("--stream-ocr-confidence-threshold", type=float,
                         default=CONFIG["stream_ocr_confidence_threshold"],
                         help="OCR confidence that finalizes an RTSP track.")
    parser.add_argument("--no-stream-preview", action="store_false", dest="show_stream_preview",
                         default=CONFIG["stream_preview"],
                         help="Disable the live OpenCV detection preview window for RTSP input.")

    parser.add_argument("--out-dir", default=CONFIG["out_dir"],
                         help="Base output folder. A dd-mm-yy folder is created inside it for "
                              "each run's results. Defaults to a 'vehicle_pipeline_output' "
                              "folder next to the source.")

    args = parser.parse_args()

    if not args.image and not args.folder and not args.rtsp_url:
        args.image = CONFIG["image"]
        args.folder = CONFIG["folder"]
        args.rtsp_url = CONFIG["rtsp_url"]

    args.show_stream_preview = bool(args.rtsp_url and args.show_stream_preview)

    if not args.image and not args.folder and not args.rtsp_url:
        sys.exit("Error: no source set. Edit CONFIG['folder'], CONFIG['image'], or CONFIG['rtsp_url'] "
                 "at the top of this file, or pass --image / --folder / --rtsp-url on the command line.")
    if args.folder and not os.path.isdir(args.folder):
        sys.exit(f"Error: folder not found: {args.folder}")
    if args.image and not os.path.isfile(args.image):
        sys.exit(f"Error: image not found: {args.image}")
    if not os.path.isfile(args.vehicle_weights):
        sys.exit(f"Error: vehicle weights not found: {args.vehicle_weights}")
    if not args.detection_only and not os.path.isfile(args.plate_weights):
        sys.exit(f"Error: plate weights not found: {args.plate_weights}")
    if args.stream_frame_skip < 1:
        sys.exit("Error: --stream-frame-skip must be at least 1.")
    if args.stream_width < 0 or args.stream_height < 0 or (args.stream_fps is not None and args.stream_fps < 0):
        sys.exit("Error: stream width, height, and FPS cannot be negative.")
    if args.stream_buffer_size < 1:
        sys.exit("Error: --stream-buffer-size must be at least 1.")
    if args.stream_preview_delay_ms < 1:
        sys.exit("Error: --stream-preview-delay-ms must be at least 1.")
    if not 0.0 <= args.stream_track_iou_threshold <= 1.0:
        sys.exit("Error: --stream-track-iou-threshold must be between 0 and 1.")
    if args.stream_track_max_center_distance < 0:
        sys.exit("Error: --stream-track-max-center-distance cannot be negative.")
    if args.stream_track_max_missing < 0:
        sys.exit("Error: --stream-track-max-missing cannot be negative.")
    if args.stream_tracking_cooldown_frames < 0:
        sys.exit("Error: --stream-tracking-cooldown-frames cannot be negative.")
    if args.stream_max_frames < 0:
        sys.exit("Error: --stream-max-frames cannot be negative.")
    if args.stream_ocr_max_attempts < 1:
        sys.exit("Error: --stream-ocr-max-attempts must be at least 1.")
    if args.stream_ocr_confidence_threshold < 0.0 or args.stream_ocr_confidence_threshold > 1.0:
        sys.exit("Error: --stream-ocr-confidence-threshold must be between 0 and 1.")
    if not 0.0 <= args.stream_vehicle_conf_threshold <= 1.0:
        sys.exit("Error: --stream-vehicle-conf-threshold must be between 0 and 1.")
    base_out_dir = args.out_dir or default_base_out_dir(args.image, args.folder, args.rtsp_url)
    date_dir = make_date_dir(base_out_dir)
    vehicle_dir = os.path.join(date_dir, "vehicle_detection")
    plate_dir = os.path.join(date_dir, "plate_detection")
    ocr_dir = os.path.join(date_dir, "ocr")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(plate_dir, exist_ok=True)
    print(f"[info] Output folder for this run: {date_dir}")

    print("[info] Loading vehicle detection model...")
    vehicle_model = detect.build_model(args.vehicle_weights, args.device)
    if args.detection_only:
        print("[info] Detection-only mode; plate model, cropping, and OCR disabled")
        plate_model = None
        ocr_engine = None
    else:
        print("[info] Loading plate detection model...")
        plate_model = detect.build_model(args.plate_weights, args.device)
        if args.skip_ocr:
            print("[info] OCR disabled; running detection and crop stages only")
            ocr_engine = None
        else:
            print(f"[info] Loading PaddleOCR engine (model tier: {args.model_tier})...")
            ocr_engine = ocr.build_ocr_engine(args.model_tier)

    if args.folder:
        image_paths = detect.list_images_in_folder(args.folder)
        if not image_paths:
            sys.exit(f"Error: no images ({sorted(detect.IMAGE_EXTENSIONS)}) found in: {args.folder}")
    elif args.image:
        image_paths = [args.image]
    else:
        image_paths = []

    work_dir = tempfile.mkdtemp(prefix="vehicle_plate_preprocess_")
    agg_rows: List[Dict[str, Any]] = []
    total_vehicles = 0
    try:
        if args.rtsp_url:
            total_vehicles += process_rtsp_stream(
                args.rtsp_url, args, vehicle_model, plate_model, ocr_engine,
                date_dir, vehicle_dir, plate_dir, ocr_dir, work_dir, agg_rows,
            )
        else:
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
    print(f"Images processed : {len(image_paths) if image_paths else 'stream'}")
    print(f"Vehicles cropped : {total_vehicles}")
    print(f"Plates read      : {plates_read}/{len(agg_rows)}")
    print(f"Log              : {log_path}")


if __name__ == "__main__":
    main()
