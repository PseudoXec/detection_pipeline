"""
local_model_infer.py
---------------------
Drop-in replacement for roboflow_workflow_infer.py: runs YOUR OWN local
YOLO model (a .pt weights file, loaded with the `ultralytics` package)
instead of calling the Roboflow serverless API. No network call, no API
key, no workspace/workflow id - just a path to weights on disk.

Everything else in the pipeline (main.py's OCR stage, the crop/upscale
logic, folder handling) is unchanged. The only thing that changed is how
detections are produced: run_local_detection() returns the exact same
list-of-dicts shape (x, y, width, height, class, confidence - center-based
box in pixel coords) that the old extract_predictions() returned from the
Roboflow response, so crop_and_save_detections() below is identical to
before.

SETUP
-----
    pip install ultralytics opencv-python numpy
"""

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Fixed config for local inference
# --------------------------------------------------------------------------
DEFAULT_WEIGHTS = "weights/plate_detector.pt"  # path to your trained .pt file
DEFAULT_DEVICE = None                          # None = ultralytics auto-picks GPU if available, else CPU
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MAX_DIM = 1600  # downscale anything larger than this (longest side)
DEFAULT_MIN_DIM = 416   # upscale anything smaller than this (longest side)


class LocalInferenceError(RuntimeError):
    """Raised when local model inference fails."""


def build_model(weights_path: str, device: Optional[str] = None) -> YOLO:
    """
    Loads a local YOLO model from weights_path (a .pt file exported by
    your own training run - e.g. runs/train/weights/best.pt). device can
    be left None to let ultralytics auto-select (GPU if one is visible,
    otherwise CPU), or set explicitly to "cpu", "cuda:0", etc.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Point --weights (or CONFIG['weights'] in main.py) at your trained .pt file."
        )
    model = YOLO(weights_path)
    if device:
        model.to(device)
    return model


# --------------------------------------------------------------------------
# Preprocessing (before detection) - identical to the old Roboflow version
# --------------------------------------------------------------------------
def preprocess_for_detection(
    image_path: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
):
    """
    Light preprocessing applied to the FULL photo before it's sent for
    detection: resize into a sane range, CLAHE contrast on luminance only,
    light denoise. Not to be confused with the separate, heavier
    preprocessing used later for OCR on individual crops.

    Returns a BGR numpy array, or None if the image couldn't be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    longest_side = max(h, w)
    if longest_side > max_dim:
        scale = max_dim / longest_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif longest_side < min_dim:
        scale = min_dim / longest_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    img = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    return img


def prepare_detection_input(
    image_path: str,
    preprocess: bool,
    work_dir: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
) -> str:
    """
    Returns the path that should actually be run through the model: either
    the original image_path unchanged, or a preprocessed copy written into
    work_dir. Falls back to the original on any preprocessing failure so a
    single bad image can't crash a batch run.
    """
    if not preprocess:
        return image_path

    processed = preprocess_for_detection(image_path, max_dim=max_dim, min_dim=min_dim)
    if processed is None:
        print(f"[warn] preprocessing failed, using original image: {image_path}")
        return image_path

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(work_dir, f"{base}_preprocessed.jpg")
    cv2.imwrite(out_path, processed)
    return out_path


# --------------------------------------------------------------------------
# Local model inference
# --------------------------------------------------------------------------
def run_local_detection(
    model: YOLO,
    image_path: str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: Optional[int] = None,
    classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs the local YOLO model on a single image and returns a list of
    prediction dicts shaped exactly like the old Roboflow output:
    {"x": cx, "y": cy, "width": w, "height": h, "class": name, "confidence": conf}
    (x/y/width/height are center-based, in pixel coordinates of image_path).

    classes, if given, is a set of class-name strings to keep - anything
    else the model detects (e.g. other object types your model was also
    trained on) is filtered out here so only your target objects get
    cropped downstream. Leave it None to keep every detected class.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    predict_kwargs: Dict[str, Any] = dict(
        source=image_path, conf=conf_threshold, iou=iou_threshold, verbose=False,
    )
    if imgsz:
        predict_kwargs["imgsz"] = imgsz

    try:
        results = model.predict(**predict_kwargs)
    except Exception as e:  # noqa: BLE001 - surface as our own error type
        raise LocalInferenceError(f"Local model inference failed on {image_path}: {e}") from e

    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names  # {class_index: class_name}
    xywh = boxes.xywh.cpu().numpy()      # Nx4: center-x, center-y, width, height (pixels, original image scale)
    confs = boxes.conf.cpu().numpy()
    cls_idxs = boxes.cls.cpu().numpy().astype(int)

    predictions: List[Dict[str, Any]] = []
    for (cx, cy, bw, bh), conf, cls_idx in zip(xywh, confs, cls_idxs):
        cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
        if classes and cls_name not in classes:
            continue
        predictions.append({
            "x": float(cx),
            "y": float(cy),
            "width": float(bw),
            "height": float(bh),
            "class": cls_name,
            "confidence": float(conf),
        })

    return predictions


def run_local_tracking(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: Optional[int] = None,
    classes: Optional[Set[str]] = None,
    tracker: str = "bytetrack.yaml",
) -> List[Dict[str, Any]]:
    """Run persistent Ultralytics tracking on one in-memory video frame."""
    track_kwargs: Dict[str, Any] = dict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        tracker=tracker,
        persist=True,
        verbose=False,
    )
    if imgsz:
        track_kwargs["imgsz"] = imgsz

    try:
        results = model.track(**track_kwargs)
    except Exception as e:  # noqa: BLE001 - surface as our own error type
        raise LocalInferenceError(f"Local model tracking failed: {e}") from e

    if not results:
        return []
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    xywh = boxes.xywh.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_idxs = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xywh)

    predictions: List[Dict[str, Any]] = []
    for (cx, cy, bw, bh), conf, cls_idx, track_id in zip(xywh, confs, cls_idxs, track_ids):
        cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
        if classes and cls_name not in classes:
            continue
        prediction = {
            "x": float(cx),
            "y": float(cy),
            "width": float(bw),
            "height": float(bh),
            "class": cls_name,
            "confidence": float(conf),
        }
        if track_id is not None:
            prediction["track_id"] = f"stream_v{int(track_id)}"
        predictions.append(prediction)
    return predictions


# --------------------------------------------------------------------------
# Cropping - identical to the old Roboflow version (box shape is the same)
# --------------------------------------------------------------------------
def crop_and_save_detections(
    image_path: str,
    predictions: List[Dict[str, Any]],
    out_dir: str,
    prefix: str,
    padding_ratio: float = 0.15,
    min_crop_height: int = 120,
    output_extension: str = ".jpg",
    simple_name: bool = False,
) -> List[str]:
    """
    Crops each detected object out of image_path (using center x/y +
    width/height boxes, clamped to image bounds) and saves it as its own
    file in out_dir. Returns the list of saved paths.

    padding_ratio expands each box by this fraction of its own width/height
    on every side before cropping (clamped to image bounds). A tight, exact
    bounding box will sometimes shave a pixel or two off the leftmost/
    rightmost character, which is enough to make a plate that's perfectly
    readable to a human unrecognizable to OCR. Default 0.15 (15%); set to 0
    to disable.

    min_crop_height upscales (INTER_CUBIC) any crop shorter than this,
    preserving aspect ratio, before saving - zooms in once on the sharpest
    available pixels (the detection input, not a re-compressed crop) so the
    OCR stage has more real detail to work with. Set to 0 to disable.

    IMPORTANT: image_path here must be whatever image was actually run
    through the model (preprocessed or original) - detection coordinates
    are relative to that image, not necessarily the original file on disk.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warn] could not read image for cropping: {image_path}")
        return []

    h, w = image.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, p in enumerate(predictions, start=1):
        cx, cy, bw, bh = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        if None in (cx, cy, bw, bh):
            continue

        pad_w = bw * padding_ratio
        pad_h = bh * padding_ratio
        x1 = max(int(cx - bw / 2 - pad_w), 0)
        y1 = max(int(cy - bh / 2 - pad_h), 0)
        x2 = min(int(cx + bw / 2 + pad_w), w)
        y2 = min(int(cy + bh / 2 + pad_h), h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        if min_crop_height and crop_h < min_crop_height:
            scale = min_crop_height / crop_h
            crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

        cls = str(p.get("class") or "object")
        safe_cls = "".join(c if c.isalnum() else "_" for c in cls)
        conf = p.get("confidence")
        conf_str = f"{conf * 100:.0f}" if isinstance(conf, (int, float)) else "NA"

        if simple_name:
            base_name = f"{prefix}_crop" if i == 1 else f"{prefix}_crop_{i}"
            out_name = f"{base_name}{output_extension}"
            counter = 2
            while os.path.exists(os.path.join(out_dir, out_name)):
                out_name = f"{base_name}_{counter}{output_extension}"
                counter += 1
        else:
            # Keep the full original filename, then tag the index/class/
            # confidence so the crop can be traced back to its source.
            out_name = f"{prefix}_cropped_{i}_{safe_cls}_{conf_str}{output_extension}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, crop)
        saved_paths.append(out_path)

    return saved_paths


def process_single_image(
    model: YOLO,
    image_path: str,
    out_dir: str,
    preprocess: bool,
    work_dir: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
    crop_padding: float = 0.15,
    min_crop_height: int = 120,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: Optional[int] = None,
    target_classes: Optional[Set[str]] = None,
) -> int:
    """
    Runs the local model on one image and crops+saves any detections.
    Returns crop count. Prints one short status line per source image:
        <filename> -> cropped (N)     - N crop(s) saved
        <filename> -> check (reason)  - nothing usable came out of it,
                                         worth a manual look
    """
    filename = os.path.basename(image_path)
    detection_input_path = prepare_detection_input(image_path, preprocess, work_dir, max_dim, min_dim)

    try:
        predictions = run_local_detection(
            model, detection_input_path, conf_threshold, iou_threshold, imgsz, target_classes,
        )
    except (FileNotFoundError, LocalInferenceError) as e:
        print(f"{filename} -> check (error: {e})")
        return 0

    if not predictions:
        print(f"{filename} -> check (no detections)")
        return 0

    # Crop names are based on the ORIGINAL filename for readability, even
    # though pixels come from the (possibly preprocessed) detection input.
    prefix = os.path.splitext(filename)[0]
    saved_paths = crop_and_save_detections(
        detection_input_path, predictions, out_dir, prefix, crop_padding, min_crop_height,
    )

    if not saved_paths:
        print(f"{filename} -> check (detections found but none croppable)")
        return 0

    print(f"{filename} -> cropped ({len(saved_paths)})")
    return len(saved_paths)


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def default_out_dir(image: Optional[str], folder: Optional[str], dir_name: str = "cropped_detections") -> str:
    """
    Builds a default output directory that lives INSIDE the source, as a
    child folder:

    --folder /data/Batch1        -> /data/Batch1/cropped_detections
    --image  /data/Batch1/plate.jpg -> /data/Batch1/cropped_detections
    """
    if folder:
        sample_folder = os.path.abspath(folder.rstrip("/\\"))
    else:
        sample_folder = os.path.abspath(os.path.dirname(image) or ".")

    return os.path.join(sample_folder, dir_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run a local YOLO model (.pt weights) on an image (or a folder of images) "
                    "and crop out detections."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a single local image file.")
    source_group.add_argument("--folder", help="Path to a folder of images to process in a batch.")

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                         help=f"Path to your trained .pt weights file (default {DEFAULT_WEIGHTS}).")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                         help="Device to run on: 'cpu', 'cuda:0', etc. Default: let ultralytics auto-pick.")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                         help=f"Minimum detection confidence to keep (default {DEFAULT_CONF_THRESHOLD}).")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                         help=f"NMS IoU threshold (default {DEFAULT_IOU_THRESHOLD}).")
    parser.add_argument("--imgsz", type=int, default=None,
                         help="Inference image size fed to the model. Default: model's own default.")
    parser.add_argument("--classes", default=None,
                         help="Comma-separated class names to keep (e.g. 'license_plate'). "
                              "Default: keep every class the model detects.")
    parser.add_argument("--out-dir", default=None,
                         help="Directory to save cropped detection images into. Defaults to "
                              "'cropped_detections' inside the source folder (or the image's "
                              "containing folder for --image).")
    parser.add_argument("--no-preprocess", action="store_true",
                         help="Send the raw image as-is; skip contrast/denoise/resize preprocessing.")
    parser.add_argument("--keep-preprocessed", action="store_true",
                         help="Save the preprocessed full images into '<out-dir>_preprocessed_full' "
                              "instead of discarding them after each run.")
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                         help=f"Downscale images larger than this on the longest side (default {DEFAULT_MAX_DIM}).")
    parser.add_argument("--min-dim", type=int, default=DEFAULT_MIN_DIM,
                         help=f"Upscale images smaller than this on the longest side (default {DEFAULT_MIN_DIM}).")
    parser.add_argument("--crop-padding", type=float, default=0.15,
                         help="Expand each detection box by this fraction of its own width/height on "
                              "every side before cropping (default 0.15 = 15%%). Set to 0 to disable.")
    parser.add_argument("--crop-min-height", type=int, default=120,
                         help="Upscale (zoom in on) any crop shorter than this many pixels before "
                              "saving (default 120). Set to 0 to disable.")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = default_out_dir(args.image, args.folder)
        print(f"[info] --out-dir not set, defaulting to: {args.out_dir}")

    model = build_model(args.weights, args.device)
    preprocess = not args.no_preprocess
    target_classes = set(c.strip() for c in args.classes.split(",") if c.strip()) if args.classes else None

    if args.keep_preprocessed:
        work_dir = args.out_dir.rstrip("/\\") + "_preprocessed_full"
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = tempfile.mkdtemp(prefix="local_preprocess_")
        cleanup_work_dir = True

    try:
        if args.folder:
            if not os.path.isdir(args.folder):
                sys.exit(f"Error: folder not found: {args.folder}")

            image_paths = list_images_in_folder(args.folder)
            if not image_paths:
                sys.exit(f"Error: no images ({sorted(IMAGE_EXTENSIONS)}) found in: {args.folder}")

            print(f"Found {len(image_paths)} image(s) in {args.folder}. "
                  f"Preprocessing: {'on' if preprocess else 'off'}. Running local model on each...")

            total_crops = 0
            for image_path in image_paths:
                total_crops += process_single_image(
                    model, image_path, args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                    args.crop_padding, args.crop_min_height, args.conf_threshold, args.iou_threshold,
                    args.imgsz, target_classes,
                )

            print(f"\n=== Done: {total_crops} object(s) cropped from {len(image_paths)} "
                  f"image(s), saved to '{args.out_dir}' ===")

        else:
            print(f"Running local model on image: {args.image} "
                  f"(preprocessing: {'on' if preprocess else 'off'})")
            total_crops = process_single_image(
                model, args.image, args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                args.crop_padding, args.crop_min_height, args.conf_threshold, args.iou_threshold,
                args.imgsz, target_classes,
            )
            print(f"\n=== Done: {total_crops} object(s) cropped, saved to '{args.out_dir}' ===")
    finally:
        if cleanup_work_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
