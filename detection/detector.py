"""
detector.py
-----------
Thin wrapper around Ultralytics YOLO: loading a model (.pt or exported
OpenVINO folder), running detection on a single image, running detection on
a batch of images in one forward pass, and running ByteTrack tracking on a
video frame.

This is the ONLY file that talks to the `ultralytics` library directly -
every other module deals in plain dicts/numpy arrays. That means if the
underlying model library ever changes, this is the only file that needs
to change with it.
"""

# os is used for simple filesystem checks (is this a .pt file or a folder?)
import os
# typing hints document exactly what shape of data every function expects
from typing import Any, Dict, List, Optional, Set, Tuple

# numpy is how every image/array is represented in this project
import numpy as np
# the actual model class from Ultralytics
from ultralytics import YOLO

# supported image file extensions for --folder mode
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InferenceError(RuntimeError):
    """Raised whenever a model forward-pass fails, so callers can catch just this."""


def is_openvino_export(path: str) -> bool:
    """True if `path` is a folder containing an exported OpenVINO model (.xml + .bin)."""
    return os.path.isdir(path) and any(name.endswith(".xml") for name in os.listdir(path))


def weights_exist(path: str) -> bool:
    """True if `path` points at either a .pt file or an OpenVINO export folder."""
    return os.path.isfile(path) or is_openvino_export(path)


def load_model(weights_path: str, device: Optional[str] = None) -> YOLO:
    """Load a YOLO model, whether it's a raw .pt checkpoint or an OpenVINO export.

    OpenVINO exports run noticeably faster on CPU-only hardware like a
    Raspberry Pi - see export_openvino.py to create one from a .pt file.
    """
    # fail loudly and early with a clear message if the path is wrong,
    # instead of letting Ultralytics raise a confusing internal error later
    if not weights_exist(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            f"Point config.model.vehicle_weights / plate_weights at a .pt file "
            f"or an exported *_openvino_model folder."
        )

    if os.path.isfile(weights_path):
        # a .pt checkpoint can be moved to a GPU device if one is configured
        model = YOLO(weights_path)
        if device:
            model.to(device)
    else:
        # OpenVINO exports are CPU-optimized and ignore the `device` setting
        model = YOLO(weights_path, task="detect")

    return model


def warmup_model(model: YOLO, imgsz: int, frame_shape: Optional[Tuple[int, int]] = None) -> None:
    """Run one throwaway inference so the real, first camera frame isn't slowed
    down by model/graph initialization costs."""
    # use the real capture resolution if we know it, otherwise fall back to imgsz
    height, width = frame_shape if frame_shape else (imgsz, imgsz)
    # a black frame is enough - we only care about paying the startup cost
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        model.predict(source=dummy_frame, imgsz=imgsz, verbose=False)
    except Exception as error:  # pragma: no cover - warmup failures are non-fatal
        print(f"[warn] model warm-up failed (continuing anyway): {error}")


def _boxes_to_predictions(result: Any, classes: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Convert one Ultralytics result object into our plain-dict box format."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    class_names = result.names
    # pull every box's center-x, center-y, width, height in one numpy call
    xywh = boxes.xywh.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_indexes = boxes.cls.cpu().numpy().astype(int)

    predictions: List[Dict[str, Any]] = []
    for (cx, cy, width, height), confidence, class_index in zip(xywh, confidences, class_indexes):
        # translate the numeric class index into its human-readable name
        class_name = class_names.get(int(class_index), str(class_index)) if isinstance(class_names, dict) else str(class_index)
        # skip this detection if we were told to only keep certain classes
        if classes and class_name not in classes:
            continue
        predictions.append({
            "x": float(cx),
            "y": float(cy),
            "width": float(width),
            "height": float(height),
            "class": class_name,
            "confidence": float(confidence),
        })

    return predictions


def detect(
    model: YOLO,
    image: "str | np.ndarray",
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Run the model on ONE image (file path or in-memory frame)."""
    try:
        # a single call handles both str paths and numpy arrays transparently
        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
    except Exception as error:
        raise InferenceError(f"Detection failed: {error}") from error

    # predict() always returns a list (one entry per input image); we sent one image
    return _boxes_to_predictions(results[0], classes) if results else []


def detect_batch(
    model: YOLO,
    images: List[np.ndarray],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    classes: Optional[Set[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """Run the model on MANY images in a single forward pass.

    Used for plate detection: instead of calling the plate model once per
    vehicle crop, every vehicle crop from the current frame is sent through
    together, which is much faster on CPU-bound hardware like a Pi.
    """
    if not images:
        return []

    # skip empty/invalid crops but remember their original position so the
    # returned list still lines up 1-to-1 with the `images` list the caller passed in
    valid_indexes = [i for i, image in enumerate(images) if isinstance(image, np.ndarray) and image.size > 0]
    if not valid_indexes:
        return [[] for _ in images]

    # A static-shape OpenVINO export (the default from export_openvino.py) is
    # compiled for exactly one input shape, e.g. [1, 3, 640, 640] - batch
    # size baked in as 1. Handing it more than one image at once then fails
    # with a shape-mismatch error. `_batch_predict_unsupported` remembers
    # that this particular model can't be batched, so we don't pay for a
    # failed attempt on every single frame afterwards.
    batch_images = [images[i] for i in valid_indexes]
    results = None
    if len(batch_images) == 1 or not getattr(model, "_batch_predict_unsupported", False):
        try:
            results = model.predict(source=batch_images, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
        except Exception as error:
            if len(batch_images) == 1:
                raise InferenceError(f"Batch detection failed on 1 image: {error}") from error
            # remember this for next time and fall through to the one-at-a-time path below
            model._batch_predict_unsupported = True

    if results is None:
        # one-at-a-time fallback: slower, but works with a static-shape export
        try:
            results = []
            for image in batch_images:
                results.extend(model.predict(source=[image], conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False))
        except Exception as error:
            raise InferenceError(f"Batch detection failed on {len(batch_images)} image(s): {error}") from error

    predictions_by_result = [_boxes_to_predictions(result, classes) for result in results]

    # rebuild a full-length output list so index i still corresponds to images[i]
    output: List[List[Dict[str, Any]]] = [[] for _ in images]
    for position, original_index in enumerate(valid_indexes):
        output[original_index] = predictions_by_result[position]
    return output


def track(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    tracker_config: str,
    classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Run persistent ByteTrack tracking on one live video frame.

    `persist=True` tells Ultralytics to remember track state between calls,
    which is what gives each vehicle a stable ID across frames instead of a
    fresh ID every time.
    """
    try:
        results = model.track(
            source=frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz,
            tracker=tracker_config, persist=True, verbose=False,
        )
    except Exception as error:
        raise InferenceError(f"Tracking failed: {error}") from error

    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    class_names = result.names
    xywh = boxes.xywh.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_indexes = boxes.cls.cpu().numpy().astype(int)
    # boxes.id is None when ByteTrack hasn't assigned IDs yet (e.g. first frame)
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xywh)

    predictions: List[Dict[str, Any]] = []
    for (cx, cy, width, height), confidence, class_index, track_id in zip(xywh, confidences, class_indexes, track_ids):
        class_name = class_names.get(int(class_index), str(class_index)) if isinstance(class_names, dict) else str(class_index)
        if classes and class_name not in classes:
            continue
        prediction = {
            "x": float(cx), "y": float(cy), "width": float(width), "height": float(height),
            "class": class_name, "confidence": float(confidence),
        }
        if track_id is not None:
            prediction["track_id"] = f"v{int(track_id)}"
        predictions.append(prediction)

    return predictions


def list_images_in_folder(folder: str) -> List[str]:
    """Return every supported image file in `folder`, sorted by filename."""
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files
