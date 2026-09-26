"""
detector.py
-----------
Thin wrapper around Ultralytics YOLO: loading a model (.pt, .onnx, or an
exported NCNN folder), running detection on a batch of images in one
forward pass, and running ByteTrack tracking on a video frame.

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


def is_ncnn_export(path: str) -> bool:
    """True if `path` is a folder containing an exported NCNN model (.ncnn.param + .ncnn.bin)."""
    return os.path.isdir(path) and any(name.endswith(".ncnn.param") for name in os.listdir(path))


def weights_exist(path: str) -> bool:
    """True if `path` points at a weights file (.pt / .onnx) or an NCNN export folder."""
    return os.path.isfile(path) or is_ncnn_export(path)


def resolve_inference_threads(configured: int) -> int:
    """0 = auto: every core but one (the last core is left for the RTSP
    decoder + HTTP threads), never below 2. Anything above 0 is used as given.

    A Raspberry Pi 5 has 4 PHYSICAL cores and no SMT, so os.cpu_count() is
    already the physical count. (The old formula halved it as if hyperthreading
    were present, which left inference on only 2 of the 4 cores.)"""
    if configured > 0:
        return configured
    return max(2, (os.cpu_count() or 4) - 1)


def normalize_imgsz(imgsz: "int | list | tuple") -> "int | List[int]":
    """int -> int (square); [h, w] / (h, w) -> [h, w] (rectangular model)."""
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0])
        return [int(imgsz[0]), int(imgsz[1])]
    return int(imgsz)


def imgsz_hw(imgsz: "int | list | tuple") -> Tuple[int, int]:
    """(height, width) of a model input given as an int or an [h, w] pair."""
    size = normalize_imgsz(imgsz)
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def limit_onnx_threads(configured: int) -> Optional[int]:
    """Cap how many CPU threads ONNX Runtime may use, for every model loaded
    AFTER this call. Returns the cap applied, or None if nothing was changed.

    Why: a 1280x1280 ONNX model on CPU keeps every core busy for over a second
    per frame. The RTSP decoder and the live-view HTTP threads share those same
    cores, so the video freezes during each inference and then jumps forward.
    Leaving a few cores free keeps decoding and streaming steady, at the cost of
    somewhat slower inference. `configured < 0` leaves ONNX Runtime untouched.

    Ultralytics builds its ONNX Runtime session internally with no thread
    option, so the cap is applied by wrapping InferenceSession's constructor.
    """
    if configured < 0:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None          # NCNN / .pt models don't use ONNX Runtime

    threads = resolve_inference_threads(configured)
    original_init = ort.InferenceSession.__init__
    if getattr(original_init, "_thread_capped", False):
        original_init = original_init._original      # re-configuring: don't stack wrappers

    def capped_init(self, path_or_bytes, sess_options=None, *args, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        if not sess_options.intra_op_num_threads:     # respect an explicit choice
            sess_options.intra_op_num_threads = threads
        try:
            # idle worker threads otherwise busy-wait, burning the cores we want free
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        original_init(self, path_or_bytes, sess_options, *args, **kwargs)

    capped_init._thread_capped = True
    capped_init._original = original_init
    ort.InferenceSession.__init__ = capped_init
    return threads


def onnx_static_input_hw(weights_path: str) -> Optional[Tuple[int, int]]:
    """(height, width) baked into a static-shape .onnx file, or None if the file is not
    an .onnx / has a dynamic input / can't be inspected.

    A static export accepts ONLY that size. The config's *_imgsz values used to have to be
    kept in sync with the file by hand, and when they drifted every inference call failed
    with "Got invalid dimensions for input" (which the pipeline then stored as "no plate")."""
    if not (os.path.isfile(weights_path) and weights_path.lower().endswith(".onnx")):
        return None
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # we only want the shape: skip the optimizer
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(weights_path, options, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape                      # [batch, 3, H, W]
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width
    except Exception:
        pass
    return None


def load_model(weights_path: str, device: Optional[str] = None) -> YOLO:
    """Load a YOLO model, whether it's a raw .pt/.onnx file or an NCNN export.

    NCNN exports run noticeably faster on CPU-only hardware like a
    Raspberry Pi - see export_ncnn.py / the *_ncnn_model folders under
    models/ (exported via `YOLO(...).export(format="ncnn")`).
    """
    # fail loudly and early with a clear message if the path is wrong,
    # instead of letting Ultralytics raise a confusing internal error later
    if not weights_exist(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            f"Point config.model.vehicle_weights / plate_weights at a .pt/.onnx file, "
            f"or an exported *_ncnn_model folder."
        )

    if os.path.isfile(weights_path):
        # a .pt checkpoint can be moved to a GPU device if one is configured
        model = YOLO(weights_path)
        if device:
            model.to(device)
    else:
        # NCNN exports are CPU-optimized and ignore the `device` setting
        model = YOLO(weights_path, task="detect")

    return model


def warmup_model(model: YOLO, imgsz: "int | list | tuple", frame_shape: Optional[Tuple[int, int]] = None) -> None:
    """Run one throwaway inference so the real, first camera frame isn't slowed
    down by model/graph initialization costs."""
    # use the real capture resolution if we know it, otherwise fall back to imgsz
    height, width = frame_shape if frame_shape else imgsz_hw(imgsz)
    # a black frame is enough - we only care about paying the startup cost
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        model.predict(source=dummy_frame, imgsz=normalize_imgsz(imgsz), verbose=False)
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

    # A static-shape export (the default from export_ncnn.py, and most .onnx
    # exports) is compiled for exactly one input shape, e.g. [1, 3, 640, 640] -
    # batch size baked in as 1. Handing it more than one image at once then
    # fails with a shape-mismatch error. `_batch_predict_unsupported` remembers
    # that this particular model can't be batched, so we don't pay for a
    # failed attempt on every single frame afterwards.
    batch_images = [images[i] for i in valid_indexes]
    results = None
    if len(batch_images) == 1 or not getattr(model, "_batch_predict_unsupported", False):
        try:
            results = model.predict(source=batch_images, conf=conf_threshold, iou=iou_threshold, imgsz=normalize_imgsz(imgsz), verbose=False)
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
                results.extend(model.predict(source=[image], conf=conf_threshold, iou=iou_threshold, imgsz=normalize_imgsz(imgsz), verbose=False))
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
    imgsz: "int | list | tuple",
    tracker_config: str,
    classes: Optional[Set[str]] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> List[Dict[str, Any]]:
    """Run persistent ByteTrack tracking on one live video frame.

    `persist=True` tells Ultralytics to remember track state between calls,
    which is what gives each vehicle a stable ID across frames instead of a
    fresh ID every time.

    `region` = (x1, y1, x2, y2) in full-frame pixels. When given, ONLY that
    window is fed to the model (the ROI + margin; see
    geometry.compute_detect_region) and the returned boxes are shifted back
    into FULL-FRAME pixels, so callers never know a crop happened. ByteTrack
    sees crop coordinates, which differ from frame coordinates by a constant
    offset, so tracking is unaffected.
    """
    source = frame
    offset_x = offset_y = 0
    if region is not None:
        region_x1, region_y1, region_x2, region_y2 = region
        if region_x2 > region_x1 and region_y2 > region_y1:
            # contiguous copy: a slice is a strided view, and OpenCV/ONNX preprocessing wants a dense array
            source = np.ascontiguousarray(frame[region_y1:region_y2, region_x1:region_x2])
            offset_x, offset_y = region_x1, region_y1

    try:
        results = model.track(
            source=source, conf=conf_threshold, iou=iou_threshold, imgsz=normalize_imgsz(imgsz),
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
            "x": float(cx) + offset_x, "y": float(cy) + offset_y,
            "width": float(width), "height": float(height),
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
