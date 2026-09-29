import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

log = logging.getLogger("pipeline")

# ---- box stabiliser -------------------------------------------------------------------------
# ByteTrack reports its own smoothed (Kalman) box. When a track gets matched to a wrong / merged
# detection that box can suddenly become huge (or tiny) for a single update. A real vehicle only
# changes size gradually, so the size of each track may change by at most this factor per update.
MAX_SIZE_STEP = 1.35        # width/height may grow at most 35 % per update
MAX_SHRINK_STEP = 2.0       # ...but may SHRINK faster: an oversized first box must settle quickly
# ByteTrack (config/bytetrack_custom.yaml: track_buffer) keeps a lost track alive internally for up
# to `track_buffer` frames before dropping it, and a re-matched track keeps its v-number. This must
# stay >= track_buffer / camera.max_fps, or a track that reappears after a longer-than-that gap skips
# the clamp entirely ("starts fresh") right when it's most likely to reappear with a bad size (its
# Kalman box was coasting, uncorrected, the whole time it was lost). 30 frames / 10 fps = 3s; kept
# with margin.
SIZE_MEMORY_SECONDS = 4.0
SIZE_PRUNE_SECONDS = 30.0
MAX_FRAME_FRACTION = 0.80   # a box wider/taller than this share of the frame is never a single vehicle

# ---- tracker lag ------------------------------------------------------------------------------
# Stock ByteTrack trusts its constant-velocity model much more than the newest detection, so the
# reported box trails a moving vehicle. A larger velocity noise makes it follow the detector faster
# (stock is 1/160). Set to None to leave ultralytics untouched.
KALMAN_VELOCITY_WEIGHT: Optional[float] = 1.0 / 25


def tune_tracker_kalman(velocity_weight: Optional[float] = KALMAN_VELOCITY_WEIGHT) -> bool:
    """Make ByteTrack follow the detector more closely. Safe no-op if ultralytics' internals differ."""
    if not velocity_weight:
        return False
    try:
        from ultralytics.trackers import byte_tracker
        from ultralytics.trackers.utils import kalman_filter
        filter_cls = kalman_filter.KalmanFilterXYAH
        if getattr(filter_cls, "_lag_tuned", False):
            return True

        original_init = filter_cls.__init__

        def tuned_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._std_weight_velocity = velocity_weight

        filter_cls.__init__ = tuned_init
        filter_cls._lag_tuned = True
        shared = getattr(byte_tracker.STrack, "shared_kalman", None)  # already built at import time
        if shared is not None:
            shared._std_weight_velocity = velocity_weight
        return True
    except Exception as error:  # different ultralytics version - keep stock behaviour
        log.warning("[detector] could not tune the tracker's Kalman filter (%s) - using stock ByteTrack", error)
        return False


class BoxStabilizer:
    """Per-track size gate: stops one bad update from turning a small vehicle into a giant box."""

    def __init__(self, max_step: float = MAX_SIZE_STEP, memory_seconds: float = SIZE_MEMORY_SECONDS):
        self.max_step = max_step
        self.memory_seconds = memory_seconds
        self._last: Dict[str, Tuple[float, float, float]] = {}   # track_id -> (w, h, time)
        self._last_prune = time.monotonic()

    def apply(self, prediction: Dict[str, Any], frame_w: float, frame_h: float, now: float) -> bool:
        """Clamp the prediction in place. Returns False if it should be dropped entirely."""
        width, height = prediction["width"], prediction["height"]
        if width <= 1 or height <= 1:
            return False
        if width > MAX_FRAME_FRACTION * frame_w or height > MAX_FRAME_FRACTION * frame_h:
            return False

        track_id = prediction.get("track_id")
        if track_id is None:
            return True

        previous = self._last.get(track_id)
        if previous is not None and now - previous[2] <= self.memory_seconds:
            prev_w, prev_h = previous[0], previous[1]
            new_w = min(max(width, prev_w / MAX_SHRINK_STEP), prev_w * self.max_step)
            new_h = min(max(height, prev_h / MAX_SHRINK_STEP), prev_h * self.max_step)
            if new_w != width or new_h != height:
                log.debug("[detector] %s size %.0fx%.0f -> %.0fx%.0f (was %.0fx%.0f)",
                          track_id, width, height, new_w, new_h, prev_w, prev_h)
                prediction["width"], prediction["height"] = new_w, new_h
                width, height = new_w, new_h
        self._last[track_id] = (width, height, now)

        if now - self._last_prune > SIZE_PRUNE_SECONDS:
            self._last_prune = now
            self._last = {t: v for t, v in self._last.items() if now - v[2] <= SIZE_PRUNE_SECONDS}
        return True


_stabilizer = BoxStabilizer()


class InferenceError(RuntimeError):
    pass


def is_ncnn_export(path: str) -> bool:
    return os.path.isdir(path) and any(name.endswith(".ncnn.param") for name in os.listdir(path))


def weights_exist(path: str) -> bool:
    return os.path.isfile(path) or is_ncnn_export(path)


def resolve_inference_threads(configured: int) -> int:
    if configured > 0:
        return configured
    return max(2, (os.cpu_count() or 4) - 1)


def normalize_imgsz(imgsz: "int | list | tuple") -> "int | List[int]":
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0])
        return [int(imgsz[0]), int(imgsz[1])]
    return int(imgsz)


def imgsz_hw(imgsz: "int | list | tuple") -> Tuple[int, int]:
    size = normalize_imgsz(imgsz)
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def limit_onnx_threads(configured: int) -> Optional[int]:
    if configured < 0:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    threads = resolve_inference_threads(configured)
    original_init = ort.InferenceSession.__init__
    if getattr(original_init, "_thread_capped", False):
        original_init = original_init._original

    def capped_init(self, path_or_bytes, sess_options=None, *args, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        if not sess_options.intra_op_num_threads:
            sess_options.intra_op_num_threads = threads
        try:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        original_init(self, path_or_bytes, sess_options, *args, **kwargs)

    capped_init._thread_capped = True
    capped_init._original = original_init
    ort.InferenceSession.__init__ = capped_init
    return threads


def onnx_static_input_hw(weights_path: str) -> Optional[Tuple[int, int]]:
    if not (os.path.isfile(weights_path) and weights_path.lower().endswith(".onnx")):
        return None
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(weights_path, options, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width
    except Exception:
        pass
    return None


def load_model(weights_path: str, device: Optional[str] = None) -> YOLO:
    if not weights_exist(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            f"Point config.model.vehicle_weights / plate_weights at a .pt/.onnx file, "
            f"or an exported *_ncnn_model folder."
        )

    if os.path.isfile(weights_path):
        model = YOLO(weights_path)
        if device:
            model.to(device)
    else:
        model = YOLO(weights_path, task="detect")

    return model


def warmup_model(model: YOLO, imgsz: "int | list | tuple", frame_shape: Optional[Tuple[int, int]] = None) -> None:
    height, width = frame_shape if frame_shape else imgsz_hw(imgsz)
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        model.predict(source=dummy_frame, imgsz=normalize_imgsz(imgsz), verbose=False)
    except Exception as error:
        print(f"[warn] model warm-up failed (continuing anyway): {error}")


def _boxes_to_predictions(result: Any, classes: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    class_names = result.names
    xywh = boxes.xywh.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_indexes = boxes.cls.cpu().numpy().astype(int)

    predictions: List[Dict[str, Any]] = []
    for (cx, cy, width, height), confidence, class_index in zip(xywh, confidences, class_indexes):
        class_name = class_names.get(int(class_index), str(class_index)) if isinstance(class_names, dict) else str(class_index)
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
    if not images:
        return []

    valid_indexes = [i for i, image in enumerate(images) if isinstance(image, np.ndarray) and image.size > 0]
    if not valid_indexes:
        return [[] for _ in images]

    batch_images = [images[i] for i in valid_indexes]
    results = None
    if len(batch_images) == 1 or not getattr(model, "_batch_predict_unsupported", False):
        try:
            results = model.predict(source=batch_images, conf=conf_threshold, iou=iou_threshold, imgsz=normalize_imgsz(imgsz), verbose=False)
        except Exception as error:
            if len(batch_images) == 1:
                raise InferenceError(f"Batch detection failed on 1 image: {error}") from error
            model._batch_predict_unsupported = True

    if results is None:
        try:
            results = []
            for image in batch_images:
                results.extend(model.predict(source=[image], conf=conf_threshold, iou=iou_threshold, imgsz=normalize_imgsz(imgsz), verbose=False))
        except Exception as error:
            raise InferenceError(f"Batch detection failed on {len(batch_images)} image(s): {error}") from error

    predictions_by_result = [_boxes_to_predictions(result, classes) for result in results]

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
    detect_conf: Optional[float] = None,
    agnostic_nms: bool = False,
) -> List[Dict[str, Any]]:
    tune_tracker_kalman()
    source = frame
    offset_x = offset_y = 0
    if region is not None:
        region_x1, region_y1, region_x2, region_y2 = region
        if region_x2 > region_x1 and region_y2 > region_y1:
            source = np.ascontiguousarray(frame[region_y1:region_y2, region_x1:region_x2])
            offset_x, offset_y = region_x1, region_y1

    try:
        floor = min(conf_threshold, detect_conf) if detect_conf else conf_threshold
        results = model.track(
            source=source, conf=floor, iou=iou_threshold, imgsz=normalize_imgsz(imgsz),
            tracker=tracker_config, persist=True, agnostic_nms=agnostic_nms, verbose=False,
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
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xywh)

    now = time.monotonic()
    predictions: List[Dict[str, Any]] = []
    for (cx, cy, width, height), confidence, class_index, track_id in zip(xywh, confidences, class_indexes, track_ids):
        class_name = class_names.get(int(class_index), str(class_index)) if isinstance(class_names, dict) else str(class_index)
        if classes and class_name not in classes:
            continue
        if float(confidence) < conf_threshold:
            continue
        prediction = {
            "x": float(cx) + offset_x, "y": float(cy) + offset_y,
            "width": float(width), "height": float(height),
            "class": class_name, "confidence": float(confidence),
        }
        if track_id is not None:
            prediction["track_id"] = f"v{int(track_id)}"
        if not _stabilizer.apply(prediction, frame.shape[1], frame.shape[0], now):
            continue
        predictions.append(prediction)

    return predictions


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files
