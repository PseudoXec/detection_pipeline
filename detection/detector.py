import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
) -> List[Dict[str, Any]]:
    source = frame
    offset_x = offset_y = 0
    if region is not None:
        region_x1, region_y1, region_x2, region_y2 = region
        if region_x2 > region_x1 and region_y2 > region_y1:
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
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files
