"""Lean vehicle detection flow optimized for Raspberry Pi:

    vehicle detect → crop → plate detect → crop → send metadata
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import detector
from core.geometry import box_edges, crop_box
from modes.base import DetectionMode, RuntimeContext
from modes.vehicle.geometry import crop_plate, is_inside_roi
from storage.buffer import VehicleRecord, encode_jpeg

log = logging.getLogger("pipeline")


class VehicleMode(DetectionMode):
    """Lean vehicle -> plate detection pipeline."""

    name = "vehicle"

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = ctx.config.vehicle
        self.buffer = ctx.buffer
        self.roi = ctx.roi
        self.camera_source = ctx.camera_source

        self.vehicle_model = None
        self.plate_model = None
        self._id_counter = 0
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stopped = False

        try:
            self._load()
        except Exception:
            self.stop()
            raise

    def _load(self) -> None:
        runtime = self.ctx.config.runtime
        config = self.config
        capped_threads = detector.limit_onnx_threads(runtime.inference_threads)
        if capped_threads:
            print(f"[vehicle] ONNX Runtime limited to {capped_threads} CPU thread(s)")

        print("[vehicle] loading vehicle model...")
        self.vehicle_model = detector.load_model(config.model.vehicle_weights, runtime.device)
        print("[vehicle] loading plate model...")
        self.plate_model = detector.load_model(config.model.plate_weights, runtime.device)
        self._sync_imgsz_with_model("vehicle", config.model.vehicle_weights, "vehicle_imgsz")
        self._sync_imgsz_with_model("plate", config.model.plate_weights, "plate_imgsz")

        classes = config.model.vehicle_classes
        self.vehicle_classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[vehicle] warming up models...")
        region = self._get_detect_region(frame_shape) if frame_shape else None
        if region is not None:
            frame_shape = (region[3] - region[1], region[2] - region[0])
        detector.warmup_model(self.vehicle_model, self.config.model.vehicle_imgsz, frame_shape)
        detector.warmup_model(self.plate_model, self.config.model.plate_imgsz)

    def _sync_imgsz_with_model(self, label: str, weights: str, attribute: str) -> None:
        actual = detector.onnx_static_input_hw(weights)
        if actual is None:
            return
        configured = detector.imgsz_hw(getattr(self.config.model, attribute))
        if actual != configured:
            log.warning("[vehicle] %s model %s static shape %dx%d != configured %dx%d; using %dx%d",
                        label, weights, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
            setattr(self.config.model, attribute, [actual[0], actual[1]])

    def _get_detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        features = self.config.features
        if not (features.roi_crop_detect and features.roi_filter):
            return None
        return self.roi.detect_region(
            frame_shape, detector.imgsz_hw(self.config.model.vehicle_imgsz),
            self.config.roi.crop_margin, label="vehicle model",
        )

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        frame_shape = frame.shape[:2]
        cfg = self.config
        features = cfg.features

        # 1. Vehicle detect
        region = self._get_detect_region(frame_shape)
        predictions = detector.detect(
            self.vehicle_model, frame,
            cfg.model.vehicle_conf_threshold, cfg.model.vehicle_iou_threshold,
            cfg.model.vehicle_imgsz, self.vehicle_classes, region=region,
            agnostic_nms=cfg.model.agnostic_nms,
        )
        predictions = [p for p in predictions if not self._looks_like_plate(p)]

        # Filter to ROI
        if features.roi_filter:
            polygon = self.roi.get_polygon()
            roi_cfg = cfg.roi
            predictions = [
                p for p in predictions
                if is_inside_roi(
                    p, frame_shape, polygon,
                    roi_cfg.min_width_ratio, roi_cfg.min_height_ratio,
                    roi_cfg.max_width_ratio, roi_cfg.max_height_ratio,
                    roi_cfg.edge_margin_ratio,
                )
            ]

        detected_count = len(predictions)
        crop_cfg = cfg.crop
        now = datetime.now()

        for prediction in predictions:
            self._id_counter += 1
            track_id = f"v{self._id_counter}"
            stem = f"{self._session_tag}_{track_id}"

            # 2. Crop vehicle
            vehicle_crop = crop_box(
                frame, prediction,
                crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height,
            )
            if vehicle_crop is None:
                continue

            # 3. Plate detect
            plate_detected = False
            plate_confidence: Optional[float] = None
            plate_crop: Optional[np.ndarray] = None
            plate_x1 = plate_y1 = plate_x2 = plate_y2 = None

            if features.plate_detection:
                plate_predictions = detector.detect(
                    self.plate_model, vehicle_crop,
                    cfg.model.plate_conf_threshold, cfg.model.vehicle_iou_threshold,
                    cfg.model.plate_imgsz,
                )
                if plate_predictions:
                    best_plate = max(plate_predictions, key=lambda p: p.get("confidence", 0.0))
                    plate_confidence = float(best_plate.get("confidence", 0.0))
                    plate_x1, plate_y1, plate_x2, plate_y2 = box_edges(best_plate)

                    # 4. Crop plate
                    plate_crop = crop_plate(
                        vehicle_crop, best_plate,
                        crop_cfg.plate_padding_pixels, crop_cfg.plate_min_crop_height,
                        crop_cfg.plate_padding_ratio,
                    )
                    if plate_crop is not None:
                        plate_detected = True

            # Encode JPEGs
            disk_files: Optional[Dict[str, bytes]] = {} if features.save_images_to_disk else None
            vehicle_jpeg = encode_jpeg(vehicle_crop, cfg.images.jpeg_quality)
            if disk_files is not None:
                disk_files[f"vehicle_detection/{stem}_{self._safe_name(prediction.get('class'))}.jpg"] = vehicle_jpeg

            plate_jpeg = None
            if plate_detected and plate_crop is not None:
                plate_jpeg = encode_jpeg(plate_crop, cfg.images.plate_jpeg_quality)
                if disk_files is not None:
                    disk_files[f"plate_detection/{stem}_plate.jpg"] = plate_jpeg

            vx1, vy1, vx2, vy2 = box_edges(prediction)

            record = VehicleRecord(
                track_id=track_id,
                camera_source=self.camera_source,
                vehicle_class=str(prediction.get("class") or "vehicle"),
                vehicle_confidence=float(prediction.get("confidence", 0.0)),
                vehicle_image_jpeg=vehicle_jpeg,
                vehicle_box_x1=vx1,
                vehicle_box_y1=vy1,
                vehicle_box_x2=vx2,
                vehicle_box_y2=vy2,
                plate_detected=plate_detected,
                plate_confidence=plate_confidence,
                plate_image_jpeg=plate_jpeg,
                plate_box_x1=plate_x1,
                plate_box_y1=plate_y1,
                plate_box_x2=plate_x2,
                plate_box_y2=plate_y2,
                detected_at=now,
                disk_files=disk_files,
            )

            # 5. Send metadata (enqueue to shared SQLite buffer)
            self.buffer.enqueue(record)

            if features.print_console:
                status = f"plate found ({plate_confidence:.2f})" if plate_detected else "no plate"
                print(f"[vehicle] {track_id} {prediction.get('class', 'vehicle')} ({prediction.get('confidence', 0):.2f}) - {status}")

        return detected_count

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        models = [self.vehicle_model, self.plate_model]
        self.vehicle_model = self.plate_model = None
        detector.release_models(*models)
        log.info("[vehicle] stopped, models released")

    @staticmethod
    def _looks_like_plate(prediction: Dict[str, Any]) -> bool:
        label = "".join(ch for ch in str(prediction.get("class") or "").lower() if ch.isalnum())
        return any(marker in label for marker in ("plate", "license", "licence"))

    @staticmethod
    def _safe_name(value: Optional[str]) -> str:
        value = value or "vehicle"
        return "".join(ch if ch.isalnum() else "_" for ch in value)
