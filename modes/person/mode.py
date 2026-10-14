"""Lean person detection flow optimized for Raspberry Pi:

    person detect → crop → face detect → crop → send metadata
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from core import detector
from core.geometry import crop_box
from modes.base import DetectionMode, RuntimeContext
from modes.person.face import YuNetFaceDetector
from modes.person.geometry import is_person_inside_roi
from storage.buffer import PersonRecord, encode_jpeg

log = logging.getLogger("pipeline")


class PersonMode(DetectionMode):
    """Lean person -> face detection pipeline."""

    name = "person"

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = ctx.config
        self.cfg = ctx.config.person
        self.buffer = ctx.buffer
        self.roi = ctx.roi
        self.camera_source = ctx.camera_source

        self.person_model = None
        self.face_detector: Optional[YuNetFaceDetector] = None
        self._id_counter = 0
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stopped = False

        try:
            self._load()
        except Exception:
            self.stop()
            raise

    def _load(self) -> None:
        cfg, config = self.cfg, self.config
        runtime = config.runtime
        capped = detector.limit_onnx_threads(runtime.inference_threads)
        if capped:
            print(f"[person] ONNX Runtime limited to {capped} CPU thread(s)")

        print("[person] loading person model...")
        self.person_model = detector.load_model(cfg.weights, runtime.device)
        self._sync_imgsz()

        if cfg.face_enabled:
            print("[person] loading face model...")
            self.face_detector = YuNetFaceDetector(
                cfg.face_weights, cfg.face_score_threshold, cfg.face_search_region,
                cfg.face_min_px, cfg.face_min_sharpness, cfg.face_ref_px, cfg.face_ref_sharpness,
            )
        else:
            log.warning("[person] face_enabled is OFF - people will be stored without faces")

        classes = cfg.classes
        self.classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def _sync_imgsz(self) -> None:
        actual = detector.onnx_static_input_hw(self.cfg.weights)
        if actual is None:
            return
        configured = detector.imgsz_hw(self.cfg.imgsz)
        if actual != configured:
            log.warning("[person] %s static shape %dx%d != configured %dx%d; using %dx%d",
                        self.cfg.weights, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
            self.cfg.imgsz = [actual[0], actual[1]]

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[person] warming up models...")
        region = self._detect_region(frame_shape) if frame_shape else None
        shape = (region[3] - region[1], region[2] - region[0]) if region else frame_shape
        detector.warmup_model(self.person_model, self.cfg.imgsz, shape)
        if self.face_detector is not None:
            dummy = np.zeros((300, 120, 3), dtype=np.uint8)
            try:
                self.face_detector.detect(dummy)
                log.info("[person] face model self-test OK")
            except Exception as error:
                log.error("[person] face model self-test failed: %s", error)

    def _detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        cfg = self.cfg
        if not (cfg.roi_crop_detect and cfg.roi_filter):
            return None
        return self.roi.detect_region(
            frame_shape, detector.imgsz_hw(cfg.imgsz), cfg.roi_crop_margin,
            top_extra=cfg.roi_top_extra, label="person model",
        )

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        cfg = self.cfg
        frame_shape = frame.shape[:2]

        # 1. Person detect
        region = self._detect_region(frame.shape)
        predictions = detector.detect(
            self.person_model, frame, cfg.conf_threshold, cfg.iou_threshold, cfg.imgsz,
            self.classes, region=region, agnostic_nms=cfg.agnostic_nms,
        )

        # Filter to ROI (feet check)
        if cfg.roi_filter:
            polygon = self.roi.get_polygon()
            sizes = (cfg.min_width_ratio, cfg.min_height_ratio, cfg.max_width_ratio, cfg.max_height_ratio)
            predictions = [
                p for p in predictions
                if is_person_inside_roi(p, frame_shape, polygon, *sizes, edge_margin_ratio=cfg.edge_margin_ratio)
            ]

        detected_count = len(predictions)
        now = datetime.now()

        for prediction in predictions:
            self._id_counter += 1
            track_id = f"p{self._id_counter}"
            stem = f"{self._session_tag}_{track_id}"
            event_uuid = str(uuid.uuid4())

            # 2. Crop person
            person_crop = crop_box(frame, prediction, cfg.padding_ratio, cfg.min_crop_height)
            if person_crop is None:
                continue

            # 3. Face detect
            face_detected = False
            face_crop_jpeg: Optional[bytes] = None
            face_landmarks: Optional[str] = None
            face_quality_score: Optional[float] = None

            if self.face_detector is not None:
                hit = self.face_detector.detect(person_crop)
                if hit is not None and hit.passed_gate:
                    face_detected = True
                    # 4. Crop face
                    fx1, fy1, fx2, fy2 = hit.box
                    margin_x, margin_y = (fx2 - fx1) * cfg.face_margin, (fy2 - fy1) * cfg.face_margin
                    crop_h, crop_w = person_crop.shape[:2]
                    x1, y1 = max(0, int(fx1 - margin_x)), max(0, int(fy1 - margin_y))
                    x2, y2 = min(crop_w, int(fx2 + margin_x)), min(crop_h, int(fy2 + margin_y))
                    face_crop = person_crop[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crop_jpeg = encode_jpeg(face_crop, cfg.face_jpeg_quality)
                    face_landmarks = json.dumps([[round(x, 1), round(y, 1)] for x, y in hit.landmarks])
                    face_quality_score = round(hit.quality, 4)

            # Encode JPEGs
            disk_files: Optional[dict] = {} if cfg.save_images_to_disk else None
            person_jpeg = encode_jpeg(person_crop, cfg.jpeg_quality)
            if disk_files is not None:
                disk_files[f"person_detection/{stem}_person.jpg"] = person_jpeg
                if face_detected and face_crop_jpeg is not None:
                    disk_files[f"face_detection/{stem}_face.jpg"] = face_crop_jpeg

            record = PersonRecord(
                event_uuid=event_uuid,
                device_id=self.config.runtime.device_id or "unknown",
                session_id=self.ctx.session_id,
                track_id=track_id,
                person_confidence=float(prediction.get("confidence", 0.0)),
                person_image_jpeg=person_jpeg,
                face_detected=face_detected,
                detected_at=now,
                face_image_jpeg=face_crop_jpeg,
                face_landmarks=face_landmarks,
                face_quality_score=face_quality_score,
                disk_files=disk_files,
            )

            # 5. Send metadata (enqueue to shared SQLite buffer)
            self.buffer.enqueue(record)

            status = f"face found (q={face_quality_score:.2f})" if face_detected else "no face"
            print(f"[person] {track_id} ({prediction.get('confidence', 0):.2f}) - {status}")

        return detected_count

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        models = [self.person_model]
        self.person_model = None
        self.face_detector = None
        detector.release_models(*models)
        log.info("[person] stopped, models released")