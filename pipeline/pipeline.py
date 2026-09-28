import logging
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detection import detector
from detection import image_ops
from config.config import PipelineConfig
from detection.geometry import (
    crop_vehicle, crop_plate, is_inside_roi, compute_detect_region, box_edges, polygon_bounds,
)
from storage.storage import DetectionStorage, DetectionRecord
from pipeline.timing import VehicleTiming, Stopwatch
from detection.tracker import FallbackTracker, PositionDeduper, needs_fallback_tracker
from api.roi_client import fetch_roi_polygon
from ocr import build_ocr_reader
from live.box_publisher import LiveBoxPublisher
from live.live_server import LiveServer
from pipeline.finalize_worker import FinalizeWorkerPool, FinalizeJob

log = logging.getLogger("pipeline")

_PRUNE_INTERVAL_SECONDS = 5.0


@dataclass
class _PlateCandidate:
    vehicle_crop: np.ndarray
    box: Tuple[float, float, float, float]
    plate_crop: np.ndarray
    plate_conf: float
    crop_ms: Optional[float]
    raw_crop: Optional[np.ndarray] = None
    ocr_text: Optional[str] = None
    ocr_score: float = 0.0
    ocr_attempted: bool = False

    @property
    def rank(self) -> Tuple[float, float]:
        return (self.ocr_score, self.plate_conf)


@dataclass
class _TrackState:
    timing: VehicleTiming
    crop: Optional[np.ndarray]
    prediction: Dict[str, Any]
    last_seen: float
    stem: str
    attempts: int = 0
    finalized: bool = False
    last_attempt: float = 0.0
    best: Optional["_PlateCandidate"] = None


class DetectionPipeline:
    def __init__(self, config: PipelineConfig, storage: DetectionStorage, camera_source: str):
        self.config = config
        self.storage = storage
        self.camera_source = camera_source

        capped_threads = detector.limit_onnx_threads(config.model.inference_threads)
        if capped_threads:
            print(f"[pipeline] ONNX Runtime limited to {capped_threads} CPU thread(s) "
                  f"(model.inference_threads: {config.model.inference_threads}, 0 = auto, -1 = no limit)")

        print("[pipeline] loading vehicle model...")
        self.vehicle_model = detector.load_model(config.model.vehicle_weights, config.model.device)
        print("[pipeline] loading plate model...")
        self.plate_model = detector.load_model(config.model.plate_weights, config.model.device)
        self._sync_imgsz_with_model("vehicle", config.model.vehicle_weights, "vehicle_imgsz")
        self._sync_imgsz_with_model("plate", config.model.plate_weights, "plate_imgsz")

        self.fallback_tracker = FallbackTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_center_distance=config.tracking.max_center_distance,
            max_missing_frames=config.tracking.max_missing_frames,
        )
        self.deduper = PositionDeduper(
            cooldown_seconds=config.tracking.dedup_cooldown_seconds,
            position_threshold=config.tracking.dedup_position_threshold,
        )

        self.ocr_reader = build_ocr_reader(config.ocr) if config.features.ocr_read else None

        self.finalize_pool: Optional[FinalizeWorkerPool] = None
        if config.features.async_ocr and self.ocr_reader is not None:
            self.finalize_pool = FinalizeWorkerPool(
                storage, worker_threads=config.ocr.worker_threads, max_queue=config.ocr.finalize_queue_size,
            ).start()

        self.live_publisher: Optional[LiveBoxPublisher] = None
        if config.features.live_boxes:
            if config.live.endpoint_url:
                self.live_publisher = LiveBoxPublisher(
                    endpoint_url=config.live.endpoint_url,
                    camera_id=config.live.camera_id,
                    max_hz=config.live.max_hz,
                    timeout_seconds=config.live.timeout_seconds,
                ).start()
            else:
                print("[pipeline] features.live_boxes is on but live.endpoint_url is not set - live feed disabled")

        self.live_server: Optional[LiveServer] = None
        if config.features.live_stream:
            self.live_server = LiveServer(
                camera_id=config.live.camera_id,
                host=config.live.serve_host,
                port=config.live.serve_port,
                stream_fps=config.live.stream_fps,
                stream_width=config.live.stream_width,
                jpeg_quality=config.live.jpeg_quality,
                box_max_age_seconds=config.live.box_max_age_seconds,
                auth_token=config.live.auth_token,
                box_visual_tracking=config.live.box_visual_tracking,
                box_extrapolate=config.live.box_extrapolate,
                box_extrapolate_max_seconds=config.live.box_extrapolate_max_seconds,
                roi_get=self.get_roi_polygon,
                roi_set=self.set_roi_polygon,
            )

        self._tracks: Dict[str, _TrackState] = {}
        self._last_prune = time.monotonic()
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._detect_region: Optional[Tuple[int, int, int, int]] = None
        self._detect_region_shape: Optional[tuple] = None
        self._roi_lock = threading.Lock()

        self._roi_poll_thread: Optional[threading.Thread] = None
        self._roi_poll_stop = threading.Event()
        if config.features.roi_fetch_from_api and config.api.roi_endpoint_url and config.api.camera_id is not None:
            self._refresh_roi_from_api()
            if config.api.roi_poll_interval_seconds > 0:
                self._roi_poll_thread = threading.Thread(
                    target=self._roi_poll_loop, name="roi-poll", daemon=True,
                )
                self._roi_poll_thread.start()
        self._last_vehicle_detect_ms: Optional[float] = None
        self._last_plate_detect_ms: Optional[float] = None

        classes = config.model.vehicle_classes
        self.vehicle_classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[pipeline] warming up models...")
        region = self._get_detect_region(frame_shape) if frame_shape else None
        if region is not None:
            frame_shape = (region[3] - region[1], region[2] - region[0])
        detector.warmup_model(self.vehicle_model, self.config.model.vehicle_imgsz, frame_shape)
        detector.warmup_model(self.plate_model, self.config.model.plate_imgsz)
        self._plate_self_test()

    def _sync_imgsz_with_model(self, label: str, weights: str, attribute: str) -> None:
        actual = detector.onnx_static_input_hw(weights)
        if actual is None:
            return
        configured = detector.imgsz_hw(getattr(self.config.model, attribute))
        if actual != configured:
            log.warning(
                "[pipeline] %s model %s is a static %dx%d (HxW) export but model.%s is %dx%d - "
                "using %dx%d so inference works. Re-export the model at the size you intend, or fix model.%s.",
                label, weights, actual[0], actual[1], attribute, configured[0], configured[1],
                actual[0], actual[1], attribute)
            setattr(self.config.model, attribute, [actual[0], actual[1]])

    def _plate_self_test(self) -> None:
        model_cfg = self.config.model
        if not self.config.features.plate_detection:
            log.warning("features.plate_detection is OFF - every vehicle will be stored WITHOUT a plate")
            return
        dummy_crop = np.zeros((300, 400, 3), dtype=np.uint8)
        try:
            with Stopwatch() as sw:
                detector.detect_batch(self.plate_model, [dummy_crop], model_cfg.plate_conf_threshold,
                                      model_cfg.vehicle_iou_threshold, model_cfg.plate_imgsz)
            log.info("[pipeline] plate model self-test OK (%.0f ms on a blank 400x300 crop)", sw.ms)
        except Exception as error:
            log.error("[pipeline] PLATE MODEL SELF-TEST FAILED - every vehicle will be stored as 'no plate': %s",
                      error, exc_info=True)

    def get_roi_polygon(self) -> List[List[float]]:
        with self._roi_lock:
            return [list(point) for point in self.config.roi.polygon]

    def set_roi_polygon(self, polygon: List[List[float]]) -> None:
        with self._roi_lock:
            self.config.roi.polygon = [[float(x), float(y)] for x, y in polygon]
            self._detect_region_shape = None

    def _refresh_roi_from_api(self) -> None:
        api, camera = self.config.api, self.config.camera
        polygon = fetch_roi_polygon(
            api.roi_endpoint_url, api.camera_id, api.roi_fetch_timeout_seconds,
            pixel_mode=api.roi_coordinates_are_pixels,
            reference_width=api.roi_reference_width or camera.frame_width,
            reference_height=api.roi_reference_height or camera.frame_height,
        )
        if polygon is not None:
            self.set_roi_polygon(polygon)

    def _roi_poll_loop(self) -> None:
        interval = self.config.api.roi_poll_interval_seconds
        while not self._roi_poll_stop.wait(interval):
            self._refresh_roi_from_api()

    def _get_detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        features = self.config.features
        if not (features.roi_crop_detect and features.roi_filter):
            return None
        shape = tuple(frame_shape[:2])
        if shape != self._detect_region_shape:
            roi = self.config.roi
            with self._roi_lock:
                polygon = roi.polygon
            roi_x_min, roi_x_max, roi_y_min, roi_y_max = polygon_bounds(polygon)
            self._detect_region = compute_detect_region(
                shape, roi_x_min, roi_x_max, roi_y_min, roi_y_max,
                roi.crop_margin, detector.imgsz_hw(self.config.model.vehicle_imgsz),
            )
            self._detect_region_shape = shape
            x1, y1, x2, y2 = self._detect_region
            print(f"[pipeline] vehicle model window: x {x1}-{x2}, y {y1}-{y2} "
                  f"({x2 - x1}x{y2 - y1} of the {shape[1]}x{shape[0]} frame)")
        return self._detect_region

    def _detect_and_track_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        model_cfg, tracking_cfg, features = self.config.model, self.config.tracking, self.config.features
        region = self._get_detect_region(frame.shape)

        with Stopwatch() as sw:
            vehicle_predictions = detector.track(
                self.vehicle_model, frame,
                model_cfg.vehicle_conf_threshold, model_cfg.vehicle_iou_threshold,
                model_cfg.vehicle_imgsz, tracking_cfg.bytetrack_config, self.vehicle_classes,
                region=region,
            )
        self._last_vehicle_detect_ms = sw.ms if features.time_vehicle_detect else None

        if features.fallback_tracker and needs_fallback_tracker(vehicle_predictions):
            self.fallback_tracker.update(vehicle_predictions)

        return vehicle_predictions

    def _filter_to_roi(self, predictions: List[Dict[str, Any]], frame_shape: tuple) -> List[Dict[str, Any]]:
        if not self.config.features.roi_filter:
            return predictions
        roi = self.config.roi
        with self._roi_lock:
            polygon = roi.polygon
        return [
            prediction for prediction in predictions
            if is_inside_roi(
                prediction, frame_shape,
                polygon,
                roi.min_width_ratio, roi.min_height_ratio,
                roi.max_width_ratio, roi.max_height_ratio,
                roi.edge_margin_ratio,
            )
        ]

    def _register_vehicle(self, frame: np.ndarray, prediction: Dict[str, Any], now: float) -> Tuple[Optional[str], bool]:
        track_id = prediction["track_id"]
        features = self.config.features

        state = self._tracks.get(track_id)
        if state is not None:
            state.last_seen = now
            state.prediction = prediction
            if features.position_dedup:
                self.deduper.refresh(track_id, prediction)
            return track_id, False

        if features.position_dedup:
            existing_track_id = self.deduper.find_existing_track(prediction)
            if existing_track_id is not None and existing_track_id in self._tracks:
                prediction["track_id"] = existing_track_id
                state = self._tracks[existing_track_id]
                state.last_seen = now
                state.prediction = prediction
                return existing_track_id, False

        crop_cfg = self.config.crop
        with Stopwatch() as sw:
            vehicle_crop = crop_vehicle(
                frame, prediction, crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height,
            )
        if vehicle_crop is None:
            return None, False

        timing = VehicleTiming(
            vehicle_detect_ms=self._last_vehicle_detect_ms,
            vehicle_crop_ms=sw.ms if features.time_vehicle_crop else None,
        )
        self._tracks[track_id] = _TrackState(
            timing=timing, crop=vehicle_crop, prediction=prediction, last_seen=now,
            stem=f"{self._session_tag}_{track_id}",
        )
        if features.position_dedup:
            self.deduper.remember_save(track_id, prediction)
        return track_id, True

    def _run_plate_detection(self, track_ids_needing_plate: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        features = self.config.features
        if not track_ids_needing_plate or not features.plate_detection:
            self._last_plate_detect_ms = None
            return {}

        model_cfg = self.config.model
        try:
            crops = [self._tracks[track_id].crop for track_id in track_ids_needing_plate]
            if features.enhance_before_plate_detect:
                crops = [image_ops.sharpen_and_denoise(crop) for crop in crops]
            with Stopwatch() as sw:
                batch_results = detector.detect_batch(
                    self.plate_model, crops,
                    model_cfg.plate_conf_threshold, model_cfg.vehicle_iou_threshold, model_cfg.plate_imgsz,
                )
            self._last_plate_detect_ms = sw.ms if features.time_plate_detect else None
        except Exception as error:
            log.warning("plate stage FAILED for %s - stored WITHOUT a plate: %s",
                        track_ids_needing_plate, error, exc_info=True)
            batch_results = [[] for _ in track_ids_needing_plate]
            self._last_plate_detect_ms = None

        for track_id in track_ids_needing_plate:
            self._tracks[track_id].timing.plate_detect_ms = self._last_plate_detect_ms

        return dict(zip(track_ids_needing_plate, batch_results))

    def _evaluate_plate(self, state: _TrackState, plate_predictions: List[Dict[str, Any]]) -> Optional[_PlateCandidate]:
        crop_cfg, preprocess_cfg, features = self.config.crop, self.config.preprocess, self.config.features
        vehicle_crop = state.crop
        if vehicle_crop is None:
            return None

        best_plate = max(plate_predictions, key=lambda p: p.get("confidence", 0.0))
        box = box_edges(best_plate)

        with Stopwatch() as sw:
            raw_crop = crop_plate(
                vehicle_crop, best_plate, crop_cfg.plate_padding_pixels, crop_cfg.plate_min_crop_height,
                crop_cfg.plate_padding_ratio,
            )
            plate_crop = raw_crop
            if raw_crop is not None and features.enhance_plate_crop:
                plate_crop = image_ops.enhance_plate_crop(raw_crop, preprocess_cfg.plate_crop_min_height)
        if raw_crop is None:
            return None

        candidate = _PlateCandidate(
            vehicle_crop=vehicle_crop, box=box, plate_crop=plate_crop,
            plate_conf=float(best_plate.get("confidence", 0.0)),
            crop_ms=sw.ms if features.time_plate_crop else None,
            raw_crop=raw_crop if raw_crop is not plate_crop else None,
        )

        if self.ocr_reader is not None and self.finalize_pool is None:
            try:
                candidate.ocr_text, candidate.ocr_score = self.ocr_reader.read_scored(plate_crop, candidate.raw_crop)
            except Exception as error:
                if features.print_console:
                    print(f"[pipeline] OCR read failed: {error}")
            candidate.ocr_attempted = True
        return candidate

    def _is_good_enough(self, candidate: Optional[_PlateCandidate]) -> bool:
        if candidate is None:
            return False
        if self.finalize_pool is not None or self.ocr_reader is None or getattr(self.ocr_reader, "_load_failed", False):
            return candidate.plate_conf >= self.config.model.plate_conf_threshold
        return bool(candidate.ocr_text) and candidate.ocr_score >= self.config.ocr.accept_score

    def _flush_stale_pending(self, now: float) -> None:
        wait = self.config.tracking.plate_stale_finalize_seconds
        if wait <= 0:
            return
        for track_id, state in list(self._tracks.items()):
            if state.finalized or state.attempts == 0 or state.crop is None:
                continue
            if now - state.last_seen > wait:
                try:
                    self._finalize_vehicle(track_id, state)
                except Exception as error:
                    log.warning("could not store stale track %s: %s", track_id, error)

    def _finalize_vehicle(self, track_id: str, state: _TrackState) -> None:
        best = state.best
        vehicle_crop = best.vehicle_crop if best is not None else state.crop
        if vehicle_crop is None:
            state.finalized = True
            return
        timing, base_prediction, stem = state.timing, state.prediction, state.stem

        if best is not None and self.finalize_pool is not None:
            job = FinalizeJob(
                track_id=track_id,
                build_record=lambda: self._build_record(track_id, best, vehicle_crop, base_prediction, timing, stem),
            )
            if self.finalize_pool.submit(job):
                state.finalized, state.crop, state.best = True, None, None
                return

        record = self._build_record(track_id, best, vehicle_crop, base_prediction, timing, stem)
        if self.config.features.store_to_sqlite and record is not None:
            self.storage.enqueue(record)
        state.finalized, state.crop, state.best = True, None, None

    def _build_record(
        self, track_id: str, best: Optional[_PlateCandidate], vehicle_crop: np.ndarray,
        base_prediction: Dict[str, Any], timing: VehicleTiming, stem: str,
    ) -> DetectionRecord:
        features = self.config.features
        disk_files: Optional[Dict[str, bytes]] = {} if features.save_images_to_disk else None

        plate_confidence: Optional[float] = None
        plate_image_bytes: Optional[bytes] = None
        plate_detected = False
        plate_x1 = plate_y1 = plate_x2 = plate_y2 = None

        ocr_process = False
        ocr_read = self.config.ocr.unrecognized_text

        if best is not None:
            if not best.ocr_attempted and self.ocr_reader is not None:
                try:
                    best.ocr_text, best.ocr_score = self.ocr_reader.read_scored(best.plate_crop, best.raw_crop)
                except Exception as error:
                    if features.print_console:
                        print(f"[pipeline] OCR read failed for {track_id}: {error}")
                best.ocr_attempted = True

            plate_x1, plate_y1, plate_x2, plate_y2 = best.box
            timing.plate_crop_ms = best.crop_ms
            plate_confidence = best.plate_conf
            plate_image_bytes = image_ops.encode_jpeg(best.plate_crop, self.config.storage.plate_jpeg_quality)
            plate_detected = True
            if disk_files is not None:
                disk_files[f"plate_detection/{stem}_plate.jpg"] = plate_image_bytes
            ocr_process = True
            ocr_read = best.ocr_text if best.ocr_text else self.config.ocr.unrecognized_text

        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = box_edges(base_prediction)

        vehicle_jpeg = None
        if features.col_vehicle_image or disk_files is not None:
            vehicle_jpeg = image_ops.encode_jpeg(vehicle_crop, self.config.storage.jpeg_quality)
            if disk_files is not None:
                disk_files[f"vehicle_detection/{stem}_{self._safe_name(base_prediction.get('class'))}.jpg"] = vehicle_jpeg

        record = DetectionRecord(
            track_id=track_id,
            camera_source=self.camera_source,
            vehicle_class=str(base_prediction.get("class") or "vehicle"),
            vehicle_confidence=float(base_prediction.get("confidence", 0.0)),
            vehicle_image_jpeg=vehicle_jpeg if features.col_vehicle_image else None,
            vehicle_box_x1=vehicle_x1 if features.col_vehicle_box else None,
            vehicle_box_y1=vehicle_y1 if features.col_vehicle_box else None,
            vehicle_box_x2=vehicle_x2 if features.col_vehicle_box else None,
            vehicle_box_y2=vehicle_y2 if features.col_vehicle_box else None,
            plate_detected=plate_detected,
            plate_confidence=plate_confidence,
            plate_image_jpeg=plate_image_bytes if features.col_plate_image else None,
            plate_box_x1=plate_x1 if features.col_plate_box else None,
            plate_box_y1=plate_y1 if features.col_plate_box else None,
            plate_box_x2=plate_x2 if features.col_plate_box else None,
            plate_box_y2=plate_y2 if features.col_plate_box else None,
            detected_at=datetime.now(),
            vehicle_detect_ms=timing.vehicle_detect_ms if features.col_vehicle_detect_ms else None,
            vehicle_crop_ms=timing.vehicle_crop_ms if features.col_vehicle_crop_ms else None,
            plate_detect_ms=timing.plate_detect_ms if features.col_plate_detect_ms else None,
            plate_crop_ms=timing.plate_crop_ms if features.col_plate_crop_ms else None,
            total_pipeline_ms=timing.total_ms if features.col_total_pipeline_ms else None,
            ocr_process=ocr_process if features.col_ocr_read else False,
            ocr_read=ocr_read if features.col_ocr_read else self.config.ocr.unrecognized_text,
            disk_files=disk_files,
        )

        if features.print_console:
            status = f"plate found ({plate_confidence:.2f}) - ocr: {ocr_read}" if plate_detected else "no plate found"
            if features.print_timing:
                breakdown = (
                    f"vehicle_detect={self._fmt_ms(timing.vehicle_detect_ms)} "
                    f"vehicle_crop={self._fmt_ms(timing.vehicle_crop_ms)} "
                    f"plate_detect={self._fmt_ms(timing.plate_detect_ms)} "
                    f"plate_crop={self._fmt_ms(timing.plate_crop_ms)} "
                    f"total={self._fmt_ms(timing.total_ms)}"
                )
                print(f"[pipeline] {track_id} finalized - {status} - {breakdown}")
            else:
                print(f"[pipeline] {track_id} finalized - {status}")

        return record

    def shutdown(self) -> None:
        self._roi_poll_stop.set()
        if self._roi_poll_thread is not None:
            self._roi_poll_thread.join(timeout=2.0)
        if self.live_publisher is not None:
            self.live_publisher.stop()
        if self.live_server is not None:
            self.live_server.stop()
        if self.finalize_pool is not None:
            self.finalize_pool.stop()

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False, captured_at: Optional[float] = None) -> int:
        frame_shape = frame.shape[:2]
        features = self.config.features
        max_attempts = max(1, self.config.tracking.max_plate_attempts)
        now = time.monotonic()

        vehicle_predictions = self._detect_and_track_vehicles(frame)
        vehicle_predictions = [p for p in vehicle_predictions if not self._looks_like_plate(p)]
        all_tracked_vehicles = vehicle_predictions
        vehicle_predictions = self._filter_to_roi(vehicle_predictions, frame_shape)

        new_vehicle_count = 0
        track_ids_needing_plate: List[str] = []

        for prediction in vehicle_predictions:
            if not prediction.get("track_id"):
                continue
            track_id, created = self._register_vehicle(frame, prediction, now)
            if track_id is None:
                continue
            state = self._tracks[track_id]
            if created:
                new_vehicle_count += 1

            if state.finalized or state.attempts >= max_attempts:
                continue

            if state.attempts > 0:
                if now - state.last_attempt < self.config.tracking.plate_retry_interval_seconds:
                    continue
                crop_cfg = self.config.crop
                fresh_crop = crop_vehicle(frame, prediction, crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height)
                if fresh_crop is not None:
                    state.crop = fresh_crop
            state.last_attempt = now
            track_ids_needing_plate.append(track_id)

        in_roi_ids = {id(p) for p in vehicle_predictions}
        for live_sink in (self.live_publisher, self.live_server):
            if live_sink is not None:
                live_sink.publish(frame_shape, all_tracked_vehicles, in_roi_ids, captured_at)

        if not features.plate_detection:
            for track_id in track_ids_needing_plate:
                self._finalize_vehicle(track_id, self._tracks[track_id])
        else:
            plate_results = self._run_plate_detection(track_ids_needing_plate)

            for track_id, plate_predictions in plate_results.items():
                state = self._tracks[track_id]
                state.attempts += 1
                attempts_exhausted = always_finalize or state.attempts >= max_attempts

                if plate_predictions:
                    candidate = self._evaluate_plate(state, plate_predictions)
                    if candidate is not None and (state.best is None or candidate.rank > state.best.rank):
                        state.best = candidate

                if self._is_good_enough(state.best) or attempts_exhausted:
                    self._finalize_vehicle(track_id, state)

            self._flush_stale_pending(now)

        if now - self._last_prune >= _PRUNE_INTERVAL_SECONDS:
            self._prune_tracks(now)

        return new_vehicle_count

    def _prune_tracks(self, now: float) -> None:
        self._last_prune = now
        ttl = self.config.tracking.track_ttl_seconds
        expired = [track_id for track_id, state in self._tracks.items() if now - state.last_seen > ttl]
        for track_id in expired:
            state = self._tracks[track_id]
            if not state.finalized and state.crop is not None:
                try:
                    self._finalize_vehicle(track_id, state)
                except Exception as error:
                    log.warning("could not store expired track %s: %s", track_id, error)
            del self._tracks[track_id]
        self.deduper.prune()

    @staticmethod
    def _looks_like_plate(prediction: Dict[str, Any]) -> bool:
        label = "".join(ch for ch in str(prediction.get("class") or "").lower() if ch.isalnum())
        return any(marker in label for marker in ("plate", "license", "licence"))

    @staticmethod
    def _fmt_ms(value: Optional[float]) -> str:
        return f"{value:.1f}ms" if value is not None else "off"

    @staticmethod
    def _safe_name(value: Optional[str]) -> str:
        value = value or "vehicle"
        return "".join(ch if ch.isalnum() else "_" for ch in value)
