"""
pipeline.py
-----------
This is where the actual "Vehicle Detect -> Crop -> Plate Detect -> Crop"
flow described in the project brief lives. Every other module is a building
block; this module wires them together for one frame at a time.

Flow for a single frame:
    1. Vehicle Detect  - run the vehicle model (with tracking) on the frame
    2. Filter          - keep only boxes inside the configured ROI and size range
    3. Crop (vehicle)  - cut each vehicle out of the frame, once per track_id
    4. Plate Detect    - run the plate model on all vehicle crops in one batch
    5. Crop (plate)    - cut the plate out of the vehicle crop, enhance it
    6. Store           - hand the finished record (metadata + both images +
                         per-vehicle timing) to the SQLite buffer
"""

# os.path helpers are used for building on-disk output paths
import os
# datetime timestamps every detection with a real wall-clock time
from datetime import datetime
from typing import Any, Dict, List, Optional

# cv2.imwrite is used only when "save images to disk" is turned on in config
import cv2
import numpy as np

from detection import detector
from detection import image_ops
from config.config import PipelineConfig
from detection.geometry import crop_vehicle, crop_plate, is_inside_roi, compute_containment, compute_iou, box_edges
from storage.storage import DetectionStorage, DetectionRecord
from pipeline.timing import VehicleTiming, Stopwatch
from detection.tracker import FallbackTracker, PositionDeduper, needs_fallback_tracker
from ocr import PlateOCRReader
from live.box_publisher import LiveBoxPublisher
from live.live_server import LiveServer


class DetectionPipeline:
    """Holds both models plus tracking/dedup state and runs the full flow
    for every frame handed to `process_frame`."""

    def __init__(self, config: PipelineConfig, storage: DetectionStorage, camera_source: str):
        self.config = config
        self.storage = storage
        # identifies which camera/source these detections came from - useful
        # once more than one Pi/camera writes into (or is compared against) the same schema
        self.camera_source = camera_source

        print("[pipeline] loading vehicle model...")
        self.vehicle_model = detector.load_model(config.model.vehicle_weights, config.model.device)
        print("[pipeline] loading plate model...")
        self.plate_model = detector.load_model(config.model.plate_weights, config.model.device)

        # fallback tracker only actually used if ByteTrack fails to tag a track_id
        self.fallback_tracker = FallbackTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_center_distance=config.tracking.max_center_distance,
            max_missing_frames=config.tracking.max_missing_frames,
        )
        self.deduper = PositionDeduper(
            cooldown_seconds=config.tracking.dedup_cooldown_seconds,
            position_threshold=config.tracking.dedup_position_threshold,
        )

        # lazy-loaded inline OCR reader - only pays the PaddleOCR import/load
        # cost if features.ocr_read is on and a plate is actually found
        self.ocr_reader = (
            PlateOCRReader(
                lang=config.ocr.lang,
                min_confidence=config.ocr.min_confidence,
                allowed_chars=config.ocr.allowed_chars,
            )
            if config.features.ocr_read
            else None
        )

        # optional live box feed for the command center; fire-and-forget on its
        # own thread, so a slow/dead server can never stall the detection loop
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

        # optional live VIEW served from the Pi (frames + boxes over HTTP). Only
        # created here; run.py starts it once it has a camera to read frames from.
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
            )

        # per-track_id bookkeeping so we never re-save a crop we already have,
        # and so we know how many times we've tried (and failed) to find a plate
        self._saved_vehicle_crops: Dict[str, np.ndarray] = {}
        self._plate_attempts: Dict[str, int] = {}
        self._finalized_tracks: set = set()
        self._timings: Dict[str, VehicleTiming] = {}
        # ms the LAST vehicle-detect / plate-detect model call took, so every
        # vehicle that was part of that same call can be stamped with it
        self._last_vehicle_detect_ms: Optional[float] = None
        self._last_plate_detect_ms: Optional[float] = None

        # comma-separated class allow-list from config, turned into a fast lookup set
        classes = config.model.vehicle_classes
        self.vehicle_classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

        if config.features.save_images_to_disk:
            os.makedirs(os.path.join(config.storage.output_dir, "vehicle_detection"), exist_ok=True)
            os.makedirs(os.path.join(config.storage.output_dir, "plate_detection"), exist_ok=True)

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        """Run one throwaway inference per model before the real loop starts."""
        print("[pipeline] warming up models...")
        detector.warmup_model(self.vehicle_model, self.config.model.vehicle_imgsz, frame_shape)
        detector.warmup_model(self.plate_model, self.config.model.plate_imgsz)

    # ------------------------------------------------------------------ #
    # Stage 1: Vehicle Detect (+ tracking)
    # ------------------------------------------------------------------ #
    def _detect_and_track_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        model_cfg, tracking_cfg, features = self.config.model, self.config.tracking, self.config.features

        with Stopwatch() as sw:
            vehicle_predictions = detector.track(
                self.vehicle_model, frame,
                model_cfg.vehicle_conf_threshold, model_cfg.vehicle_iou_threshold,
                model_cfg.vehicle_imgsz, tracking_cfg.bytetrack_config, self.vehicle_classes,
            )
        self._last_vehicle_detect_ms = sw.ms if features.time_vehicle_detect else None

        # ByteTrack occasionally can't assign an ID (e.g. right after a
        # reconnect); when that happens, fall back to our own simple tracker
        if features.fallback_tracker and needs_fallback_tracker(vehicle_predictions):
            self.fallback_tracker.update(vehicle_predictions)

        return vehicle_predictions

    def _filter_to_roi(self, predictions: List[Dict[str, Any]], frame_shape: tuple) -> List[Dict[str, Any]]:
        if not self.config.features.roi_filter:
            return predictions
        roi = self.config.roi
        return [
            prediction for prediction in predictions
            if is_inside_roi(
                prediction, frame_shape,
                roi.x_min, roi.x_max, roi.y_min, roi.y_max,
                roi.min_width_ratio, roi.min_height_ratio,
                roi.max_width_ratio, roi.max_height_ratio,
                roi.edge_margin_ratio,
            )
        ]

    # ------------------------------------------------------------------ #
    # Stage 2/3: Vehicle Crop
    # ------------------------------------------------------------------ #
    def _get_or_create_vehicle_crop(self, frame: np.ndarray, prediction: Dict[str, Any]) -> Optional[str]:
        """Returns the track_id that should be used for this detection
        (which may be remapped by the position-deduper) and ensures a crop
        exists in memory for it. Returns None if cropping failed."""
        track_id = prediction["track_id"]
        features = self.config.features

        if track_id in self._saved_vehicle_crops:
            # already have this vehicle - just refresh its dedup timestamp
            # so a vehicle sitting still (red light) doesn't get re-triggered
            if features.position_dedup:
                self.deduper.refresh(track_id, prediction)
            return track_id

        # is this "new" track_id actually the same physical vehicle as one we
        # already saved a moment ago in the same spot?
        if features.position_dedup:
            existing_track_id = self.deduper.find_existing_track(prediction)
            if existing_track_id is not None:
                prediction["track_id"] = existing_track_id
                return existing_track_id

        crop_cfg = self.config.crop
        with Stopwatch() as sw:
            vehicle_crop = crop_vehicle(
                frame, prediction, crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height,
            )
        if vehicle_crop is None:
            return None

        # record this vehicle's timing: the detect-model call that found it
        # (shared across every vehicle in this frame) plus its own crop time
        timing = VehicleTiming(
            vehicle_detect_ms=self._last_vehicle_detect_ms,
            vehicle_crop_ms=sw.ms if features.time_vehicle_crop else None,
        )
        self._timings[track_id] = timing

        self._saved_vehicle_crops[track_id] = vehicle_crop
        if features.position_dedup:
            self.deduper.remember_save(track_id, prediction)
        self._plate_attempts[track_id] = 0

        if features.save_images_to_disk:
            self._write_crop_to_disk("vehicle_detection", f"{track_id}_{self._safe_name(prediction.get('class'))}", vehicle_crop)

        return track_id

    # ------------------------------------------------------------------ #
    # Stage 4/5: Plate Detect + Crop
    # ------------------------------------------------------------------ #
    def _run_plate_detection(self, track_ids_needing_plate: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Batch-run the plate model over every vehicle crop that still needs one."""
        features = self.config.features
        if not track_ids_needing_plate or not features.plate_detection:
            self._last_plate_detect_ms = None
            return {}

        crops = [self._saved_vehicle_crops[track_id] for track_id in track_ids_needing_plate]
        if features.enhance_before_plate_detect:
            crops = [image_ops.sharpen_and_denoise(crop) for crop in crops]

        model_cfg = self.config.model
        try:
            with Stopwatch() as sw:
                batch_results = detector.detect_batch(
                    self.plate_model, crops,
                    model_cfg.plate_conf_threshold, model_cfg.vehicle_iou_threshold, model_cfg.plate_imgsz,
                )
            self._last_plate_detect_ms = sw.ms if features.time_plate_detect else None
        except detector.InferenceError as error:
            if features.print_console:
                print(f"[pipeline] plate detection error: {error}")
            batch_results = [[] for _ in track_ids_needing_plate]
            self._last_plate_detect_ms = None

        # every track in this batch shares the same plate-detect timing
        for track_id in track_ids_needing_plate:
            if track_id in self._timings:
                self._timings[track_id].plate_detect_ms = self._last_plate_detect_ms

        return dict(zip(track_ids_needing_plate, batch_results))

    def _finalize_vehicle(self, track_id: str, base_prediction: Dict[str, Any], plate_predictions: List[Dict[str, Any]]) -> None:
        """Pick the best plate (if any), crop + enhance it, and hand the
        completed record off to storage. Marks the track as finalized so it
        is never processed again."""
        crop_cfg, preprocess_cfg, features = self.config.crop, self.config.preprocess, self.config.features
        vehicle_crop = self._saved_vehicle_crops[track_id]
        timing = self._timings[track_id]

        plate_confidence: Optional[float] = None
        plate_image_bytes: Optional[bytes] = None
        plate_detected = False
        # plate box edges are relative to the VEHICLE CROP (same pixel space
        # as vehicle_image_jpeg) since that's what the plate model actually
        # saw; None until/unless a plate is actually found below
        plate_x1: Optional[float] = None
        plate_y1: Optional[float] = None
        plate_x2: Optional[float] = None
        plate_y2: Optional[float] = None

        # OCR metadata - defaults cover "no plate ever found for this
        # vehicle" (attempts_exhausted with no plate_predictions at all):
        # no OCR pass was possible, so the read is unrecognized by definition
        ocr_process = False
        ocr_read = self.config.ocr.unrecognized_text

        if plate_predictions:
            best_plate = max(plate_predictions, key=lambda p: p.get("confidence", 0.0))
            # raw detection box edges, in the vehicle crop's own pixel space -
            # exactly what's needed to draw a rectangle on the stored vehicle_image
            plate_x1, plate_y1, plate_x2, plate_y2 = box_edges(best_plate)
            with Stopwatch() as sw:
                plate_crop = crop_plate(vehicle_crop, best_plate, crop_cfg.plate_padding_pixels, crop_cfg.plate_min_crop_height)
                if plate_crop is not None and features.enhance_plate_crop:
                    plate_crop = image_ops.enhance_plate_crop(plate_crop, preprocess_cfg.plate_crop_min_height)
            timing.plate_crop_ms = sw.ms if features.time_plate_crop else None

            if plate_crop is not None:
                plate_confidence = float(best_plate.get("confidence", 0.0))
                plate_image_bytes = image_ops.encode_jpeg(plate_crop, self.config.storage.jpeg_quality)
                plate_detected = True
                if features.save_images_to_disk:
                    self._write_crop_to_disk("plate_detection", f"{track_id}_plate", plate_crop)

                # a plate crop exists - attempt an OCR read on it. Any failure
                # (OCR disabled, engine unavailable, low confidence, no text
                # found) falls back to the configured "Unrecognized" text,
                # it never blocks or crashes the finalize step.
                ocr_process = True
                ocr_text = None
                if self.ocr_reader is not None:
                    try:
                        ocr_text = self.ocr_reader.read(plate_crop)
                    except Exception as error:
                        if features.print_console:
                            print(f"[pipeline] {track_id} OCR read failed: {error}")
                        ocr_text = None
                ocr_read = ocr_text if ocr_text else self.config.ocr.unrecognized_text
            else:
                # detection fired but the crop itself failed (degenerate box) -
                # don't report box coordinates for a plate we didn't actually save
                plate_x1 = plate_y1 = plate_x2 = plate_y2 = None
                # a plate box existed but we couldn't produce a crop to OCR -
                # still "attempted" in the sense that a plate was detected,
                # but there was nothing to read
                ocr_process = True
                ocr_read = self.config.ocr.unrecognized_text

        # vehicle box edges, in the ORIGINAL FULL FRAME's pixel space - what's
        # needed to draw this vehicle's box back onto the full camera frame
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = box_edges(base_prediction)

        record = DetectionRecord(
            track_id=track_id,
            camera_source=self.camera_source,
            vehicle_class=str(base_prediction.get("class") or "vehicle"),
            vehicle_confidence=float(base_prediction.get("confidence", 0.0)),
            vehicle_image_jpeg=image_ops.encode_jpeg(vehicle_crop, self.config.storage.jpeg_quality) if features.col_vehicle_image else None,
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
        )

        if features.store_to_sqlite:
            self.storage.enqueue(record)
        self._finalized_tracks.add(track_id)

        if features.print_console:
            if plate_detected:
                status = f"plate found ({plate_confidence:.2f}) - ocr: {ocr_read}"
            else:
                status = "no plate found"
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

    # ------------------------------------------------------------------ #
    # Public entry point: process ONE frame end to end
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Stop background helpers owned by the pipeline (live box push + live view server)."""
        if self.live_publisher is not None:
            self.live_publisher.stop()
        if self.live_server is not None:
            self.live_server.stop()

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False, captured_at: Optional[float] = None) -> int:
        """Runs the full Vehicle Detect -> Crop -> Plate Detect -> Crop flow
        for one frame. Returns how many NEW vehicles were cropped this call.

        `always_finalize=True` is used for single-image/folder mode: there is
        no "next frame" to try again on, so the vehicle must be finalized
        (plate found or not) after this one and only attempt, regardless of
        `tracking.max_plate_attempts` (which exists for the live-stream case,
        where the same vehicle is seen across many frames).
        """
        frame_shape = frame.shape[:2]

        # Stage 1: detect + track every vehicle in the frame
        vehicle_predictions = self._detect_and_track_vehicles(frame)
        # Stage 1b: drop plate-labelled boxes and anything outside our ROI/size range
        vehicle_predictions = [p for p in vehicle_predictions if not self._looks_like_plate(p)]
        # keep the pre-ROI list for the live feed: the operator should see
        # vehicles approaching, flagged in_roi or not
        all_tracked_vehicles = vehicle_predictions
        vehicle_predictions = self._filter_to_roi(vehicle_predictions, frame_shape)

        new_vehicle_count = 0
        track_ids_needing_plate: List[str] = []
        base_prediction_by_track: Dict[str, Dict[str, Any]] = {}

        # Stage 2/3: make sure every currently-visible vehicle has a crop
        for prediction in vehicle_predictions:
            was_new = prediction["track_id"] not in self._saved_vehicle_crops
            track_id = self._get_or_create_vehicle_crop(frame, prediction)
            if track_id is None:
                continue
            if was_new and track_id not in self._finalized_tracks:
                new_vehicle_count += 1

            base_prediction_by_track[track_id] = prediction

            attempts = self._plate_attempts.get(track_id, 0)
            already_done = track_id in self._finalized_tracks
            attempts_exhausted = attempts >= self.config.tracking.max_plate_attempts
            if not already_done and not attempts_exhausted:
                track_ids_needing_plate.append(track_id)

        # Live feed: publish NOW, before the (slower) plate model runs, so
        # boxes go out at detector speed. By this point the dedup step has
        # already remapped track_ids in place, so they match the DB rows.
        in_roi_ids = {id(p) for p in vehicle_predictions}
        for live_sink in (self.live_publisher, self.live_server):
            if live_sink is not None:
                live_sink.publish(frame_shape, all_tracked_vehicles, in_roi_ids, captured_at)

        # Stage 4: run the plate model, once, on every vehicle crop that needs it
        plate_results = self._run_plate_detection(track_ids_needing_plate)

        # Stage 5/6: crop + save the plate (or record the attempt) for each vehicle
        for track_id, plate_predictions in plate_results.items():
            self._plate_attempts[track_id] = self._plate_attempts.get(track_id, 0) + 1
            attempts = self._plate_attempts[track_id]
            attempts_exhausted = always_finalize or attempts >= self.config.tracking.max_plate_attempts

            if plate_predictions or attempts_exhausted:
                self._finalize_vehicle(track_id, base_prediction_by_track[track_id], plate_predictions)

        return new_vehicle_count

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_plate(prediction: Dict[str, Any]) -> bool:
        """Some vehicle-model checkpoints also emit a stray 'plate' class -
        make sure those never get treated as a vehicle box."""
        label = "".join(ch for ch in str(prediction.get("class") or "").lower() if ch.isalnum())
        return any(marker in label for marker in ("plate", "license", "licence"))

    @staticmethod
    def _fmt_ms(value: Optional[float]) -> str:
        """Console-friendly rendering for a possibly-disabled timing value."""
        return f"{value:.1f}ms" if value is not None else "off"

    @staticmethod
    def _safe_name(value: Optional[str]) -> str:
        """Turn a class name into something safe to use inside a filename."""
        value = value or "vehicle"
        return "".join(ch if ch.isalnum() else "_" for ch in value)

    def _write_crop_to_disk(self, subfolder: str, stem: str, image: np.ndarray) -> None:
        """Optional convenience copy of a crop on disk, alongside the DB blob."""
        directory = os.path.join(self.config.storage.output_dir, subfolder)
        path = os.path.join(directory, f"{stem}.jpg")
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, self.config.storage.jpeg_quality])
