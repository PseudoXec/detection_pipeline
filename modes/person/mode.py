"""Person -> face detection mode.

    Person detect + track -> (feet in ROI?) -> crop -> face detect (background thread) -> crop -> store

One row per person TRACK goes to `person_detections` (see modes/person/store.py). A row is written when
the first face good enough shows up, or when the face attempts run out, or when the person leaves the
scene / the track expires / the mode is stopped - whichever comes first. A person whose face was never
found is still stored (face_detected = 0) so the dashboard knows someone was there.

Threading mirrors vehicle mode: the frame loop only detects and tracks people; the face model runs on
its own thread; JPEG encoding and the SQLite hand-off run on the finalize pool.
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import detector, image_ops
from core.finalize_worker import FinalizeJob, FinalizeWorkerPool
from core.geometry import box_edges, crop_box, point_in_polygon
from core.tracker import FallbackTracker, PositionDeduper, needs_fallback_tracker
from modes.base import DetectionMode, RuntimeContext
from modes.person.face import FaceDetectionWorker, FaceHit, FaceResult, YuNetFaceDetector
from modes.person.geometry import feet_point, is_person_inside_roi
from modes.person.store import PersonRecord

log = logging.getLogger("pipeline")

PIPELINE_VERSION = "person-1.0"
_PRUNE_INTERVAL_SECONDS = 5.0


@dataclass
class _Sighting:
    """One frame in which a person was seen: the crop and everything that belongs to that frame."""
    crop: np.ndarray
    box: Tuple[float, float, float, float]     # full-frame pixels
    confidence: float
    captured_at: Optional[float]               # unix time the frame was grabbed (None for image files)
    detect_ms: Optional[float]
    crop_ms: Optional[float]
    frame_shape: Tuple[int, int]


@dataclass
class _FaceCandidate:
    sighting: _Sighting                        # the frame the face was found in
    hit: FaceHit
    detect_ms: Optional[float]


@dataclass
class _TrackState:
    first_seen: datetime
    last_seen: float
    last_seen_wall: datetime
    stem: str
    prediction: Dict[str, Any]
    best: Optional[_Sighting] = None           # most confident sighting so far (used when no face is found)
    face: Optional[_FaceCandidate] = None      # best face so far
    attempts: int = 0
    last_attempt: float = 0.0
    finalized: bool = False


class PersonMode(DetectionMode):
    name = "person"

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = ctx.config                    # only for process-wide settings (runtime)
        self.cfg = ctx.config.person                # everything that belongs to person mode
        self.storage = ctx.stores.person
        self.roi = ctx.roi
        self.camera_source = ctx.camera_source

        self.person_model = None
        self.face_worker: Optional[FaceDetectionWorker] = None
        self.finalize_pool: Optional[FinalizeWorkerPool] = None
        self._stopped = False
        self._tracks: Dict[str, _TrackState] = {}
        self._pending_sightings: Dict[str, _Sighting] = {}   # face jobs in flight -> the frame they came from
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
        self.stabilizer = detector.BoxStabilizer()

        if cfg.face_enabled:
            print("[person] loading face model...")
            face_detector = YuNetFaceDetector(
                cfg.face_weights, cfg.face_score_threshold, cfg.face_search_region,
                cfg.face_min_px, cfg.face_min_sharpness, cfg.face_ref_px, cfg.face_ref_sharpness,
            )
            self.face_worker = FaceDetectionWorker(face_detector, max_queue=cfg.face_worker_max_queue)
            if cfg.async_face_detection:
                self.face_worker.start()
        else:
            log.warning("[person] person.face_enabled is OFF - every person is stored WITHOUT a face")

        self.fallback_tracker = FallbackTracker(
            iou_threshold=cfg.fallback_iou_threshold,
            max_center_distance=cfg.fallback_max_center_distance,
            max_missing_frames=cfg.fallback_max_missing_frames,
        )
        self.deduper = PositionDeduper(cfg.dedup_cooldown_seconds, cfg.dedup_position_threshold)
        self._aliases: Dict[str, str] = {}
        self._last_prune = time.monotonic()
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._last_person_detect_ms: Optional[float] = None

        # JPEG encoding + the SQLite hand-off happen off the frame loop
        self.finalize_pool = FinalizeWorkerPool(self.storage, worker_threads=1, max_queue=32).start()

        classes = cfg.classes
        self.classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None
        self._person_model_name = self._describe_weights(cfg.weights, cfg.imgsz)
        self._face_model_name = self._describe_weights(cfg.face_weights) if cfg.face_enabled else None

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _describe_weights(path: str, imgsz: Any = None) -> str:
        import os
        base = os.path.basename(str(path).rstrip("/\\"))
        base = base.replace("_ncnn_model", "-ncnn").replace(".onnx", "-onnx").replace(".pt", "-pt")
        if imgsz is not None:
            size = detector.normalize_imgsz(imgsz)
            base += f"-{size if isinstance(size, int) else 'x'.join(map(str, size))}"
        return base

    def _sync_imgsz(self) -> None:
        actual = detector.onnx_static_input_hw(self.cfg.weights)
        if actual is None:
            return
        configured = detector.imgsz_hw(self.cfg.imgsz)
        if actual != configured:
            log.warning("[person] %s is a static %dx%d (HxW) export but person.imgsz is %dx%d - using %dx%d",
                        self.cfg.weights, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
            self.cfg.imgsz = [actual[0], actual[1]]

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[person] warming up models...")
        region = self._detect_region(frame_shape) if frame_shape else None
        shape = (region[3] - region[1], region[2] - region[0]) if region else frame_shape
        detector.warmup_model(self.person_model, self.cfg.imgsz, shape)
        if self.face_worker is not None:
            dummy = np.zeros((300, 120, 3), dtype=np.uint8)
            result = self.face_worker.run_inline("warmup", dummy, 0)
            if result.error:
                log.error("[person] FACE MODEL SELF-TEST FAILED - people will be stored without faces: %s", result.error)
            else:
                log.info("[person] face model self-test OK (%.0f ms on a blank 120x300 crop)", result.detect_ms or 0.0)

    def _detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        cfg = self.cfg
        if not (cfg.roi_crop_detect and cfg.roi_filter):
            return None
        return self.roi.detect_region(
            frame_shape, detector.imgsz_hw(cfg.imgsz), cfg.roi_crop_margin,
            top_extra=cfg.roi_top_extra, label="person model",
        )

    # ------------------------------------------------------------------ per frame
    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        cfg = self.cfg
        frame_shape = frame.shape[:2]
        now = time.monotonic()

        predictions = self._detect_and_track(frame)
        all_tracked = predictions
        inside = self._filter_to_roi(predictions, frame_shape)

        new_people = 0
        wanting_face: List[str] = []
        for prediction in inside:
            if not prediction.get("track_id"):
                continue
            track_id, created = self._register(frame, prediction, now, captured_at)
            if track_id is None:
                continue
            if created:
                new_people += 1
            state = self._tracks[track_id]
            if state.finalized:
                continue
            self._remember_sighting(state, frame, prediction, captured_at)

            if self.face_worker is None or state.attempts >= cfg.face_max_attempts:
                continue
            if self.face_worker.is_pending(track_id) and not always_finalize:
                continue
            if state.attempts > 0 and now - state.last_attempt < cfg.face_retry_interval_seconds:
                continue
            state.last_attempt = now
            wanting_face.append(track_id)

        self._publish_live(frame_shape, all_tracked, captured_at)
        self._run_faces(frame, wanting_face, always_finalize, captured_at)
        self._flush_stale(now, always_finalize)

        if now - self._last_prune >= _PRUNE_INTERVAL_SECONDS:
            self._prune(now)
        return new_people

    def _detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        cfg = self.cfg
        started = time.perf_counter()
        predictions = detector.track(
            self.person_model, frame, cfg.conf_threshold, cfg.iou_threshold, cfg.imgsz,
            cfg.bytetrack_config, self.classes,
            region=self._detect_region(frame.shape),
            detect_conf=cfg.detector_conf_floor or None,
            agnostic_nms=cfg.agnostic_nms, id_prefix="p", stabilizer=self.stabilizer,
        )
        self._last_person_detect_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if cfg.fallback_tracker and needs_fallback_tracker(predictions):
            self.fallback_tracker.update(predictions)
        return predictions

    def _filter_to_roi(self, predictions: List[Dict[str, Any]], frame_shape: tuple) -> List[Dict[str, Any]]:
        cfg = self.cfg
        if not cfg.roi_filter:
            return predictions
        polygon = self.roi.get_polygon()
        sizes = (cfg.min_width_ratio, cfg.min_height_ratio, cfg.max_width_ratio, cfg.max_height_ratio)
        return [p for p in predictions
                if is_person_inside_roi(p, frame_shape, polygon, *sizes, edge_margin_ratio=cfg.edge_margin_ratio)]

    def _publish_live(self, frame_shape: tuple, tracked: List[Dict[str, Any]], captured_at: Optional[float]) -> None:
        # LIVE VIEW: show everyone whose feet are inside the polygon, even if the strict size filter
        # rejected them for storage - a person standing in the zone must not look "undetected".
        polygon = self.roi.get_polygon()
        height, width = frame_shape[:2]
        in_roi_ids = set()
        for prediction in tracked:
            feet_x, feet_y = feet_point(prediction)
            if point_in_polygon(feet_x / width, feet_y / height, polygon):
                in_roi_ids.add(id(prediction))
        self.ctx.live.publish(frame_shape, tracked, in_roi_ids, captured_at, "person")

    # ------------------------------------------------------------------ tracks
    def _register(self, frame: np.ndarray, prediction: Dict[str, Any], now: float,
                  captured_at: Optional[float]) -> Tuple[Optional[str], bool]:
        track_id = prediction["track_id"]
        cfg = self.cfg

        def touch(tid: str) -> Tuple[str, bool]:
            state = self._tracks[tid]
            state.last_seen, state.last_seen_wall, state.prediction = now, datetime.now(), prediction
            if cfg.position_dedup:
                self.deduper.refresh(tid, prediction)
            prediction["track_id"] = tid
            return tid, False

        if track_id in self._tracks:
            return touch(track_id)

        canonical = self._aliases.get(track_id)
        if canonical is not None and canonical in self._tracks:
            return touch(canonical)

        if cfg.position_dedup:
            existing = self.deduper.find_existing_track(prediction)
            if existing is not None and existing in self._tracks:
                self._aliases[track_id] = existing
                return touch(existing)

        self._tracks[track_id] = _TrackState(
            first_seen=datetime.now(), last_seen=now, last_seen_wall=datetime.now(),
            stem=f"{self._session_tag}_{track_id}", prediction=prediction,
        )
        if cfg.position_dedup:
            self.deduper.remember_save(track_id, prediction)
        return track_id, True

    def _crop_person(self, frame: np.ndarray, prediction: Dict[str, Any]) -> Tuple[Optional[np.ndarray], float]:
        started = time.perf_counter()
        crop = crop_box(frame, prediction, self.cfg.padding_ratio, self.cfg.min_crop_height)
        return crop, round((time.perf_counter() - started) * 1000.0, 2)

    def _remember_sighting(self, state: _TrackState, frame: np.ndarray, prediction: Dict[str, Any],
                           captured_at: Optional[float]) -> None:
        confidence = float(prediction.get("confidence", 0.0))
        if state.best is not None and confidence <= state.best.confidence:
            return
        crop, crop_ms = self._crop_person(frame, prediction)
        if crop is None:
            return
        state.best = _Sighting(
            crop=crop, box=box_edges(prediction), confidence=confidence, captured_at=captured_at,
            detect_ms=self._last_person_detect_ms, crop_ms=crop_ms, frame_shape=frame.shape[:2],
        )

    # ------------------------------------------------------------------ faces
    def _run_faces(self, frame: np.ndarray, track_ids: List[str], inline: bool, captured_at: Optional[float]) -> None:
        worker, cfg = self.face_worker, self.cfg
        if worker is None:
            return

        for track_id in track_ids:
            state = self._tracks[track_id]
            crop, crop_ms = self._crop_person(frame, state.prediction)
            if crop is None:
                continue
            sighting = _Sighting(
                crop=crop, box=box_edges(state.prediction), confidence=float(state.prediction.get("confidence", 0.0)),
                captured_at=captured_at, detect_ms=self._last_person_detect_ms, crop_ms=crop_ms, frame_shape=frame.shape[:2],
            )
            if inline or not cfg.async_face_detection:
                self._apply_face_result(worker.run_inline(track_id, crop, state.attempts + 1), sighting)
            else:
                self._pending_sightings[track_id] = sighting
                if not worker.submit(track_id, crop, state.attempts + 1):
                    self._pending_sightings.pop(track_id, None)

        if cfg.async_face_detection and not inline:
            for result in worker.poll_results():
                self._apply_face_result(result, self._pending_sightings.pop(result.track_id, None))

    def _apply_face_result(self, result: FaceResult, sighting: Optional[_Sighting]) -> None:
        state = self._tracks.get(result.track_id)
        if state is None or state.finalized:
            return
        state.attempts += 1
        exhausted = state.attempts >= self.cfg.face_max_attempts

        hit = result.hit
        if hit is not None and hit.passed_gate and sighting is not None:
            candidate = _FaceCandidate(sighting=sighting, hit=hit, detect_ms=result.detect_ms)
            if state.face is None or hit.quality > state.face.hit.quality:
                state.face = candidate

        good = state.face is not None and state.face.hit.quality >= self.cfg.face_good_enough_quality
        if good or exhausted:
            self._finalize(result.track_id, state)

    def _flush_stale(self, now: float, always_finalize: bool) -> None:
        wait = self.cfg.stale_finalize_seconds
        for track_id, state in list(self._tracks.items()):
            if state.finalized or (state.best is None and state.face is None):
                continue
            if always_finalize or (wait > 0 and now - state.last_seen > wait):
                self._finalize(track_id, state)

    def _prune(self, now: float) -> None:
        self._last_prune = now
        ttl = self.cfg.track_ttl_seconds
        for track_id in [t for t, s in self._tracks.items() if now - s.last_seen > ttl]:
            state = self._tracks[track_id]
            if not state.finalized:
                self._finalize(track_id, state)
            del self._tracks[track_id]
        self._aliases = {a: t for a, t in self._aliases.items() if t in self._tracks}
        self.deduper.prune()

    # ------------------------------------------------------------------ storing
    def _finalize(self, track_id: str, state: _TrackState) -> None:
        if state.finalized:
            return
        state.finalized = True
        sighting = state.face.sighting if state.face is not None else state.best
        if sighting is None:
            return
        face, attempts, first_seen, last_seen = state.face, state.attempts, state.first_seen, state.last_seen_wall
        stem = state.stem
        event_uuid = str(uuid.uuid4())
        detected_at = datetime.now()
        job = FinalizeJob(
            track_id=track_id,
            build_record=lambda: self._build_record(track_id, event_uuid, sighting, face, attempts,
                                                    first_seen, last_seen, detected_at, stem),
        )
        if not self.finalize_pool.submit(job):
            record = self._build_record(track_id, event_uuid, sighting, face, attempts,
                                        first_seen, last_seen, detected_at, stem)
            self.storage.enqueue(record)
        state.best = None
        state.face = None

    def _build_record(self, track_id: str, event_uuid: str, sighting: _Sighting, face: Optional[_FaceCandidate],
                      attempts: int, first_seen: datetime, last_seen: datetime, detected_at: datetime,
                      stem: str) -> PersonRecord:
        cfg, config = self.cfg, self.config
        disk: Optional[Dict[str, bytes]] = {} if cfg.save_images_to_disk else None

        encode_started = time.perf_counter()
        person_jpeg = image_ops.encode_jpeg(sighting.crop, cfg.jpeg_quality)
        person_crop_ms = round((sighting.crop_ms or 0.0) + (time.perf_counter() - encode_started) * 1000.0, 2)
        if disk is not None:
            disk[f"person_detection/{stem}_person.jpg"] = person_jpeg

        face_fields: Dict[str, Any] = {}
        face_detect_ms = face_crop_ms = None
        if face is not None:
            started = time.perf_counter()
            hit = face.hit
            fx1, fy1, fx2, fy2 = hit.box
            margin_x, margin_y = (fx2 - fx1) * cfg.face_margin, (fy2 - fy1) * cfg.face_margin
            crop_h, crop_w = sighting.crop.shape[:2]
            x1, y1 = max(0, int(fx1 - margin_x)), max(0, int(fy1 - margin_y))
            x2, y2 = min(crop_w, int(fx2 + margin_x)), min(crop_h, int(fy2 + margin_y))
            face_crop = sighting.crop[y1:y2, x1:x2]
            face_jpeg = image_ops.encode_jpeg(face_crop, cfg.face_jpeg_quality)
            face_crop_ms = round((time.perf_counter() - started) * 1000.0, 2)
            face_detect_ms = face.detect_ms
            if disk is not None:
                disk[f"face_detection/{stem}_face.jpg"] = face_jpeg
            face_fields = dict(
                face_image_jpeg=face_jpeg,
                face_landmarks=json.dumps([[round(x, 1), round(y, 1)] for x, y in hit.landmarks]),
                face_quality_score=round(hit.quality, 4),
            )

        if sighting.captured_at is not None:
            total_ms = round((time.time() - sighting.captured_at) * 1000.0, 2)
        else:
            total_ms = round(sum(v for v in (sighting.detect_ms, person_crop_ms, face_detect_ms, face_crop_ms) if v), 2)

        record = PersonRecord(
            event_uuid=event_uuid, device_id=config.runtime.device_id or "unknown", session_id=self.ctx.session_id,
            track_id=track_id,
            person_confidence=sighting.confidence, person_image_jpeg=person_jpeg,
            face_detected=face is not None, detected_at=detected_at,
            disk_files=disk, **face_fields,
        )
        print(f"[person] {track_id} stored - face {'found (q=%.2f)' % face.hit.quality if face else 'not found'}"
              f" - person_detect={_fmt(sighting.detect_ms)} total={_fmt(total_ms)}")
        return record

    # ------------------------------------------------------------------ stop
    def stop(self) -> None:
        """Hand in-flight people to storage, stop the workers, free the models."""
        if self._stopped:
            return
        self._stopped = True

        if self.finalize_pool is not None:
            for track_id, state in list(self._tracks.items()):
                if not state.finalized and (state.best is not None or state.face is not None):
                    try:
                        self._finalize(track_id, state)
                    except Exception as error:
                        log.warning("could not store person %s while stopping: %s", track_id, error)
            self.finalize_pool.stop()      # drains: those records reach the store before we return

        face_done = self.face_worker.stop() if self.face_worker is not None else True
        models = [self.person_model]
        self.person_model = None
        self.face_worker = self.finalize_pool = None
        self._tracks.clear()
        self._pending_sightings.clear()
        detector.release_models(*models)
        log.info("[person] stopped, models released%s", "" if face_done else " (face thread still finishing)")


def _fmt(value: Optional[float]) -> str:
    return f"{value:.1f}ms" if value is not None else "n/a"


def _cpu_temp_c() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="ascii") as handle:
            return round(int(handle.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return None