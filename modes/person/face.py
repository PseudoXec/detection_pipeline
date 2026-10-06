"""Face detection for the person mode, on its own background thread.

YuNet (OpenCV's `cv2.FaceDetectorYN`) looks for a face in the upper part of a person crop. It runs off
the frame loop for the same reason the plate model does in vehicle mode: a slow or backlogged face
model must never delay person detection, tracking or the live overlay. The worker never touches the
mode's track state - it only exchanges FaceJob / FaceResult objects through two queues, and the mode
folds results back in from its own thread.

`cv2.FaceDetectorYN` is not thread-safe, so exactly one thread (this worker) ever calls it.
"""
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from modes.person import quality

log = logging.getLogger("pipeline")


@dataclass
class FaceHit:
    box: Tuple[float, float, float, float]            # x1, y1, x2, y2 in PERSON-CROP pixels
    landmarks: List[Tuple[float, float]]              # 5 points, PERSON-CROP pixels
    score: float                                      # detector confidence
    sharpness: float                                  # Laplacian variance of the face region
    quality: float                                    # combined score, see quality.quality_score
    passed_gate: bool                                 # big enough and sharp enough to keep


class YuNetFaceDetector:
    """Wrapper that turns a person crop into the best FaceHit (or None)."""

    def __init__(self, weights: str, score_threshold: float, search_region: List[float],
                 min_face_px: int, min_sharpness: float, ref_px: float, ref_sharpness: float):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Face model not found at: {weights}\n"
                f"Download face_detection_yunet_2023mar.onnx from the OpenCV Zoo "
                f"(models/face_detection_yunet folder of github.com/opencv/opencv_zoo) into models/, "
                f"or set person.face_enabled: false.")
        import cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ImportError("this OpenCV build has no FaceDetectorYN - install opencv-python >= 4.8")
        self._detector = cv2.FaceDetectorYN.create(weights, "", (320, 320), score_threshold)
        self.search_region = search_region
        self.min_face_px = min_face_px
        self.min_sharpness = min_sharpness
        self.ref_px = ref_px
        self.ref_sharpness = ref_sharpness

    def detect(self, person_crop: np.ndarray) -> Optional[FaceHit]:
        if person_crop is None or person_crop.size == 0:
            return None
        height, width = person_crop.shape[:2]
        fx0, fy0, fx1, fy1 = self.search_region
        x0, y0 = max(0, int(width * fx0)), max(0, int(height * fy0))
        x1, y1 = min(width, int(width * fx1)), min(height, int(height * fy1))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None

        region = np.ascontiguousarray(person_crop[y0:y1, x0:x1])
        self._detector.setInputSize((region.shape[1], region.shape[0]))
        _, faces = self._detector.detect(region)
        if faces is None or len(faces) == 0:
            return None

        # rows: x, y, w, h, 5 landmark (x, y) pairs, score. Keep the most confident face.
        row = max(faces, key=lambda r: float(r[14]))
        fx, fy, fw, fh = (float(v) for v in row[:4])
        box = (max(0.0, fx + x0), max(0.0, fy + y0), min(float(width), fx + fw + x0), min(float(height), fy + fh + y0))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        landmarks = [(float(row[4 + 2 * i]) + x0, float(row[5 + 2 * i]) + y0) for i in range(5)]

        face_region = person_crop[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face_sharpness = quality.sharpness(face_region)
        min_side = min(box[2] - box[0], box[3] - box[1])
        score = float(row[14])
        return FaceHit(
            box=box, landmarks=landmarks, score=score, sharpness=face_sharpness,
            quality=quality.quality_score(score, min_side, face_sharpness, self.ref_px, self.ref_sharpness),
            passed_gate=quality.passes_gate(min_side, face_sharpness, self.min_face_px, self.min_sharpness),
        )


@dataclass
class FaceJob:
    track_id: str
    person_crop: np.ndarray
    attempt: int


@dataclass
class FaceResult:
    track_id: str
    person_crop: np.ndarray       # the exact crop the face was searched in (stored if this face wins)
    hit: Optional[FaceHit]
    attempt: int
    detect_ms: Optional[float]
    error: Optional[str] = None


class FaceDetectionWorker:
    def __init__(self, detector: Any, max_queue: int = 8):
        self.detector = detector
        self._jobs: "queue.Queue[FaceJob]" = queue.Queue(maxsize=max(1, max_queue))
        self._results: "queue.Queue[FaceResult]" = queue.Queue()
        self._pending: set = set()
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._last_drop_log = 0.0

    def start(self) -> "FaceDetectionWorker":
        self._thread = threading.Thread(target=self._run, name="face-worker", daemon=True)
        self._thread.start()
        log.info("[face-worker] started (max_queue=%d)", self._jobs.maxsize)
        return self

    def stop(self) -> bool:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            return not self._thread.is_alive()
        return True

    def is_pending(self, track_id: str) -> bool:
        with self._pending_lock:
            return track_id in self._pending

    def submit(self, track_id: str, person_crop: np.ndarray, attempt: int) -> bool:
        """Non-blocking. False = skipped for now (already in flight, or backlogged); the mode's retry
        cadence tries again, and no attempt is consumed until a result actually comes back."""
        if person_crop is None:
            return False
        with self._pending_lock:
            if track_id in self._pending:
                return False
            self._pending.add(track_id)
        try:
            self._jobs.put_nowait(FaceJob(track_id, person_crop, attempt))
            return True
        except queue.Full:
            with self._pending_lock:
                self._pending.discard(track_id)
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 5.0:
                log.warning("[face-worker] queue full - skipped %d face attempt(s) so far "
                            "(more people in view than the Pi can read faces for)", self._dropped)
                self._last_drop_log = now
            return False

    def poll_results(self) -> List[FaceResult]:
        out: List[FaceResult] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def run_inline(self, track_id: str, person_crop: np.ndarray, attempt: int) -> FaceResult:
        """Same work on the caller's thread (person.async_face_detection: false)."""
        return self._detect(FaceJob(track_id, person_crop, attempt))

    def _detect(self, job: FaceJob) -> FaceResult:
        started = time.monotonic()
        try:
            hit = self.detector.detect(job.person_crop)
            return FaceResult(job.track_id, job.person_crop, hit, job.attempt, (time.monotonic() - started) * 1000.0)
        except Exception as error:   # keep the worker alive; report the failure per job
            log.warning("[face-worker] face detection failed for %s: %s", job.track_id, error, exc_info=True)
            return FaceResult(job.track_id, job.person_crop, None, job.attempt,
                              (time.monotonic() - started) * 1000.0, str(error))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            result = self._detect(job)
            with self._pending_lock:
                self._pending.discard(job.track_id)
            self._results.put(result)
