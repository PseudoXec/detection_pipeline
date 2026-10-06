"""Runs plate detection on a background thread, off the frame-reading / vehicle-tracking loop.

Why this exists
----------------
`VehicleMode.process_frame()` used to call the plate model inline, once per vehicle that was
due for an attempt, before it could go back and read the next camera frame. With 2-4 vehicles in
view (normal for a street camera) and up to `tracking.max_plate_attempts` retries each, that inline
inference is the main reason the effective processing rate falls well below `camera.max_fps`:

* Boxes look laggy / low-fps: the live view's box snapshot is only published once per processed
  frame, so a slow frame loop means a slow-updating overlay even if the raw video itself is smooth.
* Boxes "slide late" / jump: with fewer, unevenly-spaced updates, a vehicle moves further between
  two processed frames, so its box has to jump further to catch up.
* Small objects get oversized boxes: a track that goes several hundred ms between real detections
  spends longer being purely Kalman-predicted (no measurement correction), which is exactly when
  its estimated size can drift before the next detection pulls it back.

Moving the (slower, more variable-cost) plate model to its own thread lets vehicle detection +
tracking + the live overlay run at the camera's actual frame rate, while plate reads happen in the
background and get folded back into the track state as they finish - a frame or two later, not
blocking the one that submitted them.

Thread-safety: this worker never touches `VehicleMode._tracks` (or any of the pipeline's own
state). It only exchanges `PlateJob` / `PlateResult` objects through two queues; the pipeline reads
results and updates its own state from its own (main) thread only, exactly as it already did before
this worker existed.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from core import detector
from modes.vehicle import plate_enhance

log = logging.getLogger("pipeline")


@dataclass
class PlateJob:
    track_id: str
    crop: np.ndarray
    attempt: int
    submitted_at: float


@dataclass
class PlateResult:
    track_id: str
    predictions: List[Dict[str, Any]]
    attempt: int
    detect_ms: Optional[float]
    error: Optional[str] = None


class PlateDetectionWorker:
    def __init__(
        self,
        model: Any,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: "int | list | tuple",
        enhance_before_detect: bool,
        max_queue: int = 12,
        batch_window_seconds: float = 0.03,
        max_batch: int = 4,
    ):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.enhance_before_detect = enhance_before_detect
        self.batch_window_seconds = batch_window_seconds
        self.max_batch = max_batch

        self._jobs: "queue.Queue[PlateJob]" = queue.Queue(maxsize=max_queue)
        self._results: "queue.Queue[PlateResult]" = queue.Queue()
        self._pending: set = set()  # track_ids currently queued or being processed
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._last_drop_log = 0.0

    def start(self) -> "PlateDetectionWorker":
        self._thread = threading.Thread(target=self._run, name="plate-worker", daemon=True)
        self._thread.start()
        log.info("[plate-worker] started (max_queue=%d, max_batch=%d)",
                 self._jobs.maxsize, self.max_batch)
        return self

    def stop(self) -> bool:
        """Stop the thread. True when it really ended (so its model can be freed safely)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            return not self._thread.is_alive()
        return True

    def is_pending(self, track_id: str) -> bool:
        with self._pending_lock:
            return track_id in self._pending

    def submit(self, track_id: str, crop: np.ndarray, attempt: int) -> bool:
        """Non-blocking. False means the attempt was skipped for now (already in flight, or the
        worker is backlogged) - the caller's normal retry cadence will try again; nothing is
        silently lost since a vehicle is still finalized on exit / TTL either way."""
        if crop is None:
            return False
        with self._pending_lock:
            if track_id in self._pending:
                return False
            self._pending.add(track_id)
        try:
            self._jobs.put_nowait(PlateJob(track_id, crop, attempt, time.monotonic()))
            return True
        except queue.Full:
            with self._pending_lock:
                self._pending.discard(track_id)
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 5.0:
                log.warning("[plate-worker] queue full - dropped %d plate attempt(s) so far "
                            "(the Pi can't keep up with plate reads at the current vehicle volume)",
                            self._dropped)
                self._last_drop_log = now
            return False

    def poll_results(self) -> List[PlateResult]:
        """Non-blocking drain - call once per frame from the main thread."""
        out: List[PlateResult] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.batch_window_seconds
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._jobs.get(timeout=remaining))
                except queue.Empty:
                    break
            self._process_batch(batch)

    def _process_batch(self, batch: List[PlateJob]) -> None:
        crops = [job.crop for job in batch]
        error = None
        detect_ms = None
        try:
            if self.enhance_before_detect:
                crops = [plate_enhance.sharpen_and_denoise(c) for c in crops]
            t0 = time.monotonic()
            results = detector.detect_batch(
                self.model, crops, self.conf_threshold, self.iou_threshold, self.imgsz,
            )
            detect_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # keep the worker alive - report the failure per job instead
            log.warning("[plate-worker] batch of %d failed: %s", len(batch), exc, exc_info=True)
            results = [[] for _ in batch]
            error = str(exc)

        for job, predictions in zip(batch, results):
            with self._pending_lock:
                self._pending.discard(job.track_id)
            self._results.put(PlateResult(job.track_id, predictions, job.attempt, detect_ms, error))
