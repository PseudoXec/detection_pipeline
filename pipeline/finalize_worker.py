"""
finalize_worker.py
-------------------
Runs the expensive TAIL of finalizing a vehicle - the OCR read, JPEG
encoding, and building + enqueuing the DetectionRecord - on background
thread(s), off the real-time detection/tracking loop.

Why this exists
----------------
PaddleOCR's CPU inference is the single most expensive thing this pipeline
does per vehicle (often 100-400+ ms on a Pi 5, sometimes for two image
variants). It used to run INLINE inside `DetectionPipeline.process_frame`,
which meant:
  * every vehicle with a plate stalled the vehicle model / ByteTrack for the
    OCR's entire duration, so the live view's *boxes* jumped or lagged even
    though the video itself stayed smooth (it comes straight from the camera
    thread) - not "seamless".
  * a second vehicle entering frame while OCR ran on the first one was
    detected late, sometimes late enough for ByteTrack to lose or fragment
    its track, hurting accuracy, not just smoothness.

`_evaluate_plate` still does the FAST work synchronously (crop + enhance,
a few ms) so the best plate BOX is picked using the plate detector's own
confidence - exactly the fallback ranking this pipeline already used for
"no OCR available". Only the slow OCR call + final record build + storage
handoff move here, and only once per vehicle (at finalize time), not once
per retry attempt.

Design rules (same spirit as live/live_server.py and live/box_publisher.py):
  * `submit()` is called from the hot loop. It never blocks: a full queue
    drops the job (loudly) rather than ever stalling detection.
  * Each job is a self-contained closure (`build_record`) that only touches
    values captured at submit time - never the shared `_tracks` dict - so
    there is nothing to lock and no race with the detection loop reusing a
    track_id.
  * A worker thread crashing on one job must never take down the others or
    the pipeline; every job is wrapped in try/except.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("pipeline")


@dataclass
class FinalizeJob:
    track_id: str
    # does the OCR read + JPEG encode + builds a DetectionRecord (or None to
    # skip storing, e.g. a degenerate crop) - runs entirely on the worker thread
    build_record: Callable[[], Any]


class FinalizeWorkerPool:
    """Fixed-size pool of daemon threads draining a bounded finalize-job queue."""

    def __init__(self, storage, worker_threads: int = 1, max_queue: int = 32):
        self.storage = storage
        self._queue: "queue.Queue[Optional[FinalizeJob]]" = queue.Queue(maxsize=max(1, max_queue))
        self._threads = [
            threading.Thread(target=self._loop, name=f"finalize-{i}", daemon=True)
            for i in range(max(1, worker_threads))
        ]

    def start(self) -> "FinalizeWorkerPool":
        for thread in self._threads:
            thread.start()
        return self

    def submit(self, job: FinalizeJob) -> bool:
        """Never blocks the caller (the detection loop). Returns False - and
        logs - if the queue is already full, i.e. OCR is falling behind the
        camera; the vehicle is lost rather than the frame loop stalling."""
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            log.warning(
                "[finalize] queue full (%d pending) - dropping %s; OCR/storage is falling "
                "behind the camera, consider ocr.worker_threads or a faster OCR model",
                self._queue.qsize(), job.track_id,
            )
            return False

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:            # shutdown sentinel
                return
            try:
                record = job.build_record()
                if record is not None:
                    self.storage.enqueue(record)
            except Exception as error:      # one bad job must never kill the worker thread
                log.warning("[finalize] job for %s failed: %s", job.track_id, error, exc_info=True)

    def stop(self, timeout: float = 5.0) -> None:
        """Lets every already-queued job finish (so nothing already accepted is
        silently dropped) before the threads exit."""
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
