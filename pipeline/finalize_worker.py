import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("pipeline")


@dataclass
class FinalizeJob:
    track_id: str
    build_record: Callable[[], Any]


class FinalizeWorkerPool:
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

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                record = job.build_record()
                if record is not None:
                    self.storage.enqueue(record)
            except Exception as error:
                log.warning("[finalize] job for %s failed: %s", job.track_id, error, exc_info=True)

    def stop(self, timeout: float = 5.0) -> None:
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
