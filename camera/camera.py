import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("pipeline")


@dataclass
class CapturedFrame:
    image: np.ndarray
    frame_number: int
    captured_at: float


def configure_decode_threads(decode_threads: int) -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;{decode_threads}"
    )


class ThreadedRTSPCamera:
    def __init__(
        self,
        rtsp_url: str,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        buffer_size: int = 1,
        reconnect_delay_seconds: float = 5.0,
        max_reconnect_attempts: int = 0,
        max_fps: float = 0.0,
    ):
        self.rtsp_url = rtsp_url
        self.max_fps = max_fps
        # Only a fallback: the camera decides the resolution, never this program. The real size is
        # read from the first decoded frame of every (re)connect and published as actual_width/height.
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.actual_width: Optional[int] = None
        self.actual_height: Optional[int] = None
        self.buffer_size = buffer_size
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_attempts = max_reconnect_attempts

        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._stop_event = threading.Event()
        self._frame_counter = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._logged_actual_size = False

    def start(self) -> "ThreadedRTSPCamera":
        try:
            self._open_capture()
            print(f"[camera] connected to {self.rtsp_url}")
        except RuntimeError as error:
            print(f"[camera] {error}")
            print(f"[camera] retrying every {self.reconnect_delay_seconds:.0f}s until the camera is reachable...")
            self._reconnect()

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self) -> None:
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # No CAP_PROP_FRAME_WIDTH/HEIGHT here: they are ignored for RTSP under the FFmpeg backend, so
        # they only pretended the size was configurable. The stream's own size is used instead.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open RTSP stream: {self.rtsp_url}")

        self._capture = capture
        self._logged_actual_size = False  # log again on every (re)connect - the camera may have changed profile

    def _grab_loop(self) -> None:
        consecutive_failures = 0
        min_interval = 1.0 / self.max_fps if self.max_fps and self.max_fps > 0 else 0.0
        last_kept = 0.0

        while not self._stop_event.is_set():
            capture = self._capture
            ok = capture.grab() if capture is not None else False
            frame = None
            if ok and min_interval:
                now = time.monotonic()
                if now - last_kept < min_interval * 0.85:
                    continue
                last_kept = now
            if ok:
                ok, frame = capture.retrieve()

            if not ok or frame is None:
                consecutive_failures += 1
                print(f"[camera] read failed ({consecutive_failures} in a row) - reconnecting...")
                self._reconnect()
                continue

            consecutive_failures = 0
            self._frame_counter += 1

            if not self._logged_actual_size:
                self._logged_actual_size = True
                self._check_actual_resolution(frame)

            with self._lock:
                self._latest = CapturedFrame(
                    image=frame, frame_number=self._frame_counter, captured_at=time.time(),
                )

    def _check_actual_resolution(self, frame: np.ndarray) -> None:
        """Record the resolution the camera really sends (the first frame after each (re)connect).
        Everything downstream (warm-up, the ROI crop window, the size reported to the command
        center) follows this, not the configured camera.frame_width/height."""
        actual_h, actual_w = frame.shape[:2]
        previous = (self.actual_width, self.actual_height)
        self.actual_width, self.actual_height = actual_w, actual_h
        if previous == (None, None):
            log.info("[camera] stream resolution confirmed: %dx%d", actual_w, actual_h)
        elif previous != (actual_w, actual_h):
            log.warning(
                "[camera] stream resolution changed from %dx%d to %dx%d after a reconnect "
                "(the camera switched profile) - following the new size",
                previous[0], previous[1], actual_w, actual_h,
            )

    def _reconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            time.sleep(self.reconnect_delay_seconds)
            try:
                self._open_capture()
                print(f"[camera] reconnected after {attempt} attempt(s)")
                return
            except RuntimeError as error:
                print(f"[camera] {error}")
                if self.max_reconnect_attempts and attempt >= self.max_reconnect_attempts:
                    print(f"[camera] giving up after {attempt} reconnect attempt(s)")
                    self._stop_event.set()
                    return

    def read_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._capture is not None:
            self._capture.release()
