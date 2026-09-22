"""
camera.py
---------
Reads frames from an RTSP camera (or a static image/folder) in a dedicated
background thread.

Why a background thread matters on a Raspberry Pi 5:
    RTSP decode (FFmpeg, software H.264 since the Pi 5 has no hardware
    decoder wired up in OpenCV) is itself CPU work. If we called
    cap.read() directly inside the main detection loop, every model
    inference call would delay the NEXT frame read, frames would queue up
    inside the OS network buffer, and we'd end up processing stale,
    delayed video instead of "now".

    Instead, this thread does nothing but grab frames as fast as the
    camera sends them and keep only the newest one. The main pipeline loop
    just asks "what's the latest frame?" whenever it's ready, which keeps
    detection results tied to (near) real time even if inference is slower
    than the camera's frame rate.
"""

# os is used to configure FFmpeg's decode-thread environment variable
import os
# threading runs the frame-grabbing loop independently of the detection loop
import threading
# time is used for reconnect back-off delays and frame timestamps
import time
# dataclass gives us a small, typed container for "one frame plus metadata"
from dataclasses import dataclass
from typing import Optional

# cv2.VideoCapture is what actually talks to the RTSP stream / files
import cv2
import numpy as np


@dataclass
class CapturedFrame:
    """One frame read from the camera, tagged with when and which number it was."""
    image: np.ndarray      # the decoded BGR frame
    frame_number: int      # monotonically increasing counter since this camera started
    captured_at: float     # time.time() timestamp of when it was read


def configure_decode_threads(decode_threads: int) -> None:
    """Pin FFmpeg's RTSP decoder to a fixed thread count.

    Must run before the first cv2.VideoCapture(...) call. Left unpinned,
    FFmpeg happily uses every core it can see, which starves the inference
    threads running alongside it and causes dropped/corrupted frames
    (visible as "Could not find ref with POC #" in OpenCV's console output).
    """
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        f"rtsp_transport;tcp|threads;{decode_threads}",
    )


class ThreadedRTSPCamera:
    """Background-threaded RTSP reader that always exposes the newest frame."""

    def __init__(
        self,
        rtsp_url: str,
        frame_width: int = 1280,
        frame_height: int = 720,
        buffer_size: int = 1,
        reconnect_delay_seconds: float = 5.0,
        max_reconnect_attempts: int = 0,
    ):
        self.rtsp_url = rtsp_url
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.buffer_size = buffer_size
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_attempts = max_reconnect_attempts

        # guards access to `_latest` between the grabber thread and callers
        self._lock = threading.Lock()
        # the newest frame we have available; None until the first frame arrives
        self._latest: Optional[CapturedFrame] = None
        # set once stop() is called, tells the grabber thread to exit
        self._stop_event = threading.Event()
        # counts frames since this camera object was created
        self._frame_counter = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedRTSPCamera":
        """Open the stream and start the background grabber thread."""
        self._open_capture()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self) -> None:
        """(Re)connect to the RTSP source. Raises RuntimeError if it can't open."""
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # ask for a specific resolution; the camera may ignore this
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        # a buffer size of 1 means OpenCV won't queue up old frames internally -
        # every read() call gets the most recently decoded frame
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open RTSP stream: {self.rtsp_url}")

        self._capture = capture

    def _grab_loop(self) -> None:
        """Runs forever in the background thread until stop() is called."""
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ok, frame = self._capture.read() if self._capture is not None else (False, None)

            if not ok or frame is None:
                consecutive_failures += 1
                print(f"[camera] read failed ({consecutive_failures} in a row) - reconnecting...")
                self._reconnect()
                continue

            consecutive_failures = 0
            self._frame_counter += 1

            # publish the new frame under the lock so callers never see a
            # half-updated frame/frame_number pair
            with self._lock:
                self._latest = CapturedFrame(
                    image=frame, frame_number=self._frame_counter, captured_at=time.time(),
                )

    def _reconnect(self) -> None:
        """Release the dead capture and try to open a fresh one, with back-off."""
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
        """Return the newest available frame, or None if nothing has arrived yet."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Signal the grabber thread to exit and release the camera handle."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._capture is not None:
            self._capture.release()
