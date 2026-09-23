"""
live_server.py
--------------
Serves the live camera view FROM the Pi, so nothing has to be pushed to a
server endpoint. (The dashboard's existing endpoint is built on the SQLite
`detections` columns - one row per finished vehicle - and has no place for
video frames. Frames are a different kind of data: continuous, throw-away,
big. So the dashboard PULLS them from here instead.)

    GET /live/stream.mjpg   multipart MJPEG, opens in a browser / WebView2 / any MJPEG client
    GET /live/frame.jpg     one JPEG (poll this from WPF if you don't want a stream parser)
    GET /live/boxes         latest tracker boxes as JSON (draw your own overlay)
    GET /live/health        {"ok": true, ...} - is the Pi alive, how old is the newest frame

    ?overlay=0|1            burn the boxes into the image (default 1; use 0 for the raw frame)
    ?token=SECRET           or header "Authorization: Bearer SECRET" (only if live.auth_token is set)

Design rules (same spirit as box_publisher.py - this must never hurt detection):
    * publish() is called from the hot loop: it only builds a small dict and
      swaps a reference. No I/O, never blocks, never raises.
    * Frames come straight from the camera thread (camera.read_latest), NOT
      from the detection loop, so video stays smooth even when inference is slow.
    * JPEG encoding happens on the HTTP threads, only while someone is
      watching, and once per frame no matter how many viewers are connected.
    * If the port can't be bound, the server logs it and stays off; the
      pipeline keeps running.
"""

import hmac
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import cv2

from live.snapshot import build_snapshot

log = logging.getLogger("pipeline")

_BOUNDARY = "frame"
_GREEN = (0, 200, 0)       # vehicle inside the ROI (will be saved)
_GRAY = (170, 170, 170)    # tracked but outside the ROI


class LiveServer:
    def __init__(
        self,
        camera_id: str,
        host: str = "0.0.0.0",
        port: int = 8090,
        stream_fps: float = 10.0,
        stream_width: int = 960,
        jpeg_quality: int = 70,
        box_max_age_seconds: float = 1.0,
        auth_token: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.min_interval = 1.0 / stream_fps if stream_fps > 0 else 0.0
        self.stream_width = stream_width
        self.jpeg_quality = int(jpeg_quality)
        self.box_max_age_seconds = box_max_age_seconds
        self.auth_token = auth_token or None
        self.session_id = uuid.uuid4().hex[:8]

        self._frame_source: Optional[Callable[[], Any]] = None   # -> CapturedFrame | None
        self._snapshot: Optional[Dict[str, Any]] = None          # newest boxes (reference swap = atomic)
        self._seq = 0

        self._enc_lock = threading.Lock()
        self._cache: Dict[bool, Tuple[Tuple, bytes]] = {}        # overlay flag -> (key, jpeg)

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    def set_frame_source(self, source: Callable[[], Any]) -> None:
        """`source()` must return an object with .image (BGR ndarray) and
        .frame_number, or None while no frame has arrived (camera.read_latest)."""
        self._frame_source = source

    def start(self) -> "LiveServer":
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        except OSError as error:
            log.error("[live] cannot listen on %s:%d (%s) - live view stays off", self.host, self.port, error)
            self._httpd = None
            return self
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="live-server", daemon=True)
        self._thread.start()
        log.info("[live] serving %s at http://%s:%d/live/stream.mjpg (also /live/frame.jpg, /live/boxes)",
                 self.camera_id, "<pi-ip>" if self.host in ("0.0.0.0", "") else self.host, self.port)
        if not self.auth_token:
            log.warning("[live] live.auth_token is not set - anyone who can reach this port can watch the camera")
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # called from the detection loop - must be fast and must never raise
    # ------------------------------------------------------------------ #
    def publish(
        self,
        frame_shape: tuple,
        predictions: List[Dict[str, Any]],
        in_roi_ids: Set[int],
        captured_at: Optional[float] = None,
    ) -> None:
        try:
            self._seq += 1
            self._snapshot = build_snapshot(
                self.camera_id, self.session_id, self._seq,
                frame_shape, predictions, in_roi_ids, captured_at,
            )
        except Exception as error:
            log.debug("[live] publish skipped: %s", error)

    # ------------------------------------------------------------------ #
    # rendering (runs on the HTTP threads)
    # ------------------------------------------------------------------ #
    def _fresh_boxes(self) -> Optional[Dict[str, Any]]:
        snap = self._snapshot
        if snap is None or time.time() - snap["processed_at"] > self.box_max_age_seconds:
            return None      # detector stalled or nothing yet: better no boxes than wrong boxes
        return snap

    def _render(self, overlay: bool) -> Optional[Tuple[Tuple, bytes]]:
        """Newest frame as JPEG bytes, cached so N viewers cost one encode."""
        captured = self._frame_source() if self._frame_source else None
        if captured is None:
            return None
        snap = self._fresh_boxes() if overlay else None
        key = (captured.frame_number, snap["seq"] if snap else 0)

        with self._enc_lock:
            cached = self._cache.get(overlay)
            if cached and cached[0] == key:
                return cached

            image = captured.image
            src_h, src_w = image.shape[:2]
            if self.stream_width and src_w > self.stream_width:
                out_w = self.stream_width
                out_h = int(round(src_h * out_w / src_w))
                image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)   # new array
            elif overlay and snap:
                image = image.copy()          # never draw on the camera thread's array
            if snap:
                self._draw(image, snap)

            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return None
            result = (key, buf.tobytes())
            self._cache[overlay] = result
            return result

    @staticmethod
    def _draw(image, snap: Dict[str, Any]) -> None:
        # boxes are in the detector frame's pixels; scale to the image we are sending
        sx = image.shape[1] / (snap["frame"]["width"] or 1)
        sy = image.shape[0] / (snap["frame"]["height"] or 1)
        for v in snap["vehicles"]:
            colour = _GREEN if v["in_roi"] else _GRAY
            p1 = (int(v["x1"] * sx), int(v["y1"] * sy))
            p2 = (int(v["x2"] * sx), int(v["y2"] * sy))
            cv2.rectangle(image, p1, p2, colour, 2)
            label = f'#{v["track_id"]} {v["class"]} {v["confidence"]:.2f}'
            cv2.putText(image, label, (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):          # keep journald quiet
                pass

            # -- helpers -------------------------------------------------
            def _authorized(self, query) -> bool:
                if not server.auth_token:
                    return True
                supplied = query.get("token", [""])[0]
                header = self.headers.get("Authorization", "")
                if header.startswith("Bearer "):
                    supplied = header[7:]
                return hmac.compare_digest(supplied.encode(), server.auth_token.encode())

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload: Any) -> None:
                self._send(code, json.dumps(payload).encode(), "application/json")

            # -- routes --------------------------------------------------
            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                try:
                    if not self._authorized(query):
                        return self._json(401, {"error": "unauthorized"})
                    overlay = query.get("overlay", ["1"])[0] not in ("0", "false", "no")

                    if url.path == "/live/health":
                        snap = server._snapshot
                        cap = server._frame_source() if server._frame_source else None
                        return self._json(200, {
                            "ok": True,
                            "camera_id": server.camera_id,
                            "session_id": server.session_id,
                            "has_frame": cap is not None,
                            "boxes_age_seconds": round(time.time() - snap["processed_at"], 2) if snap else None,
                        })

                    if url.path == "/live/boxes":
                        snap = server._snapshot
                        if snap is None:
                            return self._json(503, {"error": "no detections processed yet"})
                        return self._json(200, dict(snap, age_seconds=round(time.time() - snap["processed_at"], 3)))

                    if url.path == "/live/frame.jpg":
                        out = server._render(overlay)
                        if out is None:
                            return self._json(503, {"error": "no frame from camera yet"})
                        return self._send(200, out[1], "image/jpeg")

                    if url.path == "/live/stream.mjpg":
                        return self._stream(overlay)

                    self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass       # viewer closed the tab / app - normal

            def _stream(self, overlay: bool) -> None:
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                last_key = None
                while not server._stop_event.is_set():
                    started = time.monotonic()
                    out = server._render(overlay)
                    if out is None or out[0] == last_key:
                        time.sleep(0.03)           # nothing new yet
                        continue
                    last_key, jpeg = out
                    self.wfile.write(
                        f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    pause = server.min_interval - (time.monotonic() - started)
                    if pause > 0:
                        time.sleep(pause)

        return Handler
