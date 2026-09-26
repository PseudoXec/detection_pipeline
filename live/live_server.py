"""
live_server.py
--------------
Serves the live camera view FROM the Pi, so nothing has to be pushed to a
server endpoint. (The dashboard's existing endpoint is built on the SQLite
`detections` columns - one row per finished vehicle - and has no place for
video frames. Frames are a different kind of data: continuous, throw-away,
big. So the dashboard PULLS them from here instead.)

    GET  /live/stream.mjpg   multipart MJPEG, opens in a browser / WebView2 / any MJPEG client
    GET  /live/frame.jpg     one JPEG (poll this from WPF if you don't want a stream parser)
    GET  /live/boxes         latest tracker boxes as JSON (draw your own overlay)
    GET  /live/health        {"ok": true, ...} - is the Pi alive, how old is the newest frame
    GET  /live/roi           current ROI polygon: {"polygon": [[x,y], ...]} (normalized 0..1)
    POST /live/roi           front end POSTs a redrawn ROI polygon here, same shape as the GET
                             above ({"polygon": [[x,y], ...]}, at least 3 points). Point-in-polygon
                             (not a bounding box) decides ROI membership - see detection/geometry.py.
                             Takes effect on the very next frame; 400 if the body is missing/invalid.

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
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from live.box_flow import BoxFlow
from live.snapshot import build_snapshot

log = logging.getLogger("pipeline")

_BOUNDARY = "frame"
_GREEN = (0, 200, 0)       # every box is green (in_roi is still reported by /live/boxes)
_LINE_THICKNESS = 1        # box outline, in pixels of the streamed image
_FONT_SCALE = 0.4          # label text size
_BLEND_SECONDS = 0.4       # when a fresh detection corrects a box, glide to the new position over about this long


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
        box_extrapolate: bool = True,
        box_extrapolate_max_seconds: float = 3.5,
        box_visual_tracking: bool = True,
        roi_get: Optional[Callable[[], List[List[float]]]] = None,
        roi_set: Optional[Callable[[List[List[float]]], None]] = None,
    ):
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.min_interval = 1.0 / stream_fps if stream_fps > 0 else 0.0
        self.stream_width = stream_width
        self.jpeg_quality = int(jpeg_quality)
        self.box_max_age_seconds = box_max_age_seconds
        self.auth_token = auth_token or None
        # The detector only produces boxes every ~1-2 s on this hardware, but the
        # video runs at ~10 fps. With extrapolation on, each box is slid along the
        # vehicle's measured velocity (from its last two detections) so it keeps up
        # with the picture between detections instead of jumping once per update.
        self.box_extrapolate = box_extrapolate
        self.box_extrapolate_max_seconds = box_extrapolate_max_seconds
        # Follows the picture inside each box (optical flow) so a vehicle's box moves
        # from its FIRST detection - guessing speed from detections needs two of them.
        # The extrapolation above stays as the fallback when the picture can't be followed.
        self._flow: Optional[BoxFlow] = BoxFlow() if box_visual_tracking else None
        self._shown: Dict[Any, Dict[str, Any]] = {}      # track_id -> what we last drew (for blending)
        self.session_id = uuid.uuid4().hex[:8]

        # wired up by pipeline.py so the front end can read/redraw the ROI
        # polygon without either side needing to know about the other's internals
        self._roi_get = roi_get
        self._roi_set = roi_set

        self._frame_source: Optional[Callable[[], Any]] = None   # -> CapturedFrame | None
        self._snapshot: Optional[Dict[str, Any]] = None          # newest boxes (reference swap = atomic)
        self._seq = 0
        self._motion: Optional[Tuple[int, float, Dict[Any, Tuple[float, ...]]]] = None   # (seq, frame time, track_id -> px/s)
        self._prev_boxes: Optional[Tuple[float, Dict[Any, Tuple[float, ...]]]] = None    # (frame time, track_id -> box)

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
        log.info("[live] serving %s at http://%s:%d/live/stream.mjpg (also /live/frame.jpg, /live/boxes, /live/roi)",
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
            snapshot = build_snapshot(
                self.camera_id, self.session_id, self._seq,
                frame_shape, predictions, in_roi_ids, captured_at,
            )
            if self.box_extrapolate:
                self._motion = self._estimate_motion(snapshot)     # set BEFORE the snapshot goes live
            self._snapshot = snapshot
        except Exception as error:
            log.debug("[live] publish skipped: %s", error)

    def _estimate_motion(self, snap: Dict[str, Any]) -> Tuple[int, float, Dict[Any, Tuple[float, ...]]]:
        """Per-track velocity (x1, y1, x2, y2 in px/s) from this snapshot vs the
        previous one. Uses the time each FRAME was captured, not when inference
        finished, so a slow detector doesn't distort the speed."""
        frame_time = snap["captured_at"] if snap.get("captured_at") is not None else snap["processed_at"]
        boxes = {
            v["track_id"]: (v["x1"], v["y1"], v["x2"], v["y2"])
            for v in snap["vehicles"] if v.get("track_id") is not None
        }
        velocities: Dict[Any, Tuple[float, ...]] = {}
        previous = self._prev_boxes
        if previous is not None:
            prev_time, prev_boxes = previous
            elapsed = frame_time - prev_time
            if 0.05 <= elapsed <= 5.0:                 # ignore duplicate frames and long gaps
                for track_id, box in boxes.items():
                    old = prev_boxes.get(track_id)
                    if old is not None:
                        velocities[track_id] = tuple((box[i] - old[i]) / elapsed for i in range(4))
        self._prev_boxes = (frame_time, boxes)
        return (snap["seq"], frame_time, velocities)

    def _project_boxes(self, snap: Dict[str, Any], frame_time: Optional[float]) -> List[Dict[str, Any]]:
        """The snapshot's vehicles moved forward to the time of the frame being
        drawn. Boxes with no velocity yet (first sighting) are drawn where they were detected."""
        vehicles = snap["vehicles"]
        motion = self._motion
        if not self.box_extrapolate or motion is None or motion[0] != snap["seq"] or frame_time is None:
            return vehicles
        ahead = min(max(frame_time - motion[1], 0.0), self.box_extrapolate_max_seconds)
        if ahead <= 0 or not motion[2]:
            return vehicles
        moved_vehicles = []
        for v in vehicles:
            velocity = motion[2].get(v.get("track_id"))
            if velocity is None:
                moved_vehicles.append(v)
                continue
            x1, y1 = v["x1"] + velocity[0] * ahead, v["y1"] + velocity[1] * ahead
            x2, y2 = v["x2"] + velocity[2] * ahead, v["y2"] + velocity[3] * ahead
            if x2 <= x1 or y2 <= y1:                   # shrinking box extrapolated to nothing: don't trust it
                moved_vehicles.append(v)
                continue
            moved_vehicles.append(dict(v, x1=x1, y1=y1, x2=x2, y2=y2))
        return moved_vehicles

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

            if overlay and self._flow is not None:
                # remember every frame (small, grayscale) so a new detection can be
                # replayed forward to "now" - see live/box_flow.py
                try:
                    self._flow.add_frame(captured.image, captured.captured_at, captured.frame_number)
                except Exception as error:
                    self._disable_flow(error)

            image = captured.image
            src_h, src_w = image.shape[:2]
            if self.stream_width and src_w > self.stream_width:
                out_w = self.stream_width
                out_h = int(round(src_h * out_w / src_w))
                image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)   # new array
            elif overlay and snap:
                image = image.copy()          # never draw on the camera thread's array
            if snap:
                try:
                    vehicles = self._boxes_for_frame(snap, captured)
                except Exception as error:
                    self._disable_flow(error)    # box smoothing is a nicety: it must never break the video
                    vehicles = snap["vehicles"]
                self._draw(image, snap, vehicles)
            else:
                self._shown.clear()              # detector stalled: forget old boxes so they can't glide back in

            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return None
            result = (key, buf.tobytes())
            self._cache[overlay] = result
            return result

    def _disable_flow(self, error: Exception) -> None:
        if self._flow is not None:
            log.warning("[live] box tracking turned off after an error (%s) - boxes fall back to plain detections", error)
        self._flow = None
        self._shown.clear()

    def _boxes_for_frame(self, snap: Dict[str, Any], captured: Any) -> List[Dict[str, Any]]:
        """Where to draw each vehicle on THIS frame (detector pixels).
        Best source first: the box followed through the video itself; else the box
        moved along the speed measured from earlier detections; else where it was detected."""
        now = captured.captured_at
        if self._flow is not None:
            self._flow.sync(snap)
        estimates = self._project_boxes(snap, now)

        drawn: List[Dict[str, Any]] = []
        alive = set()
        for vehicle in estimates:
            track_id = vehicle.get("track_id")
            if track_id is None:
                drawn.append(vehicle)
                continue
            followed = self._flow.box(track_id) if self._flow is not None else None
            if followed is not None:
                source, target = "flow", followed
            else:
                source = "estimate"
                target = np.array([vehicle["x1"], vehicle["y1"], vehicle["x2"], vehicle["y2"]], dtype=np.float64)
            alive.add(track_id)
            x1, y1, x2, y2 = self._blend(track_id, snap["seq"], source, target, now)
            drawn.append(dict(vehicle, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))

        for gone in [t for t in self._shown if t not in alive]:
            del self._shown[gone]
        return drawn

    def _blend(self, track_id: Any, seq: int, source: str, target: np.ndarray, now: float) -> np.ndarray:
        """Stops a corrected box from jumping. Whenever the estimate changes (a new
        detection arrived, or we switched between following/estimating), the box
        carries on from where it was heading and glides onto the new estimate,
        instead of teleporting there."""
        state = self._shown.get(track_id)
        if state is None:
            offset, since, step = np.zeros(4), now, np.zeros(4)
        elif state["seq"] == seq and state["source"] == source:
            offset, since, step = state["offset"], state["since"], target - state["target"]
        else:
            step = state["step"]
            offset = (state["shown"] + step) - target      # where the box would be now if nothing had changed
            since = now
        shown = target + offset * math.exp(-max(0.0, now - since) / _BLEND_SECONDS)
        self._shown[track_id] = {"seq": seq, "source": source, "target": target, "shown": shown,
                                 "step": step, "offset": offset, "since": since}
        return shown

    @staticmethod
    def _draw(image, snap: Dict[str, Any], vehicles: Optional[List[Dict[str, Any]]] = None) -> None:
        # boxes are in the detector frame's pixels; scale to the image we are sending
        sx = image.shape[1] / (snap["frame"]["width"] or 1)
        sy = image.shape[0] / (snap["frame"]["height"] or 1)
        for v in (vehicles if vehicles is not None else snap["vehicles"]):
            p1 = (int(v["x1"] * sx), int(v["y1"] * sy))
            p2 = (int(v["x2"] * sx), int(v["y2"] * sy))
            cv2.rectangle(image, p1, p2, _GREEN, _LINE_THICKNESS, cv2.LINE_AA)
            label = f'#{v["track_id"]} {v["class"]} {v["confidence"]:.2f}'
            cv2.putText(image, label, (p1[0], max(10, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _GREEN, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            # TCP_NODELAY: without it Windows' Nagle + delayed-ACK can hold small
            # writes back ~40-200 ms each, which shows up as a laggy MJPEG stream.
            disable_nagle_algorithm = True

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
                            # how old the newest camera frame is: ~0 = decoder is keeping up,
                            # growing = the camera/decoder is the source of the lag
                            "frame_age_seconds": round(time.time() - cap.captured_at, 2) if cap else None,
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

                    if url.path == "/live/roi":
                        if server._roi_get is None:
                            return self._json(503, {"error": "roi not available"})
                        return self._json(200, {"polygon": server._roi_get()})

                    self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass       # viewer closed the tab / app - normal

            def do_POST(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                try:
                    if not self._authorized(query):
                        return self._json(401, {"error": "unauthorized"})

                    if url.path == "/live/roi":
                        return self._set_roi()

                    self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass       # front end closed the connection - normal

            def _set_roi(self) -> None:
                """POST body: {"polygon": [[x, y], ...]}, normalized 0..1, >= 3 points.
                This is how the dashboard front end pushes a redrawn ROI shape to
                the Pi; the vehicle/plate pipeline picks it up on the next frame."""
                if server._roi_set is None:
                    return self._json(503, {"error": "roi not available"})

                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    polygon = body["polygon"]
                    points = [(float(p[0]), float(p[1])) for p in polygon]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as error:
                    return self._json(400, {"error": f"bad request body: {error}"})

                if len(points) < 3:
                    return self._json(400, {"error": "polygon needs at least 3 points"})
                if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
                    return self._json(400, {"error": "polygon points must be normalized 0..1"})

                server._roi_set([list(p) for p in points])
                log.info("[live] ROI polygon updated by front end (%d points)", len(points))
                return self._json(200, {"ok": True, "polygon": [list(p) for p in points]})

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
                    header = f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    # ONE write per frame (header + jpeg + terminator), not three tiny ones
                    self.wfile.write(header + jpeg + b"\r\n")
                    self.wfile.flush()
                    pause = server.min_interval - (time.monotonic() - started)
                    if pause > 0:
                        time.sleep(pause)

        return Handler
