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
_GREEN = (0, 0, 255)
_GREY = (140, 140, 140)  # vehicles outside the ROI (only drawn when asked for)
_LINE_THICKNESS = 1
_FONT_SCALE = 0.4
_BLEND_SECONDS = 0.15          # was 0.4 - a long blend IS visible lag on fast vehicles
_MAX_EXTRAPOLATE_SECONDS = 1.2   # never project a box further ahead than this, however old the detection
_EXTRAPOLATE_TAU = 0.35          # seconds; projection speed decays with this constant
_MIN_MOTION_INTERVAL = 0.05     # detections closer together than this give noisy velocities
_ROI_COLOR = (0, 255, 255)
_ROI_THICKNESS = 1

_VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live view</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
header{display:flex;gap:20px;align-items:center;padding:10px 14px;background:#1c1c1c}
label{cursor:pointer;user-select:none}
img{display:block;max-width:100%;margin:0 auto}
</style></head><body>
<header>
<label><input type="checkbox" id="boxes"> Boxes</label>
<label><input type="checkbox" id="roi"> ROI polygon</label>
<label><input type="checkbox" id="outside"> Vehicles outside ROI</label>
</header>
<img id="view" alt="live stream">
<script>
var q = new URLSearchParams(location.search);
var boxes = document.getElementById("boxes");
var roi = document.getElementById("roi");
var outside = document.getElementById("outside");
var view = document.getElementById("view");
boxes.checked = q.get("overlay") !== "0";
roi.checked = q.get("roi") === "1";
outside.checked = q.get("outside") === "1";
function load() {
  var p = new URLSearchParams();
  p.set("overlay", boxes.checked ? "1" : "0");
  p.set("roi", roi.checked ? "1" : "0");
  p.set("outside", outside.checked ? "1" : "0");
  if (q.get("token")) p.set("token", q.get("token"));
  view.src = "/live/stream.mjpg?" + p.toString();
}
boxes.onchange = load;
roi.onchange = load;
outside.onchange = load;
load();
</script></body></html>
"""


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
        mode_status: Optional[Callable[[], Dict[str, Any]]] = None,
        mode_request: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        mode_choices: Optional[Callable[[], List[str]]] = None,
        live_enabled: bool = True,
        device: Optional[Any] = None,
    ):
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.min_interval = 1.0 / stream_fps if stream_fps > 0 else 0.0
        self.stream_width = stream_width
        self.jpeg_quality = int(jpeg_quality)
        self.box_max_age_seconds = box_max_age_seconds
        self.auth_token = auth_token or None
        self.box_extrapolate = box_extrapolate
        self.box_extrapolate_max_seconds = box_extrapolate_max_seconds
        self._flow: Optional[BoxFlow] = BoxFlow() if box_visual_tracking else None
        self._motion_velocity: Dict[Any, Tuple[float, ...]] = {}
        self._shown: Dict[Any, Dict[str, Any]] = {}
        self.session_id = uuid.uuid4().hex[:8]

        self._roi_get = roi_get
        self._roi_set = roi_set
        # command-center control of the detection mode (see control/mode_manager.py)
        self._mode_status = mode_status
        self._mode_request = mode_request
        self._mode_choices = mode_choices
        # device management (control/device.py): /device, /device/buffer, /control/refresh-identity, /control/restart
        self._device = device
        # False = serve only /control/* and /live/health; the camera image/boxes stay private
        self.live_enabled = live_enabled

        self._frame_source: Optional[Callable[[], Any]] = None
        self._snapshot: Optional[Dict[str, Any]] = None
        self._seq = 0
        self._motion: Optional[Tuple[int, float, Dict[Any, Tuple[float, ...]]]] = None
        self._prev_boxes: Optional[Tuple[float, Dict[Any, Tuple[float, ...]]]] = None

        self._enc_lock = threading.Lock()
        self._cache: Dict[Tuple[bool, bool], Tuple[Tuple, bytes]] = {}

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def set_frame_source(self, source: Callable[[], Any]) -> None:
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
        pi_host = "<pi-ip>" if self.host in ("0.0.0.0", "") else self.host
        if self.live_enabled:
            log.info("[live] serving %s at http://%s:%d/live/stream.mjpg (also /live/view, /live/frame.jpg, /live/boxes, /live/roi)",
                     self.camera_id, pi_host, self.port)
        if self._mode_request is not None:
            log.info("[control] detection mode can be changed with POST http://%s:%d/control/mode  {\"mode\": \"vehicle|person|idle\"}",
                     pi_host, self.port)
        if not self.auth_token:
            log.warning("[live] live.auth_token is not set - anyone who can reach this port can watch the camera"
                        + (" and switch the detection mode" if self._mode_request is not None else ""))
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def clear_boxes(self) -> None:
        """Forget every box and motion estimate. Called when the detection mode changes so the old
        mode's boxes can't linger on screen or be mistaken for the new mode's tracks."""
        with self._enc_lock:
            self._snapshot = None
            self._motion = None
            self._prev_boxes = None
            self._motion_velocity = {}
            self._shown.clear()
            self._cache.clear()
            if self._flow is not None:
                self._flow = BoxFlow()

    def publish(
        self,
        frame_shape: tuple,
        predictions: List[Dict[str, Any]],
        in_roi_ids: Set[int],
        captured_at: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> None:
        try:
            self._seq += 1
            snapshot = build_snapshot(
                self.camera_id, self.session_id, self._seq,
                frame_shape, predictions, in_roi_ids, captured_at, mode,
            )
            if self.box_extrapolate:
                self._motion = self._estimate_motion(snapshot)
            self._snapshot = snapshot
        except Exception as error:
            log.debug("[live] publish skipped: %s", error)

    def _estimate_motion(self, snap: Dict[str, Any]) -> Tuple[int, float, Dict[Any, Tuple[float, ...]]]:
        """Per-track CENTRE velocity (px/s). Size is never extrapolated - only the position moves."""
        frame_time = snap["captured_at"] if snap.get("captured_at") is not None else snap["processed_at"]
        centres = {
            v["track_id"]: ((v["x1"] + v["x2"]) / 2.0, (v["y1"] + v["y2"]) / 2.0,
                            v["x2"] - v["x1"], v["y2"] - v["y1"])
            for v in snap["vehicles"] if v.get("track_id") is not None
        }
        velocities: Dict[Any, Tuple[float, ...]] = {}
        previous = self._prev_boxes
        if previous is not None:
            prev_time, prev_centres = previous
            elapsed = frame_time - prev_time
            if _MIN_MOTION_INTERVAL <= elapsed <= 2.0:
                for track_id, (cx, cy, w, h) in centres.items():
                    old = prev_centres.get(track_id)
                    if old is None:
                        continue
                    vx, vy = (cx - old[0]) / elapsed, (cy - old[1]) / elapsed
                    earlier = self._motion_velocity.get(track_id)
                    if earlier is not None:  # blend with the last estimate so one noisy frame can't fling the box
                        vx, vy = 0.65 * vx + 0.35 * earlier[0], 0.65 * vy + 0.35 * earlier[1]
                    # a vehicle can't plausibly cross more than ~3 of its own sizes per second
                    vx = max(-3.0 * max(w, 1.0), min(3.0 * max(w, 1.0), vx))
                    vy = max(-3.0 * max(h, 1.0), min(3.0 * max(h, 1.0), vy))
                    velocities[track_id] = (vx, vy)
        self._motion_velocity = dict(velocities)
        self._prev_boxes = (frame_time, centres)
        return (snap["seq"], frame_time, velocities)

    def _project_boxes(self, snap: Dict[str, Any], frame_time: Optional[float]) -> List[Dict[str, Any]]:
        vehicles = snap["vehicles"]
        motion = self._motion
        if not self.box_extrapolate or motion is None or motion[0] != snap["seq"] or frame_time is None:
            return vehicles
        ahead = min(max(frame_time - motion[1], 0.0), self.box_extrapolate_max_seconds, _MAX_EXTRAPOLATE_SECONDS)
        if ahead <= 0 or not motion[2]:
            return vehicles
        moved_vehicles = []
        for v in vehicles:
            velocity = motion[2].get(v.get("track_id"))
            if velocity is None:
                moved_vehicles.append(v)
                continue
            width, height = v["x2"] - v["x1"], v["y2"] - v["y1"]
            # same shift for both edges -> the box keeps its size; never move it more than its own size
            # damped projection: speed decays with time constant tau (vehicles receding from the
            # camera decelerate in image space) - a straight v*t line overshoots them
            reach = _EXTRAPOLATE_TAU * (1.0 - math.exp(-ahead / _EXTRAPOLATE_TAU))
            dx = max(-0.6 * width, min(0.6 * width, velocity[0] * reach))
            dy = max(-0.6 * height, min(0.6 * height, velocity[1] * reach))
            moved_vehicles.append(dict(v, x1=v["x1"] + dx, y1=v["y1"] + dy, x2=v["x2"] + dx, y2=v["y2"] + dy))
        return moved_vehicles

    def _fresh_boxes(self) -> Optional[Dict[str, Any]]:
        snap = self._snapshot
        if snap is None or time.time() - snap["processed_at"] > self.box_max_age_seconds:
            return None
        return snap

    def _render(self, overlay: bool, roi: bool = False, outside: bool = False) -> Optional[Tuple[Tuple, bytes]]:
        captured = self._frame_source() if self._frame_source else None
        if captured is None:
            return None
        snap = self._fresh_boxes() if overlay else None
        polygon = self._roi_get() if roi and self._roi_get else None
        if polygon is not None and len(polygon) < 3:
            polygon = None
        roi_signature = tuple(tuple(point) for point in polygon) if polygon else None
        key = (captured.frame_number, snap["seq"] if snap else 0, roi_signature)

        with self._enc_lock:
            cached = self._cache.get((overlay, roi, outside))
            if cached and cached[0] == key:
                return cached

            if overlay and self._flow is not None:
                try:
                    self._flow.add_frame(captured.image, captured.captured_at, captured.frame_number)
                except Exception as error:
                    self._disable_flow(error)

            image = captured.image
            src_h, src_w = image.shape[:2]
            if self.stream_width and src_w > self.stream_width:
                out_w = self.stream_width
                out_h = int(round(src_h * out_w / src_w))
                image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            elif (overlay and snap) or polygon:
                image = image.copy()
            if polygon:
                self._draw_roi(image, polygon)
            if snap:
                try:
                    vehicles = self._boxes_for_frame(snap, captured)
                except Exception as error:
                    self._disable_flow(error)
                    vehicles = snap["vehicles"]
                self._draw(image, snap, vehicles, outside)
            else:
                self._shown.clear()

            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return None
            result = (key, buf.tobytes())
            self._cache[(overlay, roi, outside)] = result
            return result

    def _disable_flow(self, error: Exception) -> None:
        if self._flow is not None:
            log.warning("[live] box tracking turned off after an error (%s) - boxes fall back to plain detections", error)
        self._flow = None
        self._shown.clear()

    def _boxes_for_frame(self, snap: Dict[str, Any], captured: Any) -> List[Dict[str, Any]]:
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
        state = self._shown.get(track_id)
        if state is None:
            offset, since, step = np.zeros(4), now, np.zeros(4)
        elif state["seq"] == seq and state["source"] == source:
            offset, since, step = state["offset"], state["since"], target - state["target"]
        else:
            step = state["step"]
            offset = (state["shown"] + step) - target
            since = now
        shown = target + offset * math.exp(-max(0.0, now - since) / _BLEND_SECONDS)
        self._shown[track_id] = {"seq": seq, "source": source, "target": target, "shown": shown,
                                 "step": step, "offset": offset, "since": since}
        return shown

    @staticmethod
    def _draw_roi(image, polygon: List[List[float]]) -> None:
        height, width = image.shape[:2]
        points = np.array([[int(x * width), int(y * height)] for x, y in polygon], dtype=np.int32)
        cv2.polylines(image, [points], True, _ROI_COLOR, _ROI_THICKNESS, cv2.LINE_AA)
        for index, (px, py) in enumerate(points, start=1):
            cv2.circle(image, (int(px), int(py)), 4, _ROI_COLOR, -1, cv2.LINE_AA)
            cv2.putText(image, str(index), (int(px) + 6, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _ROI_COLOR, 1, cv2.LINE_AA)

    @staticmethod
    def _draw(image, snap: Dict[str, Any], vehicles: Optional[List[Dict[str, Any]]] = None,
              show_outside: bool = False) -> None:
        sx = image.shape[1] / (snap["frame"]["width"] or 1)
        sy = image.shape[0] / (snap["frame"]["height"] or 1)
        for v in (vehicles if vehicles is not None else snap["vehicles"]):
            inside = v.get("in_roi", True)
            if not inside and not show_outside:
                continue  # the pipeline ignores this vehicle, so don't show it as if it were being read
            color = _GREEN if inside else _GREY
            p1 = (int(v["x1"] * sx), int(v["y1"] * sy))
            p2 = (int(v["x2"] * sx), int(v["y2"] * sy))
            cv2.rectangle(image, p1, p2, color, _LINE_THICKNESS, cv2.LINE_AA)
            label = f'#{v["track_id"]} {v["class"]} {v["confidence"]:.2f}'
            cv2.putText(image, label, (p1[0], max(10, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, 1, cv2.LINE_AA)

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True

            def log_message(self, *args):
                pass

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

            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                try:
                    if not self._authorized(query):
                        return self._json(401, {"error": "unauthorized"})
                    overlay = query.get("overlay", ["1"])[0] not in ("0", "false", "no")
                    roi = query.get("roi", ["0"])[0] in ("1", "true", "yes")
                    outside = query.get("outside", ["0"])[0] in ("1", "true", "yes")

                    if url.path == "/live/health":
                        snap = server._snapshot
                        cap = server._frame_source() if server._frame_source else None
                        return self._json(200, {
                            "ok": True,
                            "camera_id": server.camera_id,
                            "session_id": server.session_id,
                            "has_frame": cap is not None,
                            "frame_age_seconds": round(time.time() - cap.captured_at, 2) if cap else None,
                            "boxes_age_seconds": round(time.time() - snap["processed_at"], 2) if snap else None,
                            "mode": server._mode_status() if server._mode_status else None,
                        })

                    if url.path == "/control/mode":
                        return self._mode_get()

                    if url.path in ("/device", "/device/buffer"):
                        if server._device is None:
                            return self._json(404, {"error": "device management is not enabled"})
                        return self._json(200, server._device.device() if url.path == "/device" else server._device.buffer())

                    if url.path.startswith("/live/") and url.path != "/live/health" and not server.live_enabled:
                        return self._json(404, {"error": "live view is disabled (features.live_stream is false)"})

                    if url.path == "/live/boxes":
                        snap = server._snapshot
                        if snap is None:
                            return self._json(503, {"error": "no detections processed yet"})
                        return self._json(200, dict(snap, age_seconds=round(time.time() - snap["processed_at"], 3)))

                    if url.path == "/live/frame.jpg":
                        out = server._render(overlay, roi, outside)
                        if out is None:
                            return self._json(503, {"error": "no frame from camera yet"})
                        return self._send(200, out[1], "image/jpeg")

                    if url.path == "/live/stream.mjpg":
                        return self._stream(overlay, roi, outside)

                    if url.path == "/live/view":
                        return self._send(200, _VIEWER_HTML.encode(), "text/html; charset=utf-8")

                    if url.path == "/live/roi":
                        if server._roi_get is None:
                            return self._json(503, {"error": "roi not available"})
                        return self._json(200, {"polygon": server._roi_get()})

                    self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def do_POST(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                try:
                    if not self._authorized(query):
                        return self._json(401, {"error": "unauthorized"})

                    if url.path == "/live/roi":
                        return self._set_roi()

                    if url.path == "/control/mode":
                        return self._mode_post(query)

                    if url.path in ("/control/refresh-identity", "/control/restart"):
                        if server._device is None:
                            return self._json(404, {"error": "device management is not enabled"})
                        if url.path == "/control/refresh-identity":
                            result = server._device.refresh_identity()
                            return self._json(200 if result["ok"] else 502, result)
                        result = server._device.restart(force=query.get("force", ["0"])[0] in ("1", "true", "yes"))
                        return self._json(202 if result["ok"] else 409, result)

                    self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _mode_get(self) -> None:
                if server._mode_status is None:
                    return self._json(404, {"error": "mode control is not enabled (mode.control_enabled)"})
                return self._json(200, server._mode_status())

            def _mode_post(self, query) -> None:
                if server._mode_request is None:
                    return self._json(404, {"error": "mode control is not enabled (mode.control_enabled)"})

                requested = (query.get("mode", [""])[0] or "").strip()
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length:
                    try:
                        body = json.loads(self.rfile.read(length) or b"{}")
                        requested = str(body.get("mode", requested) if isinstance(body, dict) else requested).strip()
                    except (json.JSONDecodeError, ValueError) as error:
                        return self._json(400, {"error": f"bad request body: {error}"})

                choices = server._mode_choices() if server._mode_choices else []
                if not requested:
                    return self._json(400, {"error": "send {\"mode\": \"<name>\"}", "allowed": choices})
                if choices and requested.lower() not in choices:
                    return self._json(400, {"error": f"unknown mode {requested!r}", "allowed": choices})

                status = server._mode_request(requested.lower(), "command-center")
                # 202: the switch happens in the background, poll GET /control/mode for the result
                return self._json(202, status)

            def _set_roi(self) -> None:
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

            def _stream(self, overlay: bool, roi: bool, outside: bool = False) -> None:
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
                    out = server._render(overlay, roi, outside)
                    if out is None or out[0] == last_key:
                        time.sleep(0.03)
                        continue
                    last_key, jpeg = out
                    header = f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    self.wfile.write(header + jpeg + b"\r\n")
                    self.wfile.flush()
                    pause = server.min_interval - (time.monotonic() - started)
                    if pause > 0:
                        time.sleep(pause)

        return Handler
