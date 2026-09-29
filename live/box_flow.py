from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class _Track:
    __slots__ = ("box", "pts", "gray", "t", "ok", "vel")

    def __init__(self, box: np.ndarray, gray: np.ndarray, t: float):
        self.box = box
        self.pts: Optional[np.ndarray] = None
        self.gray = gray
        self.t = t
        self.ok = False
        self.vel: Optional[np.ndarray] = None  # last accepted velocity, px/s (full-res)


class BoxFlow:
    def __init__(self, work_width: int = 640, history_seconds: float = 6.0, max_points: int = 60):
        self.work_width = work_width
        self.history_seconds = history_seconds
        self.max_points = max_points
        self._frames: Deque[Tuple[float, int, np.ndarray]] = deque()
        self._last_frame_number = -1
        self._seq: Optional[int] = None
        self._tracks: Dict[Any, _Track] = {}
        self._scale = 1.0
        self._lk = dict(
            winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def add_frame(self, image: np.ndarray, captured_at: float, frame_number: int) -> None:
        if frame_number == self._last_frame_number:
            return
        self._last_frame_number = frame_number

        height, width = image.shape[:2]
        if width > self.work_width:
            image = cv2.resize(image, (self.work_width, int(round(height * self.work_width / width))),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        self._frames.append((captured_at, frame_number, gray))
        cutoff = captured_at - self.history_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

        for track in self._tracks.values():
            if track.ok:
                self._step(track, gray, captured_at)

    def sync(self, snap: Dict[str, Any]) -> None:
        if snap["seq"] == self._seq:
            return
        self._seq = snap["seq"]
        self._tracks = {}

        detected_at = snap.get("captured_at")
        if detected_at is None or not self._frames:
            return
        frames = list(self._frames)
        start = min(range(len(frames)), key=lambda i: abs(frames[i][0] - detected_at))
        if abs(frames[start][0] - detected_at) > 0.3:
            return

        start_time, _, start_gray = frames[start]
        self._scale = start_gray.shape[1] / float(snap["frame"]["width"] or 1)
        for vehicle in snap["vehicles"]:
            track_id = vehicle.get("track_id")
            if track_id is None:
                continue
            track = _Track(np.array([vehicle["x1"], vehicle["y1"], vehicle["x2"], vehicle["y2"]], dtype=np.float64),
                           start_gray, start_time)
            next_gray = frames[start + 1][2] if start + 1 < len(frames) else None
            track.pts = self._seed(start_gray, track.box, next_gray)
            if track.pts is None:
                track.pts = self._seed(start_gray, track.box)
            track.ok = track.pts is not None
            for j in range(start + 1, len(frames)):
                if not track.ok:
                    break
                self._step(track, frames[j][2], frames[j][0])

            if not track.ok and start + 1 < len(frames):
                # A brand-new track (this is its first sighting - nothing to blend/extrapolate from
                # yet) whose corner seed failed, usually because it is still small/distant, would
                # otherwise be shown frozen at its raw detected position until the NEXT detection -
                # up to one full processing interval of visible lag on the object it just latched
                # onto. Try one direct jump from the seed frame straight to the newest cached frame:
                # a single big optical-flow step tolerates a small/low-texture box better than
                # chaining many small hops (each hop needs the seed to have survived every prior one).
                self._reseed_direct(track, frames[-1][2])
            self._tracks[track_id] = track

    def box(self, track_id: Any) -> Optional[np.ndarray]:
        track = self._tracks.get(track_id)
        return track.box.copy() if track is not None and track.ok else None

    def _seed(self, gray: np.ndarray, box: np.ndarray, next_gray: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = box * self._scale
        # capped in pixels, not just a percentage - a percentage shrink eats a THIN box (a
        # motorcycle/bicycle is often much taller than it is wide) down to almost nothing, which is
        # exactly the case that most needs seeding to succeed
        box_w0, box_h0 = x2 - x1, y2 - y1
        # big boxes: seed only the inner ~60% (road/background near the edges is stationary and
        # would drag the box back); small/thin boxes keep the small pixel-capped shrink
        shrink_x = box_w0 * 0.2 if box_w0 >= 40 else min(box_w0 * 0.15, 4.0)
        shrink_y = box_h0 * 0.2 if box_h0 >= 40 else min(box_h0 * 0.15, 4.0)
        left, top = int(max(0, x1 + shrink_x)), int(max(0, y1 + shrink_y))
        right, bottom = int(min(width, x2 - shrink_x)), int(min(height, y2 - shrink_y))
        if right - left < 8 or bottom - top < 8:
            left, top = int(max(0, x1)), int(max(0, y1))       # box itself is already tiny -
            right, bottom = int(min(width, x2)), int(min(height, y2))  # track the whole thing, unshrunk
            if right - left < 4 or bottom - top < 4:
                return None
        mask = np.zeros_like(gray)
        mask[top:bottom, left:right] = 255
        if next_gray is not None and next_gray.shape == gray.shape:
            # fixed camera: pixels that changed between two consecutive frames ARE the moving
            # vehicle. Seeding only there keeps road/background out of the tracked point set
            # (the main reason a box on a fast vehicle stalls and lags behind it).
            diff = cv2.absdiff(gray, next_gray)
            moving = (cv2.GaussianBlur(diff, (5, 5), 0) > 18).astype(np.uint8) * 255
            moving = cv2.dilate(moving, np.ones((5, 5), np.uint8))
            moving = cv2.bitwise_and(moving, mask)
            if cv2.countNonZero(moving) >= 0.12 * max(1, cv2.countNonZero(mask)):
                mask = moving
        points = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_points, qualityLevel=0.01,
                                         minDistance=4, blockSize=5, mask=mask)
        if points is None or len(points) < 4:  # _step() itself only needs 4 to keep tracking
            return None
        return points.astype(np.float32)

    def _reseed_direct(self, track: "_Track", newest_gray: np.ndarray) -> None:
        """Last resort for a track whose normal (possibly multi-hop) tracking failed: one direct
        optical-flow jump from the ORIGINAL seed frame straight to the newest cached frame. Only
        used once, right after a failed sync() catch-up - never called per render frame."""
        pts = self._seed(track.gray, track.box)
        if pts is None:
            return
        p1, status, error = cv2.calcOpticalFlowPyrLK(track.gray, newest_gray, pts, None, **self._lk)
        if p1 is None:
            return
        good = (status.reshape(-1) == 1) & (error.reshape(-1) < 45)  # a bigger jump - allow more error
        if good.sum() < 4:
            return
        shift = np.median((p1[good] - pts[good]).reshape(-1, 2), axis=0) / self._scale
        box_w, box_h = track.box[2] - track.box[0], track.box[3] - track.box[1]
        if abs(shift[0]) > 2 * box_w or abs(shift[1]) > 2 * box_h:  # more room than _step()'s 1x -
            return                                                  # this is meant to cover more ground
        track.box[[0, 2]] += shift[0]
        track.box[[1, 3]] += shift[1]
        track.pts = p1[good]
        track.gray = newest_gray
        track.ok = True

    def _step(self, track: _Track, gray: np.ndarray, t: float) -> None:
        p0 = track.pts
        p1, status, error = cv2.calcOpticalFlowPyrLK(track.gray, gray, p0, None, **self._lk)
        if p1 is None:
            track.ok = False
            return
        good = (status.reshape(-1) == 1) & (error.reshape(-1) < 40)
        if good.sum() < 4:
            track.ok = False
            return

        moves = (p1 - p0).reshape(-1, 2)
        mag = np.linalg.norm(moves, axis=1)
        pool = good
        fast = float(np.percentile(mag[good], 80))
        if fast > 2.0:
            # fixed camera: points that barely move while the rest of the box moves are road /
            # background inside an oversized box - ignore them so the box doesn't stall (= lag)
            moving = good & (mag >= 0.5 * fast)
            if moving.sum() >= 4:
                pool = moving
        median = np.median(moves[pool], axis=0)
        agrees = pool & (np.linalg.norm(moves - median, axis=1) <= max(1.5, 0.35 * float(np.linalg.norm(median)) + 1.0))
        if agrees.sum() < 4:
            track.ok = False
            return
        shift = np.median(moves[agrees], axis=0) / self._scale

        box_w, box_h = track.box[2] - track.box[0], track.box[3] - track.box[1]
        if abs(shift[0]) > 2 * box_w or abs(shift[1]) > 2 * box_h:
            track.ok = False
            return

        # a real vehicle can't change speed abruptly: when one step disagrees strongly with the
        # previous velocity (other vehicle / shadow / lost points) blend instead of jumping,
        # so the box neither races ahead of nor stalls behind the vehicle
        dt = max(t - track.t, 1e-3)
        velocity = shift / dt
        if track.vel is not None:
            jump = float(np.linalg.norm(velocity - track.vel))
            if jump > max(60.0, 0.5 * float(np.linalg.norm(track.vel))):
                velocity = 0.5 * velocity + 0.5 * track.vel
                shift = velocity * dt
        track.vel = velocity if track.vel is None else 0.6 * velocity + 0.4 * track.vel

        previous_gray = track.gray
        track.box[[0, 2]] += shift[0]
        track.box[[1, 3]] += shift[1]
        track.pts = p1[agrees]
        track.gray = gray
        track.t = t

        if len(track.pts) < 16:
            # re-seed on what MOVED since the last frame, not on the whole (stationary) box
            fresh = self._seed(gray, track.box, previous_gray)
            if fresh is None:
                fresh = self._seed(gray, track.box)
            if fresh is not None:
                track.pts = fresh
