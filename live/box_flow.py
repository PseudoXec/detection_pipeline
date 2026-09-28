from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class _Track:
    __slots__ = ("box", "pts", "gray", "t", "ok")

    def __init__(self, box: np.ndarray, gray: np.ndarray, t: float):
        self.box = box
        self.pts: Optional[np.ndarray] = None
        self.gray = gray
        self.t = t
        self.ok = False


class BoxFlow:
    def __init__(self, work_width: int = 480, history_seconds: float = 6.0, max_points: int = 40):
        self.work_width = work_width
        self.history_seconds = history_seconds
        self.max_points = max_points
        self._frames: Deque[Tuple[float, int, np.ndarray]] = deque()
        self._last_frame_number = -1
        self._seq: Optional[int] = None
        self._tracks: Dict[Any, _Track] = {}
        self._scale = 1.0
        self._lk = dict(
            winSize=(21, 21), maxLevel=3,
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
            track.pts = self._seed(start_gray, track.box)
            track.ok = track.pts is not None
            for j in range(start + 1, len(frames)):
                if not track.ok:
                    break
                self._step(track, frames[j][2], frames[j][0])
            self._tracks[track_id] = track

    def box(self, track_id: Any) -> Optional[np.ndarray]:
        track = self._tracks.get(track_id)
        return track.box.copy() if track is not None and track.ok else None

    def _seed(self, gray: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = box * self._scale
        shrink_x, shrink_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        left, top = int(max(0, x1 + shrink_x)), int(max(0, y1 + shrink_y))
        right, bottom = int(min(width, x2 - shrink_x)), int(min(height, y2 - shrink_y))
        if right - left < 12 or bottom - top < 12:
            return None
        mask = np.zeros_like(gray)
        mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_points, qualityLevel=0.01,
                                         minDistance=4, blockSize=5, mask=mask)
        if points is None or len(points) < 6:
            return None
        return points.astype(np.float32)

    def _step(self, track: _Track, gray: np.ndarray, t: float) -> None:
        p0 = track.pts
        p1, status, error = cv2.calcOpticalFlowPyrLK(track.gray, gray, p0, None, **self._lk)
        if p1 is None:
            track.ok = False
            return
        good = (status.reshape(-1) == 1) & (error.reshape(-1) < 30)
        if good.sum() < 4:
            track.ok = False
            return

        moves = (p1 - p0).reshape(-1, 2)
        median = np.median(moves[good], axis=0)
        agrees = good & (np.linalg.norm(moves - median, axis=1) <= max(1.5, 0.35 * float(np.linalg.norm(median)) + 1.0))
        if agrees.sum() < 4:
            track.ok = False
            return
        shift = np.median(moves[agrees], axis=0) / self._scale

        box_w, box_h = track.box[2] - track.box[0], track.box[3] - track.box[1]
        if abs(shift[0]) > box_w or abs(shift[1]) > box_h:
            track.ok = False
            return

        track.box[[0, 2]] += shift[0]
        track.box[[1, 3]] += shift[1]
        track.pts = p1[agrees]
        track.gray = gray
        track.t = t

        if len(track.pts) < 10:
            fresh = self._seed(gray, track.box)
            if fresh is not None:
                track.pts = fresh
