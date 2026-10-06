import time
from typing import Any, Dict, List, Optional

from core.geometry import compute_iou, center_distance_in_widths

_DEDUP_MIN_IOU = 0.5


def needs_fallback_tracker(vehicle_predictions: List[Dict[str, Any]]) -> bool:
    if not vehicle_predictions:
        return False
    return not all(prediction.get("track_id") for prediction in vehicle_predictions)


class FallbackTracker:
    def __init__(self, iou_threshold: float = 0.3, max_center_distance: float = 2.0, max_missing_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_missing_frames = max_missing_frames
        self.tracks: Dict[str, Dict[str, Any]] = {}
        self._next_track_number = 1

    def update(self, vehicle_predictions: List[Dict[str, Any]]) -> None:
        used_track_ids = set()

        for prediction in sorted(vehicle_predictions, key=lambda p: p.get("confidence", 0.0), reverse=True):
            best_track_id: Optional[str] = None
            best_iou = self.iou_threshold
            best_distance = self.max_center_distance

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                overlap = compute_iou(prediction, track["box"])
                distance = center_distance_in_widths(prediction, track["box"])
                is_match = overlap >= best_iou or (overlap > 0.0 and distance <= best_distance)
                if is_match:
                    best_iou, best_distance, best_track_id = overlap, distance, track_id

            if best_track_id is None:
                best_track_id = f"v{self._next_track_number}"
                self._next_track_number += 1
                self.tracks[best_track_id] = {"box": prediction, "missing": 0}
            else:
                self.tracks[best_track_id]["box"] = prediction
                self.tracks[best_track_id]["missing"] = 0

            prediction["track_id"] = best_track_id
            used_track_ids.add(best_track_id)

        for track_id in list(self.tracks):
            if track_id not in used_track_ids:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > self.max_missing_frames:
                    del self.tracks[track_id]


class PositionDeduper:
    def __init__(self, cooldown_seconds: float = 2.0, position_threshold: float = 1.0):
        self.cooldown_seconds = cooldown_seconds
        self.position_threshold = position_threshold
        self._recent_saves: Dict[str, Dict[str, Any]] = {}

    def find_existing_track(self, box: Dict[str, Any]) -> Optional[str]:
        now = time.time()
        best_id, best_distance = None, self.position_threshold

        for track_id, entry in self._recent_saves.items():
            if now - entry["saved_at"] > self.cooldown_seconds:
                continue
            if compute_iou(box, entry) < _DEDUP_MIN_IOU:
                continue
            distance = center_distance_in_widths(box, entry)
            if distance < best_distance:
                best_distance, best_id = distance, track_id

        return best_id

    def remember_save(self, track_id: str, box: Dict[str, Any]) -> None:
        self._recent_saves[track_id] = {
            "class": box.get("class"),
            "x": box.get("x"), "y": box.get("y"),
            "width": box.get("width"), "height": box.get("height"),
            "saved_at": time.time(),
        }

    def prune(self, max_age_seconds: float = 60.0) -> int:
        cutoff = time.time() - max(max_age_seconds, self.cooldown_seconds)
        stale = [track_id for track_id, entry in self._recent_saves.items() if entry["saved_at"] < cutoff]
        for track_id in stale:
            del self._recent_saves[track_id]
        return len(stale)

    def refresh(self, track_id: str, box: Dict[str, Any]) -> None:
        if track_id in self._recent_saves:
            entry = self._recent_saves[track_id]
            entry["x"], entry["y"] = box.get("x"), box.get("y")
            entry["width"], entry["height"] = box.get("width"), box.get("height")
            entry["saved_at"] = time.time()
