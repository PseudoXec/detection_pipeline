"""
tracker.py
----------
Keeps a stable ID on "the same physical vehicle" across frames.

Two layers work together:
  1. ByteTrack (via detector.track) is tried first - it's a proper Kalman
     filter tracker and does the best job.
  2. FallbackTracker below only kicks in if ByteTrack couldn't assign an ID
     to a detection (this happens occasionally, e.g. right after a stream
     reconnect). It's a much simpler IoU + centroid-distance matcher.

Separately, `PositionDeduper` stops us from saving a second crop of a
vehicle that is just sitting still (e.g. waiting at a red light) even if
tracking briefly loses and re-creates its ID.
"""

# time.time() gives us wall-clock timestamps for the dedup cooldown window
import time
# typing hints document the shape of the dicts/sets passed around
from typing import Any, Dict, List, Optional

# shared box-math helpers live in geometry.py so both trackers use identical math
from geometry import compute_iou, center_distance_in_widths


def needs_fallback_tracker(vehicle_predictions: List[Dict[str, Any]]) -> bool:
    """True when ByteTrack did NOT manage to tag every detection with a track_id."""
    if not vehicle_predictions:
        return False
    return not all(prediction.get("track_id") for prediction in vehicle_predictions)


class FallbackTracker:
    """Simple IoU + centroid-distance tracker used only when ByteTrack fails
    to assign IDs. Assigns and remembers a `v{n}` style track_id per vehicle."""

    def __init__(self, iou_threshold: float = 0.3, max_center_distance: float = 2.0, max_missing_frames: int = 30):
        # how much overlap (or how little movement) counts as "the same vehicle"
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        # how many frames a track can go unmatched before we forget it
        self.max_missing_frames = max_missing_frames
        # active tracks: track_id -> {"box": last known box, "missing": frame count}
        self.tracks: Dict[str, Dict[str, Any]] = {}
        # counter used to generate the next new track's number
        self._next_track_number = 1

    def update(self, vehicle_predictions: List[Dict[str, Any]]) -> None:
        """Match this frame's detections against existing tracks, assigning
        `track_id` onto each prediction dict in place."""
        used_track_ids = set()

        # process the most confident detections first so they get first pick
        # of the best-matching existing track
        for prediction in sorted(vehicle_predictions, key=lambda p: p.get("confidence", 0.0), reverse=True):
            best_track_id: Optional[str] = None
            best_iou = self.iou_threshold
            best_distance = self.max_center_distance

            # compare this detection against every still-available track
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                overlap = compute_iou(prediction, track["box"])
                distance = center_distance_in_widths(prediction, track["box"])
                # a match is either strong overlap, or the box barely moved
                is_match = overlap >= best_iou or (overlap > 0.0 and distance <= best_distance)
                if is_match:
                    best_iou, best_distance, best_track_id = overlap, distance, track_id

            if best_track_id is None:
                # nothing matched -> this is a brand-new vehicle
                best_track_id = f"v{self._next_track_number}"
                self._next_track_number += 1
                self.tracks[best_track_id] = {"box": prediction, "missing": 0}
            else:
                # matched an existing track -> refresh its last-known box
                self.tracks[best_track_id]["box"] = prediction
                self.tracks[best_track_id]["missing"] = 0

            prediction["track_id"] = best_track_id
            used_track_ids.add(best_track_id)

        # any track that didn't get matched this frame ages by one frame,
        # and is forgotten once it's been missing too long (vehicle left the scene)
        for track_id in list(self.tracks):
            if track_id not in used_track_ids:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > self.max_missing_frames:
                    del self.tracks[track_id]


class PositionDeduper:
    """Prevents saving a duplicate crop for a vehicle that hasn't actually
    moved - even if the upstream tracker briefly lost/reassigned its ID."""

    def __init__(self, cooldown_seconds: float = 2.0, position_threshold: float = 1.0):
        self.cooldown_seconds = cooldown_seconds
        self.position_threshold = position_threshold
        # track_id -> {"class", "cx", "cy", "w", "h", "saved_at"}
        self._recent_saves: Dict[str, Dict[str, Any]] = {}

    def find_existing_track(self, box: Dict[str, Any]) -> Optional[str]:
        """If this detection is really the same vehicle as one saved a moment
        ago in roughly the same spot, return that earlier track_id."""
        now = time.time()
        best_id, best_distance = None, self.position_threshold

        for track_id, entry in self._recent_saves.items():
            # ignore saves outside the cooldown window - a genuinely new
            # vehicle passing through the same spot later should NOT be suppressed
            if now - entry["saved_at"] > self.cooldown_seconds:
                continue
            # only compare vehicles of the same class (a car parking where a
            # truck just left shouldn't be treated as the same vehicle)
            if box.get("class") and entry.get("class") and box["class"] != entry["class"]:
                continue
            distance = center_distance_in_widths(box, entry)
            if distance < best_distance:
                best_distance, best_id = distance, track_id

        return best_id

    def remember_save(self, track_id: str, box: Dict[str, Any]) -> None:
        """Call this right after a vehicle crop is actually saved."""
        self._recent_saves[track_id] = {
            "class": box.get("class"),
            "x": box.get("x"), "y": box.get("y"),
            "width": box.get("width"), "height": box.get("height"),
            "saved_at": time.time(),
        }

    def refresh(self, track_id: str, box: Dict[str, Any]) -> None:
        """Call this every frame a still-visible vehicle is re-detected, so a
        vehicle that dwells for a long time (traffic light) stays suppressed."""
        if track_id in self._recent_saves:
            entry = self._recent_saves[track_id]
            entry["x"], entry["y"] = box.get("x"), box.get("y")
            entry["saved_at"] = time.time()
