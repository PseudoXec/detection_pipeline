import time
from typing import Any, Dict, List, Optional, Set


def build_snapshot(
    camera_id: str,
    session_id: str,
    seq: int,
    frame_shape: tuple,
    predictions: List[Dict[str, Any]],
    in_roi_ids: Set[int],
    captured_at: Optional[float] = None,
) -> Dict[str, Any]:
    vehicles = []
    for p in predictions:
        half_w, half_h = p["width"] / 2.0, p["height"] / 2.0
        vehicles.append({
            "track_id": p.get("track_id"),
            "class": p.get("class"),
            "confidence": round(float(p.get("confidence", 0.0)), 3),
            "x1": round(p["x"] - half_w, 1), "y1": round(p["y"] - half_h, 1),
            "x2": round(p["x"] + half_w, 1), "y2": round(p["y"] + half_h, 1),
            "in_roi": id(p) in in_roi_ids,
        })
    return {
        "camera_id": camera_id,
        "session_id": session_id,
        "seq": seq,
        "captured_at": captured_at,
        "processed_at": time.time(),
        "frame": {"width": int(frame_shape[1]), "height": int(frame_shape[0])},
        "vehicles": vehicles,
    }
