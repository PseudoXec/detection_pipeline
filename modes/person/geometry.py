"""ROI rule for people.

A vehicle counts as inside the ROI when the CENTRE of its box is inside the polygon. For a person that
rule is wrong: the polygon marks a zone on the GROUND, but a person's box is tall and its centre is
about waist height - well above the floor. What decides whether someone stands in the zone is where
their FEET are, i.e. the bottom-centre of the box, so that is the point tested here.
"""
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

from core.geometry import point_in_polygon

log = logging.getLogger("pipeline")


def feet_point(box: Dict[str, Any]) -> Tuple[float, float]:
    """Bottom-centre of a centre-format box ({x, y, width, height}), in pixels."""
    return box["x"], box["y"] + box["height"] / 2.0


def person_reject_reason(
    box: Dict[str, Any],
    frame_shape: Tuple[int, int],
    polygon: Sequence[Sequence[float]],
    min_width_ratio: float, min_height_ratio: float,
    max_width_ratio: float, max_height_ratio: float,
    edge_margin_ratio: float = 0.0,
) -> Optional[str]:
    """None when the person is inside the ROI, otherwise a short reason."""
    frame_h, frame_w = frame_shape[:2]
    if frame_h <= 0 or frame_w <= 0 or len(polygon) < 3:
        return "no valid polygon / frame"

    feet_x, feet_y = feet_point(box)
    norm_feet = (feet_x / frame_w, feet_y / frame_h)
    norm_w, norm_h = box.get("width", 0.0) / frame_w, box.get("height", 0.0) / frame_h

    if not point_in_polygon(norm_feet[0], norm_feet[1], polygon):
        return f"feet ({norm_feet[0]:.3f},{norm_feet[1]:.3f}) are outside the polygon"
    if not (min_width_ratio <= norm_w <= max_width_ratio and min_height_ratio <= norm_h <= max_height_ratio):
        return f"size {norm_w:.3f}x{norm_h:.3f} outside the allowed ratios"
    if edge_margin_ratio > 0:
        left, right = box["x"] / frame_w - norm_w / 2, box["x"] / frame_w + norm_w / 2
        if left < edge_margin_ratio or right > 1.0 - edge_margin_ratio:
            return "box touches the side of the frame"
    return None


def is_person_inside_roi(box, frame_shape, polygon, *sizes, edge_margin_ratio: float = 0.0) -> bool:
    reason = person_reject_reason(box, frame_shape, polygon, *sizes, edge_margin_ratio=edge_margin_ratio)
    if reason is not None and log.isEnabledFor(logging.DEBUG):
        log.debug("[roi] person %s rejected: %s", box.get("track_id", "?"), reason)
    return reason is None
