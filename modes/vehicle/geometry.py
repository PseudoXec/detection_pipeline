"""Vehicle-only geometry: the ROI rule for vehicles and the plate crop.

The generic helpers (boxes, IoU, polygons, crops) live in core/geometry.py.
"""
import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import box_edges, point_in_polygon

log = logging.getLogger("pipeline")

# A vehicle counts as "inside the ROI" when the CENTRE of its box is inside the polygon (plus the
# size limits and the frame-edge check). The polygon is a zone on the road but the box of a tall
# vehicle always sticks out of it in the image, so requiring the whole box (or most of it) inside
# rejects big vehicles that are clearly in the zone.
# Optional extra strictness: set this to e.g. 0.5 to ALSO require that share of the box area to lie
# inside the polygon. 0 = off (centre point only, no overlap calculation at all).
MIN_INSIDE_FRACTION = 0.0

_MASK_MAX_SIDE = 320  # polygon is rasterised at this size for the overlap test (plenty accurate)
_mask_cache: Dict[str, Any] = {"key": None, "mask": None}


def crop_plate(
    vehicle_crop: np.ndarray,
    plate_box: Dict[str, Any],
    padding_pixels: int = 8,
    min_crop_height: int = 0,
    padding_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    height, width = vehicle_crop.shape[:2]

    x1, y1, x2, y2 = box_edges(plate_box)

    pad = max(padding_pixels, int(round((y2 - y1) * padding_ratio)))

    x1 = max(int(x1) - pad, 0)
    y1 = max(int(y1) - pad, 0)
    x2 = min(int(x2) + pad, width)
    y2 = min(int(y2) + pad, height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = vehicle_crop[y1:y2, x1:x2]

    crop_h, crop_w = crop.shape[:2]
    if min_crop_height and crop_h < min_crop_height:
        scale = min_crop_height / crop_h
        crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

    return crop


def _polygon_mask(polygon: Sequence[Sequence[float]], frame_h: int, frame_w: int) -> np.ndarray:
    key = (tuple((round(float(x), 6), round(float(y), 6)) for x, y in polygon), frame_h, frame_w)
    if _mask_cache["key"] == key:
        return _mask_cache["mask"]
    scale = _MASK_MAX_SIDE / float(max(frame_h, frame_w))
    mask_h, mask_w = max(1, int(round(frame_h * scale))), max(1, int(round(frame_w * scale)))
    points = np.array([[x * (mask_w - 1), y * (mask_h - 1)] for x, y in polygon], dtype=np.float32)
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1, lineType=cv2.LINE_8)
    _mask_cache["key"], _mask_cache["mask"] = key, mask
    return mask


def box_fraction_inside_polygon(
    polygon: Sequence[Sequence[float]],
    left: float, top: float, right: float, bottom: float,
    frame_h: int, frame_w: int,
) -> float:
    """Share (0..1) of a normalised box that lies inside the polygon."""
    mask = _polygon_mask(polygon, frame_h, frame_w)
    mask_h, mask_w = mask.shape
    x1 = max(0, int(math.floor(left * (mask_w - 1))))
    x2 = min(mask_w, int(math.ceil(right * (mask_w - 1))) + 1)
    y1 = max(0, int(math.floor(top * (mask_h - 1))))
    y2 = min(mask_h, int(math.ceil(bottom * (mask_h - 1))) + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    cells_total = (int(math.ceil(right * (mask_w - 1))) + 1 - int(math.floor(left * (mask_w - 1)))) * \
                  (int(math.ceil(bottom * (mask_h - 1))) + 1 - int(math.floor(top * (mask_h - 1))))
    return float(mask[y1:y2, x1:x2].sum()) / max(1, cells_total)


def roi_reject_reason(
    box: Dict[str, Any],
    frame_shape: Tuple[int, int],
    polygon: Sequence[Sequence[float]],
    min_width_ratio: float, min_height_ratio: float,
    max_width_ratio: float, max_height_ratio: float,
    edge_margin_ratio: float,
    min_inside_fraction: float = MIN_INSIDE_FRACTION,
) -> Optional[str]:
    """None when the box is inside the ROI, otherwise a short reason string."""
    frame_h, frame_w = frame_shape[:2]
    if frame_h <= 0 or frame_w <= 0 or len(polygon) < 3:
        return "no valid polygon / frame"

    center_x = box.get("x", 0.0) / frame_w
    center_y = box.get("y", 0.0) / frame_h
    norm_w = box.get("width", 0.0) / frame_w
    norm_h = box.get("height", 0.0) / frame_h
    left, right = center_x - norm_w / 2, center_x + norm_w / 2
    top, bottom = center_y - norm_h / 2, center_y + norm_h / 2

    if not point_in_polygon(center_x, center_y, polygon):
        return f"centre ({center_x:.3f},{center_y:.3f}) is outside the polygon"
    if not (min_width_ratio <= norm_w <= max_width_ratio and min_height_ratio <= norm_h <= max_height_ratio):
        return f"size {norm_w:.3f}x{norm_h:.3f} outside the allowed ratios"

    # touching the FRAME edge means the vehicle is cut off; the polygon itself is handled below
    if left < edge_margin_ratio or right > 1.0 - edge_margin_ratio \
            or top < edge_margin_ratio or bottom > 1.0 - edge_margin_ratio:
        return "box touches the frame edge"

    if min_inside_fraction > 0.0:
        fraction = box_fraction_inside_polygon(polygon, left, top, right, bottom, frame_h, frame_w)
        if fraction < min_inside_fraction:
            return f"only {fraction:.0%} of the box is inside the polygon (need {min_inside_fraction:.0%})"
    return None


def is_inside_roi(
    box: Dict[str, Any],
    frame_shape: Tuple[int, int],
    polygon: Sequence[Sequence[float]],
    min_width_ratio: float, min_height_ratio: float,
    max_width_ratio: float, max_height_ratio: float,
    edge_margin_ratio: float,
    min_inside_fraction: float = MIN_INSIDE_FRACTION,
) -> bool:
    reason = roi_reject_reason(
        box, frame_shape, polygon,
        min_width_ratio, min_height_ratio, max_width_ratio, max_height_ratio,
        edge_margin_ratio, min_inside_fraction,
    )
    if reason is not None and log.isEnabledFor(logging.DEBUG):
        log.debug("[roi] track %s rejected: %s", box.get("track_id", "?"), reason)
    return reason is None
