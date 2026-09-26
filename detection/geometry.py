"""
geometry.py
-----------
Pure math helpers: converting a detection box into pixel edges, measuring
overlap between two boxes, and cutting a box out of an image ("cropping").

Nothing in this file talks to a camera, a model, or a database - it only
works with plain numbers and numpy arrays, which makes it trivial to unit
test in isolation.

A "box" throughout this project is a dict shaped like:
    {"x": center_x, "y": center_y, "width": w, "height": h, ...}
i.e. YOLO's native center-based format (NOT top-left/bottom-right).
"""

# math is used for center-distance calculations (hypot = sqrt(dx^2+dy^2))
import math
# typing hints keep every function signature self-documenting
from typing import Any, Dict, Optional, Sequence, Tuple

# cv2 is used for resizing crops that come out smaller than the minimum height
import cv2
# numpy arrays are how every image is represented once loaded from disk/camera
import numpy as np


def box_edges(box: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Convert a center-based box into (left, top, right, bottom) pixel edges."""
    # left edge = center x minus half the width
    left = box["x"] - box["width"] / 2
    # top edge = center y minus half the height
    top = box["y"] - box["height"] / 2
    # right edge = center x plus half the width
    right = box["x"] + box["width"] / 2
    # bottom edge = center y plus half the height
    bottom = box["y"] + box["height"] / 2
    return left, top, right, bottom


def compute_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Intersection-over-Union: 0 = no overlap, 1 = identical boxes."""
    # get pixel edges for both boxes so we can compare them directly
    ax1, ay1, ax2, ay2 = box_edges(box_a)
    bx1, by1, bx2, by2 = box_edges(box_b)

    # width of the overlapping region (0 if the boxes don't actually overlap)
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    # height of the overlapping region
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    # area shared by both boxes
    intersection = overlap_w * overlap_h

    # area of each box on its own
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    # union = both areas combined, minus the part we counted twice
    union = area_a + area_b - intersection

    # avoid a divide-by-zero if both boxes are degenerate (zero area)
    return intersection / union if union else 0.0


def center_distance_in_widths(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Distance between two box centers, scaled by box size.

    Scaling by size means the same "1.0" threshold works whether the vehicle
    is close to the camera (big box) or far away (small box).
    """
    # straight-line pixel distance between the two centers
    pixel_distance = math.hypot(box_a["x"] - box_b["x"], box_a["y"] - box_b["y"])
    # use the larger of the two boxes' width/height as the scale reference
    scale = max(box_a.get("width", 0), box_a.get("height", 0),
                box_b.get("width", 0), box_b.get("height", 0), 1.0)
    return pixel_distance / scale


def crop_vehicle(
    frame: np.ndarray,
    vehicle_box: Dict[str, Any],
    padding_ratio: float = 0.08,
    min_crop_height: int = 0,
) -> Optional[np.ndarray]:
    """Cut a vehicle out of the full camera frame, with a small safety margin."""
    # frame dimensions, needed so we never crop outside the image bounds
    frame_h, frame_w = frame.shape[:2]

    # pull the box fields out; bail out cleanly if any are missing
    cx, cy = vehicle_box.get("x"), vehicle_box.get("y")
    bw, bh = vehicle_box.get("width"), vehicle_box.get("height")
    if None in (cx, cy, bw, bh):
        return None

    # extra pixels to add around the box on every side
    pad_w, pad_h = bw * padding_ratio, bh * padding_ratio

    # left/top/right/bottom pixel coordinates, clamped to stay inside the frame
    x1 = max(int(cx - bw / 2 - pad_w), 0)
    y1 = max(int(cy - bh / 2 - pad_h), 0)
    x2 = min(int(cx + bw / 2 + pad_w), frame_w)
    y2 = min(int(cy + bh / 2 + pad_h), frame_h)

    # a degenerate box (zero or negative size) can't be cropped
    if x2 <= x1 or y2 <= y1:
        return None

    # actually slice the numpy array - this is the crop
    crop = frame[y1:y2, x1:x2]

    # if the crop came out too small, upscale it so downstream models (and
    # human reviewers) have something usable to look at
    crop_h, crop_w = crop.shape[:2]
    if min_crop_height and crop_h < min_crop_height:
        scale = min_crop_height / crop_h
        crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

    return crop


def crop_plate(
    vehicle_crop: np.ndarray,
    plate_box: Dict[str, Any],
    padding_pixels: int = 8,
    min_crop_height: int = 0,
    padding_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """Cut a plate out of a vehicle crop (plate boxes come from the plate
    model, which runs *on* the vehicle crop, not the full frame)."""
    height, width = vehicle_crop.shape[:2]

    # convert the plate's center-based box to pixel edges
    x1, y1, x2, y2 = box_edges(plate_box)

    # margin = the larger of a fixed pixel amount and a fraction of the plate's own
    # height. A detector box is often a hair tight, and a clipped first/last
    # character is the most common cause of a wrong plate read; the ratio keeps the
    # margin meaningful for big plates while the pixel floor covers tiny ones.
    pad = max(padding_pixels, int(round((y2 - y1) * padding_ratio)))

    # clamp to the vehicle crop's own bounds
    x1 = max(int(x1) - pad, 0)
    y1 = max(int(y1) - pad, 0)
    x2 = min(int(x2) + pad, width)
    y2 = min(int(y2) + pad, height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = vehicle_crop[y1:y2, x1:x2]

    # upscale tiny plate crops so preprocessing/OCR-down-the-line has enough detail
    crop_h, crop_w = crop.shape[:2]
    if min_crop_height and crop_h < min_crop_height:
        scale = min_crop_height / crop_h
        crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

    return crop


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test.

    `polygon` is a sequence of [x, y] vertices (any winding order, need not
    be convex). Casts a ray from (x, y) out to +infinity along x and counts
    how many polygon edges it crosses; odd = inside, even = outside. This is
    the standard even-odd rule and works for any simple (non-self-intersecting)
    polygon, unlike a plain min/max bounding-box check.
    """
    if len(polygon) < 3:
        return False

    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        # does the horizontal ray at height y cross the edge (x1,y1)-(x2,y2)?
        if (y1 > y) != (y2 > y):
            # x coordinate where the edge crosses height y
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) bounding box of a polygon's vertices.

    Used anywhere we still need an axis-aligned window (e.g. the region fed
    to the vehicle model) even though membership itself is decided by the
    polygon, not the box.
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def compute_detect_region(
    frame_shape: Tuple[int, int],
    roi_x_min: float, roi_x_max: float, roi_y_min: float, roi_y_max: float,
    margin_ratio: float,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """Pixel window (x1, y1, x2, y2) of the FULL frame that the vehicle model
    should look at: the ROI plus a safety margin, grown to the model's aspect
    ratio and clamped to the frame.

    Everything outside the ROI is discarded by `is_inside_roi` anyway, so
    running the model on the whole frame just burns CPU. The margin keeps
    vehicles that are only partly inside the ROI whole (needed for stable
    tracking); the aspect-ratio step means the crop maps onto the model's
    static input (e.g. 512x896) with no distortion and, when the crop is the
    same size as the input, with no rescaling at all.
    """
    frame_h, frame_w = frame_shape[:2]

    left = max(0.0, roi_x_min - margin_ratio) * frame_w
    right = min(1.0, roi_x_max + margin_ratio) * frame_w
    top = max(0.0, roi_y_min - margin_ratio) * frame_h
    bottom = min(1.0, roi_y_max + margin_ratio) * frame_h
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return 0, 0, frame_w, frame_h

    if target_hw:
        target_aspect = target_hw[1] / float(target_hw[0])       # width / height
        if width / height < target_aspect:
            width = height * target_aspect                       # too narrow -> widen
        else:
            height = width / target_aspect                       # too wide   -> heighten

    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    width, height = min(width, float(frame_w)), min(height, float(frame_h))
    left = min(max(center_x - width / 2.0, 0.0), frame_w - width)
    top = min(max(center_y - height / 2.0, 0.0), frame_h - height)
    return int(round(left)), int(round(top)), int(round(left + width)), int(round(top + height))


def is_inside_roi(
    box: Dict[str, Any],
    frame_shape: Tuple[int, int],
    polygon: Sequence[Sequence[float]],
    min_width_ratio: float, min_height_ratio: float,
    max_width_ratio: float, max_height_ratio: float,
    edge_margin_ratio: float,
) -> bool:
    """True if a vehicle box's center sits inside the configured ROI polygon
    and the box is a plausible size (not a speck in the distance or a giant
    close-up).

    `polygon` is a list of normalized [x, y] points (see RoiConfig.polygon).
    Membership is decided with a point-in-polygon test on the box's center,
    not a bounding-box comparison - this is what lets the ROI be an angled
    or irregular shape instead of only ever an axis-aligned rectangle.
    """
    frame_h, frame_w = frame_shape[:2]
    if frame_h <= 0 or frame_w <= 0 or len(polygon) < 3:
        return False

    # normalize the box to 0..1 coordinates so it can be compared to the ROI
    center_x = box.get("x", 0.0) / frame_w
    center_y = box.get("y", 0.0) / frame_h
    norm_w = box.get("width", 0.0) / frame_w
    norm_h = box.get("height", 0.0) / frame_h
    left, right = center_x - norm_w / 2, center_x + norm_w / 2
    top, bottom = center_y - norm_h / 2, center_y + norm_h / 2

    # center must land inside the configured ROI polygon (point-in-polygon)
    if not point_in_polygon(center_x, center_y, polygon):
        return False
    # size must be within the plausible range (rejects noise / bad crops)
    if not (min_width_ratio <= norm_w <= max_width_ratio and min_height_ratio <= norm_h <= max_height_ratio):
        return False

    # box must not be clipped by the edge of the ROI's bounding box (half-visible
    # vehicle). The polygon itself decides membership above; this bounding-box
    # check is only an additional "not touching the frame/ROI edge" guard.
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = polygon_bounds(polygon)
    if left < max(roi_x_min, edge_margin_ratio) or right > min(roi_x_max, 1.0 - edge_margin_ratio):
        return False
    if top < max(roi_y_min, edge_margin_ratio) or bottom > min(roi_y_max, 1.0 - edge_margin_ratio):
        return False

    return True
