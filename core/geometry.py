import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("pipeline")

def box_edges(box: Dict[str, Any]) -> Tuple[float, float, float, float]:
    left = box["x"] - box["width"] / 2
    top = box["y"] - box["height"] / 2
    right = box["x"] + box["width"] / 2
    bottom = box["y"] + box["height"] / 2
    return left, top, right, bottom


def compute_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = box_edges(box_a)
    bx1, by1, bx2, by2 = box_edges(box_b)

    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def center_distance_in_widths(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    pixel_distance = math.hypot(box_a["x"] - box_b["x"], box_a["y"] - box_b["y"])
    scale = max(box_a.get("width", 0), box_a.get("height", 0),
                box_b.get("width", 0), box_b.get("height", 0), 1.0)
    return pixel_distance / scale


def crop_box(
    frame: np.ndarray,
    vehicle_box: Dict[str, Any],
    padding_ratio: float = 0.08,
    min_crop_height: int = 0,
) -> Optional[np.ndarray]:
    frame_h, frame_w = frame.shape[:2]

    cx, cy = vehicle_box.get("x"), vehicle_box.get("y")
    bw, bh = vehicle_box.get("width"), vehicle_box.get("height")
    if None in (cx, cy, bw, bh):
        return None

    pad_w, pad_h = bw * padding_ratio, bh * padding_ratio

    x1 = max(int(cx - bw / 2 - pad_w), 0)
    y1 = max(int(cy - bh / 2 - pad_h), 0)
    x2 = min(int(cx + bw / 2 + pad_w), frame_w)
    y2 = min(int(cy + bh / 2 + pad_h), frame_h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    crop_h, crop_w = crop.shape[:2]
    if min_crop_height and crop_h < min_crop_height:
        scale = min_crop_height / crop_h
        crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

    return crop



def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    if len(polygon) < 3:
        return False

    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def compute_detect_region(
    frame_shape: Tuple[int, int],
    roi_x_min: float, roi_x_max: float, roi_y_min: float, roi_y_max: float,
    margin_ratio: float,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]

    left = max(0.0, roi_x_min - margin_ratio) * frame_w
    right = min(1.0, roi_x_max + margin_ratio) * frame_w
    top = max(0.0, roi_y_min - margin_ratio) * frame_h
    bottom = min(1.0, roi_y_max + margin_ratio) * frame_h
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return 0, 0, frame_w, frame_h

    if target_hw:
        target_aspect = target_hw[1] / float(target_hw[0])
        if width / height < target_aspect:
            width = height * target_aspect
        else:
            height = width / target_aspect

    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    width, height = min(width, float(frame_w)), min(height, float(frame_h))
    left = min(max(center_x - width / 2.0, 0.0), frame_w - width)
    top = min(max(center_y - height / 2.0, 0.0), frame_h - height)
    return int(round(left)), int(round(top)), int(round(left + width)), int(round(top + height))
