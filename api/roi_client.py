"""Fetch the ROI polygon from the command center.

Drop-in replacement for api/roi_client.py - same public functions
(`parse_roi_response`, `fetch_roi_polygon`) and same signatures, so pipeline.py
needs no change.

What changed vs. the previous version
-------------------------------------
The old version silently kept the yaml ROI whenever the response was not
*exactly* what it expected, so "the API has values but the ROI never changes"
was the visible symptom of several different mismatches:

* key names had to be exactly roI_x1..roI_y4 (case-insensitive). `roiX1`,
  `x1`, `point1`, a `polygon`/`points` list, a JSON string, or one more level of
  nesting all failed. Keys are now matched after stripping case, `_`, `-`, spaces
  and every common layout is accepted.
* values had to be 0..1 unless `api.roi_coordinates_are_pixels: true`. Percent
  (0..100) or pixel values were rejected with only a WARNING. The scale is now
  auto-detected (0..1, 0..100 percent, or pixels using the reference size).
* points sent in "zig-zag" order (TL, TR, BL, BR) make a bow-tie polygon, so
  point-in-polygon rejected every vehicle. Self-intersecting quads are re-ordered.
* all-zero / null values (the endpoint's "not set yet" state) are now reported
  clearly instead of producing a degenerate ROI.
* the request now sends `?id=<camera_id>` (the endpoint rejects `cameraId` with
  400 "invalid camera Id"), and notes (debug log) if the response's own cameraId doesn't match.
* every fetch outcome is logged, including the raw response the first time and
  whenever it changes, so a mismatch is visible in the log.
"""
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

log = logging.getLogger("pipeline")

_POINT_COUNT = 4
_MIN_AREA = 0.001          # normalized area below this = empty / all-zeros ROI
_OVERSHOOT = 0.02          # tolerated overshoot past 0..1 before rejecting

_LIST_KEYS = (
    "polygon", "points", "roi", "roipoints", "roipolygon", "roicoordinates",
    "coordinates", "corners", "vertices",
)
_SCALAR_STYLES = (
    ("roix{i}", "roiy{i}"),
    ("roi{i}x", "roi{i}y"),
    ("x{i}", "y{i}"),
    ("pointx{i}", "pointy{i}"),
    ("point{i}x", "point{i}y"),
    ("p{i}x", "p{i}y"),
)

# last thing we logged per (endpoint, camera) so a 30 s poll doesn't spam the log
_last_seen: Dict[Tuple[str, Any], Any] = {}


def _norm(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _to_float(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("value is null/empty - the ROI is not set in the command center yet")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _flatten(payload: Any, depth: int = 0) -> Dict[str, Any]:
    """normalized-key -> value for the payload and up to 3 nested levels (outer wins)."""
    flat: Dict[str, Any] = {}
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            flat.update(_flatten(payload[0], depth))
        return flat
    if not isinstance(payload, dict):
        return flat

    nested: List[Any] = []
    for key, value in payload.items():
        flat.setdefault(_norm(key), value)
        if isinstance(value, (dict, list)):
            nested.append(value)
    if depth < 3:
        for value in nested:
            for key, inner in _flatten(value, depth + 1).items():
                flat.setdefault(key, inner)
    return flat


def _point_from_item(item: Any) -> List[float]:
    if isinstance(item, dict):
        lowered = {_norm(k): v for k, v in item.items()}
        return [_to_float(lowered["x"]), _to_float(lowered["y"])]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [_to_float(item[0]), _to_float(item[1])]
    raise ValueError(f"cannot read a point from {item!r}")


def _points_from_list(value: Any) -> Optional[List[List[float]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return [_point_from_item(item) for item in value]


def _extract_points(payload: Any) -> List[List[float]]:
    flat = _flatten(payload)

    for x_fmt, y_fmt in _SCALAR_STYLES:
        keys = [(x_fmt.format(i=i), y_fmt.format(i=i)) for i in range(1, _POINT_COUNT + 1)]
        if all(xk in flat and yk in flat for xk, yk in keys):
            return [[_to_float(flat[xk]), _to_float(flat[yk])] for xk, yk in keys]

    for name in _LIST_KEYS:
        if name in flat:
            points = _points_from_list(flat[name])
            if points:
                return points

    numbered = [flat.get(f"point{i}", flat.get(f"p{i}")) for i in range(1, _POINT_COUNT + 1)]
    if all(item is not None for item in numbered):
        return [_point_from_item(item) for item in numbered]

    raise KeyError(f"no ROI points found in the response (keys seen: {sorted(flat)[:25]})")


# ---------------------------------------------------------------- geometry helpers

def _area(polygon: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1, p2, p3, p4) -> bool:
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _self_intersecting(polygon: Sequence[Sequence[float]]) -> bool:
    n = len(polygon)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return True
    return False


def _order_clockwise(polygon: List[List[float]]) -> List[List[float]]:
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    ordered = sorted(polygon, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(len(ordered)), key=lambda i: ordered[i][0] + ordered[i][1])  # top-left first
    return ordered[start:] + ordered[:start]


# ---------------------------------------------------------------- public API

def parse_roi_response(payload: Any) -> List[List[float]]:
    """Raw points exactly as the API sent them (any scale). Raises on a shape it can't read."""
    return _extract_points(payload)


def _normalize_scale(
    polygon: List[List[float]],
    pixel_mode: bool,
    reference_width: Optional[float],
    reference_height: Optional[float],
) -> Tuple[List[List[float]], str]:
    top = max(abs(v) for point in polygon for v in point)

    if top <= 1.0:
        if pixel_mode:
            log.warning("[roi] api.roi_coordinates_are_pixels is on but every value is <= 1, "
                        "so they are being treated as normalized 0..1")
        return polygon, "normalized 0..1"

    if not pixel_mode and top <= 100.0:
        return [[x / 100.0, y / 100.0] for x, y in polygon], "percent 0..100"

    if not (reference_width and reference_height):
        raise ValueError(
            f"values go up to {top:g} so they look like pixels, but no reference size is known - set "
            "api.roi_reference_width/height (or camera.frame_width/height) to the resolution the "
            "command center drew the ROI on")
    return ([[x / reference_width, y / reference_height] for x, y in polygon],
            f"pixels of {reference_width:g}x{reference_height:g}")


def _validate(polygon: List[List[float]]) -> List[List[float]]:
    if any(not (-_OVERSHOOT <= v <= 1.0 + _OVERSHOOT) for point in polygon for v in point):
        raise ValueError(f"ROI falls outside the frame after scaling ({polygon}) - "
                         "check the coordinate scale / reference size")
    polygon = [[min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in polygon]
    if _self_intersecting(polygon):  # must run before the area check: a bow-tie's signed area cancels to ~0
        ordered = _order_clockwise(polygon)
        log.info("[roi] points arrived in a crossing order %s - re-ordered to %s", polygon, ordered)
        polygon = ordered
    if _area(polygon) < _MIN_AREA:
        raise ValueError(f"ROI is empty / degenerate ({polygon}) - all-zero or all-identical points "
                         "(e.g. every value 1) mean the command center has not saved a real ROI for "
                         "this camera yet")
    return polygon


def _warn_on_camera_mismatch(body: Any, requested_id: Any) -> None:
    returned = _flatten(body).get("cameraid")
    try:
        if returned is not None and int(returned) != int(requested_id):
            # some command-center builds always echo cameraId 0 even with a real ROI, so this is
            # only a hint (debug); a blank record is caught separately by the degenerate-ROI check
            log.debug("[roi] asked for camera %s but the response says cameraId=%s",
                      requested_id, returned)
    except (TypeError, ValueError):
        pass


def _error_detail(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    return f"{error} - server said: {response.text[:300]!r}"


def fetch_roi_polygon(
    endpoint_url: str,
    camera_id: int,
    timeout_seconds: float = 5.0,
    pixel_mode: bool = False,
    reference_width: Optional[float] = None,
    reference_height: Optional[float] = None,
) -> Optional[List[List[float]]]:
    """Returns a normalized 0..1 polygon, or None (caller keeps the current ROI)."""
    if not endpoint_url:
        log.warning("[roi] api.roi_endpoint_url is not set - keeping the existing ROI polygon")
        return None

    payload_preview = ""
    try:
        response = requests.get(
            endpoint_url,
            params={"id": camera_id},  # this endpoint reads `id`; `cameraId` gives 400 "invalid camera Id"
            headers={"Accept": "application/json"},
            timeout=timeout_seconds,
        )
        payload_preview = response.text[:300]
        response.raise_for_status()
        body = response.json()
        _warn_on_camera_mismatch(body, camera_id)
        raw_points = parse_roi_response(body)
        polygon, scale = _normalize_scale(raw_points, pixel_mode, reference_width, reference_height)
        polygon = _validate(polygon)
    except requests.RequestException as error:
        log.warning("[roi] failed to fetch ROI for camera %s from %s: %s - response was %r - "
                    "keeping the existing polygon", camera_id, endpoint_url, _error_detail(error), payload_preview)
        return None
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        log.warning("[roi] ROI response for camera %s could not be used (%s: %s) - response was %r - "
                    "keeping the existing polygon", camera_id, type(error).__name__, error, payload_preview)
        return None

    signature = (tuple(tuple(round(v, 6) for v in p) for p in polygon), scale)
    cache_key = (endpoint_url, camera_id)
    if _last_seen.get(cache_key) != signature:
        _last_seen[cache_key] = signature
        log.info("[roi] camera %s ROI from API (%s): raw=%s -> polygon=%s | response=%r",
                 camera_id, scale, raw_points, polygon, payload_preview)
    else:
        log.debug("[roi] camera %s ROI unchanged: %s", camera_id, polygon)
    return polygon
