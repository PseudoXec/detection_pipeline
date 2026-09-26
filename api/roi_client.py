"""
roi_client.py
-------------
Fetches the ROI polygon FROM the C# dashboard's own API.

The dashboard's front end lets an operator draw a quadrilateral over the
live view and POSTs it there. The exact JSON shape (as shared by the C# dev):

    GET http://192.168.100.14:5086/api/AiDetection/roi-coordinates?cameraId=1

    {
      "cameraId": 1,
      "roI_x1": 0.15, "roI_y1": 0.15,
      "roI_x2": 0.85, "roI_y2": 0.15,
      "roI_x3": 0.85, "roI_y3": 0.90,
      "roI_x4": 0.15, "roI_y4": 0.90
    }

i.e. one cameraId plus 4 (x, y) anchor points ("8 anchor points" - 4 points x
2 numbers each) tracing the ROI outline in order. We turn that into the
`[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` polygon shape `RoiConfig.polygon` and
`detection/geometry.py::is_inside_roi` expect.

Coordinates are expected normalized 0..1 by default. If they turn out to be
raw pixel coordinates instead, set `api.roi_coordinates_are_pixels: true` in
config.yaml (and, if needed, `api.roi_reference_width/height`) - see
`fetch_roi_polygon` below for exactly what that does.

Called at pipeline startup and (optionally) re-polled every
`api.roi_poll_interval_seconds` - see `pipeline/pipeline.py` - so an operator
redrawing the ROI on the dashboard takes effect without restarting the Pi
service. Never called from the hot detection loop itself.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("pipeline")

# the 4 anchor points, in order, as (x_key, y_key) pairs. The API's own key
# casing is "roI_x1" (capital I, lowercase rest) - matched case-insensitively
# below anyway, so this list is really just "which 4 pairs, in what order".
_ANCHOR_KEYS = [
    ("roI_x1", "roI_y1"),
    ("roI_x2", "roI_y2"),
    ("roI_x3", "roI_y3"),
    ("roI_x4", "roI_y4"),
]


def _get_ci(payload: Dict[str, Any], key: str) -> Any:
    """Case-insensitive dict lookup - the C# API's JSON casing (roI_x1 vs
    ROI_X1 vs roi_x1) isn't guaranteed to match ours byte-for-byte."""
    if key in payload:
        return payload[key]
    lowered = key.lower()
    for actual_key, value in payload.items():
        if actual_key.lower() == lowered:
            return value
    raise KeyError(key)


def parse_roi_response(payload: Dict[str, Any]) -> List[List[float]]:
    """Turn the API's flat {"roI_x1": ..., "roI_y1": ..., ...} shape into the
    polygon list our own code uses. Raises KeyError/ValueError/TypeError on a
    malformed payload - the caller decides what "couldn't fetch" means."""
    polygon: List[List[float]] = []
    for x_key, y_key in _ANCHOR_KEYS:
        x = float(_get_ci(payload, x_key))
        y = float(_get_ci(payload, y_key))
        polygon.append([x, y])
    return polygon


def fetch_roi_polygon(
    endpoint_url: str,
    camera_id: int,
    timeout_seconds: float = 5.0,
    pixel_mode: bool = False,
    reference_width: Optional[float] = None,
    reference_height: Optional[float] = None,
) -> Optional[List[List[float]]]:
    """GETs the current ROI polygon for one camera. Returns None (and logs a
    warning) on any network/HTTP/parsing failure - the caller is expected to
    keep using whatever polygon it already had (the config.yaml default, or
    the last successful fetch) rather than crash detection over this.

    Coordinates are treated as already normalized 0..1 by default. Set
    `pixel_mode=True` (api.roi_coordinates_are_pixels in config.yaml) if the
    endpoint instead sends raw pixel coordinates - each x/y is then divided
    by `reference_width`/`reference_height` (the pixel size the operator's
    mouse coordinates were captured against, e.g. the live-view canvas) to
    get back to 0..1. In pixel_mode, reference_width/height must both be
    positive numbers or the fetch is treated as a failure (better to keep the
    last-known-good polygon than silently divide by a wrong/missing size).
    """
    if not endpoint_url:
        log.warning("[roi] api.roi_endpoint_url is not set - keeping the existing ROI polygon")
        return None
    if pixel_mode and not (reference_width and reference_height):
        log.warning(
            "[roi] roi_coordinates_are_pixels is on but no reference_width/height is available "
            "(set api.roi_reference_width/height, or camera.frame_width/height) - keeping the existing polygon"
        )
        return None

    try:
        response = requests.get(
            endpoint_url,
            params={"cameraId": camera_id},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        polygon = parse_roi_response(response.json())
    except requests.RequestException as error:
        log.warning("[roi] failed to fetch ROI for camera %s from %s: %s", camera_id, endpoint_url, error)
        return None
    except (ValueError, KeyError, TypeError) as error:
        log.warning("[roi] ROI response for camera %s was malformed (%s) - keeping the existing polygon",
                    camera_id, error)
        return None

    if pixel_mode:
        polygon = [[x / reference_width, y / reference_height] for x, y in polygon]
        log.info("[roi] fetched ROI polygon for camera %s (converted from %sx%s pixels): %s",
                  camera_id, reference_width, reference_height, polygon)
    else:
        log.info("[roi] fetched ROI polygon for camera %s: %s", camera_id, polygon)
    return polygon
