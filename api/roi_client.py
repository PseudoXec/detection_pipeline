import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("pipeline")

_ANCHOR_KEYS = [
    ("roI_x1", "roI_y1"),
    ("roI_x2", "roI_y2"),
    ("roI_x3", "roI_y3"),
    ("roI_x4", "roI_y4"),
]


def _get_ci(payload: Dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    lowered = key.lower()
    for actual_key, value in payload.items():
        if actual_key.lower() == lowered:
            return value
    raise KeyError(key)


def parse_roi_response(payload: Dict[str, Any]) -> List[List[float]]:
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
