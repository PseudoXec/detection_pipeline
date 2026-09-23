"""
api_client.py
-------------
PLACEHOLDER sender for pushing one finished detection record straight to the
C# dashboard's API instead of (or in addition to) the SQLite buffer.

The C# dev owns the real endpoint/contract; this just needs `endpoint_url`
pointed at it once it exists. Until then this fails closed (returns False,
logs once) so the SQLite buffer keeps working as the fallback.
"""

import base64
import logging
from typing import Any, Dict, Optional

import requests

from storage.storage import DetectionRecord

log = logging.getLogger("pipeline")


def _b64(data: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(data).decode("ascii") if data else None


def record_to_payload(record: DetectionRecord) -> Dict[str, Any]:
    """Turns a DetectionRecord into a JSON-serializable dict. Adjust field
    names/shape here once the C# dev shares the real request contract."""
    return {
        "track_id": record.track_id,
        "camera_source": record.camera_source,
        "vehicle_class": record.vehicle_class,
        "vehicle_confidence": record.vehicle_confidence,
        "vehicle_image_base64": _b64(record.vehicle_image_jpeg),
        "vehicle_box": {
            "x1": record.vehicle_box_x1, "y1": record.vehicle_box_y1,
            "x2": record.vehicle_box_x2, "y2": record.vehicle_box_y2,
        } if record.vehicle_box_x1 is not None else None,
        "plate_detected": record.plate_detected,
        "plate_confidence": record.plate_confidence,
        "plate_image_base64": _b64(record.plate_image_jpeg),
        "plate_box": {
            "x1": record.plate_box_x1, "y1": record.plate_box_y1,
            "x2": record.plate_box_x2, "y2": record.plate_box_y2,
        } if record.plate_box_x1 is not None else None,
        "detected_at": record.detected_at.isoformat(sep=" ", timespec="seconds"),
        "timing_ms": {
            "vehicle_detect": record.vehicle_detect_ms,
            "vehicle_crop": record.vehicle_crop_ms,
            "plate_detect": record.plate_detect_ms,
            "plate_crop": record.plate_crop_ms,
            "total": record.total_pipeline_ms,
        },
    }


def send_detection(record: DetectionRecord, endpoint_url: str, timeout_seconds: float = 5.0) -> bool:
    """POSTs one record to the dashboard API. Returns True on a 2xx response,
    False on anything else (timeout, connection error, non-2xx) - the caller
    decides what to do with the row (e.g. keep it in SQLite) when this is False."""
    if not endpoint_url:
        log.warning("[api] send_via_api is on but no api.endpoint_url is configured - skipping send")
        return False

    try:
        response = requests.post(endpoint_url, json=record_to_payload(record), timeout=timeout_seconds)
        response.raise_for_status()
        return True
    except requests.RequestException as error:
        log.warning("[api] failed to send %s: %s", record.track_id, error)
        return False
