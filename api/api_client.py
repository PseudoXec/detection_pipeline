import logging
from enum import Enum
from typing import Any, Dict, Optional

import requests

from storage.storage import DetectionRecord

log = logging.getLogger("pipeline")


class SendResult(str, Enum):
    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


def record_to_query_params(
    record: DetectionRecord,
    camera_name: Optional[str] = None,
    camera_ip: Optional[str] = None,
    camera_location: Optional[str] = None,
) -> Dict[str, Any]:
    plate_text = getattr(record, "ocr_read", "UNRECOGNIZED") if record.plate_detected else "NO_PLATE"

    params: Dict[str, Any] = {
        "PlateNumber": plate_text,
        "CameraTargetID": 1,
        "DetectionConfidence": round(record.vehicle_confidence or 0.0, 2),
        "DetectedVehicle": record.vehicle_class or "Unknown",
        "TotalPipelineMs": round(record.total_pipeline_ms, 2) if record.total_pipeline_ms is not None else None,
        "TimeStamp": record.detected_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    if camera_name:
        params["CameraName"] = camera_name
    if camera_ip:
        params["CameraIpAddress"] = camera_ip
    if camera_location:
        params["Cameralocation"] = camera_location

    return params


def send_detection(
    record: DetectionRecord,
    endpoint_url: str,
    timeout_seconds: float = 5.0,
    camera_name: Optional[str] = None,
    camera_ip: Optional[str] = None,
    camera_location: Optional[str] = None,
) -> SendResult:
    if not endpoint_url:
        log.warning("[api] send_via_api is on but no api.endpoint_url is configured - skipping send")
        return SendResult.FAILED

    if not record.plate_detected:
        log.debug("[api] skipping send for %s: no plate detected", record.track_id)
        return SendResult.SKIPPED

    try:
        image_files = None
        if record.plate_image_jpeg:
            image_files = {"Image": ("plate.jpg", record.plate_image_jpeg, "image/jpeg")}

        response = requests.post(
            endpoint_url,
            params=record_to_query_params(record, camera_name, camera_ip, camera_location),
            files=image_files,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return SendResult.SENT

    except requests.exceptions.HTTPError as error:
        error_body = error.response.text if error.response is not None else "No response body"
        log.warning("[api] failed to send %s (HTTP %s): %s", record.track_id, error.response.status_code, error_body)
        return SendResult.FAILED

    except requests.RequestException as error:
        log.warning("[api] network error sending %s: %s", record.track_id, error)
        return SendResult.FAILED
