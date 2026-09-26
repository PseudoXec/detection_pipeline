"""
api_client.py
-------------
Placeholder POST to the C# dashboard's API (see `features.send_via_api`).
Called by storage.py's writer thread right after a record is written (and
again, later, to retry anything that failed) - never from the detection
loop, so a slow/dead dashboard endpoint can never stall real-time processing.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

import requests

from storage.storage import DetectionRecord

log = logging.getLogger("pipeline")


class SendResult(str, Enum):
    """Outcome of one delivery attempt - the storage layer treats them differently:
      SENT    the dashboard accepted it        -> row is done (deleted / synced=1)
      SKIPPED nothing to send (no plate found) -> row is done, it will never be sent
      FAILED  network/HTTP error               -> keep the row and retry later
    """
    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


def record_to_query_params(
    record: DetectionRecord,
    camera_name: Optional[str] = None,
    camera_ip: Optional[str] = None,
    camera_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Maps DetectionRecord to the C# API's exact Query Parameters.

    camera_name/camera_ip/camera_location come from camera/isapi_client.py
    (fetched once at startup, straight from the camera itself - see run.py)
    rather than being hand-typed here. Any of the three can be None if the
    camera didn't answer or doesn't report it (e.g. no deviceLocation set);
    that just means the param is omitted from the request, not sent empty.
    """

    # Safely get OCR text if it exists, otherwise provide a fallback
    plate_text = getattr(record, "ocr_read", "UNRECOGNIZED") if record.plate_detected else "NO_PLATE"

    params: Dict[str, Any] = {
        "PlateNumber": plate_text,
        "CameraTargetID": 1,
        "DetectionConfidence": round(record.vehicle_confidence or 0.0, 2),
        "DetectedVehicle": record.vehicle_class or "Unknown",
        # headline "vehicle to plate crop" latency only - no per-stage ms breakdown
        # and no bounding boxes go out over the API (see CHANGES.md)
        "TotalPipelineMs": round(record.total_pipeline_ms, 2) if record.total_pipeline_ms is not None else None,
        # Format explicitly to RFC 3339 (e.g., 2017-07-21T17:32:28Z)
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
    """POSTs one record to the dashboard API."""
    if not endpoint_url:
        log.warning("[api] send_via_api is on but no api.endpoint_url is configured - skipping send")
        return SendResult.FAILED

    # Prevent API rejections by dropping records where no plate was found
    if not record.plate_detected:
        log.debug("[api] skipping send for %s: no plate detected", record.track_id)
        return SendResult.SKIPPED

    try:
        # Package the raw JPEG bytes as a file upload (multipart/form-data)
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