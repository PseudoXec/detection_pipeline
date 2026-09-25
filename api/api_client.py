import logging
from enum import Enum
from typing import Any, Dict

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


def record_to_query_params(record: DetectionRecord) -> Dict[str, Any]:
    """Maps DetectionRecord to the C# API's exact Query Parameters."""
    
    # Safely get OCR text if it exists, otherwise provide a fallback
    plate_text = getattr(record, "ocr_read", "UNRECOGNIZED") if record.plate_detected else "NO_PLATE"

    return {
        "PlateNumber": plate_text,
        "CameraTargetID": 1, 
        "DetectionConfidence": round(record.vehicle_confidence or 0.0, 2),
        "Cameralocation": "Victoria, Tarlac",
        "CameraIpAddress": "192.168.100.229",
        "CameraName": "Main Camera",
        "DetectedVehicle": record.vehicle_class or "Unknown",
        
        # Map Python's (x1, y1, x2, y2) bounding box format to the API's X1, X2, X3, X4 sequence
        "VehicleBoxX1": record.vehicle_box_x1 or 0.0,
        "VehicleBoxX2": record.vehicle_box_y1 or 0.0,
        "VehicleBoxX3": record.vehicle_box_x2 or 0.0,
        "VehicleBoxX4": record.vehicle_box_y2 or 0.0,
        
        "PlateBoxX1": record.plate_box_x1 or 0.0,
        "PlateBoxX2": record.plate_box_y1 or 0.0,
        "PlateBoxX3": record.plate_box_x2 or 0.0,
        "PlateBoxX4": record.plate_box_y2 or 0.0,
        
        # Format explicitly to RFC 3339 (e.g., 2017-07-21T17:32:28Z)
        "TimeStamp": record.detected_at.strftime('%Y-%m-%dT%H:%M:%SZ')
    }


def send_detection(record: DetectionRecord, endpoint_url: str, timeout_seconds: float = 5.0) -> SendResult:
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
            params=record_to_query_params(record),
            files=image_files,
            timeout=timeout_seconds
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