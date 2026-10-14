"""Unified API sender for vehicle and person metadata.

Both vehicle and person flows send metadata through this single sender:
- Vehicle detections: POST to api.vehicle_endpoint_url with query params and plate image file.
- Person detections: POST to api.person_endpoint_url with multipart form fields and person/face image files.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import requests

from api.result import SendResult

if TYPE_CHECKING:
    from storage.buffer import PersonRecord, VehicleRecord

log = logging.getLogger("pipeline")

_PERMANENT_4XX = {400, 413, 415, 422}


def _number(value: Optional[float], digits: int) -> Optional[str]:
    return None if value is None else f"{round(float(value), digits):.{digits}f}"


class ApiSender:
    """Sends detection metadata for both vehicle and person flows to their configured endpoints."""

    def __init__(self, api_config, camera_info=None):
        self.config = api_config
        self.camera_info = camera_info

    @property
    def camera_name(self) -> Optional[str]:
        return getattr(self.camera_info, "name", None) if self.camera_info else None

    @property
    def camera_ip(self) -> Optional[str]:
        return getattr(self.camera_info, "ip_address", None) if self.camera_info else None

    @property
    def camera_location(self) -> Optional[str]:
        return getattr(self.camera_info, "location", None) if self.camera_info else None

    def send(self, record: Any) -> Tuple[SendResult, Optional[str]]:
        """Dispatch to send_vehicle or send_person based on record type."""
        # Use class name check or isinstance
        cls_name = record.__class__.__name__
        if cls_name == "VehicleRecord" or hasattr(record, "vehicle_confidence"):
            result = self.send_vehicle(record)
            return result, None if result == SendResult.SENT else str(result.value)
        elif cls_name == "PersonRecord" or hasattr(record, "event_uuid"):
            return self.send_person(record)
        else:
            log.warning("[api] unknown record type: %s", type(record))
            return SendResult.FAILED, f"unknown record type {type(record)}"

    def send_vehicle(self, record: "VehicleRecord") -> SendResult:
        endpoint_url = self.config.vehicle_endpoint_url
        if not endpoint_url:
            log.warning("[api] send_via_api is on but no api.vehicle_endpoint_url is configured - skipping send")
            return SendResult.FAILED

        if not record.plate_detected:
            log.debug("[api] skipping send for %s: no plate detected", record.track_id)
            return SendResult.SKIPPED

        params: Dict[str, Any] = {
            "CameraTargetID": 1,
            "DetectionConfidence": round(record.vehicle_confidence or 0.0, 2),
            "DetectedVehicle": record.vehicle_class or "Unknown",
            "TimeStamp": record.detected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if self.camera_name:
            params["CameraName"] = self.camera_name
        if self.camera_ip:
            params["CameraIpAddress"] = self.camera_ip
        if self.camera_location:
            params["Cameralocation"] = self.camera_location

        image_files = None
        if record.plate_image_jpeg:
            image_files = {"Image": ("plate.jpg", record.plate_image_jpeg, "image/jpeg")}

        try:
            response = requests.post(
                endpoint_url,
                params=params,
                files=image_files,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            return SendResult.SENT
        except requests.exceptions.HTTPError as error:
            error_body = error.response.text if error.response is not None else "No response body"
            log.warning("[api] failed to send vehicle %s (HTTP %s): %s",
                        record.track_id, error.response.status_code if error.response else "?", error_body)
            return SendResult.FAILED
        except requests.RequestException as error:
            log.warning("[api] network error sending vehicle %s: %s", record.track_id, error)
            return SendResult.FAILED

    def send_person(self, record: "PersonRecord") -> Tuple[SendResult, Optional[str]]:
        endpoint_url = self.config.person_endpoint_url
        if not endpoint_url:
            return SendResult.FAILED, "api.person_endpoint_url is not set"

        fields: Dict[str, Optional[str]] = {
            "EventId": record.event_uuid,
            "DeviceId": record.device_id,
            "SessionId": record.session_id,
            "TrackId": record.track_id,
            "CameraId": None if self.config.camera_id is None else str(self.config.camera_id),
            "CameraName": self.camera_name,
            "CameraIpAddress": self.camera_ip,
            "CameraLocation": self.camera_location,
            "DetectedAt": record.detected_at.astimezone().isoformat(timespec="seconds"),
            "PersonConfidence": _number(record.person_confidence, 4),
            "FaceDetected": "true" if record.face_detected else "false",
            "FaceLandmarks": record.face_landmarks,
            "FaceQualityScore": _number(record.face_quality_score, 4),
        }
        form_data = {k: v for k, v in fields.items() if v not in (None, "")}

        files: Dict[str, Any] = {"PersonImage": ("person.jpg", record.person_image_jpeg, "image/jpeg")}
        if record.face_image_jpeg:
            files["FaceImage"] = ("face.jpg", record.face_image_jpeg, "image/jpeg")

        try:
            response = requests.post(
                endpoint_url,
                data=form_data,
                files=files,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as error:
            log.warning("[api] network error sending person %s: %s", record.track_id, error)
            return SendResult.FAILED, f"network: {error}"

        status = response.status_code
        if 200 <= status < 300 or status == 409:
            return SendResult.SENT, None

        body = response.text[:200]
        log.warning("[api] person %s rejected (HTTP %s): %s", record.track_id, status, body)
        if status in _PERMANENT_4XX:
            return SendResult.REJECTED, f"HTTP {status}: {body}"
        return SendResult.FAILED, f"HTTP {status}: {body}"
