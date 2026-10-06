"""Hand-off of a person event to the C# command center.

One multipart/form-data POST per event: metadata as form fields, the person crop and (when a face was
found) the face crop as files. The Pi does not identify anyone: it only finds, crops and sends faces, and
working out who they are happens on the server. The command center gets the face image plus its landmarks
for exactly that. The full field list, coordinate conventions and the C# model are in
docs/person-api-contract.md; this file is the only place that builds the request, so a contract change
only touches `build_form()` and the `files` dict in `send_person()`.

`EventId` is the idempotency key: the endpoint must ignore a second POST with an EventId it already
stored (a retry after a timed-out request can deliver twice). HTTP 409 is treated as delivered.

Turn it on with `send_via_api: true` in modes/person/config.yaml and `person_endpoint_url` in api/config.yaml.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import requests

from api.result import SendResult
from core.urls import redact_credentials

if TYPE_CHECKING:
    from modes.person.store import PersonRecord

log = logging.getLogger("pipeline")

# The server says THIS event is unacceptable (bad fields, too large...): retrying can never help, so it is
# parked locally. Every other error (404 wrong URL, 401/403 wrong token, 5xx, 429...) is a setup or server
# problem that gets fixed - those events must survive it, so they are retried.
_PERMANENT_4XX = {400, 413, 415, 422}


def _number(value: Optional[float], digits: int) -> Optional[str]:
    return None if value is None else f"{round(float(value), digits):.{digits}f}"


def build_form(
    record: "PersonRecord",
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    camera_ip: Optional[str] = None,
    camera_location: Optional[str] = None,
) -> Dict[str, str]:
    """Every metadata field of the event as a form field. Fields without a value are left out."""
    fields: Dict[str, Optional[str]] = {
        "EventId": record.event_uuid,
        "DeviceId": record.device_id,
        "SessionId": record.session_id,
        "TrackId": record.track_id,

        "CameraId": None if camera_id is None else str(camera_id),
        "CameraName": camera_name,
        "CameraIpAddress": camera_ip,
        "CameraLocation": camera_location,
        "CameraSource": redact_credentials(record.camera_source),     # never send the camera password

        # the Pi's local wall-clock time, with its UTC offset so the server can convert it unambiguously
        "DetectedAt": record.detected_at.astimezone().isoformat(timespec="seconds"),
        "FrameWidth": None if record.frame_width is None else str(record.frame_width),
        "FrameHeight": None if record.frame_height is None else str(record.frame_height),

        "PersonConfidence": _number(record.person_confidence, 4),
        "PersonBoxX1": _number(record.person_box_x1, 1),
        "PersonBoxY1": _number(record.person_box_y1, 1),
        "PersonBoxX2": _number(record.person_box_x2, 1),
        "PersonBoxY2": _number(record.person_box_y2, 1),

        "FaceDetected": "true" if record.face_detected else "false",
        "FaceConfidence": _number(record.face_confidence, 4),
        "FaceBoxX1": _number(record.face_box_x1, 1),
        "FaceBoxY1": _number(record.face_box_y1, 1),
        "FaceBoxX2": _number(record.face_box_x2, 1),
        "FaceBoxY2": _number(record.face_box_y2, 1),
        "FaceLandmarks": record.face_landmarks,                        # JSON text: [[x, y] x 5]
        "FaceQualityScore": _number(record.face_quality_score, 4),

        "PersonDetectMs": _number(record.person_detect_ms, 2),
        "PersonCropMs": _number(record.person_crop_ms, 2),
        "FaceDetectMs": _number(record.face_detect_ms, 2),
        "FaceCropMs": _number(record.face_crop_ms, 2),
        "TotalPipelineMs": _number(record.total_pipeline_ms, 2),
    }
    return {key: value for key, value in fields.items() if value not in (None, "")}


def send_person(
    record: "PersonRecord",
    endpoint_url: Optional[str],
    timeout_seconds: float = 5.0,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    camera_ip: Optional[str] = None,
    camera_location: Optional[str] = None,
) -> Tuple[SendResult, Optional[str]]:
    """Returns (result, detail). detail is a short error text for the sync_error column."""
    if not endpoint_url:
        return SendResult.FAILED, "api.person_endpoint_url is not set"

    files: Dict[str, Any] = {"PersonImage": ("person.jpg", record.person_image_jpeg, "image/jpeg")}
    if record.face_image_jpeg:
        files["FaceImage"] = ("face.jpg", record.face_image_jpeg, "image/jpeg")

    try:
        response = requests.post(
            endpoint_url,
            data=build_form(record, camera_id, camera_name, camera_ip, camera_location),
            files=files,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        log.warning("[api] network error sending person %s: %s", record.track_id, error)
        return SendResult.FAILED, f"network: {error}"

    status = response.status_code
    if 200 <= status < 300 or status == 409:          # 409 = the endpoint already has this EventId
        return SendResult.SENT, None
    body = response.text[:200]
    log.warning("[api] person %s rejected (HTTP %s): %s", record.track_id, status, body)
    if status in _PERMANENT_4XX:
        return SendResult.REJECTED, f"HTTP {status}: {body}"
    return SendResult.FAILED, f"HTTP {status}: {body}"
