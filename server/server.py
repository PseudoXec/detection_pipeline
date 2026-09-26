"""
server.py
---------
Runs this pipeline as an HTTP OCR service instead of the batch/CLI mode in
cli/ocr_pipeline.py.

This is the "server-sided" half of the split: the edge (detection) pipeline
no longer runs OCR itself. Instead, once it has a plate crop, it POSTs that
crop plus the rest of its detection metadata here. This service's only job
is to read the plate and fill in `ocr_process` / `ocr_read` / `ocr_text` on
that same metadata, then hand it back. Everything else about the record
(track_id, camera_source, vehicle_*, plate_box_*, detected_at, timing
columns, etc.) passes straight through unchanged, whatever fields the
caller happens to send - this service does not need to know the full
schema to do its one job.

    ocr_process  -> True if an OCR attempt was made at all (a plate_crop was
                    present and plate_detected wasn't false)
    ocr_read     -> cleaned, final plate string, or "Unrecognized"
    ocr_text     -> the raw/unfiltered text the model produced before
                    cleanup (None if no attempt was made or nothing came
                    back) - kept separate from ocr_read so a bad clean-up
                    (wrong allowed_chars, etc.) is still visible for review

Endpoints
---------
    POST /ocr/read     multipart/form-data: file field "plate_crop" (JPEG/PNG
                        bytes) + any number of other form fields (metadata),
                        all passed through untouched.
                        OR application/json: {"plate_crop": "<base64>", ...}
    GET  /health        liveness check, no auth required

Request formats
----------------
multipart/form-data (mirrors the shape api_client.py already POSTs):

    curl -X POST http://server:8500/ocr/read \\
        -H "X-API-Key: <token>" \\
        -F "plate_crop=@plate.jpg;type=image/jpeg" \\
        -F "track_id=abc123" \\
        -F "camera_source=cam1" \\
        -F "plate_detected=true" \\
        -F "detected_at=2026-09-26T10:15:00Z"

application/json (plate_crop as base64):

    {
      "track_id": "abc123",
      "camera_source": "cam1",
      "plate_detected": true,
      "plate_crop": "<base64 JPEG bytes>",
      "detected_at": "2026-09-26T10:15:00Z"
    }

Response (200) - the same metadata sent in, plus the three OCR fields:

    {
      "track_id": "abc123",
      "camera_source": "cam1",
      "plate_detected": true,
      "detected_at": "2026-09-26T10:15:00Z",
      "ocr_process": true,
      "ocr_read": "ABC123",
      "ocr_text": "abc123"
    }

If server.forward_url is set, the completed record (image included) is also
relayed on to that URL - normally the C# dashboard's own ingest endpoint.
That relay is what this request's success actually depends on: the OCR read
itself always happens and is always included in the response body, but if
forwarding to forward_url fails, this endpoint returns 502 (with the same
body plus a "forward_error" field) instead of 200 - so the edge pipeline's
own retry logic re-sends the same detection later rather than the row
silently never reaching the dashboard. With no forward_url configured, this
endpoint always returns 200 and it is up to the caller to relay the result
onward itself.

Run it
------
    python server/server.py                     # dev server, run directly - from anywhere
    python -m server.server                      # equivalent, from the OCR/ project root
    gunicorn -w 2 -b 0.0.0.0:8500 server.server:app   # production (see README) - run
                                                        # from the OCR/ project root
"""

import base64
import binascii
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

# Make this file runnable directly (`python server/server.py`, or `python
# server.py` from inside server/) as well as as a module (`python -m
# server.server`, gunicorn's `server.server:app`). This project's other
# modules are imported by absolute package path (config.ocr_config,
# ocr.ocr_reader, server.dedupe, ...), which only resolve if the project
# root - the parent of this server/ folder - is on sys.path. `-m` and
# gunicorn already put it there via the current working directory; running
# the file directly does not, so we add it ourselves before anything else
# imports from those packages.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ocr.image_enhance import enhance_plate_crop
from config.ocr_config import PipelineConfig
from ocr.ocr_reader import PlateOCRReader
from server.dedupe import TrackVoteCache

log = logging.getLogger("ocr_server")

config = PipelineConfig.load()
logging.basicConfig(level=getattr(logging, config.runtime.log_level.upper(), logging.INFO),
                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")

dedupe_cache = (
    TrackVoteCache(config.server.dedupe_window_seconds, config.server.dedupe_max_entries, config.server.dedupe_db_path)
    if config.server.dedupe_enabled else None
)

reader = PlateOCRReader(
    model_name=config.ocr.model_name,
    lang=config.ocr.lang,
    min_confidence=config.ocr.min_confidence,
    allowed_chars=config.ocr.allowed_chars,
    early_exit_score=config.ocr.early_exit_score,
    validate_plate_format=config.ocr.validate_plate_format,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.server.max_content_length_mb * 1024 * 1024

# values that mean "no plate" when sent as a form string or JSON bool/str
_FALSY_STRINGS = {"false", "0", "no", "none", "null", ""}


def _is_falsy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return True
    return str(value).strip().lower() in _FALSY_STRINGS


def _check_auth() -> Optional[Tuple[Any, int]]:
    if not config.server.auth_token:
        return None
    supplied = request.headers.get("X-API-Key", "")
    if supplied != config.server.auth_token:
        return jsonify({"error": "unauthorized"}), 401
    return None


def _decode_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    if not raw_bytes:
        return None
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _extract_request() -> Tuple[Dict[str, Any], Optional[bytes], Optional[str]]:
    """Returns (metadata_fields, plate_crop_bytes, error_message).

    A vehicle with no plate is sent with no "plate_crop" file at all (see the edge
    pipeline's api_client.py) - when there is no file to attach, requests posts a
    plain application/x-www-form-urlencoded body instead of multipart/form-data, so
    this checks request.form / request.files directly (Flask populates both for
    EITHER content type) rather than branching on content_type, which used to miss
    that case entirely and fall through to the JSON branch below with an error."""
    if request.form or request.files:
        metadata = {key: value for key, value in request.form.items()}
        file_obj = request.files.get("plate_crop")
        plate_bytes = file_obj.read() if file_obj else None
        return metadata, plate_bytes, None

    payload = request.get_json(silent=True)
    if payload is None:
        return {}, None, "expected form fields (with or without a 'plate_crop' file), or a JSON body"

    metadata = {key: value for key, value in payload.items() if key != "plate_crop"}
    encoded = payload.get("plate_crop")
    if encoded is None:
        return metadata, None, None
    try:
        plate_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        return metadata, None, f"plate_crop is not valid base64: {error}"
    return metadata, plate_bytes, None


def run_ocr(plate_bytes: Optional[bytes], plate_detected_flag: Any) -> Dict[str, Any]:
    """Mirrors the reference pipeline's rule: no plate -> no OCR attempt at all.

    Distinguishes two cases that both mean "no text", so a reviewer can tell them
    apart later: a vehicle that never had a plate to begin with (no_plate_text) vs
    one that had a plate crop but nothing readable came off it (unrecognized_text).
    """
    if _is_falsy_flag(plate_detected_flag) or not plate_bytes:
        return {"ocr_process": False, "ocr_read": config.ocr.no_plate_text, "ocr_text": None}

    image = _decode_image(plate_bytes)
    if image is None:
        log.warning("run_ocr: plate_crop bytes did not decode as an image")
        return {"ocr_process": True, "ocr_read": config.ocr.unrecognized_text, "ocr_text": None}

    variants = [image]
    if config.preprocess.enhance:
        enhanced = enhance_plate_crop(image, config.preprocess.min_crop_height)
        if config.preprocess.try_both_variants:
            variants.append(enhanced)
        else:
            variants = [enhanced]

    result = reader.read_full(*variants)
    ocr_read = result.text if result.text else config.ocr.unrecognized_text
    return {"ocr_process": True, "ocr_read": ocr_read, "ocr_text": result.raw_text}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": config.ocr.model_name})


@app.route("/ocr/read", methods=["POST"])
def ocr_read_endpoint():
    auth_error = _check_auth()
    if auth_error:
        return auth_error

    metadata, plate_bytes, error = _extract_request()
    if error:
        return jsonify({"error": error}), 400

    ocr_fields = run_ocr(plate_bytes, metadata.get("plate_detected"))
    response_payload = {**metadata, **ocr_fields}
    track_id = metadata.get("track_id")

    # de-dup check: has an EARLIER sighting of this exact track_id already been
    # successfully forwarded, within the window? If so this is a retry of a send
    # that actually went through - vote, but don't forward a second row.
    existing = dedupe_cache.lookup(track_id) if dedupe_cache is not None else None
    if existing is not None and existing.forwarded:
        majority_text = dedupe_cache.record(track_id, ocr_fields["ocr_read"], response_payload, forwarded=False)
        log.info("ocr/read: track_id=%s duplicate within dedupe window - not re-forwarding (majority=%s)",
                  track_id, majority_text)
        duplicate_payload = {**existing.response_payload, "ocr_read": majority_text, "duplicate_of_track_id": track_id}
        return jsonify(duplicate_payload), 200

    if config.server.forward_url:
        forward_ok, forward_error = _forward_to_dashboard(metadata, ocr_fields, plate_bytes)
        if not forward_ok:
            # the OCR read itself succeeded (the caller still gets it back below),
            # but relaying to the dashboard did not - fail this request so the
            # edge pipeline's own retry logic (storage.py) tries again later,
            # instead of silently losing the detection. Deliberately NOT recorded
            # as forwarded=True below, so that retry is free to actually forward.
            log.warning("forward to %s failed: %s", config.server.forward_url, forward_error)
            response_payload["forward_error"] = str(forward_error)
            if dedupe_cache is not None:
                dedupe_cache.record(track_id, ocr_fields["ocr_read"], response_payload, forwarded=False)
            return jsonify(response_payload), 502

    if dedupe_cache is not None:
        dedupe_cache.record(track_id, ocr_fields["ocr_read"], response_payload, forwarded=True)

    log.info("ocr/read: track_id=%s ocr_read=%s", track_id, response_payload.get("ocr_read"))
    return jsonify(response_payload), 200


def _forward_to_dashboard(
    metadata: Dict[str, Any], ocr_fields: Dict[str, Any], plate_bytes: Optional[bytes],
) -> Tuple[bool, Optional[str]]:
    """Relays the completed record on to config.server.forward_url (normally the
    C# dashboard's own ingest endpoint). Mirrors the shape that endpoint already
    expects - query params + a multipart image file - unless forward_as_json is
    set. Returns (success, error_message)."""
    server_cfg = config.server
    forward_fields = {k: v for k, v in metadata.items() if k not in server_cfg.forward_drop_fields}
    if server_cfg.forward_plate_number_field:
        forward_fields[server_cfg.forward_plate_number_field] = ocr_fields.get("ocr_read")

    try:
        if server_cfg.forward_as_json:
            response = requests.post(
                server_cfg.forward_url, json={**forward_fields, **ocr_fields},
                timeout=server_cfg.forward_timeout_seconds,
            )
        else:
            files = None
            if plate_bytes:
                files = {server_cfg.forward_image_field: ("plate.jpg", plate_bytes, "image/jpeg")}
            response = requests.post(
                server_cfg.forward_url, params=forward_fields, files=files,
                timeout=server_cfg.forward_timeout_seconds,
            )
        response.raise_for_status()
    except requests.RequestException as error:
        return False, str(error)
    return True, None


if __name__ == "__main__":
    log.info("Starting OCR server on %s:%s (model=%s)",
              config.server.host, config.server.port, config.ocr.model_name)
    app.run(host=config.server.host, port=config.server.port)
