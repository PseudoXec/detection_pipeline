"""
camera_source_client.py
------------------------
Fetches this camera's RTSP URL (+ credentials) FROM the command center's own
API, as an alternative to hand-typing `camera.rtsp_url` into config.yaml.

This is the RTSP counterpart to `api/roi_client.py`. The command center
already knows every camera's RTSP info (that's how its own live view finds
them), so instead of duplicating `192.168.100.229:554/stream1` + credentials
into this Pi's config.yaml by hand, the pipeline can ask for it - the exact
same way it already asks for the ROI polygon.

NOTE ON THE ENDPOINT: the C# side does not have this endpoint built yet.
The JSON shape below is a *proposed* extension of the existing ROI response
(same URL as `api.roi_endpoint_url`, same `?cameraId=` query param) rather
than a brand-new endpoint, since the two are naturally fetched together:

    GET http://192.168.100.14:5086/api/AiDetection/roi-coordinates?cameraId=1

    {
      "cameraId": 1,
      "roI_x1": 0.15, "roI_y1": 0.15, ... ,      <- already exists, see roi_client.py
      "rtspUrl": "rtsp://192.168.100.229:554/stream1",
      "username": "admin",
      "password": "Victoria2313*"
    }

`rtspUrl` may or may not already have `user:pass@` embedded - both are
handled (see `_build_rtsp_url`). `username`/`password` can be omitted
entirely if `rtspUrl` already carries them, or if the camera needs no auth.
Field names are matched case-insensitively, same as roi_client.py, since
the C# side's exact casing isn't confirmed yet either.

Once the real endpoint exists and its shape is confirmed, only
`parse_camera_source_response` below should need editing - nothing calling
`fetch_camera_source()` needs to change.

Never raises: any network/HTTP/parsing problem returns None, and the caller
(see run.py's `resolve_camera_source`) is expected to fall back to whatever
static `camera.rtsp_url` is already in config.yaml, if any.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests

log = logging.getLogger("pipeline")


@dataclass
class CameraSourceInfo:
    """What the command center told us about this camera's video source."""

    rtsp_url: str                      # always a fully-formed rtsp:// URL, credentials included if any
    camera_name: Optional[str] = None  # only set if the endpoint happens to include one


def _get_ci(payload: Dict[str, Any], key: str) -> Any:
    """Case-insensitive dict lookup (same helper as roi_client.py - kept as
    its own copy here since the two modules are meant to stay independent)."""
    if key in payload:
        return payload[key]
    lowered = key.lower()
    for actual_key, value in payload.items():
        if actual_key.lower() == lowered:
            return value
    raise KeyError(key)


def _find_ci(payload: Dict[str, Any], key: str) -> Optional[Any]:
    """Like _get_ci, but returns None instead of raising when absent - for
    fields that are genuinely optional (username/password/camera name)."""
    try:
        return _get_ci(payload, key)
    except KeyError:
        return None


def _build_rtsp_url(raw_url: str, username: Optional[str], password: Optional[str]) -> str:
    """Injects username/password into raw_url if they were sent as separate
    fields and raw_url doesn't already have credentials embedded. If raw_url
    already has a user:pass@ (or no separate credentials were sent at all),
    it's returned unchanged."""
    parsed = urlparse(raw_url)
    if parsed.username or not (username and password):
        return raw_url

    host_port = parsed.hostname or ""
    if parsed.port:
        host_port += f":{parsed.port}"
    netloc = f"{username}:{password}@{host_port}"
    return urlunparse(parsed._replace(netloc=netloc))


def parse_camera_source_response(payload: Dict[str, Any]) -> CameraSourceInfo:
    """Turns the command center's JSON into a CameraSourceInfo. Raises
    KeyError/ValueError/TypeError on a malformed payload (e.g. no rtspUrl at
    all) - the caller decides what "couldn't fetch" means."""
    raw_url = _get_ci(payload, "rtspUrl")
    if not raw_url or not str(raw_url).strip():
        raise ValueError("rtspUrl was empty")

    username = _find_ci(payload, "username")
    password = _find_ci(payload, "password")
    camera_name = _find_ci(payload, "cameraName")

    return CameraSourceInfo(
        rtsp_url=_build_rtsp_url(str(raw_url).strip(), username, password),
        camera_name=str(camera_name).strip() if camera_name else None,
    )


def fetch_camera_source(
    endpoint_url: str,
    camera_id: int,
    timeout_seconds: float = 5.0,
) -> Optional[CameraSourceInfo]:
    """GETs the current RTSP source for one camera. Returns None (and logs a
    warning) on any network/HTTP/parsing failure - the caller is expected to
    fall back to config.yaml's static camera.rtsp_url, if one is set, rather
    than crash the pipeline over this."""
    if not endpoint_url:
        log.warning("[camera-source] no endpoint configured - keeping the static camera.rtsp_url")
        return None

    try:
        response = requests.get(
            endpoint_url,
            params={"cameraId": camera_id},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        info = parse_camera_source_response(response.json())
    except requests.RequestException as error:
        log.warning("[camera-source] failed to fetch camera %s source from %s: %s", camera_id, endpoint_url, error)
        return None
    except (ValueError, KeyError, TypeError) as error:
        log.warning("[camera-source] response for camera %s was malformed (%s) - "
                    "keeping the static camera.rtsp_url", camera_id, error)
        return None

    # never log the resolved URL at info level - it can carry credentials
    log.info("[camera-source] fetched RTSP source for camera %s from the command center", camera_id)
    return info
