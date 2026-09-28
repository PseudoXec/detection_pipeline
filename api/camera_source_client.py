import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests

log = logging.getLogger("pipeline")


@dataclass
class CameraSourceInfo:
    rtsp_url: str
    camera_name: Optional[str] = None


def _get_ci(payload: Dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    lowered = key.lower()
    for actual_key, value in payload.items():
        if actual_key.lower() == lowered:
            return value
    raise KeyError(key)


def _find_ci(payload: Dict[str, Any], key: str) -> Optional[Any]:
    try:
        return _get_ci(payload, key)
    except KeyError:
        return None


def _build_rtsp_url(raw_url: str, username: Optional[str], password: Optional[str]) -> str:
    parsed = urlparse(raw_url)
    if parsed.username or not (username and password):
        return raw_url

    host_port = parsed.hostname or ""
    if parsed.port:
        host_port += f":{parsed.port}"
    netloc = f"{username}:{password}@{host_port}"
    return urlunparse(parsed._replace(netloc=netloc))


def parse_camera_source_response(payload: Dict[str, Any]) -> CameraSourceInfo:
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

    log.info("[camera-source] fetched RTSP source for camera %s from the command center", camera_id)
    return info
