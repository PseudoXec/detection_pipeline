import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

log = logging.getLogger("pipeline")

_DEVICE_INFO_PATH = "/ISAPI/System/deviceInfo"
_CHANNEL_INPUT_PATH = "/ISAPI/System/Video/inputs/channels/{channel_id}"
_CHANNEL_STREAM_PATH = "/ISAPI/Streaming/channels/{stream_id}"
_NS_WILDCARD = "{*}"


@dataclass
class CameraDeviceInfo:
    ip_address: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None


def extract_host(rtsp_url: Optional[str]) -> Optional[str]:
    if not rtsp_url:
        return None
    try:
        return urlparse(rtsp_url).hostname
    except Exception:
        return None


def extract_credentials(rtsp_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not rtsp_url:
        return None, None
    try:
        parsed = urlparse(rtsp_url)
        return parsed.username, parsed.password
    except Exception:
        return None, None


def _get_xml(url: str, attempts: list, timeout_seconds: float):
    last_error: Optional[Exception] = None
    for auth in attempts:
        try:
            response = requests.get(url, auth=auth, timeout=timeout_seconds, verify=False)
            response.raise_for_status()
            return ElementTree.fromstring(response.content), None
        except Exception as error:
            last_error = error
    return None, last_error


def _text(root: ElementTree.Element, tag: str) -> Optional[str]:
    value = root.findtext(f"{_NS_WILDCARD}{tag}")
    return value.strip() if value and value.strip() else None


def _fetch_channel_name(base_url: str, channel_id: int, attempts: list, timeout_seconds: float) -> Optional[str]:
    candidates = (
        (_CHANNEL_INPUT_PATH.format(channel_id=channel_id), "name"),
        (_CHANNEL_STREAM_PATH.format(stream_id=channel_id * 100 + 1), "channelName"),
    )
    for path, tag in candidates:
        root, _ = _get_xml(base_url + path, attempts, timeout_seconds)
        if root is not None:
            name = _text(root, tag)
            if name:
                return name
    return None


def fetch_device_info(
    host: Optional[str],
    port: int = 80,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_https: bool = False,
    timeout_seconds: float = 4.0,
    channel_id: int = 1,
) -> CameraDeviceInfo:
    info = CameraDeviceInfo(ip_address=host)
    if not host:
        return info

    scheme = "https" if use_https else "http"
    base_url = f"{scheme}://{host}:{port}"

    attempts: List = [HTTPDigestAuth(username, password), HTTPBasicAuth(username, password)] if username else [None]

    root, last_error = _get_xml(base_url + _DEVICE_INFO_PATH, attempts, timeout_seconds)
    if root is not None:
        info.name = _text(root, "deviceName")
        info.location = _text(root, "deviceLocation")

    channel_name = _fetch_channel_name(base_url, channel_id, attempts, timeout_seconds)
    if channel_name:
        info.name = channel_name
    elif root is not None:
        log.warning("[isapi] %s has no readable channel %d name - using device name %r", host, channel_id, info.name)

    if root is None and not channel_name:
        log.warning(
            "[isapi] could not fetch camera info from %s:%s - camera name/location will be "
            "unavailable, falling back to the IP address alone: %s", host, port, last_error,
        )
        return info

    log.info("[isapi] %s -> name=%r location=%r", host, info.name, info.location)
    return info
