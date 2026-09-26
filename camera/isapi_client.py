"""
isapi_client.py
----------------
Looks up a camera's own configured name and (if the camera reports one) its
installation location directly from the camera itself, over Hikvision's
ISAPI, instead of hand-typing them into config.yaml or api_client.py. Used
once at startup (see run.py) so the metadata sent to the dashboard API
(api/api_client.py) reflects what's actually programmed into the camera.

ISAPI basics this module relies on:
  * GET http://<camera-ip>:<port>/ISAPI/System/deviceInfo, HTTP Digest auth
    (most Hikvision cameras reject Basic auth by default; a few older/
    rebranded ones only accept Basic - both are tried).
  * The response is a small, namespaced XML document. `deviceName` is
    always present; `deviceLocation` is camera/firmware dependent, so it's
    treated as optional - a missing tag just means "not set on this camera",
    not an error.

Never raises: a camera that's offline, not ISAPI-compatible, or protected
with the wrong credentials must not stop the pipeline from starting - it
just means the returned CameraDeviceInfo has name/location left as None,
and the caller falls back to the IP address alone.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

log = logging.getLogger("pipeline")

_DEVICE_INFO_PATH = "/ISAPI/System/deviceInfo"
# ISAPI responses are namespaced (e.g. xmlns="http://www.isapi.org/ver20/XMLSchema"),
# so tags must be looked up as "{namespace}tag" - "*" matches whatever namespace
# (or none) the specific camera/firmware happens to use.
_NS_WILDCARD = "{*}"


@dataclass
class CameraDeviceInfo:
    """What ISAPI's deviceInfo endpoint can tell us about a camera. Fields
    are None whenever the camera didn't answer or didn't report that value -
    callers should treat this as "unavailable", not an error."""

    ip_address: Optional[str] = None
    name: Optional[str] = None          # <deviceName>
    location: Optional[str] = None      # <deviceLocation> - not all cameras/firmware set this


def extract_host(rtsp_url: Optional[str]) -> Optional[str]:
    """Pulls just the IP/hostname out of an rtsp://user:pass@host:port/path URL."""
    if not rtsp_url:
        return None
    try:
        return urlparse(rtsp_url).hostname
    except Exception:
        return None


def extract_credentials(rtsp_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Pulls the username/password out of an rtsp://user:pass@host/... URL, so
    ISAPI can reuse the same login the stream already uses when no separate
    isapi_username/isapi_password is configured."""
    if not rtsp_url:
        return None, None
    try:
        parsed = urlparse(rtsp_url)
        return parsed.username, parsed.password
    except Exception:
        return None, None


def fetch_device_info(
    host: Optional[str],
    port: int = 80,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_https: bool = False,
    timeout_seconds: float = 4.0,
) -> CameraDeviceInfo:
    """GETs /ISAPI/System/deviceInfo and pulls out the name + (optional)
    location. Always returns a CameraDeviceInfo - ip_address is filled in
    even on failure; name/location stay None if the request or XML parsing
    didn't succeed, which is exactly what "location if available" means."""
    info = CameraDeviceInfo(ip_address=host)
    if not host:
        return info

    scheme = "https" if use_https else "http"
    url = f"{scheme}://{host}:{port}{_DEVICE_INFO_PATH}"

    # Digest first (the ISAPI norm), Basic as a fallback for older/rebranded
    # devices; with no credentials configured, try once unauthenticated -
    # some cameras leave deviceInfo world-readable.
    attempts = [HTTPDigestAuth(username, password), HTTPBasicAuth(username, password)] if username else [None]

    last_error: Optional[Exception] = None
    for auth in attempts:
        try:
            response = requests.get(url, auth=auth, timeout=timeout_seconds, verify=False)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            name = root.findtext(f"{_NS_WILDCARD}deviceName")
            location = root.findtext(f"{_NS_WILDCARD}deviceLocation")
            info.name = name.strip() if name and name.strip() else None
            info.location = location.strip() if location and location.strip() else None
            log.info("[isapi] %s -> name=%r location=%r", host, info.name, info.location)
            return info
        except Exception as error:
            last_error = error

    log.warning(
        "[isapi] could not fetch deviceInfo from %s:%s - camera name/location will be "
        "unavailable, falling back to the IP address alone: %s", host, port, last_error,
    )
    return info
