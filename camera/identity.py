"""Camera identity: ask the camera over ISAPI ONCE, save it on the Pi, reuse it afterwards.

Flow at startup (called from camera/resolve.py):
    1. data/camera_identity.json exists and matches this camera's IP  -> load it, NO ISAPI call
    2. otherwise                                                      -> one ISAPI run, save the result
    3. ISAPI fails but an old file exists                             -> use the old file (camera offline at boot)

Delete the file (or call refresh_identity) to force a new ISAPI run, e.g. after renaming the camera.
"""
import json
import logging
import os
import tempfile
from dataclasses import asdict, fields
from typing import Callable, Optional

from camera.isapi_client import CameraDeviceInfo

log = logging.getLogger("pipeline")

IDENTITY_FILE = os.path.join("data", "camera_identity.json")


def load_identity(host: Optional[str], path: str = IDENTITY_FILE) -> Optional[CameraDeviceInfo]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        known = {f.name for f in fields(CameraDeviceInfo)}
        info = CameraDeviceInfo(**{k: v for k, v in data.items() if k in known})
    except (OSError, ValueError, TypeError):
        return None
    if host and info.ip_address and info.ip_address != host:
        log.info("[identity] saved identity is for %s but the camera is now %s - ignoring it", info.ip_address, host)
        return None
    return info


def save_identity(info: CameraDeviceInfo, path: str = IDENTITY_FILE) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(info), handle, indent=2)
        os.replace(tmp, path)          # atomic: a power cut never leaves half a file
    except OSError as error:
        log.warning("[identity] could not save %s: %s", path, error)


def get_identity(host: Optional[str], fetch: Callable[[], CameraDeviceInfo], path: str = IDENTITY_FILE,
                 force_refresh: bool = False) -> CameraDeviceInfo:
    cached = None if force_refresh else load_identity(host, path)
    if cached is not None and cached.name:
        log.info("[identity] loaded from %s - no ISAPI call (name=%r location=%r)", path, cached.name, cached.location)
        return cached

    fresh = fetch()
    if fresh.name or fresh.location:
        save_identity(fresh, path)
        log.info("[identity] fetched over ISAPI once and saved to %s", path)
        return fresh

    if cached is not None:
        log.warning("[identity] ISAPI unavailable - using the saved identity")
        return cached
    return fresh                        # IP only; NOT saved, so the next start tries ISAPI again
