"""Device management for the Pi and the camera attached to it, served to the command center.

    GET  /device                    Pi health + camera identity (ISAPI, cached) + active mode, in one call
    GET  /device/buffer             SQLite buffer / delivery queue status per mode
    POST /control/refresh-identity  ask the camera over ISAPI again and re-save data/camera_identity.json
    POST /control/restart           restart the pipeline process (needs the systemd service, see deploy/)

The camera identity comes from camera/identity.py (one ISAPI run, saved on the Pi) - polling /device
never touches the camera. The RTSP URL / password is never included in any response.
"""
import logging
import os
import shutil
import socket
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from camera.isapi_client import extract_host
from camera.resolve import resolve_camera_info

log = logging.getLogger("pipeline")

RESTART_EXIT_CODE = 3        # non-zero, so `Restart=on-failure` in the systemd unit brings the service back


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read().strip().strip("\x00")
    except OSError:
        return None


class DeviceService:
    def __init__(self, ctx, manager, request_restart: Callable[[], None]):
        self.ctx = ctx
        self.manager = manager
        self._request_restart = request_restart
        self.frame_source: Optional[Callable[[], Any]] = None     # set by live/outputs.py
        self.started_at = time.time()
        self._identity_lock = threading.Lock()
        self._identity_refreshed_at: Optional[float] = None

    # ------------------------------------------------------------------ GET /device
    def device(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "time": datetime.now().isoformat(timespec="seconds"),
            "pi": self._pi(),
            "camera": self._camera(),
            "pipeline": self._pipeline(),
            "mode": self.manager.status(),
        }

    def _pi(self) -> Dict[str, Any]:
        cpu_temp = _read("/sys/class/thermal/thermal_zone0/temp")
        meminfo: Dict[str, int] = {}
        for line in (_read("/proc/meminfo") or "").splitlines():
            key, _, rest = line.partition(":")
            try:
                meminfo[key] = int(rest.split()[0]) // 1024
            except (ValueError, IndexError):
                pass
        disk = shutil.disk_usage(os.getcwd())
        uptime = _read("/proc/uptime")
        load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        return {
            "hostname": socket.gethostname(),
            "ip_address": self._local_ip(),
            "model": _read("/proc/device-tree/model"),
            "os": " ".join(p for p in (os.uname().sysname, os.uname().release)) if hasattr(os, "uname") else None,
            "cpu_temp_c": round(int(cpu_temp) / 1000, 1) if cpu_temp and cpu_temp.isdigit() else None,
            "cpu_cores": os.cpu_count(),
            "load_avg": [round(v, 2) for v in load] if load[0] is not None else None,
            "memory_mb": {"total": meminfo.get("MemTotal"), "available": meminfo.get("MemAvailable")},
            "disk_gb": {"total": round(disk.total / 1e9, 1), "free": round(disk.free / 1e9, 1)},
            "uptime_seconds": int(float(uptime.split()[0])) if uptime else None,
        }

    def _local_ip(self) -> Optional[str]:
        """The address the Pi uses to reach the camera (no packet is sent by a UDP connect)."""
        target = extract_host(self.ctx.config.camera.rtsp_url) or "8.8.8.8"
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((target, 9))
            return probe.getsockname()[0]
        except OSError:
            return None
        finally:
            probe.close()

    def _camera(self) -> Dict[str, Any]:
        info = self.ctx.camera_info
        config = self.ctx.config
        captured = self.frame_source() if self.frame_source else None
        age = round(time.time() - captured.captured_at, 2) if captured else None
        return {
            "id": config.api.camera_id,
            "live_id": config.live.camera_id,
            "name": getattr(info, "name", None),
            "ip_address": getattr(info, "ip_address", None),
            "location": getattr(info, "location", None),
            "model": getattr(info, "model", None),
            "serial_number": getattr(info, "serial_number", None),
            "firmware_version": getattr(info, "firmware_version", None),
            "mac_address": getattr(info, "mac_address", None),
            "stream": {
                "connected": age is not None and age < 5.0,
                "frame_age_seconds": age,
                "width": config.camera.frame_width, "height": config.camera.frame_height,
            },
            "identity_refreshed_at": (
                datetime.fromtimestamp(self._identity_refreshed_at).isoformat(timespec="seconds")
                if self._identity_refreshed_at else None),
        }

    def _pipeline(self) -> Dict[str, Any]:
        return {
            "session_id": self.ctx.session_id,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
            "uptime_seconds": int(time.time() - self.started_at),
            "supervised": self.supervised(),
        }

    # ------------------------------------------------------------------ GET /device/buffer
    def buffer(self) -> Dict[str, Any]:
        stores = {name: store.status() for name, store in self.ctx.stores.stores.items()}
        db_path = self.ctx.config.storage.database_path
        try:
            db_mb = round(os.path.getsize(db_path) / 1e6, 2)
        except OSError:
            db_mb = None
        return {
            "ok": True,
            "database": {"path": db_path, "size_mb": db_mb},
            "waiting_total": sum(s.get("waiting", 0) for s in stores.values()),
            "stores": stores,
        }

    # ------------------------------------------------------------------ POST /control/refresh-identity
    def refresh_identity(self) -> Dict[str, Any]:
        with self._identity_lock:
            fresh = resolve_camera_info(self.ctx.config, force_refresh=True)
            if fresh is None or not (fresh.name or fresh.location):
                return {"ok": False, "error": "the camera did not answer over ISAPI - keeping the old identity",
                        "camera": self._camera()}
            old = self.ctx.camera_info
            if old is not None:                         # same object the modes hold -> they see it at once
                for key, value in vars(fresh).items():
                    setattr(old, key, value)
            else:
                self.ctx.camera_info = fresh
            info = self.ctx.camera_info
            for store in self.ctx.stores.stores.values():   # new events are stamped with the new name
                store.camera_name, store.camera_ip, store.camera_location = info.name, info.ip_address, info.location
            self._identity_refreshed_at = time.time()
            log.info("[device] camera identity refreshed over ISAPI: %r", info.name)
            return {"ok": True, "camera": self._camera()}

    # ------------------------------------------------------------------ POST /control/restart
    @staticmethod
    def supervised() -> bool:
        """True when systemd started us (it sets INVOCATION_ID) - only then does a restart come back up."""
        return bool(os.environ.get("INVOCATION_ID"))

    def restart(self, force: bool = False) -> Dict[str, Any]:
        if not self.supervised() and not force:
            return {"ok": False, "error": "not running under systemd - a restart would only stop the pipeline. "
                                          "Install deploy/detection-pipeline.service, or send ?force=1 to stop it anyway."}
        log.warning("[device] restart requested by the command center")
        threading.Timer(0.5, self._request_restart).start()   # let the HTTP reply go out first
        return {"ok": True, "restarting": True, "supervised": self.supervised(),
                "note": "the pipeline flushes its buffer and exits; systemd starts it again in ~5 s"}
