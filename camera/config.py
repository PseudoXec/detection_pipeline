"""Camera settings: where the video comes from, how it is read, and the camera's ROI polygon."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraConfig:
    rtsp_url: Optional[str] = None
    image: Optional[str] = None
    folder: Optional[str] = None

    source_mode: str = "static"          # "static" | "api" (ask the command center which camera to use)

    frame_width: int = 1280
    frame_height: int = 720

    capture_buffer_size: int = 1
    decode_threads: int = 2
    max_fps: float = 0.0

    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 0

    isapi_enabled: bool = True
    isapi_port: int = 80
    isapi_https: bool = False
    isapi_channel_id: int = 1
    isapi_username: Optional[str] = None
    isapi_password: Optional[str] = None
    isapi_timeout_seconds: float = 4.0


@dataclass
class RoiConfig:
    """The zone drawn for this camera. It belongs to the camera, not to a detection mode: every mode
    uses the same polygon, each with its own rules for what counts as inside (see modes/*/config.yaml)."""
    polygon: List[List[float]] = field(default_factory=lambda: [
        [0.15, 0.15], [0.85, 0.15], [0.85, 0.90], [0.15, 0.90],
    ])
