"""Live view / live boxes settings."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LiveConfig:
    camera_id: str = "cam1"

    endpoint_url: Optional[str] = None
    max_hz: float = 10.0
    timeout_seconds: float = 1.0

    serve_host: str = "0.0.0.0"
    serve_port: int = 8090
    stream_fps: float = 10.0
    stream_width: int = 960
    jpeg_quality: int = 70
    box_max_age_seconds: float = 3.0
    box_visual_tracking: bool = True
    box_extrapolate: bool = True
    box_extrapolate_max_seconds: float = 3.5
    auth_token: Optional[str] = None
