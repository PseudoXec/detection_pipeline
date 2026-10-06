"""Command-center (C#) API settings: every endpoint URL lives here, whichever mode uses it."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ApiConfig:
    vehicle_endpoint_url: Optional[str] = None   # vehicle / plate events
    person_endpoint_url: Optional[str] = None    # person events (placeholder contract, see api/person_client.py)
    timeout_seconds: float = 5.0

    roi_endpoint_url: Optional[str] = None       # also used to look up the camera source
    camera_id: Optional[int] = None
    roi_fetch_timeout_seconds: float = 5.0
    roi_coordinates_are_pixels: bool = False
    roi_reference_width: Optional[int] = None
    roi_reference_height: Optional[int] = None
    roi_poll_interval_seconds: float = 30.0
