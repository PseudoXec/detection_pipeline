"""What every detection mode (vehicle, person, idle, ...) looks like to the rest of the program.

A mode OWNS everything that is specific to its job - its models, tracker state, background workers,
in-flight tracks. Everything shared (camera, ROI polygon, SQLite stores, live view) is handed in through
RuntimeContext and is never owned, started or stopped by a mode. That split is what lets the command
center switch modes without restarting the process, the camera connection or the live stream.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from config.config import PipelineConfig
from camera.roi_manager import RoiManager
from live.sinks import LiveSinks
from storage.hub import StorageHub


@dataclass
class RuntimeContext:
    config: PipelineConfig
    camera_source: str
    camera_info: Optional[Any]        # camera.isapi_client.CameraDeviceInfo (name / ip / location) or None
    stores: StorageHub
    roi: RoiManager
    live: LiveSinks
    session_id: str                   # one id per program run, stamped on person events


class DetectionMode:
    """Base class and contract.

    * __init__(ctx)   load models and start this mode's workers. If it raises, it must first release
                      whatever it already started - the manager treats a raising constructor as
                      "this mode failed to load" and keeps the previous mode running.
    * warmup(shape)   optional first inference so the first real frame isn't slow.
    * process_frame   called from ONE thread (the main loop), one frame at a time.
    * stop()          stop every worker thread, hand in-flight tracks to storage so nothing is lost,
                      drop the models and free their memory. Must be safe to call twice.
    """

    name = "base"

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        pass

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        return 0

    def stop(self) -> None:
        pass
