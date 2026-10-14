"""Base definitions for detection modes."""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from camera.roi_manager import RoiManager
from config.config import PipelineConfig
from storage.buffer import SQLiteBuffer


@dataclass
class RuntimeContext:
    config: PipelineConfig
    camera_source: str
    camera_info: Optional[Any]
    buffer: SQLiteBuffer
    roi: RoiManager
    session_id: str


class DetectionMode:
    """Base class for detection modes."""

    name = "base"

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        pass

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        return 0

    def stop(self) -> None:
        pass
