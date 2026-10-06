from typing import Optional

import numpy as np

from modes.base import DetectionMode


class IdleMode(DetectionMode):
    """No models, no workers, no CPU: the camera and live view keep running, nothing is detected.

    Used when the command center pauses detection, and as the safe landing spot if a mode breaks.
    """

    name = "idle"

    def __init__(self, ctx=None):
        self.ctx = ctx

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        return 0
