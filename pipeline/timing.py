"""
timing.py
---------
Measures how long each stage of the pipeline takes, per vehicle:
    vehicle_detect_ms  - the vehicle model's inference call for the frame
                         this vehicle was first seen in
    vehicle_crop_ms    - cutting the vehicle out of the frame
    plate_detect_ms    - the plate model's inference call for the batch
                         this vehicle's crop was part of
    plate_crop_ms      - cutting (+ enhancing) the plate out of the vehicle crop

Each stage is only measured if its `features.time_*` toggle is on (see
config.py); a disabled stage is simply left as None and excluded from
`total_ms` and from printing/storage.

time.perf_counter() is used (not time.time()) because it's a monotonic,
high-resolution clock meant specifically for measuring elapsed durations -
it isn't affected by the system clock being adjusted (e.g. NTP sync).
"""

from time import perf_counter
from dataclasses import dataclass
from typing import Optional


class Stopwatch:
    """Tiny context manager: `with Stopwatch() as sw: ...` then read `sw.ms`.

    Used to time one stage's actual work (a model call, a crop) rather than
    the wall-clock gap between two unrelated events.
    """

    def __enter__(self) -> "Stopwatch":
        self.ms: Optional[float] = None
        self._start = perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.ms = round((perf_counter() - self._start) * 1000.0, 2)


@dataclass
class VehicleTiming:
    """Per-vehicle timing record. Each field is set directly (in ms) by the
    pipeline as each stage completes; any stage left None was either skipped
    (feature off) or hasn't happened yet."""

    vehicle_detect_ms: Optional[float] = None
    vehicle_crop_ms: Optional[float] = None
    plate_detect_ms: Optional[float] = None
    plate_crop_ms: Optional[float] = None

    @property
    def total_ms(self) -> Optional[float]:
        """Sum of every stage that was actually measured. None if none were."""
        parts = [
            value for value in (
                self.vehicle_detect_ms, self.vehicle_crop_ms,
                self.plate_detect_ms, self.plate_crop_ms,
            ) if value is not None
        ]
        return round(sum(parts), 2) if parts else None
