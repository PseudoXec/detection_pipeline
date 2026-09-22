"""
timing.py
---------
Measures how long the pipeline takes to go from "vehicle first detected" to
"plate crop finished" for a single vehicle, broken down by stage.

This answers exactly the question "how fast did the whole detection finish
per vehicle, from vehicle to plate crop?" and gives the breakdown needed to
find which stage is the bottleneck on Raspberry Pi hardware.

time.perf_counter() is used (not time.time()) because it's a monotonic,
high-resolution clock meant specifically for measuring elapsed durations -
it isn't affected by the system clock being adjusted (e.g. NTP sync).
"""

# perf_counter() is a monotonic clock, ideal for measuring elapsed time
from time import perf_counter
# dataclass gives us a small typed record instead of a loose dict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleTiming:
    """Stopwatch for one vehicle, from the moment it's detected to the
    moment its plate crop (or "no plate found") is finalized."""

    # perf_counter() value taken the instant the vehicle box was detected
    _started_at: float = field(default_factory=perf_counter)
    # perf_counter() value taken right after the vehicle crop was saved
    _vehicle_cropped_at: Optional[float] = None
    # perf_counter() value taken right after plate detection finished running
    _plate_detected_at: Optional[float] = None
    # perf_counter() value taken right after the plate crop was saved (or
    # right after we gave up looking for one)
    _finished_at: Optional[float] = None

    def mark_vehicle_cropped(self) -> None:
        """Call this the instant the vehicle crop has been written."""
        self._vehicle_cropped_at = perf_counter()

    def mark_plate_detected(self) -> None:
        """Call this the instant the plate-detection model call returns."""
        self._plate_detected_at = perf_counter()

    def mark_finished(self) -> None:
        """Call this the instant the plate crop is saved (or the vehicle is
        finalized with no plate found)."""
        self._finished_at = perf_counter()

    def _elapsed_ms(self, start: Optional[float], end: Optional[float]) -> Optional[float]:
        """Milliseconds between two perf_counter() readings, or None if either is missing."""
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 2)

    @property
    def vehicle_crop_ms(self) -> Optional[float]:
        """Time from vehicle detection to the vehicle crop being saved."""
        return self._elapsed_ms(self._started_at, self._vehicle_cropped_at)

    @property
    def plate_detect_ms(self) -> Optional[float]:
        """Time spent running the plate-detection model on the vehicle crop."""
        return self._elapsed_ms(self._vehicle_cropped_at, self._plate_detected_at)

    @property
    def plate_crop_ms(self) -> Optional[float]:
        """Time from plate detection finishing to the plate crop being saved."""
        return self._elapsed_ms(self._plate_detected_at, self._finished_at)

    @property
    def total_ms(self) -> Optional[float]:
        """The headline number: vehicle detected -> plate crop finished, end to end."""
        return self._elapsed_ms(self._started_at, self._finished_at)
