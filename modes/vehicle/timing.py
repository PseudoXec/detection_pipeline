from time import perf_counter
from dataclasses import dataclass
from typing import Optional


class Stopwatch:
    def __enter__(self) -> "Stopwatch":
        self.ms: Optional[float] = None
        self._start = perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.ms = round((perf_counter() - self._start) * 1000.0, 2)


@dataclass
class VehicleTiming:
    vehicle_detect_ms: Optional[float] = None
    vehicle_crop_ms: Optional[float] = None
    plate_detect_ms: Optional[float] = None
    plate_crop_ms: Optional[float] = None

    @property
    def total_ms(self) -> Optional[float]:
        parts = [
            value for value in (
                self.vehicle_detect_ms, self.vehicle_crop_ms,
                self.plate_detect_ms, self.plate_crop_ms,
            ) if value is not None
        ]
        return round(sum(parts), 2) if parts else None
