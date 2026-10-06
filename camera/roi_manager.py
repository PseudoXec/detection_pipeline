"""The ROI polygon for this camera, shared by every detection mode (lives with the camera code).

The polygon belongs to the CAMERA, not to a detection pipeline: the command center draws it and the
Pi keeps it current no matter which mode is running. It used to live inside the vehicle pipeline; it
moved here so switching modes neither loses it nor restarts the polling thread.
"""
import logging
import threading
from typing import List, Optional, Tuple

from api.roi_client import fetch_roi_polygon
from config.config import PipelineConfig
from core.geometry import compute_detect_region, polygon_bounds

log = logging.getLogger("pipeline")


class RoiManager:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lock = threading.Lock()
        self._version = 0                        # bumps on every change; modes use it to drop cached windows
        self._region_cache: dict = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

    # ------------------------------------------------------------------ polygon
    def get_polygon(self) -> List[List[float]]:
        with self._lock:
            return [list(point) for point in self.config.roi.polygon]

    def set_polygon(self, polygon: List[List[float]]) -> None:
        new_polygon = [[float(x), float(y)] for x, y in polygon]
        with self._lock:
            if new_polygon == self.config.roi.polygon:
                return
            self.config.roi.polygon = new_polygon
            self._version += 1
            self._region_cache.clear()
        log.info("[roi] polygon updated: %s", new_polygon)

    # ------------------------------------------------------------------ command-center sync
    def refresh_from_api(self) -> None:
        api, camera = self.config.api, self.config.camera
        try:
            polygon = fetch_roi_polygon(
                api.roi_endpoint_url, api.camera_id, api.roi_fetch_timeout_seconds,
                pixel_mode=api.roi_coordinates_are_pixels,
                reference_width=api.roi_reference_width or camera.frame_width,
                reference_height=api.roi_reference_height or camera.frame_height,
            )
        except Exception as error:
            log.warning("[roi] unexpected error while fetching the ROI: %s", error, exc_info=True)
            return
        if polygon is not None:
            self.set_polygon(polygon)

    def start_polling(self) -> None:
        cfg = self.config
        if not (cfg.switches.roi_fetch_from_api and cfg.api.roi_endpoint_url and cfg.api.camera_id is not None):
            return
        self.refresh_from_api()
        interval = cfg.api.roi_poll_interval_seconds
        if interval > 0:
            self._poll_thread = threading.Thread(target=self._poll_loop, args=(interval,), name="roi-poll", daemon=True)
            self._poll_thread.start()

    def _poll_loop(self, interval: float) -> None:
        while not self._poll_stop.wait(interval):
            self.refresh_from_api()

    def stop(self) -> None:
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)

    # ------------------------------------------------------------------ detection window
    def detect_region(
        self,
        frame_shape: tuple,
        target_hw: Optional[Tuple[int, int]],
        margin_ratio: float,
        top_extra: float = 0.0,
        label: str = "model",
    ) -> Tuple[int, int, int, int]:
        """Crop window (x1, y1, x2, y2) that covers the ROI plus margin, shaped like the model input.

        `top_extra` grows the window upwards (as a fraction of the frame height): a person's feet are
        in the polygon but their head is well above it.
        """
        shape = tuple(frame_shape[:2])
        key = (shape, target_hw, margin_ratio, top_extra, self._version)
        with self._lock:
            cached = self._region_cache.get(key)
            polygon = self.config.roi.polygon
        if cached is not None:
            return cached

        x_min, x_max, y_min, y_max = polygon_bounds(polygon)
        region = compute_detect_region(
            shape, x_min, x_max, max(0.0, y_min - top_extra), y_max, margin_ratio, target_hw,
        )
        with self._lock:
            self._region_cache[key] = region
        x1, y1, x2, y2 = region
        print(f"[roi] {label} window: x {x1}-{x2}, y {y1}-{y2} "
              f"({x2 - x1}x{y2 - y1} of the {shape[1]}x{shape[0]} frame)")
        return region
