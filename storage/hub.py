"""Owns the SQLite stores of every detection mode, for the whole life of the process.

Stores are not started and stopped with a mode: rows queued by the mode you just switched away from
(and rows still waiting to be re-sent to the dashboard) must keep being written and delivered.
Each store is just one thread blocked on a queue, which costs nothing while idle.
"""
import logging
import threading
from typing import Dict, Optional

from config.config import PipelineConfig
from modes.person.store import PersonStorage
from modes.vehicle.store import DetectionStorage
from storage.buffer import BufferedStore

log = logging.getLogger("pipeline")


class StorageHub:
    def __init__(self, config: PipelineConfig, camera_info=None):
        cam_name = camera_info.name if camera_info else None
        cam_ip = camera_info.ip_address if camera_info else None
        cam_location = camera_info.location if camera_info else None
        storage, vehicle, api, person = config.storage, config.vehicle.features, config.api, config.person
        self.config = config

        self.vehicle = DetectionStorage(
            database_path=storage.database_path,
            batch_size=storage.write_batch_size,
            flush_interval_seconds=storage.write_flush_interval_seconds,
            send_via_api=vehicle.send_via_api,
            delete_row_after_api_send=vehicle.delete_row_after_api_send,
            api_endpoint_url=api.vehicle_endpoint_url,
            api_timeout_seconds=api.timeout_seconds,
            output_dir=storage.output_dir if vehicle.save_images_to_disk else None,
            delete_disk_images_after_send=storage.delete_disk_images_after_send,
            send_retry_seconds=storage.send_retry_seconds,
            camera_name=cam_name, camera_ip=cam_ip, camera_location=cam_location,
        )
        self.person = PersonStorage(
            database_path=storage.database_path,
            batch_size=storage.write_batch_size,
            flush_interval_seconds=storage.write_flush_interval_seconds,
            send_via_api=person.send_via_api,
            delete_row_after_api_send=person.delete_row_after_api_send,
            send_rows_without_face=person.send_rows_without_face,
            api_endpoint_url=api.person_endpoint_url,
            api_timeout_seconds=api.timeout_seconds,
            output_dir=storage.output_dir if person.save_images_to_disk else None,
            delete_disk_images_after_send=storage.delete_disk_images_after_send,
            send_retry_seconds=storage.send_retry_seconds,
            camera_name=cam_name, camera_ip=cam_ip, camera_location=cam_location,
            camera_id=api.camera_id,
        )
        self._housekeeping_stop = threading.Event()
        self._housekeeping_thread: Optional[threading.Thread] = None

    @property
    def stores(self) -> Dict[str, BufferedStore]:
        return {"vehicle": self.vehicle, "person": self.person}

    def start(self) -> "StorageHub":
        for store in self.stores.values():
            store.start()
        self._start_housekeeping()
        return self

    def stop(self) -> None:
        self._housekeeping_stop.set()
        for store in self.stores.values():
            store.stop()

    # ------------------------------------------------------------------ retention
    def _start_housekeeping(self) -> None:
        retention_days = self.config.storage.retention_days
        max_unsynced_days = self.config.storage.max_unsynced_days
        if retention_days <= 0 and max_unsynced_days <= 0:
            return
        self._housekeeping_thread = threading.Thread(
            target=self._housekeeping_loop, args=(retention_days, max_unsynced_days),
            name="retention", daemon=True,
        )
        self._housekeeping_thread.start()

    def _housekeeping_loop(self, retention_days: int, max_unsynced_days: int) -> None:
        while not self._housekeeping_stop.is_set():
            try:
                synced = unsynced = 0
                for store in self.stores.values():
                    synced += store.delete_synced_older_than(retention_days)
                    unsynced += store.delete_unsynced_older_than(max_unsynced_days)
                files = self.vehicle.sweep_output_dir(retention_days) or self.person.sweep_output_dir(retention_days)
                if synced or unsynced or files:
                    log.info("retention cleanup: removed %d synced row(s), %d undelivered row(s), %d image file(s)",
                             synced, unsynced, files)
            except Exception as error:
                log.warning("retention cleanup failed: %s", error)
            self._housekeeping_stop.wait(6 * 60 * 60)
