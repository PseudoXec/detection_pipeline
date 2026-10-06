"""SQLite buffer for the vehicle mode: table `detections` (what the C# dashboard reads)."""
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from api.result import SendResult
from storage.buffer import BufferedStore

log = logging.getLogger("pipeline")

_MAX_SEND_RETRIES = 5
_RETRY_BATCH_ROWS = 50


@dataclass
class DetectionRecord:
    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    vehicle_image_jpeg: bytes
    vehicle_box_x1: float
    vehicle_box_y1: float
    vehicle_box_x2: float
    vehicle_box_y2: float
    plate_detected: bool
    plate_confidence: Optional[float]
    plate_image_jpeg: Optional[bytes]
    plate_box_x1: Optional[float]
    plate_box_y1: Optional[float]
    plate_box_x2: Optional[float]
    plate_box_y2: Optional[float]
    detected_at: datetime
    vehicle_detect_ms: Optional[float]
    vehicle_crop_ms: Optional[float]
    plate_detect_ms: Optional[float]
    plate_crop_ms: Optional[float]
    total_pipeline_ms: Optional[float]
    ocr_process: bool
    ocr_read: str
    disk_files: Optional[Dict[str, bytes]] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id            TEXT NOT NULL,
    camera_source       TEXT NOT NULL,
    vehicle_class       TEXT NOT NULL,
    vehicle_confidence  REAL NOT NULL,
    vehicle_image       BLOB NOT NULL,
    vehicle_box_x1      REAL NOT NULL,
    vehicle_box_y1      REAL NOT NULL,
    vehicle_box_x2      REAL NOT NULL,
    vehicle_box_y2      REAL NOT NULL,
    plate_detected      INTEGER NOT NULL,
    plate_confidence    REAL,
    plate_image         BLOB,
    plate_box_x1        REAL,
    plate_box_y1        REAL,
    plate_box_x2        REAL,
    plate_box_y2        REAL,
    detected_at         TEXT NOT NULL,
    vehicle_detect_ms   REAL,
    vehicle_crop_ms     REAL,
    plate_detect_ms     REAL,
    plate_crop_ms       REAL,
    total_pipeline_ms   REAL,
    ocr_process          INTEGER NOT NULL DEFAULT 0,
    ocr_read             TEXT,
    camera_name          TEXT,
    camera_ip            TEXT,
    camera_location      TEXT,
    synced              INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections (synced);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections (detected_at);
"""


class DetectionStorage(BufferedStore):
    TABLE = "detections"
    SCHEMA = _SCHEMA

    def __init__(
        self,
        database_path: str,
        batch_size: int = 8,
        flush_interval_seconds: float = 1.0,
        send_via_api: bool = False,
        delete_row_after_api_send: bool = True,
        api_endpoint_url: Optional[str] = None,
        api_timeout_seconds: float = 5.0,
        output_dir: Optional[str] = None,
        delete_disk_images_after_send: bool = True,
        send_retry_seconds: float = 60.0,
        camera_name: Optional[str] = None,
        camera_ip: Optional[str] = None,
        camera_location: Optional[str] = None,
    ):
        self.delete_row_after_api_send = delete_row_after_api_send
        self.api_endpoint_url = api_endpoint_url
        self.api_timeout_seconds = api_timeout_seconds
        self._retry_failures: Dict[int, int] = {}
        super().__init__(
            database_path, batch_size, flush_interval_seconds, send_via_api, send_retry_seconds,
            output_dir, delete_disk_images_after_send, camera_name, camera_ip, camera_location,
        )

    def migrate(self, connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(detections)")}
        if "vehicle_detect_ms" not in existing:
            connection.execute("ALTER TABLE detections ADD COLUMN vehicle_detect_ms REAL")
        if "ocr_process" not in existing:
            connection.execute("ALTER TABLE detections ADD COLUMN ocr_process INTEGER NOT NULL DEFAULT 0")
        if "ocr_read" not in existing:
            connection.execute("ALTER TABLE detections ADD COLUMN ocr_read TEXT")
        for column in ("camera_name", "camera_ip", "camera_location"):
            if column not in existing:
                connection.execute(f"ALTER TABLE detections ADD COLUMN {column} TEXT")

    def insert_records(self, connection: sqlite3.Connection, records: List[DetectionRecord]) -> List[int]:
        inserted_ids: List[int] = []
        with connection:
            cursor = connection.cursor()
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO detections (
                        track_id, camera_source, vehicle_class, vehicle_confidence,
                        vehicle_image, vehicle_box_x1, vehicle_box_y1, vehicle_box_x2, vehicle_box_y2,
                        plate_detected, plate_confidence, plate_image,
                        plate_box_x1, plate_box_y1, plate_box_x2, plate_box_y2,
                        detected_at, vehicle_detect_ms, vehicle_crop_ms, plate_detect_ms, plate_crop_ms,
                        total_pipeline_ms, ocr_process, ocr_read,
                        camera_name, camera_ip, camera_location
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.track_id, record.camera_source, record.vehicle_class,
                        record.vehicle_confidence, record.vehicle_image_jpeg,
                        record.vehicle_box_x1, record.vehicle_box_y1, record.vehicle_box_x2, record.vehicle_box_y2,
                        1 if record.plate_detected else 0, record.plate_confidence, record.plate_image_jpeg,
                        record.plate_box_x1, record.plate_box_y1, record.plate_box_x2, record.plate_box_y2,
                        record.detected_at.isoformat(sep=" ", timespec="seconds"),
                        record.vehicle_detect_ms, record.vehicle_crop_ms,
                        record.plate_detect_ms, record.plate_crop_ms, record.total_pipeline_ms,
                        1 if record.ocr_process else 0, record.ocr_read,
                        self.camera_name, self.camera_ip, self.camera_location,
                    ),
                )
                inserted_ids.append(cursor.lastrowid)
        return inserted_ids

    def send_new(self, connection: sqlite3.Connection, records: List[DetectionRecord], row_ids: List[int]) -> None:
        from api import vehicle_client

        sent_ids: List[int] = []
        skipped_ids: List[int] = []
        for record, row_id in zip(records, row_ids):
            result = vehicle_client.send_detection(
                record, self.api_endpoint_url, self.api_timeout_seconds,
                camera_name=self.camera_name, camera_ip=self.camera_ip, camera_location=self.camera_location,
            )
            if result == SendResult.SENT:
                sent_ids.append(row_id)
                if self.delete_disk_images_after_send:
                    self._delete_disk_files(record)
            elif result == SendResult.SKIPPED:
                skipped_ids.append(row_id)
        self.finish_rows(connection, sent_ids, skipped_ids, self.delete_row_after_api_send)

    def retry_unsent(self, connection: sqlite3.Connection) -> None:
        if not self.api_endpoint_url:
            return
        from api import vehicle_client

        leftovers = [row[0] for row in connection.execute(
            "SELECT id FROM detections WHERE synced = 0 AND plate_detected = 0")]
        self.finish_rows(connection, [], leftovers, self.delete_row_after_api_send)

        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM detections WHERE synced = 0 AND plate_detected = 1 ORDER BY id LIMIT ?",
                (_RETRY_BATCH_ROWS,),
            ).fetchall()
        finally:
            connection.row_factory = None

        sent_ids: List[int] = []
        for row in rows:
            row_id = row["id"]
            if self._retry_failures.get(row_id, 0) >= _MAX_SEND_RETRIES:
                continue
            result = vehicle_client.send_detection(
                self._row_to_record(row), self.api_endpoint_url, self.api_timeout_seconds,
                camera_name=row["camera_name"] or self.camera_name,
                camera_ip=row["camera_ip"] or self.camera_ip,
                camera_location=row["camera_location"] or self.camera_location,
            )
            if result == SendResult.FAILED:
                self._retry_failures[row_id] = self._retry_failures.get(row_id, 0) + 1
                break
            sent_ids.append(row_id)
            self._retry_failures.pop(row_id, None)
        self.finish_rows(connection, sent_ids, [], self.delete_row_after_api_send)
        if sent_ids:
            log.info("[storage] re-sent %d earlier failed record(s)", len(sent_ids))

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DetectionRecord:
        return DetectionRecord(
            track_id=row["track_id"], camera_source=row["camera_source"],
            vehicle_class=row["vehicle_class"], vehicle_confidence=row["vehicle_confidence"],
            vehicle_image_jpeg=None,
            vehicle_box_x1=row["vehicle_box_x1"], vehicle_box_y1=row["vehicle_box_y1"],
            vehicle_box_x2=row["vehicle_box_x2"], vehicle_box_y2=row["vehicle_box_y2"],
            plate_detected=bool(row["plate_detected"]), plate_confidence=row["plate_confidence"],
            plate_image_jpeg=row["plate_image"],
            plate_box_x1=row["plate_box_x1"], plate_box_y1=row["plate_box_y1"],
            plate_box_x2=row["plate_box_x2"], plate_box_y2=row["plate_box_y2"],
            detected_at=datetime.fromisoformat(row["detected_at"]),
            vehicle_detect_ms=row["vehicle_detect_ms"], vehicle_crop_ms=row["vehicle_crop_ms"],
            plate_detect_ms=row["plate_detect_ms"], plate_crop_ms=row["plate_crop_ms"],
            total_pipeline_ms=row["total_pipeline_ms"],
            ocr_process=bool(row["ocr_process"]), ocr_read=row["ocr_read"] or "Unrecognized",
        )
