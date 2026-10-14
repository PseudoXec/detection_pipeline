"""Unified SQLite buffer for both vehicle and person detection flows.

Both flows share this exact same buffer:
- Background writer thread that batches inserts (frame loop never blocks on disk or network)
- WAL mode for concurrent reading/writing
- Optional JPEG copies written to disk for spot checking
- Unified API sender integration for sending detections to command center
- Automatic retry of unsent records with backoff
- Retention cleanup (sweeping synced/stale rows and output directory)
"""
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from api.result import SendResult

log = logging.getLogger("pipeline")

_MAX_SEND_RETRIES = 5
_RETRY_BATCH_ROWS = 50
_MAX_BACKOFF_SECONDS = 3600.0


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode an OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    return buffer.tobytes()


@dataclass
class VehicleRecord:
    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    vehicle_image_jpeg: Optional[bytes] = None
    vehicle_box_x1: Optional[float] = None
    vehicle_box_y1: Optional[float] = None
    vehicle_box_x2: Optional[float] = None
    vehicle_box_y2: Optional[float] = None
    plate_detected: bool = False
    plate_confidence: Optional[float] = None
    plate_image_jpeg: Optional[bytes] = None
    plate_box_x1: Optional[float] = None
    plate_box_y1: Optional[float] = None
    plate_box_x2: Optional[float] = None
    plate_box_y2: Optional[float] = None
    detected_at: datetime = field(default_factory=datetime.now)
    disk_files: Optional[Dict[str, bytes]] = None


@dataclass
class PersonRecord:
    event_uuid: str
    device_id: str
    session_id: str
    track_id: str
    person_confidence: float
    person_image_jpeg: bytes
    face_detected: bool
    detected_at: datetime = field(default_factory=datetime.now)
    face_image_jpeg: Optional[bytes] = None
    face_landmarks: Optional[str] = None          # JSON: [[x, y] x 5]
    face_quality_score: Optional[float] = None
    disk_files: Optional[Dict[str, bytes]] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id            TEXT NOT NULL,
    camera_source       TEXT NOT NULL,
    vehicle_class       TEXT NOT NULL,
    vehicle_confidence  REAL NOT NULL,
    vehicle_image       BLOB,
    vehicle_box_x1      REAL,
    vehicle_box_y1      REAL,
    vehicle_box_x2      REAL,
    vehicle_box_y2      REAL,
    plate_detected      INTEGER NOT NULL,
    plate_confidence    REAL,
    plate_image         BLOB,
    plate_box_x1        REAL,
    plate_box_y1        REAL,
    plate_box_x2        REAL,
    plate_box_y2        REAL,
    detected_at         TEXT NOT NULL,
    camera_name         TEXT,
    camera_ip           TEXT,
    camera_location     TEXT,
    synced              INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections (synced);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections (detected_at);

CREATE TABLE IF NOT EXISTS person_detections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid           TEXT NOT NULL UNIQUE,
    device_id            TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    track_id             TEXT NOT NULL,
    camera_name          TEXT,
    camera_ip            TEXT,
    camera_location      TEXT,
    person_confidence    REAL NOT NULL,
    person_image         BLOB NOT NULL,
    face_detected        INTEGER NOT NULL CHECK (face_detected IN (0, 1)),
    face_image           BLOB,
    face_landmarks       TEXT,
    face_quality_score   REAL,
    detected_at          TEXT NOT NULL,
    synced               INTEGER NOT NULL DEFAULT 0,
    sync_attempts        INTEGER NOT NULL DEFAULT 0,
    last_sync_attempt_at TEXT,
    synced_at            TEXT,
    sync_error           TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_person_synced ON person_detections (synced, id);
CREATE INDEX IF NOT EXISTS idx_person_detected_at ON person_detections (detected_at);
"""


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _ts(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(sep=" ", timespec="seconds") if value is not None else None


class SQLiteBuffer:
    """Unified SQLite buffer used by both vehicle and person flows."""

    def __init__(
        self,
        config,
        camera_info=None,
        api_sender=None,
    ):
        self.config = config
        self.storage_cfg = config.storage
        self.database_path = self.storage_cfg.database_path
        self.output_dir = self.storage_cfg.output_dir
        self.batch_size = getattr(self.storage_cfg, "write_batch_size", 8)
        self.flush_interval_seconds = getattr(self.storage_cfg, "write_flush_interval_seconds", 1.0)
        self.send_retry_seconds = self.storage_cfg.send_retry_seconds
        self.delete_disk_images_after_send = self.storage_cfg.delete_disk_images_after_send
        self.retention_days = self.storage_cfg.retention_days
        self.max_unsynced_days = self.storage_cfg.max_unsynced_days

        self.camera_info = camera_info
        self.api_sender = api_sender

        self.camera_name = getattr(camera_info, "name", None) if camera_info else None
        self.camera_ip = getattr(camera_info, "ip_address", None) if camera_info else None
        self.camera_location = getattr(camera_info, "location", None) if camera_info else None

        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)) or ".", exist_ok=True)

        self._queue: "queue.Queue[Union[VehicleRecord, PersonRecord]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # retry tracker for vehicle rows
        self._vehicle_retry_failures: Dict[int, int] = {}

        connection = self._connect()
        connection.executescript(_SCHEMA)
        connection.commit()
        connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def start(self) -> "SQLiteBuffer":
        self._thread = threading.Thread(target=self._writer_loop, name="sqlite-buffer", daemon=True)
        self._thread.start()
        log.info("[storage] unified SQLite buffer started (%s)", self.database_path)
        return self

    def enqueue(self, record: Union[VehicleRecord, PersonRecord]) -> None:
        self._queue.put(record)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        log.info("[storage] unified SQLite buffer stopped")

    def _writer_loop(self) -> None:
        connection = self._connect()
        last_flush = time.time()
        next_retry = time.time() + self.send_retry_seconds
        last_retention = time.time()
        pending: List[Union[VehicleRecord, PersonRecord]] = []

        while not self._stop_event.is_set() or not self._queue.empty() or pending:
            try:
                pending.append(self._queue.get(timeout=0.25))
            except queue.Empty:
                pass

            should_flush = len(pending) >= self.batch_size or (
                pending and (time.time() - last_flush) >= self.flush_interval_seconds
            )
            if should_flush:
                self._write_batch(connection, pending)
                pending = []
                last_flush = time.time()

            now = time.time()
            if not self._stop_event.is_set() and self._queue.empty() and not pending and now >= next_retry:
                try:
                    self._retry_unsent(connection)
                except Exception as error:
                    log.warning("[storage] retry pass failed: %s", error)
                next_retry = now + self.send_retry_seconds

            if not self._stop_event.is_set() and now - last_retention >= 6 * 3600:
                try:
                    self._run_retention(connection)
                except Exception as error:
                    log.warning("[storage] retention cleanup failed: %s", error)
                last_retention = now

        connection.close()

    def _write_batch(self, connection: sqlite3.Connection, records: List[Union[VehicleRecord, PersonRecord]]) -> None:
        for record in records:
            self._write_disk_files(record)

        v_records: List[Tuple[VehicleRecord, int]] = []
        p_records: List[Tuple[PersonRecord, int]] = []

        with connection:
            cursor = connection.cursor()
            for record in records:
                if isinstance(record, VehicleRecord):
                    cursor.execute(
                        """
                        INSERT INTO detections (
                            track_id, camera_source, vehicle_class, vehicle_confidence,
                            vehicle_image, vehicle_box_x1, vehicle_box_y1, vehicle_box_x2, vehicle_box_y2,
                            plate_detected, plate_confidence, plate_image,
                            plate_box_x1, plate_box_y1, plate_box_x2, plate_box_y2,
                            detected_at, camera_name, camera_ip, camera_location
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.track_id, record.camera_source, record.vehicle_class,
                            record.vehicle_confidence, record.vehicle_image_jpeg,
                            record.vehicle_box_x1, record.vehicle_box_y1, record.vehicle_box_x2, record.vehicle_box_y2,
                            1 if record.plate_detected else 0, record.plate_confidence, record.plate_image_jpeg,
                            record.plate_box_x1, record.plate_box_y1, record.plate_box_x2, record.plate_box_y2,
                            _ts(record.detected_at), self.camera_name, self.camera_ip, self.camera_location,
                        ),
                    )
                    v_records.append((record, cursor.lastrowid))
                elif isinstance(record, PersonRecord):
                    cursor.execute(
                        """
                        INSERT INTO person_detections (
                            event_uuid, device_id, session_id, track_id,
                            camera_name, camera_ip, camera_location,
                            person_confidence, person_image,
                            face_detected, face_image, face_landmarks, face_quality_score,
                            detected_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.event_uuid, record.device_id, record.session_id, record.track_id,
                            self.camera_name, self.camera_ip, self.camera_location,
                            record.person_confidence, record.person_image_jpeg,
                            1 if record.face_detected else 0, record.face_image_jpeg,
                            record.face_landmarks, record.face_quality_score,
                            _ts(record.detected_at),
                        ),
                    )
                    p_records.append((record, cursor.lastrowid))

        if self.api_sender is not None and self.config.switches.api:
            if v_records and self.config.vehicle.features.send_via_api:
                self._send_new_vehicles(connection, v_records)
            if p_records and self.config.person.send_via_api:
                self._send_new_people(connection, p_records)

    def _send_new_vehicles(self, connection: sqlite3.Connection, records_with_ids: List[Tuple[VehicleRecord, int]]) -> None:
        delete_sent = self.config.vehicle.features.delete_row_after_api_send
        sent_ids: List[int] = []
        skipped_ids: List[int] = []

        for record, row_id in records_with_ids:
            result = self.api_sender.send_vehicle(record)
            if result == SendResult.SENT:
                sent_ids.append(row_id)
                if self.delete_disk_images_after_send:
                    self._delete_disk_files(record)
            elif result == SendResult.SKIPPED:
                skipped_ids.append(row_id)

        self._finish_rows(connection, "detections", sent_ids, skipped_ids, delete_sent)

    def _send_new_people(self, connection: sqlite3.Connection, records_with_ids: List[Tuple[PersonRecord, int]]) -> None:
        delete_sent = self.config.person.delete_row_after_api_send
        sent_ids: List[int] = []
        skipped_ids: List[int] = []

        for record, row_id in records_with_ids:
            if not self.config.person.send_rows_without_face and not record.face_detected:
                skipped_ids.append(row_id)
                continue
            outcome = self.api_sender.send_person(record)
            result, detail = outcome
            if result == SendResult.SENT:
                sent_ids.append(row_id)
                if self.delete_disk_images_after_send:
                    self._delete_disk_files(record)
                with connection:
                    connection.execute("UPDATE person_detections SET synced_at = ?, sync_error = NULL WHERE id = ?",
                                       (_now(), row_id))
            elif result == SendResult.SKIPPED:
                skipped_ids.append(row_id)
            else:
                new_state = 2 if result == SendResult.REJECTED else 0
                with connection:
                    connection.execute(
                        "UPDATE person_detections SET sync_attempts = sync_attempts + 1, last_sync_attempt_at = ?, "
                        "sync_error = ?, synced = ? WHERE id = ?",
                        (_now(), (detail or "")[:300], new_state, row_id),
                    )

        self._finish_rows(connection, "person_detections", sent_ids, skipped_ids, delete_sent)

    def _retry_unsent(self, connection: sqlite3.Connection) -> None:
        if self.api_sender is None or not self.config.switches.api:
            return

        # Retry vehicle rows
        if self.config.vehicle.features.send_via_api and self.config.api.vehicle_endpoint_url:
            leftovers = [r[0] for r in connection.execute("SELECT id FROM detections WHERE synced = 0 AND plate_detected = 0")]
            self._finish_rows(connection, "detections", [], leftovers, self.config.vehicle.features.delete_row_after_api_send)

            connection.row_factory = sqlite3.Row
            try:
                v_rows = connection.execute(
                    "SELECT * FROM detections WHERE synced = 0 AND plate_detected = 1 ORDER BY id LIMIT ?",
                    (_RETRY_BATCH_ROWS,),
                ).fetchall()
            finally:
                connection.row_factory = None

            sent_v_ids: List[int] = []
            for row in v_rows:
                row_id = row["id"]
                if self._vehicle_retry_failures.get(row_id, 0) >= _MAX_SEND_RETRIES:
                    continue
                record = VehicleRecord(
                    track_id=row["track_id"], camera_source=row["camera_source"],
                    vehicle_class=row["vehicle_class"], vehicle_confidence=row["vehicle_confidence"],
                    plate_detected=bool(row["plate_detected"]), plate_confidence=row["plate_confidence"],
                    plate_image_jpeg=row["plate_image"],
                    plate_box_x1=row["plate_box_x1"], plate_box_y1=row["plate_box_y1"],
                    plate_box_x2=row["plate_box_x2"], plate_box_y2=row["plate_box_y2"],
                    detected_at=datetime.fromisoformat(row["detected_at"]),
                )
                res = self.api_sender.send_vehicle(record)
                if res == SendResult.FAILED:
                    self._vehicle_retry_failures[row_id] = self._vehicle_retry_failures.get(row_id, 0) + 1
                    break
                sent_v_ids.append(row_id)
                self._vehicle_retry_failures.pop(row_id, None)

            self._finish_rows(connection, "detections", sent_v_ids, [], self.config.vehicle.features.delete_row_after_api_send)
            if sent_v_ids:
                log.info("[storage] re-sent %d earlier failed vehicle record(s)", len(sent_v_ids))

        # Retry person rows
        if self.config.person.send_via_api and self.config.api.person_endpoint_url:
            connection.row_factory = sqlite3.Row
            try:
                p_rows = connection.execute(
                    "SELECT * FROM person_detections WHERE synced = 0 AND (last_sync_attempt_at IS NULL OR "
                    "(julianday('now', 'localtime') - julianday(last_sync_attempt_at)) * 86400.0 >= "
                    "MIN(?, ? * (1 << MIN(sync_attempts, 6)))) ORDER BY id LIMIT ?",
                    (_MAX_BACKOFF_SECONDS, self.send_retry_seconds, _RETRY_BATCH_ROWS),
                ).fetchall()
            finally:
                connection.row_factory = None

            sent_p_ids: List[int] = []
            skipped_p_ids: List[int] = []
            for row in p_rows:
                record = PersonRecord(
                    event_uuid=row["event_uuid"], device_id=row["device_id"], session_id=row["session_id"],
                    track_id=row["track_id"], person_confidence=row["person_confidence"],
                    person_image_jpeg=row["person_image"], face_detected=bool(row["face_detected"]),
                    detected_at=datetime.fromisoformat(row["detected_at"]),
                    face_image_jpeg=row["face_image"], face_landmarks=row["face_landmarks"],
                    face_quality_score=row["face_quality_score"],
                )
                if not self.config.person.send_rows_without_face and not record.face_detected:
                    skipped_p_ids.append(row["id"])
                    continue
                outcome = self.api_sender.send_person(record)
                result, detail = outcome
                if result == SendResult.SENT:
                    sent_p_ids.append(row["id"])
                    with connection:
                        connection.execute("UPDATE person_detections SET synced_at = ?, sync_error = NULL WHERE id = ?",
                                           (_now(), row["id"]))
                elif result == SendResult.SKIPPED:
                    skipped_p_ids.append(row["id"])
                else:
                    new_state = 2 if result == SendResult.REJECTED else 0
                    with connection:
                        connection.execute(
                            "UPDATE person_detections SET sync_attempts = sync_attempts + 1, last_sync_attempt_at = ?, "
                            "sync_error = ?, synced = ? WHERE id = ?",
                            (_now(), (detail or "")[:300], new_state, row["id"]),
                        )
                    if (detail or "").startswith("network"):
                        break

            self._finish_rows(connection, "person_detections", sent_p_ids, skipped_p_ids, self.config.person.delete_row_after_api_send)
            if sent_p_ids:
                log.info("[storage] re-sent %d earlier failed person record(s)", len(sent_p_ids))

    def _finish_rows(self, connection: sqlite3.Connection, table: str, sent_ids: List[int],
                     skipped_ids: List[int], delete_sent: bool) -> None:
        if not (sent_ids or skipped_ids):
            return
        with connection:
            if sent_ids:
                if delete_sent:
                    connection.executemany(f"DELETE FROM {table} WHERE id = ?", [(r,) for r in sent_ids])
                else:
                    connection.executemany(f"UPDATE {table} SET synced = 1 WHERE id = ?", [(r,) for r in sent_ids])
            if skipped_ids:
                connection.executemany(f"UPDATE {table} SET synced = 1 WHERE id = ?", [(r,) for r in skipped_ids])

    def _write_disk_files(self, record: Any) -> None:
        files: Optional[Dict[str, bytes]] = getattr(record, "disk_files", None)
        if not self.output_dir or not files:
            return
        for relative_path, data in files.items():
            try:
                path = os.path.join(self.output_dir, relative_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as error:
                log.warning("[storage] could not write %s: %s", relative_path, error)

    def _delete_disk_files(self, record: Any) -> None:
        files: Optional[Dict[str, bytes]] = getattr(record, "disk_files", None)
        if not self.output_dir or not files:
            return
        for relative_path in files:
            try:
                os.remove(os.path.join(self.output_dir, relative_path))
            except OSError:
                pass

    def _run_retention(self, connection: sqlite3.Connection) -> None:
        if self.retention_days <= 0 and self.max_unsynced_days <= 0:
            return
        synced_deleted = 0
        unsynced_deleted = 0

        for table in ("detections", "person_detections"):
            if self.retention_days > 0:
                cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat(sep=" ", timespec="seconds")
                with connection:
                    cur = connection.execute(f"DELETE FROM {table} WHERE synced = 1 AND detected_at < ?", (cutoff,))
                    synced_deleted += cur.rowcount
            if self.max_unsynced_days > 0:
                cutoff = (datetime.now() - timedelta(days=self.max_unsynced_days)).isoformat(sep=" ", timespec="seconds")
                with connection:
                    cur = connection.execute(f"DELETE FROM {table} WHERE synced = 0 AND detected_at < ?", (cutoff,))
                    unsynced_deleted += cur.rowcount

        files_deleted = 0
        if self.retention_days > 0 and self.output_dir and os.path.isdir(self.output_dir):
            cutoff_ts = time.time() - self.retention_days * 86400
            for folder, _dirs, files in os.walk(self.output_dir):
                for name in files:
                    path = os.path.join(folder, name)
                    try:
                        if os.path.getmtime(path) < cutoff_ts:
                            os.remove(path)
                            files_deleted += 1
                    except OSError:
                        pass

        if synced_deleted or unsynced_deleted or files_deleted:
            log.info("[storage] retention cleanup: removed %d synced, %d unsynced row(s), %d file(s)",
                     synced_deleted, unsynced_deleted, files_deleted)

    def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "database_path": self.database_path,
            "queue_depth": self._queue.qsize(),
            "writer_alive": bool(self._thread and self._thread.is_alive()),
        }
        connection = self._connect()
        try:
            for table in ("detections", "person_detections"):
                rows = connection.execute(f"SELECT synced, COUNT(*) FROM {table} GROUP BY synced").fetchall()
                counts = {int(k): int(v) for k, v in rows}
                info[table] = {
                    "total": sum(counts.values()),
                    "waiting": counts.get(0, 0),
                    "delivered": counts.get(1, 0),
                    "rejected": counts.get(2, 0),
                }
        except sqlite3.Error as error:
            info["error"] = str(error)
        finally:
            connection.close()
        return info
