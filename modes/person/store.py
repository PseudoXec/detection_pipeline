"""SQLite buffer for the person mode: table `person_detections`.

One row per person TRACK (not per frame): the best person crop, the best face crop when a face was
found, and only what the command center needs (landmarks, face quality). The sync part sends rows to
the dashboard API and records how that went.

Coordinates:
* face_landmarks     pixels inside the stored PERSON crop (person_image).

Times are local wall-clock text "YYYY-MM-DD HH:MM:SS" like the vehicle table, so retention and the
dashboard treat both tables the same way.

synced: 0 = waiting, 1 = delivered or nothing to deliver, 2 = rejected by the API (kept for inspection).
"""
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from api.result import SendResult
from storage.buffer import BufferedStore

log = logging.getLogger("pipeline")

_RETRY_BATCH_ROWS = 50
_MAX_BACKOFF_SECONDS = 3600.0


@dataclass
class PersonRecord:
    event_uuid: str
    device_id: str
    session_id: str
    track_id: str
    person_confidence: float
    person_image_jpeg: bytes
    face_detected: bool
    detected_at: datetime
    face_image_jpeg: Optional[bytes] = None
    face_landmarks: Optional[str] = None          # JSON: [[x, y] x 5], pixels inside person_image
    face_quality_score: Optional[float] = None
    disk_files: Optional[Dict[str, bytes]] = None


_SCHEMA = """
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


def _ts(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(sep=" ", timespec="seconds") if value is not None else None


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


class PersonStorage(BufferedStore):
    TABLE = "person_detections"
    SCHEMA = _SCHEMA

    def __init__(
        self,
        database_path: str,
        batch_size: int = 8,
        flush_interval_seconds: float = 1.0,
        send_via_api: bool = False,
        delete_row_after_api_send: bool = True,
        send_rows_without_face: bool = True,
        api_endpoint_url: Optional[str] = None,
        api_timeout_seconds: float = 5.0,
        output_dir: Optional[str] = None,
        delete_disk_images_after_send: bool = True,
        send_retry_seconds: float = 60.0,
        camera_name: Optional[str] = None,
        camera_ip: Optional[str] = None,
        camera_location: Optional[str] = None,
        camera_id: Optional[int] = None,
    ):
        self.camera_id = camera_id                   # the command center's id for this camera (api.camera_id)
        self.delete_row_after_api_send = delete_row_after_api_send
        self.send_rows_without_face = send_rows_without_face
        self.api_endpoint_url = api_endpoint_url
        self.api_timeout_seconds = api_timeout_seconds
        super().__init__(
            database_path, batch_size, flush_interval_seconds, send_via_api, send_retry_seconds,
            output_dir, delete_disk_images_after_send, camera_name, camera_ip, camera_location,
        )

    def can_send(self) -> bool:
        return bool(self.send_via_api and self.api_endpoint_url)

    # ------------------------------------------------------------------ writing
    def insert_records(self, connection: sqlite3.Connection, records: List[PersonRecord]) -> List[int]:
        ids: List[int] = []
        with connection:
            cursor = connection.cursor()
            for r in records:
                cursor.execute(
                    """
                    INSERT INTO person_detections (
                        event_uuid, device_id, session_id, track_id,
                        camera_name, camera_ip, camera_location,
                        person_confidence, person_image,
                        face_detected, face_image, face_landmarks, face_quality_score,
                        detected_at
                    ) VALUES (?,?,?,?, ?,?,?, ?,?, ?,?,?,?, ?)
                    """,
                    (
                        r.event_uuid, r.device_id, r.session_id, r.track_id,
                        self.camera_name, self.camera_ip, self.camera_location,
                        r.person_confidence, r.person_image_jpeg,
                        1 if r.face_detected else 0, r.face_image_jpeg, r.face_landmarks, r.face_quality_score,
                        _ts(r.detected_at),
                    ),
                )
                ids.append(cursor.lastrowid)
        return ids

    # ------------------------------------------------------------------ sending
    def send_new(self, connection: sqlite3.Connection, records: List[PersonRecord], row_ids: List[int]) -> None:
        sent: List[int] = []
        skipped: List[int] = []
        for record, row_id in zip(records, row_ids):
            outcome = self._send_one(record)
            self._apply_outcome(connection, row_id, outcome, sent, skipped)
            if outcome[0] == SendResult.SENT and self.delete_disk_images_after_send:
                self._delete_disk_files(record)
        self.finish_rows(connection, sent, skipped, self.delete_row_after_api_send)

    def retry_unsent(self, connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        try:
            # a row that keeps failing waits longer each time (send_retry_seconds x 2, x 4, ... up to an hour)
            # but is NEVER given up on: an API outage of any length must not strand events
            rows = connection.execute(
                "SELECT * FROM person_detections WHERE synced = 0 AND (last_sync_attempt_at IS NULL OR "
                "(julianday('now', 'localtime') - julianday(last_sync_attempt_at)) * 86400.0 >= "
                "MIN(?, ? * (1 << MIN(sync_attempts, 6)))) ORDER BY id LIMIT ?",
                (_MAX_BACKOFF_SECONDS, self.send_retry_seconds, _RETRY_BATCH_ROWS),
            ).fetchall()
        finally:
            connection.row_factory = None

        sent: List[int] = []
        skipped: List[int] = []
        for row in rows:
            outcome = self._send_one(self._row_to_record(row))
            self._apply_outcome(connection, row["id"], outcome, sent, skipped)
            if outcome[0] == SendResult.FAILED and (outcome[1] or "").startswith("network"):
                break                                  # the server cannot be reached - try again next pass
        self.finish_rows(connection, sent, skipped, self.delete_row_after_api_send)
        if sent:
            log.info("[storage:person] re-sent %d earlier failed record(s)", len(sent))

    def _send_one(self, record: PersonRecord):
        from api import person_client

        if not self.send_rows_without_face and not record.face_detected:
            return SendResult.SKIPPED, None
        return person_client.send_person(
            record, self.api_endpoint_url, self.api_timeout_seconds,
            camera_id=self.camera_id, camera_name=self.camera_name, camera_ip=self.camera_ip,
            camera_location=self.camera_location,
        )

    def _apply_outcome(self, connection: sqlite3.Connection, row_id: int, outcome, sent: List[int], skipped: List[int]) -> None:
        result, detail = outcome
        if result == SendResult.SENT:
            sent.append(row_id)
            with connection:
                connection.execute("UPDATE person_detections SET synced_at = ?, sync_error = NULL WHERE id = ?", (_now(), row_id))
        elif result == SendResult.SKIPPED:
            skipped.append(row_id)
        else:
            new_state = 2 if result == SendResult.REJECTED else 0
            with connection:
                connection.execute(
                    "UPDATE person_detections SET sync_attempts = sync_attempts + 1, last_sync_attempt_at = ?, "
                    "sync_error = ?, synced = ? WHERE id = ?",
                    (_now(), (detail or "")[:300], new_state, row_id),
                )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PersonRecord:
        def when(value):
            return datetime.fromisoformat(value) if value else None
        return PersonRecord(
            event_uuid=row["event_uuid"], device_id=row["device_id"], session_id=row["session_id"],
            track_id=row["track_id"],
            person_confidence=row["person_confidence"], person_image_jpeg=row["person_image"],
            face_detected=bool(row["face_detected"]), detected_at=when(row["detected_at"]),
            face_image_jpeg=row["face_image"], face_landmarks=row["face_landmarks"],
            face_quality_score=row["face_quality_score"],
        )