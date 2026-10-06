"""SQLite buffer for the person mode: table `person_detections`.

One row per person TRACK (not per frame): the best person crop, the best face crop when a face was
found, and everything needed to judge or redo the work later (boxes, landmarks, quality, timings,
model versions). The sync part sends rows to the dashboard API and records how that went.

Coordinates:
* person_box_*       pixels in the FULL camera frame (frame_width x frame_height).
* face_box_*, face_landmarks   pixels inside the stored PERSON crop (person_image) - the same
  convention the vehicle table uses for plate boxes, so a viewer can draw them straight onto the image.

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
    camera_source: str
    person_confidence: float
    person_image_jpeg: bytes
    person_box_x1: float
    person_box_y1: float
    person_box_x2: float
    person_box_y2: float
    face_detected: bool
    detected_at: datetime
    track_first_seen_at: Optional[datetime] = None
    track_last_seen_at: Optional[datetime] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    face_confidence: Optional[float] = None
    face_image_jpeg: Optional[bytes] = None
    face_box_x1: Optional[float] = None
    face_box_y1: Optional[float] = None
    face_box_x2: Optional[float] = None
    face_box_y2: Optional[float] = None
    face_landmarks: Optional[str] = None          # JSON: [[x, y] x 5], pixels inside person_image
    face_sharpness: Optional[float] = None
    face_quality_score: Optional[float] = None
    face_attempts: Optional[int] = None
    image_format: str = "jpeg"
    person_detect_ms: Optional[float] = None
    person_crop_ms: Optional[float] = None
    face_detect_ms: Optional[float] = None
    face_crop_ms: Optional[float] = None
    total_pipeline_ms: Optional[float] = None
    pipeline_version: Optional[str] = None
    person_model: Optional[str] = None
    face_model: Optional[str] = None
    cpu_temp_c: Optional[float] = None
    disk_files: Optional[Dict[str, bytes]] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS person_detections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid           TEXT NOT NULL UNIQUE,
    device_id            TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    track_id             TEXT NOT NULL,
    track_first_seen_at  TEXT,
    track_last_seen_at   TEXT,

    camera_source        TEXT NOT NULL,
    camera_name          TEXT,
    camera_ip            TEXT,
    camera_location      TEXT,
    frame_width          INTEGER,
    frame_height         INTEGER,

    person_confidence    REAL NOT NULL,
    person_image         BLOB NOT NULL,
    person_box_x1        REAL NOT NULL,
    person_box_y1        REAL NOT NULL,
    person_box_x2        REAL NOT NULL,
    person_box_y2        REAL NOT NULL,

    face_detected        INTEGER NOT NULL CHECK (face_detected IN (0, 1)),
    face_confidence      REAL,
    face_image           BLOB,
    face_box_x1          REAL,
    face_box_y1          REAL,
    face_box_x2          REAL,
    face_box_y2          REAL,
    face_landmarks       TEXT,
    face_sharpness       REAL,
    face_quality_score   REAL,
    face_attempts        INTEGER,
    image_format         TEXT NOT NULL DEFAULT 'jpeg',

    detected_at          TEXT NOT NULL,
    person_detect_ms     REAL,
    person_crop_ms       REAL,
    face_detect_ms       REAL,
    face_crop_ms         REAL,
    total_pipeline_ms    REAL,

    pipeline_version     TEXT,
    person_model         TEXT,
    face_model           TEXT,
    cpu_temp_c           REAL,

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
                        event_uuid, device_id, session_id, track_id, track_first_seen_at, track_last_seen_at,
                        camera_source, camera_name, camera_ip, camera_location, frame_width, frame_height,
                        person_confidence, person_image,
                        person_box_x1, person_box_y1, person_box_x2, person_box_y2,
                        face_detected, face_confidence, face_image,
                        face_box_x1, face_box_y1, face_box_x2, face_box_y2,
                        face_landmarks, face_sharpness, face_quality_score, face_attempts, image_format,
                        detected_at, person_detect_ms, person_crop_ms, face_detect_ms, face_crop_ms, total_pipeline_ms,
                        pipeline_version, person_model, face_model, cpu_temp_c
                    ) VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?)
                    """,
                    (
                        r.event_uuid, r.device_id, r.session_id, r.track_id,
                        _ts(r.track_first_seen_at), _ts(r.track_last_seen_at),
                        r.camera_source, self.camera_name, self.camera_ip, self.camera_location,
                        r.frame_width, r.frame_height,
                        r.person_confidence, r.person_image_jpeg,
                        r.person_box_x1, r.person_box_y1, r.person_box_x2, r.person_box_y2,
                        1 if r.face_detected else 0, r.face_confidence, r.face_image_jpeg,
                        r.face_box_x1, r.face_box_y1, r.face_box_x2, r.face_box_y2,
                        r.face_landmarks, r.face_sharpness, r.face_quality_score, r.face_attempts, r.image_format,
                        _ts(r.detected_at), r.person_detect_ms, r.person_crop_ms, r.face_detect_ms, r.face_crop_ms,
                        r.total_pipeline_ms,
                        r.pipeline_version, r.person_model, r.face_model, r.cpu_temp_c,
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
            track_id=row["track_id"], camera_source=row["camera_source"],
            person_confidence=row["person_confidence"], person_image_jpeg=row["person_image"],
            person_box_x1=row["person_box_x1"], person_box_y1=row["person_box_y1"],
            person_box_x2=row["person_box_x2"], person_box_y2=row["person_box_y2"],
            face_detected=bool(row["face_detected"]), detected_at=when(row["detected_at"]),
            track_first_seen_at=when(row["track_first_seen_at"]), track_last_seen_at=when(row["track_last_seen_at"]),
            frame_width=row["frame_width"], frame_height=row["frame_height"],
            face_confidence=row["face_confidence"], face_image_jpeg=row["face_image"],
            face_box_x1=row["face_box_x1"], face_box_y1=row["face_box_y1"],
            face_box_x2=row["face_box_x2"], face_box_y2=row["face_box_y2"],
            face_landmarks=row["face_landmarks"], face_sharpness=row["face_sharpness"],
            face_quality_score=row["face_quality_score"], face_attempts=row["face_attempts"],
            image_format=row["image_format"],
            person_detect_ms=row["person_detect_ms"], person_crop_ms=row["person_crop_ms"],
            face_detect_ms=row["face_detect_ms"], face_crop_ms=row["face_crop_ms"],
            total_pipeline_ms=row["total_pipeline_ms"],
            pipeline_version=row["pipeline_version"], person_model=row["person_model"],
            face_model=row["face_model"], cpu_temp_c=row["cpu_temp_c"],
        )
