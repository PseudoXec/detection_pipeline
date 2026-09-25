"""
storage.py
----------
Writes every finished vehicle+plate detection into a local SQLite database
that acts as a hand-off BUFFER between this Python pipeline and the C#
dashboard application. Metadata (timestamps, class, confidence, timing) and
the actual crop images (as JPEG BLOBs) both live in the same row, so the C#
side has everything it needs from one query - no separate file transfer step.

Design choices that matter for a 24/7 Raspberry Pi deployment:
  * WAL (Write-Ahead Logging) journal mode lets the C# dashboard read the
    database freely at the same time this process keeps writing to it.
    Without WAL, SQLite's default rollback-journal mode can make the
    reader and writer briefly block each other.
  * Writes happen on a background thread through a queue, so a slow disk
    write (SD cards can be slow) never stalls the detection loop.
  * A `synced` flag lets the dashboard mark rows as "already picked up";
    `mark_synced()` / `delete_synced_older_than()` let the dashboard (or
    this pipeline's own cleanup pass) keep the buffer from growing forever.
"""

# queue.Queue is a thread-safe hand-off between the detection loop and the writer thread
import queue
# sqlite3 is Python's built-in SQLite driver - no extra dependency needed
import sqlite3
# threading runs the background writer loop independently of detection
import threading
# time is used for the periodic flush timer
import time
# datetime gives us a human-readable, sortable timestamp for each row
from datetime import datetime, timedelta
import logging
# os.makedirs ensures the database's parent folder exists before SQLite opens it
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

log = logging.getLogger("pipeline")

# a row whose POST keeps failing is retried this many times, then left alone
# (until restart) so one poisoned row can't block the rows behind it forever
_MAX_SEND_RETRIES = 5
_RETRY_BATCH_ROWS = 50


@dataclass
class DetectionRecord:
    """Everything about one fully-processed vehicle, ready to be stored."""

    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    vehicle_image_jpeg: bytes            # JPEG bytes of the vehicle crop
    # pixel edges (left, top, right, bottom) of the vehicle box IN THE
    # ORIGINAL FULL CAMERA FRAME - lets the dashboard draw the box on the
    # full frame, or know where in the scene this vehicle was
    vehicle_box_x1: float
    vehicle_box_y1: float
    vehicle_box_x2: float
    vehicle_box_y2: float
    plate_detected: bool
    plate_confidence: Optional[float]    # None if no plate was found
    plate_image_jpeg: Optional[bytes]    # None if no plate was found
    # pixel edges (left, top, right, bottom) of the plate box IN THE
    # VEHICLE CROP (i.e. relative to `vehicle_image_jpeg`, not the full
    # frame) - lets the dashboard draw a rectangle around the plate on top
    # of the stored vehicle image. None if no plate was found.
    plate_box_x1: Optional[float]
    plate_box_y1: Optional[float]
    plate_box_x2: Optional[float]
    plate_box_y2: Optional[float]
    detected_at: datetime                # wall-clock time the vehicle was first seen
    vehicle_detect_ms: Optional[float]
    vehicle_crop_ms: Optional[float]
    plate_detect_ms: Optional[float]
    plate_crop_ms: Optional[float]
    total_pipeline_ms: Optional[float]   # sum of whichever stages above were measured
    # optional spot-check copies: {path relative to output_dir: JPEG bytes}. The writer
    # thread puts them on disk (the detection loop no longer does), and removes them
    # again once the record has been delivered.
    disk_files: Optional[Dict[str, bytes]] = None


# the exact table layout the C# dashboard will read from
_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id            TEXT NOT NULL,
    camera_source       TEXT NOT NULL,
    vehicle_class       TEXT NOT NULL,
    vehicle_confidence  REAL NOT NULL,
    vehicle_image       BLOB NOT NULL,
    vehicle_box_x1      REAL NOT NULL,          -- vehicle box, pixel coords in the FULL FRAME
    vehicle_box_y1      REAL NOT NULL,
    vehicle_box_x2      REAL NOT NULL,
    vehicle_box_y2      REAL NOT NULL,
    plate_detected      INTEGER NOT NULL,       -- 0 or 1
    plate_confidence    REAL,                   -- NULL if plate_detected = 0
    plate_image         BLOB,                   -- NULL if plate_detected = 0
    plate_box_x1        REAL,                   -- plate box, pixel coords in the VEHICLE CROP
    plate_box_y1        REAL,                   -- (i.e. relative to vehicle_image, not the full frame)
    plate_box_x2        REAL,                   -- all four NULL if plate_detected = 0
    plate_box_y2        REAL,
    detected_at         TEXT NOT NULL,          -- ISO-8601 timestamp
    vehicle_detect_ms   REAL,
    vehicle_crop_ms     REAL,
    plate_detect_ms     REAL,
    plate_crop_ms       REAL,
    total_pipeline_ms   REAL,                   -- headline "vehicle to plate crop" latency
    synced              INTEGER NOT NULL DEFAULT 0,  -- set to 1 by the consuming dashboard
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- speeds up the dashboard's "give me everything not yet synced" query
CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections (synced);
-- speeds up date-range queries / retention cleanup
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections (detected_at);
"""


class DetectionStorage:
    """Owns the SQLite connection and a background thread that drains a
    write queue into the database in small batches."""

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
    ):
        self.database_path = database_path
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        # placeholder API delivery - see api_client.py; safe to leave off
        self.send_via_api = send_via_api
        self.delete_row_after_api_send = delete_row_after_api_send
        self.api_endpoint_url = api_endpoint_url
        self.api_timeout_seconds = api_timeout_seconds
        self.output_dir = output_dir
        self.delete_disk_images_after_send = delete_disk_images_after_send
        self.send_retry_seconds = send_retry_seconds
        # row id -> failed retry count (in memory only; bounded by the number of undelivered rows)
        self._retry_failures: Dict[int, int] = {}

        # make sure the folder for the .db file actually exists first
        os.makedirs(os.path.dirname(os.path.abspath(database_path)) or ".", exist_ok=True)

        # unbounded queue: detection loop pushes records, writer thread pops them
        self._queue: "queue.Queue[DetectionRecord]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # open the connection once here (in the constructor) purely to create
        # the schema up front, so the database file/tables exist immediately
        # even before the first detection comes in
        connection = self._connect()
        connection.executescript(_SCHEMA)
        self._migrate_add_missing_columns(connection)
        connection.commit()
        connection.close()

    def _migrate_add_missing_columns(self, connection: sqlite3.Connection) -> None:
        """Adds any columns introduced after a database file already existed
        (e.g. `vehicle_detect_ms`), so an older pipeline_buffer.db on a Pi
        upgrades in place instead of erroring out."""
        existing = {row[1] for row in connection.execute("PRAGMA table_info(detections)")}
        if "vehicle_detect_ms" not in existing:
            connection.execute("ALTER TABLE detections ADD COLUMN vehicle_detect_ms REAL")

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection tuned for a single-writer/many-reader setup."""
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        # WAL mode: writers and readers don't block each other (critical since
        # the C# dashboard reads this same file while we keep writing to it)
        connection.execute("PRAGMA journal_mode=WAL;")
        # NORMAL sync is a good durability/speed trade-off for a buffer database
        # (full durability isn't critical here - the source of truth is the model,
        # not this buffer - and it reduces SD-card wear on a Raspberry Pi)
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def start(self) -> "DetectionStorage":
        """Start the background writer thread."""
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        return self

    def enqueue(self, record: DetectionRecord) -> None:
        """Called from the detection loop - never touches the disk directly,
        so a slow write can never stall real-time frame processing."""
        self._queue.put(record)

    def _writer_loop(self) -> None:
        """Runs on the background thread: batches queued records into one
        SQLite transaction at a time."""
        connection = self._connect()
        last_flush = time.time()
        next_retry = time.time() + self.send_retry_seconds
        pending: List[DetectionRecord] = []

        while not self._stop_event.is_set() or not self._queue.empty() or pending:
            try:
                # wait briefly for the next record; this timeout is what lets
                # the loop periodically check the stop flag / flush timer
                record = self._queue.get(timeout=0.25)
                pending.append(record)
            except queue.Empty:
                pass

            should_flush = len(pending) >= self.batch_size or (
                pending and (time.time() - last_flush) >= self.flush_interval_seconds
            )
            if should_flush:
                self._write_batch(connection, pending)
                pending = []
                last_flush = time.time()

            # API mode: re-send rows whose POST failed earlier (server was down, timeout...).
            # Only when nothing fresh is waiting, and never while shutting down.
            if (self.send_via_api and self.api_endpoint_url and not self._stop_event.is_set()
                    and self._queue.empty() and not pending and time.time() >= next_retry):
                try:
                    self._retry_unsent(connection)
                except Exception as error:      # housekeeping must never kill the writer thread
                    log.warning("[storage] retry pass failed: %s", error)
                next_retry = time.time() + self.send_retry_seconds

        connection.close()

    def _write_batch(self, connection: sqlite3.Connection, records: List[DetectionRecord]) -> None:
        """Insert a batch of records (one INSERT per record, in one
        transaction, so we can capture each row's own id) then, if
        `send_via_api` is on, POST each record and delete its row on success."""
        # spot-check JPEG copies go to disk from HERE (writer thread), not the detection loop
        for record in records:
            self._write_disk_files(record)

        inserted_ids: List[int] = []
        try:
            with connection:  # commits automatically on success, rolls back on error
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
                            total_pipeline_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ),
                    )
                    inserted_ids.append(cursor.lastrowid)
        except sqlite3.Error as error:
            # a storage failure should never crash the whole pipeline - log and move on
            print(f"[storage] failed to write {len(records)} record(s): {error}")
            return

        if self.send_via_api:
            self._send_and_maybe_delete(connection, records, inserted_ids)

    def _send_and_maybe_delete(
        self, connection: sqlite3.Connection, records: List[DetectionRecord], row_ids: List[int],
    ) -> None:
        """API hand-off: POST each just-written record to the C# dashboard.

          SENT    -> the row is DONE: deleted (delete_row_after_api_send) or marked synced=1,
                     and its on-disk JPEG copies are removed.
          SKIPPED -> no plate, so nothing to send. Marked synced=1 so the normal retention
                     sweep removes it after `retention_days` (it used to stay synced=0 forever).
          FAILED  -> left synced=0; `_retry_unsent` tries again later.
        """
        from api import api_client  # imported here to avoid a hard dependency when the feature is off

        sent_ids: List[int] = []
        skipped_ids: List[int] = []
        for record, row_id in zip(records, row_ids):
            result = api_client.send_detection(record, self.api_endpoint_url, self.api_timeout_seconds)
            if result == api_client.SendResult.SENT:
                sent_ids.append(row_id)
                if self.delete_disk_images_after_send:
                    self._delete_disk_files(record)
            elif result == api_client.SendResult.SKIPPED:
                skipped_ids.append(row_id)
        self._finish_rows(connection, sent_ids, skipped_ids)

    def _finish_rows(self, connection: sqlite3.Connection, sent_ids: List[int], skipped_ids: List[int]) -> None:
        """Close out rows that no longer need delivering."""
        if not sent_ids and not skipped_ids:
            return
        with connection:
            if sent_ids:
                if self.delete_row_after_api_send:
                    connection.executemany("DELETE FROM detections WHERE id = ?", [(rid,) for rid in sent_ids])
                else:
                    connection.executemany("UPDATE detections SET synced = 1 WHERE id = ?", [(rid,) for rid in sent_ids])
            if skipped_ids:
                connection.executemany("UPDATE detections SET synced = 1 WHERE id = ?", [(rid,) for rid in skipped_ids])

    def _retry_unsent(self, connection: sqlite3.Connection) -> None:
        """Re-send plate rows that are still synced=0 (earlier POST failed), and
        close out no-plate rows an older version left synced=0 forever."""
        from api import api_client

        # legacy / leftover rows that can never be sent: mark them so retention can remove them
        leftovers = [row[0] for row in connection.execute(
            "SELECT id FROM detections WHERE synced = 0 AND plate_detected = 0")]
        self._finish_rows(connection, [], leftovers)

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
            result = api_client.send_detection(self._row_to_record(row), self.api_endpoint_url, self.api_timeout_seconds)
            if result == api_client.SendResult.FAILED:
                self._retry_failures[row_id] = self._retry_failures.get(row_id, 0) + 1
                break            # server probably still down - try again next cycle instead of hammering it
            sent_ids.append(row_id)
            self._retry_failures.pop(row_id, None)
        self._finish_rows(connection, sent_ids, [])
        if sent_ids:
            log.info("[storage] re-sent %d earlier failed record(s)", len(sent_ids))

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DetectionRecord:
        """Rebuild the fields the API needs from a stored row."""
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
        )

    # ------------------------------------------------------------------ #
    # on-disk spot-check copies
    # ------------------------------------------------------------------ #
    def _write_disk_files(self, record: DetectionRecord) -> None:
        if not self.output_dir or not record.disk_files:
            return
        for relative_path, data in record.disk_files.items():
            try:
                path = os.path.join(self.output_dir, relative_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as error:
                print(f"[storage] could not write {relative_path}: {error}")

    def _delete_disk_files(self, record: DetectionRecord) -> None:
        if not self.output_dir or not record.disk_files:
            return
        for relative_path in record.disk_files:
            try:
                os.remove(os.path.join(self.output_dir, relative_path))
            except OSError:
                pass

    def sweep_output_dir(self, days: int) -> int:
        """Delete spot-check JPEGs older than `days` (backstop for anything that was
        never delivered/cleaned up, and for the non-API mode). Returns files removed."""
        if days <= 0 or not self.output_dir or not os.path.isdir(self.output_dir):
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for folder, _dirs, files in os.walk(self.output_dir):
            for name in files:
                path = os.path.join(folder, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
        return removed

    def delete_unsynced_older_than(self, days: int) -> int:
        """Last-resort cap for rows that were never delivered. 0 = disabled."""
        if days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ", timespec="seconds")
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM detections WHERE synced = 0 AND detected_at < ?", (cutoff,),
                )
                return cursor.rowcount
        finally:
            connection.close()

    def delete_synced_older_than(self, days: int) -> int:
        """Housekeeping: remove rows the dashboard has already consumed
        (synced = 1) that are older than `days`. Returns rows deleted."""
        if days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ", timespec="seconds")
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM detections WHERE synced = 1 AND detected_at < ?", (cutoff,),
                )
                return cursor.rowcount
        finally:
            connection.close()

    def stop(self) -> None:
        """Signal the writer thread to drain the queue and exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
