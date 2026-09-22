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
# os.makedirs ensures the database's parent folder exists before SQLite opens it
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectionRecord:
    """Everything about one fully-processed vehicle, ready to be stored."""

    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    vehicle_image_jpeg: bytes            # JPEG bytes of the vehicle crop
    plate_detected: bool
    plate_confidence: Optional[float]    # None if no plate was found
    plate_image_jpeg: Optional[bytes]    # None if no plate was found
    detected_at: datetime                # wall-clock time the vehicle was first seen
    vehicle_crop_ms: Optional[float]
    plate_detect_ms: Optional[float]
    plate_crop_ms: Optional[float]
    total_pipeline_ms: Optional[float]   # vehicle detected -> plate crop finished


# the exact table layout the C# dashboard will read from
_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id            TEXT NOT NULL,
    camera_source       TEXT NOT NULL,
    vehicle_class       TEXT NOT NULL,
    vehicle_confidence  REAL NOT NULL,
    vehicle_image       BLOB NOT NULL,
    plate_detected      INTEGER NOT NULL,       -- 0 or 1
    plate_confidence    REAL,                   -- NULL if plate_detected = 0
    plate_image         BLOB,                   -- NULL if plate_detected = 0
    detected_at         TEXT NOT NULL,          -- ISO-8601 timestamp
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
    ):
        self.database_path = database_path
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

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
        connection.commit()
        connection.close()

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

        connection.close()

    def _write_batch(self, connection: sqlite3.Connection, records: List[DetectionRecord]) -> None:
        """Insert a batch of records in a single transaction (fast + crash-safe)."""
        rows = [
            (
                record.track_id,
                record.camera_source,
                record.vehicle_class,
                record.vehicle_confidence,
                record.vehicle_image_jpeg,
                1 if record.plate_detected else 0,
                record.plate_confidence,
                record.plate_image_jpeg,
                record.detected_at.isoformat(sep=" ", timespec="seconds"),
                record.vehicle_crop_ms,
                record.plate_detect_ms,
                record.plate_crop_ms,
                record.total_pipeline_ms,
            )
            for record in records
        ]
        try:
            with connection:  # commits automatically on success, rolls back on error
                connection.executemany(
                    """
                    INSERT INTO detections (
                        track_id, camera_source, vehicle_class, vehicle_confidence,
                        vehicle_image, plate_detected, plate_confidence, plate_image,
                        detected_at, vehicle_crop_ms, plate_detect_ms, plate_crop_ms,
                        total_pipeline_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.Error as error:
            # a storage failure should never crash the whole pipeline - log and move on
            print(f"[storage] failed to write {len(rows)} record(s): {error}")

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
