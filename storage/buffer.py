"""Shared machinery for the SQLite buffers.

Every detection mode has its own table (vehicle -> `detections`, person -> `person_detections`) in the
same database file, and its own small subclass of BufferedStore that knows its columns and how to send
a row to the dashboard API. Everything else lives here once:

* a background writer thread that batches inserts (the frame loop never waits on the disk),
* WAL mode, so the C# dashboard can read while we write,
* optional JPEG copies on disk for spot checking,
* retention (delete synced / stale rows, sweep the output folder),
* the periodic "re-send what failed" pass.

The writer threads are deliberately NOT tied to a detection mode: switching modes must never strand
rows that are still waiting to be sent, so the stores keep running for the life of the process.
"""
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline")


class BufferedStore:
    TABLE = ""            # subclasses: table name
    SCHEMA = ""           # subclasses: CREATE TABLE / INDEX statements

    def __init__(
        self,
        database_path: str,
        batch_size: int = 8,
        flush_interval_seconds: float = 1.0,
        send_via_api: bool = False,
        send_retry_seconds: float = 60.0,
        output_dir: Optional[str] = None,
        delete_disk_images_after_send: bool = True,
        camera_name: Optional[str] = None,
        camera_ip: Optional[str] = None,
        camera_location: Optional[str] = None,
    ):
        self.database_path = database_path
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.send_via_api = send_via_api
        self.send_retry_seconds = send_retry_seconds
        self.output_dir = output_dir
        self.delete_disk_images_after_send = delete_disk_images_after_send
        self.camera_name = camera_name
        self.camera_ip = camera_ip
        self.camera_location = camera_location

        os.makedirs(os.path.dirname(os.path.abspath(database_path)) or ".", exist_ok=True)

        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        connection = self._connect()
        connection.executescript(self.SCHEMA)
        self.migrate(connection)
        connection.commit()
        connection.close()

    # ------------------------------------------------------------------ hooks for subclasses
    def migrate(self, connection: sqlite3.Connection) -> None:
        """Add columns that older database files don't have yet."""

    def insert_records(self, connection: sqlite3.Connection, records: List[Any]) -> List[int]:
        raise NotImplementedError

    def send_new(self, connection: sqlite3.Connection, records: List[Any], row_ids: List[int]) -> None:
        """Called right after a batch was written, when send_via_api is on."""

    def retry_unsent(self, connection: sqlite3.Connection) -> None:
        """Called periodically while the queue is idle, when send_via_api is on."""

    def can_send(self) -> bool:
        return self.send_via_api

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def start(self) -> "BufferedStore":
        self._thread = threading.Thread(target=self._writer_loop, name=f"store-{self.TABLE}", daemon=True)
        self._thread.start()
        return self

    def enqueue(self, record: Any) -> None:
        self._queue.put(record)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _writer_loop(self) -> None:
        connection = self._connect()
        last_flush = time.time()
        next_retry = time.time() + self.send_retry_seconds
        pending: List[Any] = []

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

            if (self.can_send() and not self._stop_event.is_set()
                    and self._queue.empty() and not pending and time.time() >= next_retry):
                try:
                    self.retry_unsent(connection)
                except Exception as error:
                    log.warning("[storage:%s] retry pass failed: %s", self.TABLE, error)
                next_retry = time.time() + self.send_retry_seconds

        connection.close()

    def _write_batch(self, connection: sqlite3.Connection, records: List[Any]) -> None:
        for record in records:
            self._write_disk_files(record)
        try:
            row_ids = self.insert_records(connection, records)
        except sqlite3.Error as error:
            print(f"[storage:{self.TABLE}] failed to write {len(records)} record(s): {error}")
            return
        if self.can_send():
            self.send_new(connection, records, row_ids)

    # ------------------------------------------------------------------ marking rows
    def finish_rows(self, connection: sqlite3.Connection, sent_ids: List[int], skipped_ids: List[int],
                    delete_sent: bool, rejected_ids: Optional[List[int]] = None) -> None:
        """sent -> delete (or synced=1); skipped -> synced=1; rejected -> synced=2 (kept for inspection)."""
        rejected_ids = rejected_ids or []
        if not (sent_ids or skipped_ids or rejected_ids):
            return
        with connection:
            if sent_ids:
                if delete_sent:
                    connection.executemany(f"DELETE FROM {self.TABLE} WHERE id = ?", [(r,) for r in sent_ids])
                else:
                    connection.executemany(f"UPDATE {self.TABLE} SET synced = 1 WHERE id = ?", [(r,) for r in sent_ids])
            if skipped_ids:
                connection.executemany(f"UPDATE {self.TABLE} SET synced = 1 WHERE id = ?", [(r,) for r in skipped_ids])
            if rejected_ids:
                connection.executemany(f"UPDATE {self.TABLE} SET synced = 2 WHERE id = ?", [(r,) for r in rejected_ids])

    # ------------------------------------------------------------------ disk copies
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
                print(f"[storage:{self.TABLE}] could not write {relative_path}: {error}")

    def _delete_disk_files(self, record: Any) -> None:
        files: Optional[Dict[str, bytes]] = getattr(record, "disk_files", None)
        if not self.output_dir or not files:
            return
        for relative_path in files:
            try:
                os.remove(os.path.join(self.output_dir, relative_path))
            except OSError:
                pass

    # ------------------------------------------------------------------ retention
    def sweep_output_dir(self, days: int) -> int:
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

    def _delete_where(self, synced_clause: str, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ", timespec="seconds")
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    f"DELETE FROM {self.TABLE} WHERE {synced_clause} AND detected_at < ?", (cutoff,),
                )
                return cursor.rowcount
        finally:
            connection.close()

    def delete_synced_older_than(self, days: int) -> int:
        return self._delete_where("synced = 1", days)

    def delete_unsynced_older_than(self, days: int) -> int:
        return self._delete_where("synced = 0", days)
