"""
dedupe.py
---------
track_id de-duplication for server.py, with majority voting across repeated
submissions of the same track_id (see ServerConfig.dedupe_* in ocr_config.py).

Why this exists: the edge pipeline retries a send whenever it doesn't get a
clean 200 back (timeout, dropped connection, this service restarting mid
request, etc.) - even when the original POST actually went through and was
already forwarded to the dashboard. Without this, that retry would forward a
SECOND row for the same vehicle. TrackVoteCache remembers, for a while, which
track_ids have already been successfully forwarded, and on a repeat, returns
the same result instead of forwarding again.

"Majority voting" here means: every sighting of a track_id (the original call
plus any retries) casts a vote for whatever ocr_read it produced, and the text
returned to the caller is always the majority across all votes seen so far -
not just whichever arrived first.

Backed by a small SQLite file (server.dedupe_db_path) rather than an
in-process dict, specifically so this works correctly under
`gunicorn -w N` with N > 1: each worker is a separate process with its own
memory, so an in-memory cache would miss a duplicate that lands on a
different worker than the original. SQLite's file is the one thing every
worker (and every restart) shares. WAL mode lets multiple worker processes
read/write it concurrently without stepping on each other, the same
approach the edge pipeline's own storage.py already uses for the same
reason.

Two things worth knowing before relying on this for more than "avoid an
obvious duplicate row":

1. The row already sent to the dashboard is never corrected. The FIRST
   successful forward is what the dashboard gets; if a later duplicate's OCR
   read differs and even becomes the majority, that later text is reflected
   back to the edge pipeline and in this service's own logs, but it is not
   re-sent to the dashboard (this service only creates rows, it doesn't know
   how to update one by track_id). In practice this rarely matters: repeat
   submissions of the same track_id are almost always the exact same image
   (a network retry of the identical request), which reads the same way every
   time, so the "majority" and the "first" answer are usually identical anyway.
2. This is still only a best-effort filter, not a guarantee - the lookup()
   check and the later record(forwarded=True) call straddle the actual
   network call to the dashboard, so two near-simultaneous duplicates can
   both pass the check before either finishes forwarding. The durable fix
   for "no duplicate rows, ever" is a unique constraint / upsert on
   track_id in the dashboard's own database (its system of record, and the
   one place that can enforce this with a real transaction) - see the
   "Avoiding duplicate rows for good" section in README.md for ready-to-use
   SQL. Treat this cache as cutting down the common case cheaply, not as a
   substitute for that.
"""

import json
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrackEntry:
    first_seen: float
    votes: Counter = field(default_factory=Counter)
    response_payload: Dict[str, Any] = field(default_factory=dict)  # payload from the forwarded sighting
    forwarded: bool = False


class TrackVoteCache:
    """SQLite-backed TTL store of recently-seen track_ids, shared across every
    worker process pointed at the same db_path (see module docstring)."""

    def __init__(self, window_seconds: float, max_entries: int, db_path: str):
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # one connection per call below (sqlite3 connections aren't safe to share
        # across threads without check_same_thread=False, and Flask/gunicorn serve
        # requests on multiple threads) - this table is tiny and touched at most
        # once per detected vehicle, so the per-call connect() cost is a non-issue
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        if not self._initialized:
            with self._init_lock:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dedupe_entries (
                        track_id         TEXT PRIMARY KEY,
                        first_seen       REAL NOT NULL,
                        forwarded        INTEGER NOT NULL DEFAULT 0,
                        response_payload TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dedupe_votes (
                        track_id TEXT NOT NULL,
                        ocr_read TEXT NOT NULL,
                        count    INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (track_id, ocr_read)
                    )
                    """
                )
                connection.commit()
                self._initialized = True
        return connection

    def _purge_expired(self, connection: sqlite3.Connection, now: float) -> None:
        cutoff = now - self.window_seconds
        expired = [row[0] for row in connection.execute(
            "SELECT track_id FROM dedupe_entries WHERE first_seen < ?", (cutoff,))]
        if expired:
            connection.executemany("DELETE FROM dedupe_entries WHERE track_id = ?", [(t,) for t in expired])
            connection.executemany("DELETE FROM dedupe_votes WHERE track_id = ?", [(t,) for t in expired])
        # bound total size regardless of TTL, in case window_seconds is set very
        # high - trim the oldest entries once we're comfortably over max_entries
        count = connection.execute("SELECT COUNT(*) FROM dedupe_entries").fetchone()[0]
        if count > self.max_entries:
            overflow = count - self.max_entries
            oldest = [row[0] for row in connection.execute(
                "SELECT track_id FROM dedupe_entries ORDER BY first_seen ASC LIMIT ?", (overflow,))]
            connection.executemany("DELETE FROM dedupe_entries WHERE track_id = ?", [(t,) for t in oldest])
            connection.executemany("DELETE FROM dedupe_votes WHERE track_id = ?", [(t,) for t in oldest])

    def lookup(self, track_id: Optional[str]) -> Optional[TrackEntry]:
        """Returns the current (unexpired) entry for track_id, or None if this is
        a new sighting. Callers should forward normally unless the returned entry
        has forwarded=True - that means an earlier sighting of this SAME track_id
        already made it to the dashboard, so this one should not be forwarded again."""
        if not track_id:
            return None
        now = time.time()
        connection = self._connect()
        try:
            with connection:
                self._purge_expired(connection, now)
                row = connection.execute(
                    "SELECT first_seen, forwarded, response_payload FROM dedupe_entries WHERE track_id = ?",
                    (track_id,),
                ).fetchone()
                if row is None:
                    return None
                votes = Counter(dict(connection.execute(
                    "SELECT ocr_read, count FROM dedupe_votes WHERE track_id = ?", (track_id,)).fetchall()))
                first_seen, forwarded, payload_json = row
                return TrackEntry(
                    first_seen=first_seen, votes=votes,
                    response_payload=json.loads(payload_json) if payload_json else {},
                    forwarded=bool(forwarded),
                )
        finally:
            connection.close()

    def record(
        self, track_id: Optional[str], ocr_read: str, response_payload: Dict[str, Any], forwarded: bool,
    ) -> str:
        """Records one sighting (one vote) of track_id and returns the majority
        ocr_read across every sighting recorded for it so far. `forwarded` should
        be True only once this specific sighting was actually (successfully)
        relayed to the dashboard - the FIRST sighting recorded with forwarded=True
        is the one later duplicates are served back (see TrackEntry.response_payload)."""
        if not track_id:
            return ocr_read

        now = time.time()
        connection = self._connect()
        try:
            with connection:
                self._purge_expired(connection, now)

                existing = connection.execute(
                    "SELECT forwarded, response_payload FROM dedupe_entries WHERE track_id = ?", (track_id,),
                ).fetchone()

                if existing is None:
                    connection.execute(
                        "INSERT INTO dedupe_entries (track_id, first_seen, forwarded, response_payload) "
                        "VALUES (?, ?, ?, ?)",
                        (track_id, now, 1 if forwarded else 0, json.dumps(response_payload) if forwarded else None),
                    )
                elif forwarded and not existing[0]:
                    # first sighting that actually made it to the dashboard - this
                    # is the payload later duplicates should be served
                    connection.execute(
                        "UPDATE dedupe_entries SET forwarded = 1, response_payload = ? WHERE track_id = ?",
                        (json.dumps(response_payload), track_id),
                    )

                connection.execute(
                    "INSERT INTO dedupe_votes (track_id, ocr_read, count) VALUES (?, ?, 1) "
                    "ON CONFLICT(track_id, ocr_read) DO UPDATE SET count = count + 1",
                    (track_id, ocr_read),
                )

                majority_text = connection.execute(
                    "SELECT ocr_read FROM dedupe_votes WHERE track_id = ? ORDER BY count DESC, ocr_read ASC LIMIT 1",
                    (track_id,),
                ).fetchone()[0]
                return majority_text
        finally:
            connection.close()
