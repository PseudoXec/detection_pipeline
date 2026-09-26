"""
box_publisher.py
----------------
Sends the tracker's per-frame boxes to the server so the command center can
draw them over its own copy of the video.

Design rules (they exist so this can never hurt the detection loop):
    * publish() is called from the hot loop. It only builds a tiny dict and
      drops it in a one-slot mailbox - NO network I/O, never blocks, never raises.
    * A separate thread does the HTTP POST. If the server is slow or down,
      the mailbox is simply overwritten with the newest snapshot - stale
      snapshots are never queued or replayed (latest wins).
    * Sends are rate-limited (live.max_hz) and failures back off, so a dead
      server costs the Pi almost nothing.
    * Everything is behind features.live_boxes (off by default).
"""

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import requests

from live.snapshot import build_snapshot

log = logging.getLogger("pipeline")


class LiveBoxPublisher:
    def __init__(
        self,
        endpoint_url: str,
        camera_id: str,
        max_hz: float = 10.0,
        timeout_seconds: float = 1.0,
        idle_heartbeat_seconds: float = 1.0,
    ):
        self.endpoint_url = endpoint_url
        self.camera_id = camera_id
        # changes on every Pi restart: ByteTrack IDs restart from 1, so the
        # consumer must key tracks by (camera_id, session_id, track_id)
        self.session_id = uuid.uuid4().hex[:8]
        self.min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self.timeout_seconds = timeout_seconds
        self.idle_heartbeat_seconds = idle_heartbeat_seconds

        self._cond = threading.Condition()
        self._slot: Optional[Dict[str, Any]] = None   # newest unsent snapshot
        self._seq = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    def start(self) -> "LiveBoxPublisher":
        self._thread = threading.Thread(target=self._run, name="live-box-publisher", daemon=True)
        self._thread.start()
        log.info("[live] publishing boxes for %s to %s (session %s, max %.1f Hz)",
                 self.camera_id, self.endpoint_url, self.session_id,
                 (1.0 / self.min_interval) if self.min_interval else 0.0)
        return self

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # called from the detection loop - must be fast and must never raise
    # ------------------------------------------------------------------ #
    def publish(
        self,
        frame_shape: tuple,
        predictions: List[Dict[str, Any]],
        in_roi_ids: Set[int],
        captured_at: Optional[float] = None,
    ) -> None:
        try:
            self._seq += 1
            snapshot = build_snapshot(
                self.camera_id, self.session_id, self._seq,
                frame_shape, predictions, in_roi_ids, captured_at,
            )
            with self._cond:
                self._slot = snapshot                    # overwrite: latest wins
                self._cond.notify()
        except Exception as error:                       # never let this hurt the pipeline
            log.debug("[live] publish skipped: %s", error)

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        session = requests.Session()                     # keep-alive: no reconnect per POST
        last_attempt = 0.0
        last_sent = 0.0
        last_sent_empty = False
        failures = 0
        last_warn = 0.0

        while not self._stop_event.is_set():
            # rate limit first, THEN grab the newest snapshot, so what we send is fresh
            wait = self.min_interval - (time.monotonic() - last_attempt)
            if wait > 0 and self._stop_event.wait(wait):
                break

            with self._cond:
                if self._slot is None:
                    self._cond.wait(timeout=0.5)
                snapshot, self._slot = self._slot, None
            if snapshot is None:
                continue

            # nothing on the road: send one "empty" so the viewer clears its
            # boxes, then only a slow heartbeat so it knows the Pi is alive
            is_empty = not snapshot["vehicles"]
            now = time.monotonic()
            if is_empty and last_sent_empty and now - last_sent < self.idle_heartbeat_seconds:
                continue

            last_attempt = now
            try:
                response = session.post(self.endpoint_url, json=snapshot, timeout=self.timeout_seconds)
                response.raise_for_status()
                if failures:
                    log.info("[live] server reachable again after %d failed send(s)", failures)
                failures = 0
                last_sent, last_sent_empty = now, is_empty
            except requests.RequestException as error:
                failures += 1
                if failures == 1 or time.monotonic() - last_warn > 30.0:
                    log.warning("[live] send failed (%d in a row): %s", failures, error)
                    last_warn = time.monotonic()
                self._stop_event.wait(min(2.0, 0.2 * failures))   # back off, stay quiet
