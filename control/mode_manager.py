"""Switches the active detection mode (vehicle / person / idle) while the program keeps running.

What "seamless" and "CPU friendly" mean here
--------------------------------------------
* Only ONE mode is ever processing frames. The other is not "paused": it is stopped - its worker
  threads are joined and its models are freed - so an unused pipeline costs no CPU and no RAM.
* The camera reader, ROI polling, SQLite stores and live view are shared and never restart, so the
  stream keeps flowing during a switch.
* With `mode.preload_on_switch` (default) the NEW mode is loaded and warmed up in the background while
  the OLD mode keeps detecting. The swap itself is one pointer change between two frames, so there is
  no gap in detection. The old mode is then stopped in the background: its in-flight tracks are handed
  to storage (nothing is lost) and its memory is released. Cost: both sets of models are in RAM for the
  few seconds the load takes.
* A switch that fails (missing weights, import error...) leaves the old mode running and is reported
  in the status; it never takes detection down.
* Requests are non-blocking and coalesce: if the command center sends A, B, C quickly, only the latest
  one wins.
* The last choice is saved to `mode.state_file`, so a reboot or crash resumes in the mode the command
  center picked.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from modes.base import DetectionMode, RuntimeContext
from modes.idle import IdleMode
from modes.registry import build_mode

log = logging.getLogger("pipeline")

_MAX_CONSECUTIVE_FRAME_ERRORS = 10


def load_saved_mode(state_file: Optional[str], allowed: List[str]) -> Optional[str]:
    """The mode the command center chose last time, or None."""
    if not state_file or not os.path.isfile(state_file):
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as handle:
            name = str(json.load(handle).get("mode", "")).strip().lower()
        return name if name in allowed else None
    except (OSError, ValueError, AttributeError) as error:
        log.warning("[mode] could not read %s (%s) - using the configured default", state_file, error)
        return None


def choose_initial_mode(config, override=None) -> str:
    """Start-up order: the --mode flag, then the command center's saved choice, then mode.default."""
    if override:
        name = override.strip().lower()
        if name not in config.mode.allowed:
            raise SystemExit(f"--mode {override!r} is not one of mode.allowed: {config.mode.allowed}")
        return name
    if config.mode.remember_last:
        saved = load_saved_mode(config.mode.state_file, config.mode.allowed)
        if saved:
            log.info("[mode] resuming the last mode chosen by the command center: %s", saved)
            return saved
    return config.mode.default


class ModeManager:
    def __init__(
        self,
        ctx: RuntimeContext,
        allowed: List[str],
        state_file: Optional[str] = None,
        remember_last: bool = True,
        preload_on_switch: bool = True,
    ):
        self.ctx = ctx
        self.allowed = [name.lower() for name in allowed]
        self.state_file = state_file
        self.remember_last = remember_last
        self.preload_on_switch = preload_on_switch

        self._run_lock = threading.Lock()            # held while a frame is processed and during the swap
        self._active: DetectionMode = IdleMode(ctx)
        self._active_name = "idle"
        self._since = time.time()

        self._cond = threading.Condition()
        self._target: Optional[str] = None
        self._target_source = "api"
        self._switching_to: Optional[str] = None
        self._last_switch: Optional[Dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_shape: Optional[tuple] = None
        self._frame_errors = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self, mode_name: str, frame_shape: Optional[tuple] = None) -> None:
        """Load the first mode synchronously. If it can't load, run idle and say why."""
        self._last_shape = frame_shape
        mode_name = mode_name.lower()
        began = time.monotonic()
        try:
            mode = build_mode(mode_name, self.ctx)
            self._warm(mode, frame_shape)
            self._active, self._active_name, self._since = mode, mode_name, time.time()
            self._last_switch = self._record(mode_name, True, None, began, "startup")
            log.info("[mode] started in %s mode", mode_name)
        except Exception as error:
            log.error("[mode] could not start %s mode (%s) - running idle until the command center picks "
                      "another mode", mode_name, error, exc_info=True)
            self._last_switch = self._record(mode_name, False, str(error), began, "startup")
        self._thread = threading.Thread(target=self._switch_loop, name="mode-switcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        with self._run_lock:
            mode, self._active = self._active, IdleMode(self.ctx)
            self._active_name = "idle"
        self._safe_stop(mode)

    # ------------------------------------------------------------------ frame path (main thread)
    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        self._last_shape = frame.shape[:2]
        with self._run_lock:
            mode, name = self._active, self._active_name
            try:
                result = mode.process_frame(frame, always_finalize=always_finalize, captured_at=captured_at)
                self._frame_errors = 0
                return result
            except Exception as error:
                self._frame_errors += 1
                error_text = f"{type(error).__name__}: {error}"
                log.error("[mode] %s failed on a frame (%d in a row): %s", name, self._frame_errors, error_text,
                          exc_info=self._frame_errors <= 3)
                broken = self._frame_errors >= _MAX_CONSECUTIVE_FRAME_ERRORS
        if broken:
            self._frame_errors = 0
            log.error("[mode] %s keeps failing - falling back to idle (pick a mode again to retry)", name)
            self._last_switch = self._record(name, False, error_text, time.monotonic(), "auto-fallback")
            self._swap_in(IdleMode(self.ctx), "idle", "auto-fallback", persist=False)
        return 0

    # ------------------------------------------------------------------ command center API
    def request(self, mode_name: str, source: str = "api") -> Dict[str, Any]:
        """Ask for a mode. Returns immediately; the switch happens in the background."""
        mode_name = mode_name.strip().lower()
        if mode_name not in self.allowed:
            raise ValueError(f"unknown mode {mode_name!r} (allowed: {', '.join(self.allowed)})")
        with self._cond:
            already = (mode_name == self._active_name and self._switching_to is None and self._target is None)
            building_it = (self._switching_to == mode_name and self._target is None)
            if not (already or building_it):
                self._target, self._target_source = mode_name, source
                self._cond.notify_all()
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._cond:
            switching, target = self._switching_to, self._target
            return {
                "mode": self._active_name,
                "state": "switching" if (switching or target) else ("idle" if self._active_name == "idle" else "running"),
                "requested": switching or target,
                "since": datetime.fromtimestamp(self._since).isoformat(timespec="seconds"),
                "allowed": list(self.allowed),
                "last_switch": dict(self._last_switch) if self._last_switch else None,
            }

    def choices(self) -> List[str]:
        return list(self.allowed)

    # ------------------------------------------------------------------ switching (background thread)
    def _switch_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._cond:
                while self._target is None and not self._stop_event.is_set():
                    self._cond.wait(timeout=1.0)
                if self._stop_event.is_set():
                    return
                name, source = self._target, self._target_source
                self._target = None
                if name == self._active_name:
                    continue
                self._switching_to = name
            try:
                self._do_switch(name, source)
            except Exception as error:   # never let the switcher thread die
                log.error("[mode] unexpected error switching to %s: %s", name, error, exc_info=True)
            finally:
                with self._cond:
                    self._switching_to = None

    def _do_switch(self, name: str, source: str) -> None:
        began = time.monotonic()
        log.info("[mode] switching %s -> %s (requested by %s)...", self._active_name, name, source)

        if not self.preload_on_switch:
            # low-memory path: free the old mode first. There is a gap in detection while the new one
            # loads, and a failed load leaves us idle (the old mode is already gone).
            self._swap_in(IdleMode(self.ctx), "idle", source, persist=False)

        new_mode: Optional[DetectionMode] = None
        try:
            new_mode = build_mode(name, self.ctx)
            self._warm(new_mode, self._last_shape)
        except Exception as error:
            log.error("[mode] could not load %s mode (%s) - staying on %s", name, error, self._active_name,
                      exc_info=True)
            if new_mode is not None:
                self._safe_stop(new_mode)
            self._last_switch = self._record(name, False, f"{type(error).__name__}: {error}", began, source)
            return

        self._swap_in(new_mode, name, source, persist=True)
        self._last_switch = self._record(name, True, None, began, source)
        log.info("[mode] now running %s (switch took %.1fs)", name, time.monotonic() - began)

    def _swap_in(self, new_mode: DetectionMode, name: str, source: str, persist: bool) -> None:
        # one pointer change between two frames; waits (at most one frame) for the frame in progress
        with self._run_lock:
            old_mode, old_name = self._active, self._active_name
            self._active, self._active_name, self._since = new_mode, name, time.time()
        self.ctx.live.clear()                      # old mode's boxes must not linger on screen
        self._safe_stop(old_mode)                  # in the background: flush its tracks, free its models
        if persist and self.remember_last:
            self._save_state(name, source)
        log.info("[mode] %s stopped and released", old_name)

    # ------------------------------------------------------------------ helpers
    def _warm(self, mode: DetectionMode, frame_shape: Optional[tuple]) -> None:
        mode.warmup(frame_shape)

    @staticmethod
    def _safe_stop(mode: DetectionMode) -> None:
        try:
            mode.stop()
        except Exception as error:
            log.warning("[mode] error while stopping %s: %s", getattr(mode, "name", "?"), error, exc_info=True)

    @staticmethod
    def _record(name: str, ok: bool, error: Optional[str], began: float, source: str) -> Dict[str, Any]:
        return {
            "to": name, "ok": ok, "error": error, "source": source,
            "seconds": round(time.monotonic() - began, 2),
            "at": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_state(self, name: str, source: str) -> None:
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file)) or ".", exist_ok=True)
            temp_path = self.state_file + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump({"mode": name, "changed_by": source,
                           "changed_at": datetime.now().isoformat(timespec="seconds")}, handle)
            os.replace(temp_path, self.state_file)   # atomic: a power cut can't leave half a file
        except OSError as error:
            log.warning("[mode] could not save the chosen mode to %s: %s", self.state_file, error)
