"""
ocr_reader.py
-------------
Thin wrapper around FastPlateOCR's `LicensePlateRecognizer`: loads the
model once (lazily, on first read), runs it over one or more image
variants of the same plate crop (typically the raw crop and the
CLAHE+sharpen version from image_enhance.py), and turns the raw model
output into the cleaned/scored result the rest of this project expects.

Both run modes use this the same way:
    reader = PlateOCRReader(model_name=..., lang=..., min_confidence=...,
                             allowed_chars=..., early_exit_score=...)
    text, score      = reader.read_scored(*variants)   # cli/ocr_pipeline.py
    result           = reader.read_full(*variants)      # server/server.py
                        # result.text, result.raw_text, result.confidence

Design notes
------------
- Lazy model load: the ONNX session isn't created until the first read, so
  just importing/constructing this class (server/CLI startup) is cheap,
  and a bad model name only breaks things once someone actually tries a
  read - it never crashes startup.
- Never raises out of read_scored/read_full: a missing `fast_plate_ocr`
  install, a bad model name, a model load failure, a bad image, or an
  inference error never crashes the caller - it comes back as an empty
  read (`("", 0.0)` / an OCRResult with `text=""`, `confidence=0.0`), with
  the reason logged. This matches this project's documented behaviour
  (see README's "Notes" section) - the affected image just comes back as
  "Unrecognized", the run keeps going.
- Colour handling: FastPlateOCR expects in-memory arrays to already be in
  the colour mode its model config declares (`rgb` or `grayscale`) - it
  only does the BGR->RGB conversion itself when reading a path from disk
  (see fast_plate_ocr.core.process.read_plate_image). Both callers here
  hand us BGR arrays (cv2.imdecode/cv2.imread), so this wrapper converts
  to whatever `model.config.image_color_mode` actually asks for on every
  read, rather than assuming one mode - the model name is a config value,
  and different FastPlateOCR models declare different colour modes.
- Multiple variants are all read (unless an earlier one already clears
  `early_exit_score`, in which case the rest are skipped), and whichever
  scores highest wins.
- Confidence is the mean of the model's per-character probabilities
  (`return_confidence=True`) over the non-pad characters only, so a
  short, confidently-read plate isn't dragged down by high-confidence
  "this slot is padding" predictions on unused character slots.
- Philippine plate format gate: once a read clears min_confidence, it is
  also checked against `ocr.plate_format` (see plate_format.py) - a
  confident read that still isn't shaped like any real Philippine plate
  (reversed letters/digits, wrong grouping, a stray leftover character)
  is reported the same as an unreadable plate: the caller's `text`/
  `ocr_read` comes back empty (-> "Unrecognized"), but `raw_text`/
  `ocr_text` still carries what the model actually produced, so a bad
  read is visible for review instead of silently reaching the dashboard
  as a real-looking but wrong plate number. Set
  `ocr.validate_plate_format: false` in ocr_config.yaml to turn this off
  (e.g. if you operate outside the Philippines).
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ocr.plate_format import is_valid_ph_plate

log = logging.getLogger("ocr_pipeline.ocr_reader")


@dataclass
class OCRResult:
    text: str                  # cleaned, final text - "" if nothing readable
    raw_text: Optional[str]    # pre-cleanup text - None if nothing came back at all
    confidence: float


class PlateOCRReader:
    """Wraps a single FastPlateOCR model. One instance is created per
    process (see server.py / cli/ocr_pipeline.py) and reused across reads -
    do not construct one per request."""

    def __init__(
        self,
        model_name: str,
        lang: str = "en",
        min_confidence: float = 0.5,
        allowed_chars: Optional[str] = None,
        early_exit_score: float = 0.85,
        validate_plate_format: bool = True,
    ):
        self.model_name = model_name
        self.lang = lang  # not used by FastPlateOCR itself; kept for config/log parity
        self.min_confidence = min_confidence
        self.allowed_chars = allowed_chars.upper() if allowed_chars else None
        self.early_exit_score = early_exit_score
        self.validate_plate_format = validate_plate_format
        self._model = None
        self._load_failed = False

    # ------------------------------------------------------------------ #
    # model loading (lazy, cached)
    # ------------------------------------------------------------------ #
    def _get_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError:
            log.error(
                "fast_plate_ocr is not installed (see requirements.txt) - "
                "every read will come back Unrecognized until it is."
            )
            self._load_failed = True
            return None

        try:
            # model_name is normally a model-zoo name (e.g.
            # "cct-xs-v1-global-model"), but config/ocr_config.py's docstring
            # also allows a path to your own exported ONNX model - detect
            # that case and look for a plate-config YAML next to it.
            if os.path.isfile(self.model_name) or self.model_name.lower().endswith(".onnx"):
                config_path = self._infer_plate_config_path(self.model_name)
                if config_path is None:
                    log.error(
                        "PlateOCRReader: %r looks like a custom ONNX model path, but no "
                        "matching *_plate_config.yaml / plate_config.yaml was found next to it.",
                        self.model_name,
                    )
                    self._load_failed = True
                    return None
                self._model = LicensePlateRecognizer(
                    onnx_model_path=self.model_name, plate_config_path=config_path
                )
            else:
                self._model = LicensePlateRecognizer(self.model_name)
        except Exception:
            log.error("PlateOCRReader: failed to load model %r", self.model_name, exc_info=True)
            self._load_failed = True
            return None
        return self._model

    @staticmethod
    def _infer_plate_config_path(onnx_path: str) -> Optional[str]:
        directory = os.path.dirname(onnx_path) or "."
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        candidates = [
            os.path.join(directory, f"{stem}_plate_config.yaml"),
            os.path.join(directory, "plate_config.yaml"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #
    def _clean(self, raw_text: str) -> str:
        text = raw_text.upper().strip()
        if self.allowed_chars:
            text = "".join(ch for ch in text if ch in self.allowed_chars)
        return text

    # ------------------------------------------------------------------ #
    # single-variant read
    # ------------------------------------------------------------------ #
    def _read_variant(self, image: Optional[np.ndarray]) -> Tuple[str, float]:
        """Runs the model on one image variant. Returns (raw_text,
        confidence); raw_text is pad-stripped but NOT allowed_chars-cleaned
        yet. Never raises - returns ("", 0.0) on any failure."""
        if image is None or getattr(image, "size", 0) == 0:
            return "", 0.0

        model = self._get_model()
        if model is None:
            return "", 0.0

        try:
            import cv2

            color_mode = getattr(model.config, "image_color_mode", "rgb")
            if image.ndim == 2:
                converted = image if color_mode == "grayscale" else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif color_mode == "grayscale":
                converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictions = model.run(converted, return_confidence=True, remove_pad_char=False)
            if not predictions:
                return "", 0.0
            prediction = predictions[0]
        except Exception:
            log.warning("PlateOCRReader: inference failed on one image variant", exc_info=True)
            return "", 0.0

        pad_char = getattr(model.config, "pad_char", "_")
        char_probs = prediction.char_probs
        kept_chars = []
        kept_probs = []
        for index, char in enumerate(prediction.plate):
            if char == pad_char:
                continue
            kept_chars.append(char)
            if char_probs is not None and index < len(char_probs):
                kept_probs.append(float(char_probs[index]))

        raw_text = "".join(kept_chars)
        if not raw_text:
            return "", 0.0
        confidence = float(np.mean(kept_probs)) if kept_probs else 0.0
        return raw_text, confidence

    # ------------------------------------------------------------------ #
    # multi-variant read: best-of, with early exit
    # ------------------------------------------------------------------ #
    def _best_of(self, variants: Tuple[np.ndarray, ...]) -> Tuple[str, float]:
        best_text, best_score = "", 0.0
        for variant in variants:
            text, score = self._read_variant(variant)
            if score > best_score:
                best_text, best_score = text, score
            if best_score >= self.early_exit_score:
                break
        return best_text, best_score

    # ------------------------------------------------------------------ #
    # confidence + plate-format gate (shared by both public methods)
    # ------------------------------------------------------------------ #
    def _finalize(self, raw_text: str, score: float) -> str:
        """Turns a raw read into the final reported text: "" if it doesn't
        clear min_confidence, or (when validate_plate_format is on) if it
        doesn't match a recognized Philippine plate format - reversed
        letters/digits, wrong grouping, a stray leftover character, etc.
        In every "" case the caller still has raw_text/score to report for
        review; only the final text is suppressed."""
        if not raw_text or score < self.min_confidence:
            return ""
        cleaned = self._clean(raw_text)
        if self.validate_plate_format and not is_valid_ph_plate(cleaned):
            log.info(
                "PlateOCRReader: read %r (score=%.2f) does not match a recognized "
                "Philippine plate format - reporting as unrecognized",
                cleaned, score,
            )
            return ""
        return cleaned

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def read_scored(self, *variants: np.ndarray) -> Tuple[str, float]:
        """Used by cli/ocr_pipeline.py. Returns (text, confidence) - text
        is the cleaned, format-validated read, or "" if it didn't clear
        min_confidence or doesn't match a recognized PH plate format
        (confidence is still the real score either way, so a caller can
        tell "read something, just rejected" from "nothing at all")."""
        raw_text, score = self._best_of(variants)
        return self._finalize(raw_text, score), score

    def read_full(self, *variants: np.ndarray) -> OCRResult:
        """Used by server/server.py. Same read as read_scored, but returned
        as an OCRResult that also carries the pre-cleanup raw_text, which
        server.py reports back separately as `ocr_text` - so a low-confidence
        read, a bad `allowed_chars` cleanup, or a plate-format rejection are
        all still visible for review even though `text`/`ocr_read` comes
        back empty (-> "Unrecognized")."""
        raw_text, score = self._best_of(variants)
        if not raw_text:
            return OCRResult(text="", raw_text=None, confidence=score)
        return OCRResult(text=self._finalize(raw_text, score), raw_text=raw_text, confidence=score)
