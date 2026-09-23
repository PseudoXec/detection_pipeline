"""
plate_reader.py
----------------
Inline, synchronous plate-text OCR used by pipeline.py right after a plate
crop is produced. This is deliberately separate from ocr_worker.py:

  * ocr_worker.py runs PaddleOCR-VL (a heavy doc-parsing / GPU model) as its
    own long-running process, pulling crops from a folder or API queue -
    meant for a beefier machine, not the same process as the real-time
    camera loop.
  * PlateOCRReader here runs PaddleOCR's plain detection+recognition
    pipeline (much lighter, CPU-friendly) directly inside the detection
    loop so every stored record gets a `ocr_read` value immediately instead
    of waiting on a separate service.

The model is loaded lazily (first call to `.read()`) and only once per
process, so importing this module never pays the load cost, and a pipeline
run with `features.ocr_read: false` never even imports PaddleOCR.
"""

import logging
import os
import re
from typing import Optional

import numpy as np

log = logging.getLogger("plate_reader")

# Some CPU-only PaddlePaddle builds (seen on Windows) have a bug where the
# newer "PIR" executor combined with oneDNN can't convert certain op
# attributes (e.g. "ConvertPirAttribute2RuntimeAttribute ... not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]"), making every single OCR
# call fail regardless of image quality. Both flags must be set BEFORE
# `paddle`/`paddleocr` is first imported anywhere in the process - Paddle
# reads them once at framework init - so this needs to run at module import
# time, not inside a function called later.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")


class PlateOCRReader:
    """Wraps PaddleOCR's rec+det pipeline to read text off a small plate
    crop. Never raises - any failure (missing dependency, bad crop, model
    error) is caught and surfaces as a `None` read, which the caller turns
    into the configured "Unrecognized" text."""

    def __init__(self, lang: str = "en", min_confidence: float = 0.5, allowed_chars: Optional[str] = None):
        self.lang = lang
        self.min_confidence = min_confidence
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self._engine = None
        self._load_failed = False

    def _get_engine(self):
        if self._engine is not None or self._load_failed:
            return self._engine
        try:
            # imported here (not at module level) so this heavy dependency
            # is only paid for if OCR is actually enabled/used
            from paddleocr import PaddleOCR
        except Exception as error:  # pragma: no cover - environment dependent
            log.error("PlateOCRReader: paddleocr is not installed, plate reads will be Unrecognized: %s", error)
            self._load_failed = True
            return None

        # PaddleOCR 3.x renamed use_angle_cls -> use_textline_orientation and
        # dropped show_log entirely; PaddleOCR 2.x doesn't know the new name.
        # Try the modern kwargs first, fall back to the old ones, so this
        # works regardless of which major version ends up installed instead
        # of silently failing to load and turning EVERY plate into
        # "Unrecognized" (which is what happened before this fix).
        attempts = (
            {"use_textline_orientation": True, "lang": self.lang, "enable_mkldnn": False},
            {"use_textline_orientation": True, "lang": self.lang},
            {"use_angle_cls": True, "lang": self.lang, "show_log": False},
            {"lang": self.lang},
        )
        last_error = None
        for kwargs in attempts:
            try:
                self._engine = PaddleOCR(**kwargs)
                self._engine_api = "predict" if hasattr(self._engine, "predict") else "ocr"
                log.info("PlateOCRReader: PaddleOCR engine ready (lang=%s, kwargs=%s, api=%s)",
                         self.lang, kwargs, self._engine_api)
                return self._engine
            except Exception as error:
                last_error = error
                continue

        log.error("PlateOCRReader: failed to load PaddleOCR engine with any known API, "
                  "plate reads will be Unrecognized: %s", last_error)
        self._load_failed = True
        self._engine = None
        return None

    def _clean(self, text: str) -> str:
        text = text.upper().strip()
        if self.allowed_chars:
            text = "".join(ch for ch in text if ch in self.allowed_chars)
        else:
            text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def read(self, plate_crop: Optional[np.ndarray]) -> Optional[str]:
        """Returns the best-guess plate text, or None if nothing readable
        was found (missing crop, no OCR hits, or every hit below
        min_confidence)."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        engine = self._get_engine()
        if engine is None:
            return None

        try:
            kept = self._run_ocr(engine, plate_crop)
        except Exception as error:
            log.warning("PlateOCRReader: OCR call failed: %s", error)
            return None

        if not kept:
            return None

        combined = self._clean("".join(text for text, _ in kept))
        return combined or None

    def _run_ocr(self, engine, plate_crop: np.ndarray) -> list:
        """Runs OCR and returns a list of (text, confidence) pairs above
        min_confidence, handling both the PaddleOCR 2.x `.ocr()` return
        shape and the PaddleOCR 3.x `.predict()` return shape."""
        if getattr(self, "_engine_api", "ocr") == "predict":
            # PaddleOCR 3.x: predict() returns a list of result objects,
            # each dict-like with 'rec_texts' / 'rec_scores' lists
            results = engine.predict(plate_crop)
            kept = []
            for res in results:
                data = res if isinstance(res, dict) else getattr(res, "json", {}) or {}
                texts = data.get("rec_texts") or []
                scores = data.get("rec_scores") or []
                for text, conf in zip(texts, scores):
                    if conf >= self.min_confidence:
                        kept.append((text, conf))
            return kept

        # PaddleOCR 2.x: ocr() returns [[ [box, (text, confidence)], ... ]] -
        # one list per input image
        try:
            result = engine.ocr(plate_crop, cls=True)
        except TypeError:
            # some 2.x point releases dropped the `cls` kwarg
            result = engine.ocr(plate_crop)
        lines = result[0] if result else None
        if not lines:
            return []
        # keep every line that clears the confidence bar, in reading order
        # (top-to-bottom, which is how PaddleOCR returns them) - most plates
        # are a single line, but some are stacked two-line plates
        return [(text, conf) for _, (text, conf) in lines if conf >= self.min_confidence]
