"""
plate_reader.py
----------------
Inline, synchronous plate-text OCR used by pipeline.py right after a plate
crop is produced.

Runs FastPlateOCR's lightweight ONNX plate-recognition model directly
inside the detection loop, so every stored record gets an `ocr_read` value
immediately instead of waiting on a separate service. Unlike a general
document-OCR engine, FastPlateOCR is trained specifically on cropped
plates: it takes the whole crop and returns the plate string in one shot,
so there are no per-box text fragments to reassemble into reading order.

The model is loaded lazily (first call to `.read()`) and only once per
process, so importing this module never pays the load cost, and a pipeline
run with `features.ocr_read: false` never even imports fast_plate_ocr.
"""

import logging
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("plate_reader")

# a read whose length falls outside this range is usually a partial / noisy read,
# so it ranks lower when several image variants are compared (it is not rejected)
_PLAUSIBLE_LENGTH = (5, 8)
_IMPLAUSIBLE_PENALTY = 0.85

# lightweight default: small, fast on CPU (Pi 5), good fit for the real-time loop.
# Pick any model from the FastPlateOCR model zoo via OcrConfig.model_name.
_DEFAULT_MODEL = "cct-xs-v1-global-model"


class PlateOCRReader:
    """Wraps FastPlateOCR's ONNX plate recognizer to read text off a small
    plate crop. Never raises - any failure (missing dependency, bad crop,
    model error) is caught and surfaces as a `None` read, which the caller
    turns into the configured "Unrecognized" text."""

    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.5,
        allowed_chars: Optional[str] = None,
        early_exit_score: float = 0.85,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.lang = lang  # kept for config compatibility; FastPlateOCR models aren't language-keyed
        self.min_confidence = min_confidence
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        # once one image variant reads at or above this score, the remaining variants are skipped
        self.early_exit_score = early_exit_score
        self.model_name = model_name
        self._engine = None
        self._load_failed = False

    # ------------------------------------------------------------------ #
    # engine loading
    # ------------------------------------------------------------------ #
    def _get_engine(self):
        if self._engine is not None or self._load_failed:
            return self._engine
        try:
            # imported here (not at module level) so this dependency is only
            # paid for if OCR is actually enabled/used
            from fast_plate_ocr import LicensePlateRecognizer
        except Exception as error:  # pragma: no cover - environment dependent
            log.error("PlateOCRReader: fast_plate_ocr is not installed, plate reads will be Unrecognized: %s", error)
            self._load_failed = True
            return None

        try:
            self._engine = LicensePlateRecognizer(self.model_name)
            log.info("PlateOCRReader: FastPlateOCR engine ready (model=%s)", self.model_name)
        except Exception as error:
            log.error("PlateOCRReader: failed to load FastPlateOCR model %r, "
                      "plate reads will be Unrecognized: %s", self.model_name, error)
            self._load_failed = True
            self._engine = None
        return self._engine

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def read(self, plate_crop: Optional[np.ndarray]) -> Optional[str]:
        """Returns the best-guess plate text, or None if nothing readable
        was found (missing crop, no OCR hits, or every hit below
        min_confidence)."""
        text, _score = self.read_scored(plate_crop)
        return text

    def read_scored(
        self, plate_crop: Optional[np.ndarray], *alternates: Optional[np.ndarray],
    ) -> Tuple[Optional[str], float]:
        """Like `read`, but also returns a 0..1 quality score (confidence-weighted, lower for
        implausible lengths) so the caller can decide whether the read is good enough or
        whether another frame is worth trying.

        `alternates` are other renderings of the SAME plate (e.g. the un-enhanced crop next
        to the CLAHE-sharpened one). Each is tried in turn and the best read wins, because
        enhancement helps some plates and hurts others."""
        images = [image for image in (plate_crop, *alternates) if image is not None and image.size > 0]
        if not images:
            return None, 0.0

        engine = self._get_engine()
        if engine is None:
            return None, 0.0

        best_text: Optional[str] = None
        best_score = 0.0
        for variant in self._variants(images):
            try:
                candidate = self._run_ocr(engine, variant)
            except Exception as error:
                log.warning("PlateOCRReader: OCR call failed: %s", error)
                continue
            if candidate is None:
                continue
            text, score = candidate
            if text and score > best_score:
                best_text, best_score = text, score
            if best_text and best_score >= self.early_exit_score:
                break
        return best_text, best_score

    # ------------------------------------------------------------------ #
    # image variants
    # ------------------------------------------------------------------ #
    @staticmethod
    def _variants(images: List[np.ndarray]) -> Iterator[np.ndarray]:
        """Yield each image as RGB with a border added. FastPlateOCR expects a
        channels_last RGB (or grayscale) array, and a tight plate crop has
        letters right at the edge; replicating the edge pixels outward avoids
        clipped characters without adding fake content."""
        for image in images:
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pad = max(6, int(round(rgb.shape[0] * 0.10)))
            yield cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------ #
    # OCR + result handling
    # ------------------------------------------------------------------ #
    def _clean(self, text: str) -> str:
        text = text.upper().strip()
        if self.allowed_chars:
            text = "".join(ch for ch in text if ch in self.allowed_chars)
        else:
            text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _run_ocr(self, engine, plate_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Runs FastPlateOCR on one image variant and returns (cleaned_text,
        quality score), or None if nothing readable came back."""
        results = engine.run(plate_crop, return_confidence=True)
        if not results:
            return None
        pred = results[0]
        raw_text = getattr(pred, "plate", None)
        if not raw_text:
            return None
        cleaned = self._clean(str(raw_text))
        if not cleaned:
            return None

        char_probs = getattr(pred, "char_probs", None)
        conf = float(np.mean(char_probs)) if char_probs is not None and len(char_probs) else 1.0
        if conf < self.min_confidence:
            return None

        low, high = _PLAUSIBLE_LENGTH
        score = conf * (1.0 if low <= len(cleaned) <= high else _IMPLAUSIBLE_PENALTY)
        return cleaned, score
