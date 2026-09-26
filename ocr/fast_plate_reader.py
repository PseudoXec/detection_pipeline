"""
fast_plate_reader.py
---------------------
The pipeline's only OCR backend, using the `fast-plate-ocr` package (MIT
licensed, free, pip-installable: https://github.com/ankandrew/fast-plate-ocr).

PaddleOCR has been removed from this project entirely (it's no longer an
installed dependency, a config option, or reachable through any code path).
fast-plate-ocr is a single small ONNX classifier purpose-built for plate
reading: one plate crop in, one plate string out, no text-detection stage,
no document-orientation classifier - several times less work than a full
document-OCR engine like PaddleOCR needed, and typically 5-10x faster per
read on a Pi 5's CPU.

The trade-off to know about
----------------------------
fast-plate-ocr's hub models assume ONE line of text (they classify a fixed
number of character "slots" in left-to-right order). A 2-row/stacked plate
(common on motorcycles) will be read as both rows concatenated in whatever
order the crop presents them, which is usually wrong. There is currently no
built-in fallback for this in the pipeline - if your camera sees a
meaningful number of 2-row plates, this is a known accuracy gap to plan
around (e.g. flagging low-confidence reads for manual review) rather than
something a config toggle can fix.

Setup
-----
    pip install "fast-plate-ocr[onnx]"

The first call downloads the chosen hub model (a few MB) to a local cache;
after that it runs fully offline. See config.yaml `ocr.fast_plate_ocr_model`
for which hub model is used - `cct-xs-v2-global-model` (smallest/fastest) is
the default; `cct-s-v2-global-model` trades a bit of speed for accuracy.

Interface
---------
This class exposes a `read_scored(plate_crop, *alternates)` method and a
`_load_failed` attribute, which is what pipeline.py expects (see
ocr/__init__.py's build_ocr_reader()).
"""

import logging
import re
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("fast_plate_reader")

_PLAUSIBLE_LENGTH = (5, 8)
_IMPLAUSIBLE_PENALTY = 0.85


class FastPlateOCRReader:
    """Wraps fast-plate-ocr's ONNX inference for a small plate crop. Never
    raises - any failure (missing dependency, bad crop, model error) is
    caught and surfaces as a `None` read instead."""

    def __init__(
        self,
        hub_model: str = "cct-xs-v2-global-model",
        min_confidence: float = 0.5,
        allowed_chars: Optional[str] = None,
        early_exit_score: float = 0.85,
        cpu_threads: int = 1,
    ):
        self.hub_model = hub_model
        self.min_confidence = min_confidence
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.early_exit_score = early_exit_score
        self.cpu_threads = cpu_threads
        self._engine = None
        self._load_failed = False

    # ------------------------------------------------------------------ #
    # engine loading
    # ------------------------------------------------------------------ #
    def _get_engine(self):
        if self._engine is not None or self._load_failed:
            return self._engine
        try:
            # imported here (not at module load) so the dependency is only
            # paid for when a reader is actually built (features.ocr_read: true)
            import onnxruntime as ort
            from fast_plate_ocr import LicensePlateRecognizer
        except Exception as error:  # pragma: no cover - environment dependent
            log.error(
                "FastPlateOCRReader: fast-plate-ocr is not installed (pip install "
                "'fast-plate-ocr[onnx]') - plate reads will be Unrecognized: %s", error,
            )
            self._load_failed = True
            return None

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = max(1, self.cpu_threads)
            try:
                # same reasoning as detector.limit_onnx_threads: an idle worker
                # thread otherwise busy-waits, burning a core we want free
                sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass
            self._engine = LicensePlateRecognizer(
                hub_ocr_model=self.hub_model,
                device="cpu",
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            )
            log.info("FastPlateOCRReader: engine ready (model=%s, cpu_threads=%d)",
                     self.hub_model, self.cpu_threads)
            return self._engine
        except Exception as error:
            log.error("FastPlateOCRReader: failed to load hub model %s - plate reads will "
                      "be Unrecognized: %s", self.hub_model, error, exc_info=True)
            self._load_failed = True
            self._engine = None
            return None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def read(self, plate_crop: Optional[np.ndarray]) -> Optional[str]:
        text, _score = self.read_scored(plate_crop)
        return text

    def read_scored(
        self, plate_crop: Optional[np.ndarray], *alternates: Optional[np.ndarray],
    ) -> Tuple[Optional[str], float]:
        """Tries each image variant
        (the enhanced crop, then any alternates like the un-enhanced one) and
        keeps the best-scoring read, stopping early once one is good enough."""
        images = [image for image in (plate_crop, *alternates) if image is not None and image.size > 0]
        if not images:
            return None, 0.0

        engine = self._get_engine()
        if engine is None:
            return None, 0.0

        best_text: Optional[str] = None
        best_score = 0.0
        for variant in images:
            try:
                prepared = self._prepare(engine, variant)
                prediction = engine.run_one(prepared, return_confidence=True)
            except Exception as error:
                log.warning("FastPlateOCRReader: OCR call failed: %s", error)
                continue
            text, score = self._clean_and_score(prediction)
            if text and score > best_score:
                best_text, best_score = text, score
            if best_text and best_score >= self.early_exit_score:
                break
        return best_text, best_score

    # ------------------------------------------------------------------ #
    # image prep + result handling
    # ------------------------------------------------------------------ #
    def _prepare(self, engine, image: np.ndarray) -> np.ndarray:
        """fast-plate-ocr expects the color mode its model was trained with
        (almost always grayscale) - our crops come out of OpenCV as BGR."""
        import cv2

        mode = getattr(engine.config, "image_color_mode", "grayscale")
        if image.ndim == 3 and image.shape[2] == 3:
            if mode == "grayscale":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _clean(self, text: str) -> str:
        text = text.upper().strip()
        if self.allowed_chars:
            text = "".join(ch for ch in text if ch in self.allowed_chars)
        else:
            text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _clean_and_score(self, prediction) -> Tuple[Optional[str], float]:
        text = self._clean(prediction.plate or "")
        if not text:
            return None, 0.0
        # mean per-character confidence when the model returned it, else a
        # neutral mid-point score (fast-plate-ocr doesn't reject low-confidence
        # chars itself - min_confidence/accept_score downstream decide)
        if prediction.char_probs is not None and len(prediction.char_probs):
            confidence = float(np.mean(prediction.char_probs))
        else:
            confidence = 0.6
        if confidence < self.min_confidence:
            return None, 0.0
        low, high = _PLAUSIBLE_LENGTH
        return text, confidence * (1.0 if low <= len(text) <= high else _IMPLAUSIBLE_PENALTY)
