import logging
import re
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("fast_plate_reader")

_PLAUSIBLE_LENGTH = (5, 8)
_IMPLAUSIBLE_PENALTY = 0.85


class FastPlateOCRReader:
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

    def _get_engine(self):
        if self._engine is not None or self._load_failed:
            return self._engine
        try:
            import onnxruntime as ort
            from fast_plate_ocr import LicensePlateRecognizer
        except Exception as error:
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

    def read(self, plate_crop: Optional[np.ndarray]) -> Optional[str]:
        text, _score = self.read_scored(plate_crop)
        return text

    def read_scored(
        self, plate_crop: Optional[np.ndarray], *alternates: Optional[np.ndarray],
    ) -> Tuple[Optional[str], float]:
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

    def _prepare(self, engine, image: np.ndarray) -> np.ndarray:
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
        if prediction.char_probs is not None and len(prediction.char_probs):
            confidence = float(np.mean(prediction.char_probs))
        else:
            confidence = 0.6
        if confidence < self.min_confidence:
            return None, 0.0
        low, high = _PLAUSIBLE_LENGTH
        return text, confidence * (1.0 if low <= len(text) <= high else _IMPLAUSIBLE_PENALTY)
