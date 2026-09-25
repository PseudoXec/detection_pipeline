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

Why reads used to come out "reversed" (right characters, wrong order):
  1. PaddleOCR 3.x switches on a *document* orientation classifier and a
     document unwarping model by default. A plate is not a document: the
     classifier can rotate the crop 180 degrees, which flips the left-to-right
     order of the text boxes while the per-line classifier keeps every
     fragment's own characters upright. Both models are now switched off.
  2. Fragments (e.g. "850" + "V21", or the two rows of a stacked plate) were
     simply joined in whatever order the detector returned them. They are now
     sorted into reading order from their box positions: rows top-to-bottom,
     fragments inside a row left-to-right.
"""

import logging
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import Iterator, List, Optional, Tuple

import cv2
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

# a read whose length falls outside this range is usually a partial / noisy read,
# so it ranks lower when several image variants are compared (it is not rejected)
_PLAUSIBLE_LENGTH = (5, 8)
_IMPLAUSIBLE_PENALTY = 0.85


@dataclass
class _Fragment:
    """One recognized text box: its cleaned text, confidence and where it sat in the image."""

    text: str
    conf: float
    cx: Optional[float] = None      # box centre x / y and height, in image pixels
    cy: Optional[float] = None
    height: Optional[float] = None


class PlateOCRReader:
    """Wraps PaddleOCR's rec+det pipeline to read text off a small plate
    crop. Never raises - any failure (missing dependency, bad crop, model
    error) is caught and surfaces as a `None` read, which the caller turns
    into the configured "Unrecognized" text."""

    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.5,
        allowed_chars: Optional[str] = None,
        early_exit_score: float = 0.85,
        cpu_threads: int = 1,
    ):
        self.lang = lang
        self.min_confidence = min_confidence
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        # once one image variant reads at or above this score, the remaining variants are skipped
        self.early_exit_score = early_exit_score
        # Paddle otherwise defaults to using every core it can see for its internal
        # OpenMP/MKL math ops - on a Pi 5 that fights the vehicle/plate model threads
        # even though OCR runs on its own Python thread, because the contention is
        # for CPU cores, not the GIL. Capped both ways below: the env vars affect
        # Paddle's own thread pools, `cpu_threads` is PaddleOCR's own kwarg for it.
        self.cpu_threads = max(1, cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.cpu_threads))
        self._engine = None
        self._engine_api = "ocr"
        self._load_failed = False

    # ------------------------------------------------------------------ #
    # engine loading
    # ------------------------------------------------------------------ #
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
        #
        # The FIRST attempt also switches off the document orientation
        # classifier and the document unwarping model (3.x only): they are built
        # for scanned pages, and on a small plate crop they can rotate the image
        # 180 degrees or bend the characters, which is what produced reversed reads.
        attempts = (
            {"use_doc_orientation_classify": False, "use_doc_unwarping": False,
             "use_textline_orientation": True, "lang": self.lang, "enable_mkldnn": False,
             "cpu_threads": self.cpu_threads},
            {"use_doc_orientation_classify": False, "use_doc_unwarping": False,
             "use_textline_orientation": True, "lang": self.lang, "cpu_threads": self.cpu_threads},
            {"use_textline_orientation": True, "lang": self.lang, "enable_mkldnn": False,
             "cpu_threads": self.cpu_threads},
            {"use_textline_orientation": True, "lang": self.lang, "cpu_threads": self.cpu_threads},
            {"use_angle_cls": True, "lang": self.lang, "show_log": False, "cpu_threads": self.cpu_threads},
            {"lang": self.lang, "cpu_threads": self.cpu_threads},
            {"lang": self.lang},   # last-ditch: some point releases reject cpu_threads entirely
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
                fragments = self._run_ocr(engine, variant)
            except Exception as error:
                log.warning("PlateOCRReader: OCR call failed: %s", error)
                continue
            text, score = self._assemble(fragments)
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
        """Yield each image with a border added. PaddleOCR's text detector tends to
        drop or clip characters that touch the image edge, and a tight plate crop
        has letters right at the edge; replicating the edge pixels outward fixes
        that without adding fake content."""
        for image in images:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            pad = max(6, int(round(image.shape[0] * 0.10)))
            yield cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

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

    def _run_ocr(self, engine, plate_crop: np.ndarray) -> List[_Fragment]:
        """Runs OCR and returns the fragments above min_confidence (with their box
        positions when available), handling both the PaddleOCR 2.x `.ocr()` return
        shape and the PaddleOCR 3.x `.predict()` return shape."""
        fragments: List[_Fragment] = []

        if self._engine_api == "predict":
            # PaddleOCR 3.x: predict() returns a list of result objects, each dict-like
            # with 'rec_texts' / 'rec_scores' and box lists ('rec_polys' / 'rec_boxes')
            # aligned index-for-index with the texts
            for res in engine.predict(plate_crop):
                data = res if isinstance(res, dict) else getattr(res, "json", None) or {}
                if isinstance(data, dict) and isinstance(data.get("res"), dict):
                    data = data["res"]         # some builds nest everything under "res"
                texts = data.get("rec_texts")
                scores = data.get("rec_scores")
                texts = [] if texts is None else list(texts)
                scores = [] if scores is None else list(scores)
                geometry = self._geometry_from(data, len(texts))
                for index, (text, conf) in enumerate(zip(texts, scores)):
                    if conf < self.min_confidence:
                        continue
                    cleaned = self._clean(str(text))
                    if cleaned:
                        cx, cy, height = geometry[index] if geometry else (None, None, None)
                        fragments.append(_Fragment(cleaned, float(conf), cx, cy, height))
            return fragments

        # PaddleOCR 2.x: ocr() returns [[ [box, (text, confidence)], ... ]] -
        # one list per input image
        try:
            result = engine.ocr(plate_crop, cls=True)
        except TypeError:
            # some 2.x point releases dropped the `cls` kwarg
            result = engine.ocr(plate_crop)
        lines = result[0] if result else None
        for box, (text, conf) in lines or []:
            if conf < self.min_confidence:
                continue
            cleaned = self._clean(str(text))
            if cleaned:
                cx, cy, height = self._poly_geometry(box)
                fragments.append(_Fragment(cleaned, float(conf), cx, cy, height))
        return fragments

    @staticmethod
    def _poly_geometry(poly) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            points = np.asarray(poly, dtype=float).reshape(-1, 2)
            return float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 1].max() - points[:, 1].min())
        except Exception:
            return None, None, None

    @classmethod
    def _geometry_from(cls, data: dict, count: int):
        """Per-fragment (cx, cy, height) from a 3.x result, or None when the result
        has no boxes lined up with the texts (then the original order is kept)."""
        if not count:
            return None
        boxes = data.get("rec_boxes")
        if boxes is not None and len(boxes) == count:
            try:
                arr = np.asarray(boxes, dtype=float).reshape(count, 4)          # x1, y1, x2, y2
                return [((x1 + x2) / 2, (y1 + y2) / 2, abs(y2 - y1)) for x1, y1, x2, y2 in arr]
            except Exception:
                pass
        polys = data.get("rec_polys")
        if polys is not None and len(polys) == count:
            return [cls._poly_geometry(poly) for poly in polys]
        return None

    @staticmethod
    def _reading_order(fragments: List[_Fragment]) -> List[_Fragment]:
        """Rows top-to-bottom, fragments inside a row left-to-right. Without box
        positions the detector's own order is kept."""
        if len(fragments) < 2 or any(f.cx is None or f.cy is None for f in fragments):
            return fragments

        heights = [f.height for f in fragments if f.height]
        row_tolerance = 0.6 * (median(heights) if heights else 0.0)

        rows: List[List[_Fragment]] = []
        for fragment in sorted(fragments, key=lambda f: f.cy):
            if rows and abs(fragment.cy - sum(f.cy for f in rows[-1]) / len(rows[-1])) <= row_tolerance:
                rows[-1].append(fragment)
            else:
                rows.append([fragment])
        return [f for row in rows for f in sorted(row, key=lambda f: f.cx)]

    def _assemble(self, fragments: List[_Fragment]) -> Tuple[Optional[str], float]:
        """Fragments -> (plate text in reading order, quality score)."""
        if not fragments:
            return None, 0.0
        ordered = self._reading_order(fragments)
        text = "".join(f.text for f in ordered)
        if not text:
            return None, 0.0
        total_chars = sum(len(f.text) for f in ordered)
        mean_conf = sum(f.conf * len(f.text) for f in ordered) / total_chars     # longer fragments count more
        low, high = _PLAUSIBLE_LENGTH
        return text, mean_conf * (1.0 if low <= len(text) <= high else _IMPLAUSIBLE_PENALTY)
