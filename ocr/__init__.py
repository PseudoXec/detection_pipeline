"""
Picks which OCR backend the pipeline uses, based on `config.ocr.engine`:

    "fast_plate_ocr" (default) - a small, single-purpose ONNX plate classifier.
        Much faster on a Pi 5's CPU than PaddleOCR (no text-detection stage),
        but assumes a single line of text - see fast_plate_reader.py's module
        docstring for the 2-row/motorcycle-plate caveat before relying on it
        for those.
    "paddleocr" - the original full det+rec engine. Slower per read, but
        handles multi-row plates via its own fragment-reordering logic.

Both classes expose the same `read_scored(plate_crop, *alternates)` /
`_load_failed` interface, so pipeline.py never needs to know which one it got.
"""

import logging

from config.config import OcrConfig
from ocr.plate_reader import PlateOCRReader
from ocr.fast_plate_reader import FastPlateOCRReader

log = logging.getLogger("pipeline")

__all__ = ["PlateOCRReader", "FastPlateOCRReader", "build_ocr_reader"]


def build_ocr_reader(ocr_cfg: OcrConfig):
    """Returns a ready-to-use (lazily-loaded) OCR reader instance for
    `ocr_cfg.engine`, or falls back to PaddleOCR with a warning on an
    unrecognized value rather than crashing startup over a typo."""
    engine = (ocr_cfg.engine or "fast_plate_ocr").strip().lower()

    if engine == "fast_plate_ocr":
        return FastPlateOCRReader(
            hub_model=ocr_cfg.fast_plate_ocr_model,
            min_confidence=ocr_cfg.min_confidence,
            allowed_chars=ocr_cfg.allowed_chars,
            early_exit_score=ocr_cfg.early_exit_score,
            cpu_threads=ocr_cfg.cpu_threads,
        )

    if engine != "paddleocr":
        log.warning("[ocr] unknown ocr.engine %r - falling back to paddleocr", ocr_cfg.engine)

    return PlateOCRReader(
        lang=ocr_cfg.lang,
        min_confidence=ocr_cfg.min_confidence,
        allowed_chars=ocr_cfg.allowed_chars,
        early_exit_score=ocr_cfg.early_exit_score,
        cpu_threads=ocr_cfg.cpu_threads,
    )
