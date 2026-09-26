"""
Builds the plate OCR reader.

PaddleOCR support has been removed - fast_plate_ocr (a small, single-purpose
ONNX plate classifier) is the only backend. It's much faster on a Pi 5's
CPU than PaddleOCR was (no text-detection stage), but assumes a single line
of text - see fast_plate_reader.py's module docstring for the 2-row/
motorcycle-plate caveat.
"""

from config.config import OcrConfig
from ocr.fast_plate_reader import FastPlateOCRReader

__all__ = ["FastPlateOCRReader", "build_ocr_reader"]


def build_ocr_reader(ocr_cfg: OcrConfig) -> FastPlateOCRReader:
    """Returns a ready-to-use (lazily-loaded) FastPlateOCRReader instance."""
    return FastPlateOCRReader(
        hub_model=ocr_cfg.fast_plate_ocr_model,
        min_confidence=ocr_cfg.min_confidence,
        allowed_chars=ocr_cfg.allowed_chars,
        early_exit_score=ocr_cfg.early_exit_score,
        cpu_threads=ocr_cfg.cpu_threads,
    )
