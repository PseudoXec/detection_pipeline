from modes.vehicle.config import OcrConfig
from modes.vehicle.ocr.fast_plate_reader import FastPlateOCRReader

__all__ = ["FastPlateOCRReader", "build_ocr_reader"]


def build_ocr_reader(ocr_cfg: OcrConfig) -> FastPlateOCRReader:
    return FastPlateOCRReader(
        hub_model=ocr_cfg.fast_plate_ocr_model,
        min_confidence=ocr_cfg.min_confidence,
        allowed_chars=ocr_cfg.allowed_chars,
        early_exit_score=ocr_cfg.early_exit_score,
        cpu_threads=ocr_cfg.cpu_threads,
    )
