"""
image_enhance.py
-----------------
Optional pre-OCR cleanup for a single plate crop: upscale it if it's
smaller than the configured minimum height, run a CLAHE contrast boost,
then a light unsharp-mask sharpen. Used by both run modes
(server/server.py and cli/ocr_pipeline.py) when config.preprocess.enhance
is true (see config/ocr_config.py) - each does:

    enhance_plate_crop(image, config.preprocess.min_crop_height)

and, if config.preprocess.try_both_variants is also true, OCRs both the
raw crop and this enhanced version, keeping whichever PlateOCRReader
scores higher (some plates read better raw; low-contrast/small ones
usually read better enhanced).

Input/output: a BGR uint8 image (as decoded by cv2.imdecode/cv2.imread),
same shape and channel count out as in. Never raises - a crop that can't
be enhanced for any reason comes back as close to the original as we got
before the failure, rather than crashing the read (matching this
project's "OCR never crashes on a bad image" rule - see README's Notes
section).
"""

import logging

import cv2
import numpy as np

log = logging.getLogger("ocr_pipeline.image_enhance")


def _upscale_if_small(image: np.ndarray, min_crop_height: int) -> np.ndarray:
    """Upscales (preserving aspect ratio) if the crop is shorter than
    min_crop_height. Small crops lose too much character detail if OCR'd
    at native resolution. Downscaling never happens here - only crops
    below the threshold are touched."""
    height, width = image.shape[:2]
    if min_crop_height <= 0 or height >= min_crop_height or height <= 0:
        return image
    scale = min_crop_height / float(height)
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (new_width, min_crop_height), interpolation=cv2.INTER_CUBIC)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the luminance
    channel only (via LAB), so colour isn't distorted the way a naive
    global cv2.equalizeHist() on each channel independently would be.
    Helps low-contrast crops (backlit plates, faded paint, hazy footage)
    without blowing out already-decent ones the way global stretching can."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _unsharp_mask(image: np.ndarray, amount: float = 0.6, blur_ksize: int = 3) -> np.ndarray:
    """Light unsharp mask (adds back amount * (original - blurred)) rather
    than a raw sharpening kernel, since a raw kernel amplifies the
    JPEG/sensor noise small, upscaled plate crops tend to have."""
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def enhance_plate_crop(image: np.ndarray, min_crop_height: int = 64) -> np.ndarray:
    """Runs the full pre-OCR cleanup pass on one plate crop: upscale (if
    needed) -> CLAHE -> unsharp mask. See module docstring for how the two
    run modes use this. Never raises - each step falls back to its input
    unchanged on failure, and the furthest-along successful result is
    always what's returned."""
    if image is None or image.size == 0:
        return image

    working = image

    try:
        working = _upscale_if_small(working, min_crop_height)
    except Exception:
        log.warning("enhance_plate_crop: upscale step failed, using prior image", exc_info=True)

    try:
        working = _apply_clahe(working)
    except Exception:
        log.warning("enhance_plate_crop: CLAHE step failed, using prior image", exc_info=True)

    try:
        working = _unsharp_mask(working)
    except Exception:
        log.warning("enhance_plate_crop: sharpen step failed, using prior image", exc_info=True)

    return working
