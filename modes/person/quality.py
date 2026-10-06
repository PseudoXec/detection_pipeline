"""Face quality: how sharp, how large, how sure. Pure functions, no state."""
import cv2
import numpy as np


def sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image. Higher = sharper. 0 for an empty image."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def passes_gate(min_side_px: float, face_sharpness: float, min_face_px: int, min_sharpness: float) -> bool:
    return min_side_px >= min_face_px and face_sharpness >= min_sharpness


def quality_score(detector_score: float, min_side_px: float, face_sharpness: float,
                  ref_px: float, ref_sharpness: float) -> float:
    """detector score x size factor x sharpness factor, each capped at 1. Monotonic in all three."""
    return float(detector_score) * clamp01(min_side_px / max(ref_px, 1.0)) * clamp01(face_sharpness / max(ref_sharpness, 1e-6))
