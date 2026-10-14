"""Face detection for the person mode.

YuNet (OpenCV's `cv2.FaceDetectorYN`) looks for a face in the upper part of a person crop.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from modes.person import quality

log = logging.getLogger("pipeline")


@dataclass
class FaceHit:
    box: Tuple[float, float, float, float]            # x1, y1, x2, y2 in PERSON-CROP pixels
    landmarks: List[Tuple[float, float]]              # 5 points, PERSON-CROP pixels
    score: float                                      # detector confidence
    sharpness: float                                  # Laplacian variance of the face region
    quality: float                                    # combined score, see quality.quality_score
    passed_gate: bool                                 # big enough and sharp enough to keep


class YuNetFaceDetector:
    """Wrapper that turns a person crop into the best FaceHit (or None)."""

    def __init__(self, weights: str, score_threshold: float, search_region: List[float],
                 min_face_px: int, min_sharpness: float, ref_px: float = 64.0, ref_sharpness: float = 80.0):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Face model not found at: {weights}\n"
                f"Download face_detection_yunet_2023mar.onnx into models/, "
                f"or set person.face_enabled: false.")
        import cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ImportError("this OpenCV build has no FaceDetectorYN - install opencv-python >= 4.8")
        self._detector = cv2.FaceDetectorYN.create(weights, "", (320, 320), score_threshold)
        self.search_region = search_region
        self.min_face_px = min_face_px
        self.min_sharpness = min_sharpness
        self.ref_px = ref_px
        self.ref_sharpness = ref_sharpness

    def detect(self, person_crop: np.ndarray) -> Optional[FaceHit]:
        if person_crop is None or person_crop.size == 0:
            return None
        height, width = person_crop.shape[:2]
        fx0, fy0, fx1, fy1 = self.search_region
        x0, y0 = max(0, int(width * fx0)), max(0, int(height * fy0))
        x1, y1 = min(width, int(width * fx1)), min(height, int(height * fy1))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None

        region = np.ascontiguousarray(person_crop[y0:y1, x0:x1])
        self._detector.setInputSize((region.shape[1], region.shape[0]))
        _, faces = self._detector.detect(region)
        if faces is None or len(faces) == 0:
            return None

        # rows: x, y, w, h, 5 landmark (x, y) pairs, score. Keep the most confident face.
        row = max(faces, key=lambda r: float(r[14]))
        fx, fy, fw, fh = (float(v) for v in row[:4])
        box = (max(0.0, fx + x0), max(0.0, fy + y0), min(float(width), fx + fw + x0), min(float(height), fy + fh + y0))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        landmarks = [(float(row[4 + 2 * i]) + x0, float(row[5 + 2 * i]) + y0) for i in range(5)]

        face_region = person_crop[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face_sharpness = quality.sharpness(face_region)
        min_side = min(box[2] - box[0], box[3] - box[1])
        score = float(row[14])
        return FaceHit(
            box=box, landmarks=landmarks, score=score, sharpness=face_sharpness,
            quality=quality.quality_score(score, min_side, face_sharpness, self.ref_px, self.ref_sharpness),
            passed_gate=quality.passes_gate(min_side, face_sharpness, self.min_face_px, self.min_sharpness),
        )
