import cv2
import numpy as np


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    return buffer.tobytes()
