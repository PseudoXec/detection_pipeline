import cv2
import numpy as np


def sharpen_and_denoise(image: np.ndarray, clahe_clip: float = 3.0, sharpen_amount: float = 0.5) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1 + sharpen_amount, blurred, -sharpen_amount, 0)

    return sharpened


def enhance_plate_crop(plate_crop: np.ndarray, min_height: int = 64) -> np.ndarray:
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    image = plate_crop.copy()
    height, width = image.shape[:2]

    if height < min_height:
        scale = min_height / max(1, height)
        image = cv2.resize(image, (max(1, int(width * scale)), min_height), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(enhanced, 5, 30, 30)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)

    return sharpened


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    return buffer.tobytes()
