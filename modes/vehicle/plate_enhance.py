"""Plate-image enhancement (vehicle mode only): sharpen/denoise before plate detection and before OCR."""
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


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _richardson_lucy(channel: np.ndarray, sigma: float, iterations: int) -> np.ndarray:
    """Light Lucy-Richardson deblur (Gaussian PSF) on one 8-bit channel. Small crops -> a few ms."""
    observed = channel.astype(np.float32) / 255.0 + 1e-4
    estimate = observed.copy()
    for _ in range(iterations):
        reblurred = cv2.GaussianBlur(estimate, (0, 0), sigma) + 1e-4
        # the Gaussian PSF is symmetric, so its flipped copy is itself
        estimate *= cv2.GaussianBlur(observed / reblurred, (0, 0), sigma)
    return np.clip((estimate - 1e-4) * 255.0, 0, 255).astype(np.uint8)


def enhance_plate_crop(plate_crop: np.ndarray, min_height: int = 64) -> np.ndarray:
    """Make a plate crop clearer for reading: upscale, denoise, deblur (only if soft), local contrast, sharpen."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    image = plate_crop.copy()
    height, width = image.shape[:2]

    # 1) upscale small crops (Lanczos keeps character edges cleaner than cubic)
    target_height = max(min_height, 96)
    if height < target_height:
        scale = target_height / max(1, height)
        image = cv2.resize(image, (max(1, int(round(width * scale))), target_height), interpolation=cv2.INTER_LANCZOS4)

    # 2) gentle denoise first - deblurring amplifies noise
    image = cv2.bilateralFilter(image, 5, 25, 25)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    # 3) deblur only crops that are actually soft (already-sharp ones would just get ringing)
    softness = _sharpness(lightness)
    if softness < 600.0:
        sigma = 1.6 if softness < 150.0 else 1.2
        lightness = _richardson_lucy(lightness, sigma, iterations=8)

    # 4) local contrast
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    # 5) final light unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
