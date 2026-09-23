"""
image_ops.py
------------
Image enhancement only. No detection, no I/O beyond receiving/returning a
numpy array. Kept separate so it can be tuned (or swapped out) without
touching the detection/tracking/storage logic.
"""

# cv2 provides every enhancement operation used below (CLAHE, blur, resize)
import cv2
# numpy arrays are how images are represented everywhere in this project
import numpy as np


def sharpen_and_denoise(image: np.ndarray, clahe_clip: float = 3.0, sharpen_amount: float = 0.5) -> np.ndarray:
    """General-purpose contrast/denoise/sharpen pass.

    Used on vehicle crops right before plate detection - a bit of contrast
    boost measurably helps the plate model find small, low-contrast plates.
    """
    if image is None or image.size == 0:
        return image

    # LAB color space separates brightness (L) from color (A/B); enhancing
    # only L avoids introducing color casts into the plate/vehicle image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    # CLAHE = local (tile-based) contrast enhancement, better than a single
    # global contrast stretch because lighting is rarely uniform across a frame
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    # merge the enhanced brightness channel back with the original color
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    # bilateral filter removes sensor noise while keeping edges sharp
    # (a normal Gaussian blur would also blur the plate character edges)
    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)

    # unsharp mask: blur the image, then subtract the blur from the original
    # to emphasize edges - this is what makes plate characters look crisper
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1 + sharpen_amount, blurred, -sharpen_amount, 0)

    return sharpened


def enhance_plate_crop(plate_crop: np.ndarray, min_height: int = 64) -> np.ndarray:
    """Final polish applied to a plate crop right before it's saved/stored.

    Slightly gentler than sharpen_and_denoise (a plate crop is small and
    over-sharpening it destroys character strokes instead of clarifying them).
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    image = plate_crop.copy()
    height, width = image.shape[:2]

    # plates captured from a distance can be only a few pixels tall - upscale
    # them first so every following step has more pixels to work with
    if height < min_height:
        scale = min_height / max(1, height)
        image = cv2.resize(image, (max(1, int(width * scale)), min_height), interpolation=cv2.INTER_CUBIC)

    # same CLAHE-on-luminance trick as sharpen_and_denoise, slightly gentler clip
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    # light denoise + light sharpen - the goal here is legibility, not drama
    denoised = cv2.bilateralFilter(enhanced, 5, 30, 30)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)

    return sharpened


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a numpy image to JPEG bytes, ready to store in SQLite or on disk."""
    # cv2.imencode returns (success_flag, numpy_buffer); we only need the bytes
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    # .tobytes() turns the numpy buffer into a plain Python bytes object
    return buffer.tobytes()
