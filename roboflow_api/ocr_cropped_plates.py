import argparse
import csv
import os

import cv2
from paddleocr import PaddleOCR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_HEIGHT = 64  # upscale anything shorter than this


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _upscale(img, min_height: int):
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, (int(w * scale), min_height), interpolation=cv2.INTER_CUBIC)
    return img


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _gentle_sharpen(img):
    # Unsharp mask instead of a hard [-1,5,-1] kernel - much less prone to
    # ringing/haloing around character edges, which is what tends to flip a
    # recognizer on a plate that a human reads just fine.
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, 1.4, blurred, -0.4, 0)


def generate_ocr_variants(image_path: str, min_height: int = MIN_HEIGHT):
    """
    Builds several differently-processed versions of the same crop instead
    of betting everything on one fixed pipeline. Deep-learning OCR models
    can be surprisingly sensitive to preprocessing choices that look
    harmless to a human eye (denoise strength, sharpening, contrast), so a
    crop that's perfectly legible to a person can still fail on one
    specific variant while succeeding on another.

    Returns a list of (variant_name, BGR numpy array) tuples, or an empty
    list if the image couldn't be read. Callers should run OCR on each and
    keep whichever gives the highest confidence.

    Variants:
      - "raw":   just upscaled, nothing else. Best when the source photo is
                 already clear - CLAHE/denoise/sharpen can sometimes hurt an
                 already-good image more than they help.
      - "light": upscaled + CLAHE contrast only. Helps glare/shadow without
                 the risk sharpening adds.
      - "full":  upscaled + denoise + CLAHE + a gentle unsharp mask. Helps
                 soft/blurry crops.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    raw = _upscale(img.copy(), min_height)
    light = _clahe(raw.copy())

    full = cv2.fastNlMeansDenoisingColored(raw.copy(), None, h=3, hColor=3,
                                            templateWindowSize=7, searchWindowSize=21)
    full = _clahe(full)
    full = _gentle_sharpen(full)

    return [("raw", raw), ("light", light), ("full", full)]


def preprocess_for_ocr(image_path: str, min_height: int = MIN_HEIGHT):
    """
    Kept for backward compatibility / --save-preprocessed debugging: returns
    just the "full" variant (upscale + denoise + CLAHE + gentle sharpen).
    The main pipeline now uses generate_ocr_variants() instead so it isn't
    committed to a single preprocessing choice.
    """
    variants = generate_ocr_variants(image_path, min_height)
    if not variants:
        return None
    return dict(variants)["full"]


# --------------------------------------------------------------------------
# Filename helpers
# --------------------------------------------------------------------------
def sanitize_for_filename(text: str, fallback: str = "unrecognized") -> str:
    """
    Turns recognized OCR text into something safe to use as a filename:
    keeps only alnum characters and uppercases (plates are conventionally
    shown uppercase). Falls back to a placeholder if OCR found nothing.
    """
    cleaned = "".join(c for c in text if c.isalnum()).upper()
    return cleaned if cleaned else fallback


def unique_destination(directory: str, base_name: str, ext: str) -> str:
    """
    Builds a full path for `base_name + ext` inside `directory`, appending
    _2, _3, ... if that name is already taken (e.g. two plates both read as
    the same text, or a rename collides with an existing file).
    """
    candidate = os.path.join(directory, f"{base_name}{ext}")
    if not os.path.exists(candidate):
        return candidate

    counter = 2
    while True:
        candidate = os.path.join(directory, f"{base_name}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
MODEL_TIERS = {
    # lang="en" looks like the obvious choice but is a trap: PaddleOCR 3.x
    # routes it to en_PP-OCRv5_mobile_rec, a lightweight model optimized for
    # speed, not accuracy. For plates - short, high-stakes alphanumeric
    # strings - the larger server models read markedly better.
    "server": dict(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
    ),
    "mobile-en": dict(lang="en"),
    # Uses whatever PaddleOCR ships as its own default for the installed
    # version (PP-OCRv6_medium on newer releases) instead of pinning names.
    "default": dict(),
}


def build_ocr_engine(model_tier: str = "server") -> PaddleOCR:
    """
    Plate crops are already tightly cropped single objects, so we skip the
    document-oriented preprocessing PaddleOCR normally does (orientation
    classification, unwarping) - that's built for full-page scans, not small
    crops, and just adds latency here. Text-line orientation is kept on
    since plates can be photographed at an angle.

    model_tier controls which recognition/detection models are loaded - see
    MODEL_TIERS. "server" is the default because it's meaningfully more
    accurate than the mobile English model for short alphanumeric strings
    like plates, at the cost of a slightly larger download and slower
    inference per image.
    """
    if model_tier not in MODEL_TIERS:
        raise ValueError(f"Unknown model_tier {model_tier!r}; choose from {sorted(MODEL_TIERS)}")

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        # Works around a known PaddlePaddle 3.3.x bug on Windows/CPU where the
        # default MKL-DNN (oneDNN) backend throws:
        #   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
        #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
        # See: https://github.com/PaddlePaddle/Paddle/issues/77340
        # Slightly slower, but avoids the crash. Remove this once Paddle fixes it upstream.
        enable_mkldnn=False,
        **MODEL_TIERS[model_tier],
    )


def read_plate_text(ocr_engine: PaddleOCR, preprocessed_img) -> tuple[str, float]:
    """
    Runs PaddleOCR on a preprocessed BGR numpy array and returns
    (joined_text, mean_confidence). Concatenates multiple detected text
    lines with a space, since a plate can sometimes be split into two lines
    by the recognizer.
    """
    results = ocr_engine.predict(preprocessed_img)
    if not results:
        return "", 0.0

    res = results[0]
    texts = res.get("rec_texts", []) if hasattr(res, "get") else getattr(res, "rec_texts", [])
    scores = res.get("rec_scores", []) if hasattr(res, "get") else getattr(res, "rec_scores", [])

    if not texts:
        return "", 0.0

    joined_text = " ".join(texts)
    mean_conf = sum(scores) / len(scores) if scores else 0.0
    return joined_text, mean_conf


def read_plate_text_best(ocr_engine: PaddleOCR, variants) -> tuple[str, float, str]:
    """
    Runs OCR on every (variant_name, image) pair from generate_ocr_variants()
    and keeps whichever gives the highest confidence. Returns
    (best_text, best_confidence, winning_variant_name). If every variant
    comes back empty, returns ("", 0.0, "none").
    """
    best_text, best_conf, best_name = "", 0.0, "none"
    for name, img in variants:
        text, conf = read_plate_text(ocr_engine, img)
        if text and conf > best_conf:
            best_text, best_conf, best_name = text, conf, name
    return best_text, best_conf, best_name


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and OCR every cropped plate image in a folder using PaddleOCR."
    )
    parser.add_argument("--input-dir", required=True, help="Folder of cropped detection images.")
    parser.add_argument("--output", default=None,
                         help="CSV file to write the scan list to. Defaults to "
                              "'ocr_scan_log.csv' inside --input-dir.")
    parser.add_argument("--no-rename", action="store_true",
                         help="Don't rename the source images; just report recognized text.")
    parser.add_argument("--save-preprocessed", action="store_true",
                         help="Also save the preprocessed images (for debugging) into <input-dir>_preprocessed.")
    parser.add_argument("--model-tier", choices=sorted(MODEL_TIERS), default="server",
                         help="Which PaddleOCR models to load. 'server' (default) is the most accurate "
                              "for short alphanumeric strings like plates. 'mobile-en' is the old, faster "
                              "but less accurate default. 'default' uses whatever PaddleOCR ships as its "
                              "own default for the installed version.")
    parser.add_argument("--single-pass", action="store_true",
                         help="Run OCR once using only the 'full' preprocessing pipeline instead of "
                              "trying raw/light/full variants and keeping the best. Faster, but less "
                              "accurate on crops where heavier preprocessing hurts more than it helps.")
    parser.add_argument("--min-confidence", type=float, default=0.80,
                         help="Confidence (0-1) below which a result is flagged 'LOWCONF_' in its "
                              "filename/log instead of trusted outright, so it's easy to spot for manual "
                              "review. Default 0.80.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Error: folder not found: {args.input_dir}")

    output_path = args.output or os.path.join(args.input_dir, "ocr_scan_log.csv")

    image_files = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise SystemExit(f"Error: no images found in: {args.input_dir}")

    print(f"Loading PaddleOCR engine (model tier: {args.model_tier}; "
          f"first run downloads model weights, this can take a while)...")
    ocr_engine = build_ocr_engine(args.model_tier)

    preprocessed_dir = None
    if args.save_preprocessed:
        preprocessed_dir = args.input_dir.rstrip("/\\") + "_preprocessed"
        os.makedirs(preprocessed_dir, exist_ok=True)

    rows = []
    print(f"\nRunning OCR on {len(image_files)} image(s) in {args.input_dir}...\n")
    for original_filename in image_files:
        image_path = os.path.join(args.input_dir, original_filename)
        variants = generate_ocr_variants(image_path)
        if not variants:
            print(f"[warn] could not read: {original_filename}")
            continue

        if preprocessed_dir:
            variant_dict = dict(variants)
            cv2.imwrite(os.path.join(preprocessed_dir, original_filename), variant_dict["full"])

        if args.single_pass:
            text, conf = read_plate_text(ocr_engine, dict(variants)["full"])
            winning_variant = "full"
        else:
            text, conf, winning_variant = read_plate_text_best(ocr_engine, variants)

        low_confidence = conf < args.min_confidence

        new_filename = original_filename
        if not args.no_rename:
            ext = os.path.splitext(original_filename)[1]
            base_name = sanitize_for_filename(text)
            if low_confidence:
                base_name = f"LOWCONF_{base_name}"
            new_path = unique_destination(args.input_dir, base_name, ext)
            os.rename(image_path, new_path)
            new_filename = os.path.basename(new_path)

        flag = "  [LOW CONFIDENCE - review]" if low_confidence else ""
        print(f"{original_filename} -> {new_filename}: '{text}' "
              f"(confidence {conf * 100:.1f}%, variant '{winning_variant}'){flag}")
        rows.append({
            "original_file": original_filename,
            "renamed_file": new_filename,
            "text": text,
            "confidence": f"{conf:.4f}",
            "variant": winning_variant,
            "low_confidence": str(low_confidence),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["original_file", "renamed_file", "text", "confidence", "variant", "low_confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Done: {len(rows)} image(s) scanned, list saved to {output_path} ===")


if __name__ == "__main__":
    main()
