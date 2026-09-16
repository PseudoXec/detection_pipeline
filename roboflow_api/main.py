"""
main.py
-------
Centralized pipeline that runs the two scripts back-to-back:

    1. roboflow_workflow_infer.py  -> detects plates in the source image(s)
                                       and crops each detection into --out-dir
    2. ocr_cropped_plates.py       -> OCRs every crop in --out-dir, renames
                                       each file to its recognized text, and
                                       writes a scan log CSV
SETUP
-----
    pip install inference-sdk paddleocr paddlepaddle opencv-python numpy
    $env:ROBOFLOW_API_KEY="your_private_api_key"       # Windows PowerShell
"""

import argparse
import os
import shutil
import sys
import tempfile

import roboflow_workflow_infer as detect
import ocr_cropped_plates as ocr

CONFIG = {
    # --- Source: set exactly ONE of these, leave the other as None ---
    "image": None,                      # e.g. r"C:\plates\plate.jpg"
    "folder": r"C:\Users\User\Documents\Detection_Inference\data\Batch1",                # e.g. r"C:\plates\samples"

    # --- Roboflow ---
    "workspace": detect.WORKSPACE_NAME,
    "workflow": detect.WORKFLOW_ID,
    # ROBOFLOW_API_KEY is still read from the environment for security
    # (don't hardcode API keys in a file you might commit/share). Uncomment
    # below only if you understand that risk:
    # "roboflow_api_key": "your_private_api_key",
    "roboflow_api_key": "YdI3yJ8U00nKQGsRWnvP",

    # --- Detection stage ---
    "out_dir": None,                    # None = auto sibling folder (see default_out_dir)
    "no_preprocess": False,             # True = skip contrast/denoise/resize before detection
    "keep_preprocessed": False,         # True = keep the full preprocessed photos on disk
    "max_dim": detect.DEFAULT_MAX_DIM,
    "min_dim": detect.DEFAULT_MIN_DIM,
    "crop_padding": 0.15,               # margin added around each detection box
    "crop_min_height": 120,             # zoom small crops up to at least this height (px)

    # --- OCR stage ---
    "ocr_output": None,                 # None = "<out_dir>/ocr_scan_log.csv"
    "no_rename": False,                 # True = don't rename crops, just report text
    "save_preprocessed": False,         # True = dump OCR-preprocessed debug images
    "model_tier": "server",             # "server" (accurate, default) | "mobile-en" (fast) | "default"
    "single_pass": False,               # True = skip the raw/light/full best-of comparison
    "min_confidence": 0.80,             # below this, filename gets a LOWCONF_ prefix

    # --- Pipeline control ---
    "skip_detection": False,            # True = OCR only, using existing files in out_dir
    "skip_ocr": False,                  # True = detection/crop only, no OCR
}
# ==========================================================================


# --------------------------------------------------------------------------
# Stage 1: detection + cropping (thin wrapper around roboflow_workflow_infer)
# --------------------------------------------------------------------------
def run_detection_stage(args) -> int:
    """
    Runs detection+cropping on --image or --folder using the imported
    functions from roboflow_workflow_infer.py. Returns the number of crops
    produced.
    """
    api_key = args.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "Error: no Roboflow API key found.\n"
            "  export ROBOFLOW_API_KEY=your_private_api_key   (macOS/Linux)\n"
            '  $env:ROBOFLOW_API_KEY="your_private_api_key"    (PowerShell)\n'
            "  or set CONFIG[\"roboflow_api_key\"] in main.py"
        )

    client = detect.build_client(api_key)
    preprocess = not args.no_preprocess

    if args.keep_preprocessed:
        work_dir = args.out_dir.rstrip("/\\") + "_preprocessed_full"
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = tempfile.mkdtemp(prefix="rf_preprocess_")
        cleanup_work_dir = True

    total_crops = 0
    try:
        if args.folder:
            if not os.path.isdir(args.folder):
                sys.exit(f"Error: folder not found: {args.folder}")

            image_paths = detect.list_images_in_folder(args.folder)
            if not image_paths:
                sys.exit(f"Error: no images ({sorted(detect.IMAGE_EXTENSIONS)}) found in: {args.folder}")

            print(f"[1/2] Found {len(image_paths)} image(s) in {args.folder}. "
                  f"Preprocessing: {'on' if preprocess else 'off'}. "
                  f"Running workflow '{args.workflow}' on each...")

            for image_path in image_paths:
                total_crops += detect.process_single_image(
                    client, image_path, args.workspace, args.workflow,
                    args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                    args.crop_padding, args.crop_min_height,
                )

            print(f"\n=== [1/2] Detection done: {total_crops} object(s) cropped from "
                  f"{len(image_paths)} image(s), saved to '{args.out_dir}' ===")

        else:
            print(f"[1/2] Running workflow '{args.workflow}' in workspace '{args.workspace}' "
                  f"on image: {args.image} (preprocessing: {'on' if preprocess else 'off'})")
            total_crops = detect.process_single_image(
                client, args.image, args.workspace, args.workflow,
                args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                args.crop_padding, args.crop_min_height,
            )
            print(f"\n=== [1/2] Detection done: {total_crops} object(s) cropped, "
                  f"saved to '{args.out_dir}' ===")
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    return total_crops


# --------------------------------------------------------------------------
# Stage 2: OCR + rename (thin wrapper around ocr_cropped_plates)
# --------------------------------------------------------------------------
def run_ocr_stage(args) -> int:
    """
    Runs OCR on every crop in args.out_dir using the imported functions from
    ocr_cropped_plates.py. Returns the number of images scanned.
    """
    input_dir = args.out_dir
    output_path = args.ocr_output or os.path.join(input_dir, "ocr_scan_log.csv")

    image_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in ocr.IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"[2/2] No crops found in {input_dir}, skipping OCR stage.")
        return 0

    print(f"\n[2/2] Loading PaddleOCR engine (model tier: {args.model_tier}; "
          f"first run downloads model weights)...")
    ocr_engine = ocr.build_ocr_engine(args.model_tier)

    preprocessed_dir = None
    if args.save_preprocessed:
        preprocessed_dir = input_dir.rstrip("/\\") + "_preprocessed"
        os.makedirs(preprocessed_dir, exist_ok=True)

    import csv
    rows = []
    print(f"[2/2] Running OCR on {len(image_files)} image(s) in {input_dir}...\n")
    for original_filename in image_files:
        image_path = os.path.join(input_dir, original_filename)
        variants = ocr.generate_ocr_variants(image_path)
        if not variants:
            print(f"[warn] could not read: {original_filename}")
            continue

        variant_dict = dict(variants)
        if preprocessed_dir:
            import cv2
            cv2.imwrite(os.path.join(preprocessed_dir, original_filename), variant_dict["full"])

        if args.single_pass:
            text, conf = ocr.read_plate_text(ocr_engine, variant_dict["full"])
            winning_variant = "full"
        else:
            text, conf, winning_variant = ocr.read_plate_text_best(ocr_engine, variants)

        low_confidence = conf < args.min_confidence

        new_filename = original_filename
        if not args.no_rename:
            ext = os.path.splitext(original_filename)[1]
            base_name = ocr.sanitize_for_filename(text)
            if low_confidence:
                base_name = f"LOWCONF_{base_name}"
            new_path = ocr.unique_destination(input_dir, base_name, ext)
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

    print(f"\n=== [2/2] OCR done: {len(rows)} image(s) scanned, list saved to {output_path} ===")
    return len(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Roboflow plate detection+crop, then PaddleOCR + rename. "
                     "All defaults come from the CONFIG dict at the top of this file - edit that "
                     "and run with no flags, or override anything here for a one-off run."
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", default=CONFIG["image"],
                               help="Path to a single local image file.")
    source_group.add_argument("--folder", default=CONFIG["folder"],
                               help="Path to a folder of images to process in a batch.")

    # Roboflow
    parser.add_argument("--workspace", default=CONFIG["workspace"], help="Roboflow workspace slug.")
    parser.add_argument("--workflow", default=CONFIG["workflow"], help="Roboflow workflow id/slug.")
    parser.add_argument("--roboflow-api-key", default=CONFIG["roboflow_api_key"],
                         help="Roboflow API key. Prefer the ROBOFLOW_API_KEY env var over this flag "
                              "so the key doesn't end up in shell history.")

    # Detection-stage options (mirrors roboflow_workflow_infer.py)
    parser.add_argument("--out-dir", default=CONFIG["out_dir"],
                         help="Directory to save cropped detection images into (also the OCR input dir). "
                              "Defaults to 'cropped_detections' created as a sibling of the source "
                              "folder's parent (or the image's containing folder's parent for --image).")
    parser.add_argument("--no-preprocess", action="store_true", default=CONFIG["no_preprocess"],
                         help="Send the raw image as-is; skip detection-stage contrast/denoise/resize.")
    parser.add_argument("--keep-preprocessed", action="store_true", default=CONFIG["keep_preprocessed"],
                         help="Save the preprocessed full images into '<out-dir>_preprocessed_full' "
                              "instead of discarding them after each run.")
    parser.add_argument("--max-dim", type=int, default=CONFIG["max_dim"],
                         help=f"Downscale images larger than this on the longest side "
                              f"(default {CONFIG['max_dim']}).")
    parser.add_argument("--min-dim", type=int, default=CONFIG["min_dim"],
                         help=f"Upscale images smaller than this on the longest side "
                              f"(default {CONFIG['min_dim']}).")
    parser.add_argument("--crop-padding", type=float, default=CONFIG["crop_padding"],
                         help="Expand each detection box by this fraction of its own width/height on "
                              "every side before cropping. Prevents tight boxes from clipping character "
                              "edges. Set to 0 to disable.")
    parser.add_argument("--crop-min-height", type=int, default=CONFIG["crop_min_height"],
                         help="Upscale (zoom in on) any crop shorter than this many pixels, preserving "
                              "aspect ratio, before saving. Set to 0 to disable.")

    # OCR-stage options (mirrors ocr_cropped_plates.py)
    parser.add_argument("--ocr-output", default=CONFIG["ocr_output"],
                         help="CSV file to write the OCR scan list to. Defaults to "
                              "'ocr_scan_log.csv' inside --out-dir.")
    parser.add_argument("--no-rename", action="store_true", default=CONFIG["no_rename"],
                         help="Don't rename the cropped images; just report recognized text.")
    parser.add_argument("--save-preprocessed", action="store_true", default=CONFIG["save_preprocessed"],
                         help="Also save the OCR-preprocessed crops (for debugging) into "
                              "'<out-dir>_preprocessed'.")
    parser.add_argument("--model-tier", choices=sorted(ocr.MODEL_TIERS), default=CONFIG["model_tier"],
                         help="Which PaddleOCR models to load. 'server' (default) is the most accurate "
                              "for short alphanumeric strings like plates. 'mobile-en' is the old, faster "
                              "but less accurate default. 'default' uses whatever PaddleOCR ships as its "
                              "own default for the installed version.")
    parser.add_argument("--single-pass", action="store_true", default=CONFIG["single_pass"],
                         help="Run OCR once using only the 'full' preprocessing pipeline instead of "
                              "trying raw/light/full variants and keeping the best result. Faster but "
                              "less accurate on crops where heavier preprocessing hurts more than it helps.")
    parser.add_argument("--min-confidence", type=float, default=CONFIG["min_confidence"],
                         help="Confidence (0-1) below which a result is flagged 'LOWCONF_' in its "
                              "filename/log for manual review.")

    # Pipeline control
    parser.add_argument("--skip-detection", action="store_true", default=CONFIG["skip_detection"],
                         help="Skip stage 1 and run OCR directly on whatever is already in --out-dir.")
    parser.add_argument("--skip-ocr", action="store_true", default=CONFIG["skip_ocr"],
                         help="Skip stage 2 and stop after detection+cropping.")

    args = parser.parse_args()

    if args.skip_detection and args.skip_ocr:
        sys.exit("Error: --skip-detection and --skip-ocr can't both be set (nothing to do).")

    if not args.skip_detection:
        if args.image and args.folder:
            sys.exit("Error: set only one of --image / --folder (or CONFIG['image'] / CONFIG['folder']).")
        if not args.image and not args.folder:
            sys.exit(
                "Error: no source set. Edit CONFIG['folder'] or CONFIG['image'] at the top of main.py, "
                "or pass --image / --folder on the command line."
            )
        if args.folder and not os.path.isdir(args.folder):
            sys.exit(f"Error: folder not found: {args.folder}\n"
                      f"(check CONFIG['folder'] at the top of main.py)")
        if args.image and not os.path.isfile(args.image):
            sys.exit(f"Error: image not found: {args.image}\n"
                      f"(check CONFIG['image'] at the top of main.py)")

    if args.out_dir is None:
        args.out_dir = detect.default_out_dir(args.image, args.folder)
        print(f"[info] --out-dir not set, defaulting to: {args.out_dir}")

    total_crops = None
    if not args.skip_detection:
        total_crops = run_detection_stage(args)
        if total_crops == 0:
            print("\nNo detections were cropped; skipping OCR stage.")
            return
    else:
        if not os.path.isdir(args.out_dir):
            sys.exit(f"Error: --skip-detection set but --out-dir not found: {args.out_dir}")
        print(f"[1/2] Skipped (--skip-detection). Using existing crops in '{args.out_dir}'.")

    if not args.skip_ocr:
        run_ocr_stage(args)
    else:
        print("[2/2] Skipped (--skip-ocr).")

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
