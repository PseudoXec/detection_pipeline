import argparse
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# --------------------------------------------------------------------------
# Fixed config for this workflow (from Roboflow deploy tab)
# --------------------------------------------------------------------------
API_URL = "https://serverless.roboflow.com"
WORKSPACE_NAME = "asdasd-qisot"
WORKFLOW_ID = "platenumberdetect-vplatenumberdetect-0mqc6-2-yolo11n-t1-logic"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MAX_DIM = 1600  # downscale anything larger than this (longest side)
DEFAULT_MIN_DIM = 416   # upscale anything smaller than this (longest side)


class WorkflowInferenceError(RuntimeError):
    """Raised when the workflow call fails after all retries."""


def build_client(api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    # Send the API key as an Authorization: Bearer header, not a query param.
    client.configure(InferenceConfiguration(api_key_transport="header"))
    return client


# --------------------------------------------------------------------------
# Preprocessing (before detection)
# --------------------------------------------------------------------------
def preprocess_for_detection(
    image_path: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
):
    """
    Light preprocessing applied to the FULL photo before it's sent for
    detection. Meant to help the model on harsh outdoor lighting, shadows,
    or low-resolution source photos - not to be confused with the separate,
    heavier preprocessing used later for OCR on individual crops.

    Steps:
      1. Resize: downscale very large images (faster, more consistent
         uploads) or upscale very small ones (so small/distant objects
         aren't missed). Skipped if already in a reasonable range.
      2. CLAHE contrast enhancement on the luminance channel only - pulls
         detail out of shadows/glare without shifting color balance.
      3. Light denoise to clean up sensor/compression noise.

    Returns a BGR numpy array, or None if the image couldn't be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    longest_side = max(h, w)
    if longest_side > max_dim:
        scale = max_dim / longest_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif longest_side < min_dim:
        scale = min_dim / longest_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    img = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    return img


def prepare_detection_input(
    image_path: str,
    preprocess: bool,
    work_dir: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
) -> str:
    """
    Returns the path that should actually be sent to the workflow: either
    the original image_path unchanged, or a preprocessed copy written into
    work_dir. Falls back to the original on any preprocessing failure so a
    single bad image can't crash a batch run.
    """
    if not preprocess:
        return image_path

    processed = preprocess_for_detection(image_path, max_dim=max_dim, min_dim=min_dim)
    if processed is None:
        print(f"[warn] preprocessing failed, using original image: {image_path}")
        return image_path

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(work_dir, f"{base}_preprocessed.jpg")
    cv2.imwrite(out_path, processed)
    return out_path


# --------------------------------------------------------------------------
# Workflow call
# --------------------------------------------------------------------------
def run_plate_detection_workflow(
    client: InferenceHTTPClient,
    image_path: str,
    workspace_name: str = WORKSPACE_NAME,
    workflow_id: str = WORKFLOW_ID,
    parameters: Optional[Dict[str, Any]] = None,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Runs the workflow on a single local image and returns the first result
    dict (workflows_run's response is a list, one entry per input image).

    Retries with exponential backoff on transient failures. Raises
    WorkflowInferenceError if every attempt fails.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            result = client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": image_path},
                parameters=parameters or {},
            )
            if not result:
                raise WorkflowInferenceError("Workflow returned an empty result list.")
            return result[0]
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry on anything
            last_error = e
            if attempt < max_retries:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"[warn] attempt {attempt} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)

    raise WorkflowInferenceError(
        f"Workflow call failed after {max_retries} attempts: {last_error}"
    ) from last_error


def extract_predictions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pulls the detection list out of the confirmed output shape:
    result["predictions"]["predictions"]. Falls back gracefully if the
    workflow output ever changes shape.
    """
    preds_field = result.get("predictions")
    if isinstance(preds_field, dict) and isinstance(preds_field.get("predictions"), list):
        return preds_field["predictions"]
    if isinstance(preds_field, list):
        return preds_field
    return []


def crop_and_save_detections(
    image_path: str,
    predictions: List[Dict[str, Any]],
    out_dir: str,
    prefix: str,
    padding_ratio: float = 0.15,
    min_crop_height: int = 120,
) -> List[str]:
    """
    Crops each detected object out of image_path (using center x/y +
    width/height boxes, clamped to image bounds) and saves it as its own
    file in out_dir. Returns the list of saved paths.

    padding_ratio expands each box by this fraction of its own width/height
    on every side before cropping (clamped to image bounds). A tight,
    exact-to-the-model bounding box will sometimes shave a pixel or two off
    the leftmost/rightmost character, which is enough to make a plate that's
    perfectly readable to a human unrecognizable to OCR. Default 0.15 (15%)
    is a comfortable margin; set to 0 to disable.

    min_crop_height upscales (INTER_CUBIC) any crop shorter than this,
    preserving aspect ratio, before saving. Plate detections are often small
    relative to the source photo - saving them at native pixel size bakes in
    that softness for every downstream step (including the OCR-stage's own
    upscaling, which then has less real detail to work with). Zooming in
    here, once, on the sharpest available pixels (the original detection
    input, not a re-compressed crop) gives the OCR stage a better starting
    point. Set to 0 to disable.

    IMPORTANT: image_path here must be whatever image was actually sent to
    the workflow (preprocessed or original) - detection coordinates are
    relative to that image, not necessarily the original file on disk.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warn] could not read image for cropping: {image_path}")
        return []

    h, w = image.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, p in enumerate(predictions, start=1):
        cx, cy, bw, bh = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        if None in (cx, cy, bw, bh):
            continue

        pad_w = bw * padding_ratio
        pad_h = bh * padding_ratio
        x1 = max(int(cx - bw / 2 - pad_w), 0)
        y1 = max(int(cy - bh / 2 - pad_h), 0)
        x2 = min(int(cx + bw / 2 + pad_w), w)
        y2 = min(int(cy + bh / 2 + pad_h), h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        if min_crop_height and crop_h < min_crop_height:
            scale = min_crop_height / crop_h
            crop = cv2.resize(crop, (int(crop_w * scale), min_crop_height), interpolation=cv2.INTER_CUBIC)

        cls = str(p.get("class") or p.get("class_name") or "object")
        safe_cls = "".join(c if c.isalnum() else "_" for c in cls)
        conf = p.get("confidence")
        conf_str = f"{conf * 100:.0f}" if isinstance(conf, (int, float)) else "NA"

        out_name = f"{prefix}_{i}_{safe_cls}_{conf_str}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, crop)
        saved_paths.append(out_path)

    return saved_paths


def process_single_image(
    client: InferenceHTTPClient,
    image_path: str,
    workspace: str,
    workflow: str,
    out_dir: str,
    preprocess: bool,
    work_dir: str,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
    crop_padding: float = 0.15,
    min_crop_height: int = 120,
) -> int:
    """Runs the workflow on one image and crops+saves any detections. Returns crop count."""
    print(f"\n--- {image_path} ---")

    detection_input_path = prepare_detection_input(image_path, preprocess, work_dir, max_dim, min_dim)
    if preprocess and detection_input_path != image_path:
        print(f"  preprocessed -> {detection_input_path}")

    try:
        result = run_plate_detection_workflow(client, detection_input_path, workspace, workflow)
    except (FileNotFoundError, WorkflowInferenceError) as e:
        print(f"[error] {e}")
        return 0

    predictions = extract_predictions(result)
    if not predictions:
        print("No detections.")
        return 0

    # Crop names are based on the ORIGINAL filename for readability, even
    # though pixels come from the (possibly preprocessed) detection input.
    prefix = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = crop_and_save_detections(
        detection_input_path, predictions, out_dir, prefix, crop_padding, min_crop_height,
    )

    print(f"Cropped {len(saved_paths)} object(s):")
    for path in saved_paths:
        print(f"  -> {path}")

    return len(saved_paths)


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def default_out_dir(image: Optional[str], folder: Optional[str], dir_name: str = "cropped_detections") -> str:
    """
    Builds a default output directory that lives INSIDE the source, as a
    child folder, instead of dropping "cropped_detections" into the current
    working directory or next to the source.

    --folder /data/Batch1        -> /data/Batch1/cropped_detections
    --image  /data/Batch1/plate.jpg -> /data/Batch1/cropped_detections

    In both cases the source's containing folder (e.g. "Batch1") is treated
    as the batch folder, and the crop dir is created as a child of it - so
    each batch folder ends up self-contained with its own crops alongside
    its raw images.
    """
    if folder:
        sample_folder = os.path.abspath(folder.rstrip("/\\"))
    else:
        sample_folder = os.path.abspath(os.path.dirname(image) or ".")

    return os.path.join(sample_folder, dir_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run a Roboflow Workflow on an image (or a folder of images) and crop out detections."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a single local image file.")
    source_group.add_argument("--folder", help="Path to a folder of images to process in a batch.")

    parser.add_argument("--workspace", default=WORKSPACE_NAME, help="Roboflow workspace slug.")
    parser.add_argument("--workflow", default=WORKFLOW_ID, help="Roboflow workflow id/slug.")
    parser.add_argument("--out-dir", default=None,
                         help="Directory to save cropped detection images into. Defaults to "
                              "'cropped_detections' created as a subfolder inside the source folder "
                              "(or inside the image's containing folder for --image) - e.g. "
                              "Batch1/cropped_detections.")
    parser.add_argument("--no-preprocess", action="store_true",
                         help="Send the raw image as-is; skip contrast/denoise/resize preprocessing.")
    parser.add_argument("--keep-preprocessed", action="store_true",
                         help="Save the preprocessed full images into '<out-dir>_preprocessed_full' "
                              "instead of discarding them after each run.")
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                         help=f"Downscale images larger than this on the longest side (default {DEFAULT_MAX_DIM}).")
    parser.add_argument("--min-dim", type=int, default=DEFAULT_MIN_DIM,
                         help=f"Upscale images smaller than this on the longest side (default {DEFAULT_MIN_DIM}).")
    parser.add_argument("--crop-padding", type=float, default=0.15,
                         help="Expand each detection box by this fraction of its own width/height on "
                              "every side before cropping (default 0.15 = 15%%). Prevents tight boxes "
                              "from clipping character edges, which can make an otherwise-clear plate "
                              "unreadable to OCR. Set to 0 to disable.")
    parser.add_argument("--crop-min-height", type=int, default=120,
                         help="Upscale (zoom in on) any crop shorter than this many pixels, preserving "
                              "aspect ratio, before saving (default 120). Small plate detections saved "
                              "at native size give the OCR stage less real detail to work with. "
                              "Set to 0 to disable.")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ROBOFLOW_API_KEY is not set.\n"
            "  export ROBOFLOW_API_KEY=your_private_api_key   (macOS/Linux)\n"
            '  $env:ROBOFLOW_API_KEY="your_private_api_key"    (PowerShell)'
        )

    if args.out_dir is None:
        args.out_dir = default_out_dir(args.image, args.folder)
        print(f"[info] --out-dir not set, defaulting to: {args.out_dir}")

    client = build_client(api_key)
    preprocess = not args.no_preprocess

    if args.keep_preprocessed:
        work_dir = args.out_dir.rstrip("/\\") + "_preprocessed_full"
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = tempfile.mkdtemp(prefix="rf_preprocess_")
        cleanup_work_dir = True

    try:
        if args.folder:
            if not os.path.isdir(args.folder):
                sys.exit(f"Error: folder not found: {args.folder}")

            image_paths = list_images_in_folder(args.folder)
            if not image_paths:
                sys.exit(f"Error: no images ({sorted(IMAGE_EXTENSIONS)}) found in: {args.folder}")

            print(f"Found {len(image_paths)} image(s) in {args.folder}. "
                  f"Preprocessing: {'on' if preprocess else 'off'}. "
                  f"Running workflow '{args.workflow}' on each...")

            total_crops = 0
            for image_path in image_paths:
                total_crops += process_single_image(
                    client, image_path, args.workspace, args.workflow,
                    args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                    args.crop_padding, args.crop_min_height,
                )

            print(f"\n=== Done: {total_crops} object(s) cropped from {len(image_paths)} "
                  f"image(s), saved to '{args.out_dir}' ===")

        else:
            print(f"Running workflow '{args.workflow}' in workspace '{args.workspace}' "
                  f"on image: {args.image} (preprocessing: {'on' if preprocess else 'off'})")
            total_crops = process_single_image(
                client, args.image, args.workspace, args.workflow,
                args.out_dir, preprocess, work_dir, args.max_dim, args.min_dim,
                args.crop_padding, args.crop_min_height,
            )
            print(f"\n=== Done: {total_crops} object(s) cropped, saved to '{args.out_dir}' ===")
    finally:
        if cleanup_work_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
