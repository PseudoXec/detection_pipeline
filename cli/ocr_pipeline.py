"""
ocr_pipeline.py
----------------
Standalone OCR-only pipeline built on FastPlateOCR.

This intentionally does NOT run vehicle or plate detection. It expects to
be pointed at an image (or a folder of images) that is ALREADY a plate
crop - e.g. output from a separate detection stage, or a folder of plate
photos - and its only job is: read each one, score the read, write out the
results.

Usage (run from the OCR/ project root)
-----
    python -m cli.ocr_pipeline --image path/to/plate.jpg
    python -m cli.ocr_pipeline --folder path/to/plate_crops/ --recursive
    python -m cli.ocr_pipeline --folder plates/ --config my_config.yaml
    python -m cli.ocr_pipeline --folder plates/ --csv out.csv --json out.json

See config/ocr_config.py / config/ocr_config.yaml for every tunable value
(model choice, confidence threshold, enhancement, output paths, etc).
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2

from ocr.image_enhance import enhance_plate_crop
from config.ocr_config import PipelineConfig
from ocr.ocr_reader import PlateOCRReader

log = logging.getLogger("ocr_pipeline")


@dataclass
class OcrResult:
    image_path: str
    text: Optional[str]
    confidence: float
    elapsed_ms: float
    error: Optional[str] = None

    @property
    def display_text(self) -> str:
        return self.text if self.text else "Unrecognized"


class OcrPipeline:
    """Drives PlateOCRReader over a batch of plate images with no detection
    step involved. Every image is treated independently."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reader = PlateOCRReader(
            model_name=config.ocr.model_name,
            lang=config.ocr.lang,
            min_confidence=config.ocr.min_confidence,
            allowed_chars=config.ocr.allowed_chars,
            early_exit_score=config.ocr.early_exit_score,
            validate_plate_format=config.ocr.validate_plate_format,
        )

    # ------------------------------------------------------------------ #
    # image discovery
    # ------------------------------------------------------------------ #
    def discover_images(self) -> List[Path]:
        input_cfg = self.config.input
        if input_cfg.image:
            path = Path(input_cfg.image)
            if not path.is_file():
                raise FileNotFoundError(f"--image path does not exist: {path}")
            return [path]

        if input_cfg.folder:
            root = Path(input_cfg.folder)
            if not root.is_dir():
                raise FileNotFoundError(f"--folder path does not exist or is not a directory: {root}")
            pattern_iter: Iterator[Path] = root.rglob("*") if input_cfg.recursive else root.glob("*")
            exts = {e.lower() for e in input_cfg.extensions}
            images = sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in exts)
            if not images:
                log.warning("No images with extensions %s found under %s", input_cfg.extensions, root)
            return images

        raise ValueError("No input configured - set input.image or input.folder (or pass --image/--folder).")

    # ------------------------------------------------------------------ #
    # single-image OCR
    # ------------------------------------------------------------------ #
    def read_one(self, image_path: Path) -> OcrResult:
        start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            return OcrResult(str(image_path), None, 0.0, 0.0, error="failed to load image")

        variants = [image]
        if self.config.preprocess.enhance and self.config.preprocess.try_both_variants:
            variants.append(enhance_plate_crop(image, self.config.preprocess.min_crop_height))
        elif self.config.preprocess.enhance:
            variants = [enhance_plate_crop(image, self.config.preprocess.min_crop_height)]

        try:
            text, score = self.reader.read_scored(*variants)
        except Exception as error:  # PlateOCRReader already guards internally, but stay defensive here too
            elapsed_ms = (time.perf_counter() - start) * 1000
            return OcrResult(str(image_path), None, 0.0, elapsed_ms, error=str(error))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OcrResult(str(image_path), text, score, elapsed_ms)

    # ------------------------------------------------------------------ #
    # batch run
    # ------------------------------------------------------------------ #
    def run(self) -> List[OcrResult]:
        images = self.discover_images()
        if not images:
            return []

        results: List[OcrResult] = []
        workers = max(1, self.config.runtime.workers)
        if workers == 1:
            for path in images:
                result = self.read_one(path)
                results.append(result)
                self._log_result(result)
        else:
            # Note: ONNX Runtime sessions aren't guaranteed free-threaded, so
            # keep runtime.workers at 1 unless you've verified your build/
            # model handles concurrent .run() calls safely.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(self.read_one, images):
                    results.append(result)
                    self._log_result(result)

        self._write_outputs(results)
        return results

    def _log_result(self, result: OcrResult) -> None:
        if not self.config.output.print_console:
            return
        if result.error:
            print(f"{result.image_path}\tERROR: {result.error}")
        else:
            print(f"{result.image_path}\t{result.display_text}\tscore={result.confidence:.2f}\t{result.elapsed_ms:.1f}ms")

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    def _write_outputs(self, results: List[OcrResult]) -> None:
        out_cfg = self.config.output

        if out_cfg.results_csv:
            csv_path = Path(out_cfg.results_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_path", "text", "confidence", "elapsed_ms", "error"])
                for r in results:
                    writer.writerow([r.image_path, r.display_text, f"{r.confidence:.4f}", f"{r.elapsed_ms:.2f}", r.error or ""])
            log.info("Wrote %d results to %s", len(results), csv_path)

        if out_cfg.results_json:
            json_path = Path(out_cfg.results_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "image_path": r.image_path,
                    "text": r.text,
                    "display_text": r.display_text,
                    "confidence": round(r.confidence, 4),
                    "elapsed_ms": round(r.elapsed_ms, 2),
                    "error": r.error,
                }
                for r in results
            ]
            with open(json_path, "w") as handle:
                json.dump(payload, handle, indent=2)
            log.info("Wrote %d results to %s", len(results), json_path)

        if out_cfg.save_annotated:
            self._write_annotated(results)

    def _write_annotated(self, results: List[OcrResult]) -> None:
        annotated_dir = Path(self.config.output.annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.error:
                continue
            image = cv2.imread(r.image_path)
            if image is None:
                continue
            label = f"{r.display_text} ({r.confidence:.2f})"
            cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            out_path = annotated_dir / Path(r.image_path).name
            cv2.imwrite(str(out_path), image)
        log.info("Wrote annotated copies to %s", annotated_dir)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR-only pipeline for license plate crops (FastPlateOCR).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (defaults to config/ocr_config.yaml if present).")
    parser.add_argument("--image", type=str, default=None, help="Path to a single plate-crop image.")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of plate-crop images.")
    parser.add_argument("--recursive", action="store_true", help="Search --folder recursively.")
    parser.add_argument("--model", type=str, default=None, help="FastPlateOCR model name or path to a custom ONNX model.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Reads below this become 'Unrecognized'.")
    parser.add_argument("--no-enhance", action="store_true", help="Skip CLAHE/sharpen preprocessing before OCR.")
    parser.add_argument("--csv", dest="csv_path", type=str, default=None, help="Path to write a results CSV (default: output/ocr_results.csv).")
    parser.add_argument("--json", dest="json_path", type=str, default=None, help="Path to write a results JSON file.")
    parser.add_argument("--no-csv", action="store_true", help="Don't write a CSV.")
    parser.add_argument("--workers", type=int, default=None, help="Thread pool size for batch runs (default 1).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image console lines.")
    return parser


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.image:
        config.input.image = args.image
    if args.folder:
        config.input.folder = args.folder
    if args.recursive:
        config.input.recursive = True
    if args.model:
        config.ocr.model_name = args.model
    if args.min_confidence is not None:
        config.ocr.min_confidence = args.min_confidence
    if args.no_enhance:
        config.preprocess.enhance = False
    if args.csv_path:
        config.output.results_csv = args.csv_path
    if args.no_csv:
        config.output.results_csv = None
    if args.json_path:
        config.output.results_json = args.json_path
    if args.workers is not None:
        config.runtime.workers = args.workers
    if args.quiet:
        config.output.print_console = False
    return config


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PipelineConfig.load(args.config)
    config = apply_cli_overrides(config, args)

    logging.basicConfig(level=getattr(logging, config.runtime.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not config.input.image and not config.input.folder:
        build_arg_parser().error("one of --image or --folder is required (or set input.image/input.folder in the config).")

    pipeline = OcrPipeline(config)
    try:
        results = pipeline.run()
    except (FileNotFoundError, ValueError) as error:
        log.error(str(error))
        return 1

    read_count = sum(1 for r in results if r.text)
    log.info("Done: %d/%d images read successfully.", read_count, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
