#!/usr/bin/env python3
"""
Install:
    python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
    python -m pip install pyyaml requests
    
"""

import argparse
import base64
import logging
import shutil
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Optional

import requests
import yaml
from paddleocr import PaddleOCRVL

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

log = logging.getLogger("ocr_worker")


# =========================================================================
# Config loading
# =========================================================================

@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=raw)

    def get(self, *keys, default=None):
        """Safe nested lookup, e.g. cfg.get('source', 'folder', 'input_dir')."""
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


# =========================================================================
# Sources — where cropped images come FROM
# =========================================================================

class ImageItem:
    """One unit of work: an id, a local path to the image bytes, and
    whatever metadata the source needs to ack/cleanup later."""

    def __init__(self, item_id: str, local_path: Path, meta: Optional[dict] = None):
        self.item_id = item_id
        self.local_path = local_path
        self.meta = meta or {}


class Source(ABC):
    @abstractmethod
    def fetch_batch(self, batch_size: int) -> list:
        """Return up to batch_size ImageItems ready to process."""

    @abstractmethod
    def ack_success(self, item: ImageItem):
        """Called after an item is processed successfully."""

    @abstractmethod
    def ack_failure(self, item: ImageItem):
        """Called after an item permanently fails (retries exhausted)."""


class FolderSource(Source):
    def __init__(self, cfg: Config):
        input_dir = cfg.get("source", "folder", "input_dir", default="./incoming")
        self.input_dir = Path(input_dir)
        processed_dir = cfg.get("source", "folder", "processed_dir")
        failed_dir = cfg.get("source", "folder", "failed_dir")
        self.processed_dir = Path(processed_dir) if processed_dir else self.input_dir / "_processed"
        self.failed_dir = Path(failed_dir) if failed_dir else self.input_dir / "_failed"
        for d in (self.input_dir, self.processed_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def fetch_batch(self, batch_size: int) -> list:
        files = sorted(
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )[:batch_size]
        return [ImageItem(item_id=p.name, local_path=p) for p in files]

    def ack_success(self, item: ImageItem):
        dest = self.processed_dir / item.local_path.name
        if item.local_path.exists():
            shutil.move(str(item.local_path), str(dest))

    def ack_failure(self, item: ImageItem):
        dest = self.failed_dir / item.local_path.name
        if item.local_path.exists():
            shutil.move(str(item.local_path), str(dest))


class APISource(Source):
    """Polls a REST endpoint for pending images, downloads each one to a
    local temp file, and optionally acks/deletes it on the server after
    processing. Adjust the endpoint contract in config.yaml and, if your
    API shape differs, tweak _parse_list_response / _download below."""

    def __init__(self, cfg: Config):
        self.base_url = cfg.get("source", "api", "base_url", default="").rstrip("/")
        self.list_endpoint = cfg.get("source", "api", "list_endpoint", default="/images/pending")
        self.ack_endpoint = cfg.get("source", "api", "ack_endpoint")
        self.headers = cfg.get("source", "api", "headers", default={}) or {}
        self.tmp_dir = Path("./_api_tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def fetch_batch(self, batch_size: int) -> list:
        try:
            resp = requests.get(
                f"{self.base_url}{self.list_endpoint}",
                headers=self.headers,
                params={"limit": batch_size},
                timeout=15,
            )
            resp.raise_for_status()
            records = resp.json()
        except Exception:
            log.exception("Failed to fetch pending image list from API")
            return []

        items = []
        for rec in records[:batch_size]:
            item_id = str(rec.get("id"))
            try:
                local_path = self._download(rec)
                items.append(ImageItem(item_id=item_id, local_path=local_path, meta=rec))
            except Exception:
                log.exception("Failed to download image %s", item_id)
        return items

    def _download(self, rec: dict) -> Path:
        """Supports either a direct URL or an inline base64 image field.
        Adjust this to match your actual API response shape."""
        item_id = str(rec.get("id"))
        dest = self.tmp_dir / f"{item_id}.png"

        if "image_base64" in rec:
            dest.write_bytes(base64.b64decode(rec["image_base64"]))
        elif "url" in rec:
            r = requests.get(rec["url"], headers=self.headers, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
        else:
            raise ValueError(f"Record {item_id} has neither 'url' nor 'image_base64'")
        return dest

    def ack_success(self, item: ImageItem):
        self._cleanup_local(item)
        self._ack(item, status="done")

    def ack_failure(self, item: ImageItem):
        self._cleanup_local(item)
        self._ack(item, status="failed")

    def _ack(self, item: ImageItem, status: str):
        if not self.ack_endpoint:
            return
        try:
            requests.post(
                f"{self.base_url}{self.ack_endpoint}/{item.item_id}",
                headers=self.headers,
                json={"status": status},
                timeout=15,
            )
        except Exception:
            log.exception("Failed to ack item %s (status=%s)", item.item_id, status)

    def _cleanup_local(self, item: ImageItem):
        try:
            if item.local_path.exists():
                item.local_path.unlink()
        except Exception:
            log.exception("Failed to delete temp file for %s", item.item_id)


def build_source(cfg: Config) -> Source:
    source_type = cfg.get("source", "type", default="folder")
    if source_type == "folder":
        return FolderSource(cfg)
    if source_type == "api":
        return APISource(cfg)
    raise ValueError(f"Unknown source.type: {source_type!r} (expected 'folder' or 'api')")


# =========================================================================
# Sinks — where OCR results go TO
# =========================================================================

class Sink(ABC):
    @abstractmethod
    def write(self, item: ImageItem, result: Any):
        """Persist one OCR result."""


class FolderSink(Sink):
    def __init__(self, cfg: Config):
        output_dir = cfg.get("output", "folder", "output_dir", default="./results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, item: ImageItem, result: Any):
        # `result` here is the list of PaddleOCRVL result objects; each
        # knows how to save itself.
        for res in result:
            res.save_to_json(save_path=str(self.output_dir))
            res.save_to_markdown(save_path=str(self.output_dir))


class APISink(Sink):
    def __init__(self, cfg: Config):
        self.base_url = cfg.get("output", "api", "base_url", default="").rstrip("/")
        self.results_endpoint = cfg.get("output", "api", "results_endpoint", default="/ocr/results")
        self.headers = cfg.get("output", "api", "headers", default={}) or {}

    def write(self, item: ImageItem, result: Any):
        payload_texts = []
        for res in result:
            # PaddleOCRVL result objects support .json (dict-like) access;
            # fall back to str() if the exact structure differs by version.
            try:
                payload_texts.append(res.json)
            except AttributeError:
                payload_texts.append(str(res))
        try:
            requests.post(
                f"{self.base_url}{self.results_endpoint}",
                headers=self.headers,
                json={"id": item.item_id, "result": payload_texts},
                timeout=30,
            )
        except Exception:
            log.exception("Failed to POST result for %s", item.item_id)
            raise


def build_sink(cfg: Config) -> Sink:
    output_type = cfg.get("output", "type", default="folder")
    if output_type == "folder":
        return FolderSink(cfg)
    if output_type == "api":
        return APISink(cfg)
    raise ValueError(f"Unknown output.type: {output_type!r} (expected 'folder' or 'api')")


# =========================================================================
# Worker
# =========================================================================

class OCRWorker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.source = build_source(cfg)
        self.sink = build_sink(cfg)

        self.poll_interval = cfg.get("worker", "poll_interval", default=1.0)
        self.batch_size = cfg.get("worker", "batch_size", default=8)
        self.recycle_after = cfg.get("worker", "recycle_after", default=None)
        self.retry_count = cfg.get("worker", "retry_count", default=2)

        self.pipeline_version = cfg.get("pipeline", "version", default="v1.6")
        self.vl_rec_backend = cfg.get("pipeline", "vl_rec_backend")
        self.vl_rec_server_url = cfg.get("pipeline", "vl_rec_server_url")

        self._stop = Event()
        self._pipeline = None
        self._processed_count = 0

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        log.info("Shutdown signal received, finishing current batch then exiting...")
        self._stop.set()

    def _load_pipeline(self):
        kwargs = {"pipeline_version": self.pipeline_version}
        if self.vl_rec_backend:
            kwargs["vl_rec_backend"] = self.vl_rec_backend
        if self.vl_rec_server_url:
            kwargs["vl_rec_server_url"] = self.vl_rec_server_url
        log.info("Loading PaddleOCR-VL-%s pipeline (happens once per process lifetime)...", self.pipeline_version)
        t0 = time.time()
        self._pipeline = PaddleOCRVL(**kwargs)
        log.info("Pipeline ready in %.1fs", time.time() - t0)

    def _process_one(self, item: ImageItem) -> bool:
        last_exc = None
        for attempt in range(1, self.retry_count + 2):  # +1 initial try, +1 for range inclusivity
            try:
                output = self._pipeline.predict(str(item.local_path))
                result = list(output)
                self.sink.write(item, result)
                self.source.ack_success(item)
                self._processed_count += 1
                return True
            except Exception as exc:
                last_exc = exc
                log.warning("Attempt %d/%d failed for %s: %s", attempt, self.retry_count + 1, item.item_id, exc)
        log.error("Giving up on %s after %d attempts: %s", item.item_id, self.retry_count + 1, last_exc)
        self.source.ack_failure(item)
        return False

    def run(self):
        self._load_pipeline()
        log.info("Worker started. Source=%s Output=%s", self.cfg.get("source", "type"), self.cfg.get("output", "type"))

        while not self._stop.is_set():
            batch = self.source.fetch_batch(self.batch_size)

            if not batch:
                time.sleep(self.poll_interval)
                continue

            t0 = time.time()
            ok = 0
            for item in batch:
                if self._stop.is_set():
                    break
                if self._process_one(item):
                    ok += 1
            dt = time.time() - t0

            log.info(
                "Processed %d/%d images in %.2fs (%.2f img/s) | total processed: %d",
                ok, len(batch), dt, (len(batch) / dt if dt > 0 else 0.0), self._processed_count,
            )

            if self.recycle_after and self._processed_count >= self.recycle_after:
                log.info("Reached recycle threshold (%d images) — reloading pipeline to release memory.", self.recycle_after)
                self._pipeline = None
                self._processed_count = 0
                self._load_pipeline()

        log.info("Worker stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Continuous, config-driven PaddleOCR-VL-1.6 worker.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    logging.basicConfig(
        level=getattr(logging, str(cfg.get("worker", "log_level", default="INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    worker = OCRWorker(cfg)
    worker.run()


if __name__ == "__main__":
    main()
