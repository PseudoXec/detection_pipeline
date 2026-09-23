"""
run.py
------
The executable entry point. Everything above this file is a building block;
this file just wires them together and runs the loop.

Usage on a Raspberry Pi (24/7 RTSP mode, reads config.yaml automatically):
    python3 run.py

Usage for a quick local test against a single image or a folder of images
(no camera needed, config.yaml's rtsp_url is ignored when these are passed):
    python3 run.py --image sample.jpg
    python3 run.py --folder ./sample_images

Override just the RTSP URL without editing config.yaml:
    python3 run.py --rtsp-url "rtsp://user:pass@192.168.1.50:554/stream1"

Stop cleanly with Ctrl+C (SIGINT) or `systemctl stop <service>` (SIGTERM) -
both are caught below and trigger an orderly shutdown so the SQLite writer
thread gets to flush whatever is still queued.
"""

# argparse handles the small number of CLI overrides listed above
import argparse
# logging gives us leveled, timestamped console output (works well with journald on a Pi)
import logging
# signal lets us catch Ctrl+C and `systemctl stop` and shut down cleanly
import signal
# sys.exit is used for fatal startup errors (bad path, can't open camera, etc.)
import sys
# threading runs the periodic "delete old synced rows" housekeeping job
import threading
# time.sleep paces the housekeeping loop and the reconnect-retry loop
import time

import cv2

from camera.camera import ThreadedRTSPCamera, configure_decode_threads
from config.config import PipelineConfig
from pipeline.pipeline import DetectionPipeline
from storage.storage import DetectionStorage
from detection import detector

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    """Only the handful of things you'd realistically want to override from
    the command line; everything else belongs in config.yaml."""
    parser = argparse.ArgumentParser(description="Vehicle & license plate detection pipeline")
    parser.add_argument("--config", help="Path to a config.yaml (defaults to the one next to this file)")
    parser.add_argument("--rtsp-url", help="Override the RTSP URL from config.yaml")
    parser.add_argument("--image", help="Process a single image instead of a live stream")
    parser.add_argument("--folder", help="Process every image in a folder instead of a live stream")
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    """Simple, journald/systemd-friendly console logging."""
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_on_images(pipeline: DetectionPipeline, image_paths: list) -> None:
    """Test/offline mode: run the same pipeline used for live video over a
    fixed list of image files. Each image is treated as one independent frame."""
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            log.warning("could not read image: %s", path)
            continue
        # single image/folder mode gets exactly one attempt per vehicle - there
        # is no "next frame" to retry plate detection on, so finalize now
        new_vehicles = pipeline.process_frame(frame, always_finalize=True)
        log.info("%s -> %d new vehicle(s)", path, new_vehicles)


def run_on_stream(pipeline: DetectionPipeline, config: PipelineConfig) -> None:
    """Production mode: pull frames from the RTSP camera forever, feeding each
    one through the pipeline as soon as it's the newest available frame."""
    configure_decode_threads(config.camera.decode_threads)

    camera = ThreadedRTSPCamera(
        rtsp_url=config.camera.rtsp_url,
        frame_width=config.camera.frame_width,
        frame_height=config.camera.frame_height,
        buffer_size=config.camera.capture_buffer_size,
        reconnect_delay_seconds=config.camera.reconnect_delay_seconds,
        max_reconnect_attempts=config.camera.max_reconnect_attempts,
    ).start()

    # warm the models up using the real capture resolution so the very first
    # genuine frame isn't slowed down by lazy graph initialization
    pipeline.warmup(frame_shape=(config.camera.frame_height, config.camera.frame_width))

    stop_event = threading.Event()

    def handle_shutdown(signum, _frame):
        log.info("received signal %s - shutting down...", signum)
        stop_event.set()

    # SIGINT = Ctrl+C, SIGTERM = what systemd sends on `systemctl stop`
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    last_processed_frame_number = -1
    frames_processed = 0
    max_frames = config.runtime.max_frames

    log.info("pipeline running - press Ctrl+C to stop")
    try:
        while not stop_event.is_set():
            captured = camera.read_latest()

            # nothing decoded yet (e.g. still connecting) - wait a beat and retry
            if captured is None:
                time.sleep(0.05)
                continue

            # never re-process a frame we've already handled; this is what
            # lets the pipeline naturally skip frames when inference is
            # slower than the camera's frame rate, instead of queuing up a
            # backlog of stale video
            if captured.frame_number <= last_processed_frame_number:
                time.sleep(0.01)
                continue
            last_processed_frame_number = captured.frame_number

            pipeline.process_frame(captured.image)
            frames_processed += 1

            if config.features.show_preview:
                cv2.imshow("pipeline preview", captured.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frames_processed >= max_frames:
                log.info("reached configured max_frames=%d - stopping", max_frames)
                break
    finally:
        camera.stop()
        if config.features.show_preview:
            cv2.destroyAllWindows()


def start_retention_housekeeping(storage: DetectionStorage, retention_days: int) -> None:
    """Once a day, delete rows the dashboard has already marked synced=1 that
    are older than `retention_days`, so the buffer database doesn't grow
    forever on a device with limited storage like a Raspberry Pi's SD card."""
    if retention_days <= 0:
        return

    def loop():
        while True:
            time.sleep(24 * 60 * 60)
            deleted = storage.delete_synced_older_than(retention_days)
            if deleted:
                log.info("retention cleanup: removed %d old synced row(s)", deleted)

    threading.Thread(target=loop, daemon=True).start()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)

    # CLI flags win over whatever is in config.yaml, for quick one-off testing
    if args.rtsp_url:
        config.camera.rtsp_url = args.rtsp_url
    if args.image:
        config.camera.image = args.image
    if args.folder:
        config.camera.folder = args.folder

    setup_logging(config.runtime.log_level)

    storage = DetectionStorage(
        database_path=config.storage.database_path,
        batch_size=config.storage.write_batch_size,
        flush_interval_seconds=config.storage.write_flush_interval_seconds,
        send_via_api=config.features.send_via_api,
        delete_row_after_api_send=config.features.delete_row_after_api_send,
        api_endpoint_url=config.api.endpoint_url,
        api_timeout_seconds=config.api.timeout_seconds,
    ).start()
    start_retention_housekeeping(storage, config.storage.retention_days)

    camera_source = config.camera.rtsp_url or config.camera.folder or config.camera.image or "unknown"
    pipeline = DetectionPipeline(config, storage, camera_source)

    try:
        if config.camera.folder:
            image_paths = detector.list_images_in_folder(config.camera.folder)
            if not image_paths:
                sys.exit(f"No images found in folder: {config.camera.folder}")
            pipeline.warmup()
            run_on_images(pipeline, image_paths)
        elif config.camera.image:
            pipeline.warmup()
            run_on_images(pipeline, [config.camera.image])
        elif config.camera.rtsp_url:
            run_on_stream(pipeline, config)
        else:
            sys.exit("No source configured - set camera.rtsp_url, camera.image, or camera.folder in config.yaml")
    finally:
        # give the background writer thread a chance to flush anything queued
        log.info("flushing storage buffer before exit...")
        storage.stop()


if __name__ == "__main__":
    main()
