"""Feeding frames to the detection mode manager: from a live stream, one image, or a folder of images."""
import logging
import os
import signal
import threading
import time
from typing import List

import cv2

from camera.camera import ThreadedRTSPCamera, configure_decode_threads
from config.config import PipelineConfig

log = logging.getLogger("pipeline")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def run_on_images(manager, image_paths: list) -> None:
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            log.warning("could not read image: %s", path)
            continue
        new_detections = manager.process_frame(frame, always_finalize=True)
        log.info("%s -> %d new detection(s)", path, new_detections)


def run_on_stream(manager, config: PipelineConfig, live_server, initial_mode: str) -> None:
    configure_decode_threads(config.camera.decode_threads)

    camera = ThreadedRTSPCamera(
        rtsp_url=config.camera.rtsp_url,
        frame_width=config.camera.frame_width,
        frame_height=config.camera.frame_height,
        buffer_size=config.camera.capture_buffer_size,
        reconnect_delay_seconds=config.camera.reconnect_delay_seconds,
        max_reconnect_attempts=config.camera.max_reconnect_attempts,
        max_fps=config.camera.max_fps,
    ).start()

    if live_server is not None:
        live_server.set_frame_source(camera.read_latest)
        live_server.start()

    manager.start(initial_mode, frame_shape=(config.camera.frame_height, config.camera.frame_width))

    stop_event = threading.Event()
    shutdown_requests = 0

    def handle_shutdown(signum, _frame):
        nonlocal shutdown_requests
        shutdown_requests += 1
        if shutdown_requests == 1:
            log.info("received signal %s - shutting down... (press Ctrl+C again to force-quit immediately)", signum)
            stop_event.set()
        else:
            log.info("second interrupt received - forcing immediate exit")
            os._exit(1)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    last_processed_frame_number = -1
    frames_processed = 0
    max_frames = config.runtime.max_frames
    heartbeat_interval_seconds = 15.0
    last_heartbeat = time.time()
    last_heartbeat_frame_count = 0
    waited_for_first_frame = False

    log.info("pipeline running - press Ctrl+C to stop")
    try:
        while not stop_event.is_set():
            captured = camera.read_latest()

            if captured is None:
                if not waited_for_first_frame and time.time() - last_heartbeat > 5.0:
                    log.info("still waiting on the first frame from the camera...")
                    last_heartbeat = time.time()
                time.sleep(0.05)
                continue
            waited_for_first_frame = True

            if captured.frame_number <= last_processed_frame_number:
                time.sleep(0.01)
                continue
            last_processed_frame_number = captured.frame_number

            manager.process_frame(captured.image, captured_at=captured.captured_at)
            frames_processed += 1

            if time.time() - last_heartbeat >= heartbeat_interval_seconds:
                new_frames = frames_processed - last_heartbeat_frame_count
                log.info(
                    "heartbeat: %d frame(s) processed in the last %.0fs (%d total) - still running",
                    new_frames, heartbeat_interval_seconds, frames_processed,
                )
                last_heartbeat = time.time()
                last_heartbeat_frame_count = frames_processed

            if config.switches.show_preview:
                cv2.imshow("pipeline preview", captured.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frames_processed >= max_frames:
                log.info("reached configured max_frames=%d - stopping", max_frames)
                break
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt caught directly - shutting down...")
    finally:
        camera.stop()
        if config.switches.show_preview:
            cv2.destroyAllWindows()
