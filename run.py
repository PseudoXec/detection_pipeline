"""Entry point: reads the settings, starts the services every detection mode shares, then feeds frames
to the mode manager. Everything with a real job lives in its own folder - this file only wires them up.

    python run.py                       run from the camera, in the mode the command center last chose
    python run.py --mode person         start in a specific mode
    python run.py --image a.jpg         one image          python run.py --folder ./imgs    a folder of images
"""
import argparse
import logging
import sys
import uuid

import cv2

from camera.frame_loop import list_images_in_folder, run_on_images, run_on_stream
from camera.resolve import resolve_camera_info, resolve_camera_source
from camera.roi_manager import RoiManager
from config.config import PipelineConfig
from control.mode_manager import ModeManager, choose_initial_mode
from live.outputs import build_live_outputs
from live.sinks import LiveSinks
from modes.base import RuntimeContext
from storage.hub import StorageHub

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection pipeline (vehicle / person), switchable from the command center")
    parser.add_argument("--config", help="Optional extra config file, applied after all the config.yaml files (see config/config.py)")
    parser.add_argument("--rtsp-url", help="Override camera.rtsp_url")
    parser.add_argument("--image", help="Process a single image instead of a live stream")
    parser.add_argument("--folder", help="Process every image in a folder instead of a live stream")
    parser.add_argument("--mode", help="Start in this detection mode (vehicle | person | idle), overriding the saved choice and mode.default")
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)

    if args.rtsp_url:
        config.camera.rtsp_url = args.rtsp_url
    if args.image:
        config.camera.image = args.image
    if args.folder:
        config.camera.folder = args.folder

    setup_logging(config.runtime.log_level)

    if not config.switches.api:
        log.info("[api] switches.api is off - ROI fetch, camera source fetch and data sending are off")

    cv2.setNumThreads(max(1, config.runtime.opencv_threads))

    if not args.rtsp_url:
        resolve_camera_source(config)

    camera_info = resolve_camera_info(config)
    camera_source = config.camera.rtsp_url or config.camera.folder or config.camera.image or "unknown"
    initial_mode = choose_initial_mode(config, args.mode)

    # ---- everything below is SHARED by all detection modes and lives for the whole run -----------
    stores = StorageHub(config, camera_info).start()
    roi = RoiManager(config)
    roi.start_polling()
    sinks = LiveSinks()
    ctx = RuntimeContext(
        config=config, camera_source=camera_source, camera_info=camera_info,
        stores=stores, roi=roi, live=sinks, session_id=uuid.uuid4().hex,
    )
    manager = ModeManager(
        ctx, allowed=config.mode.allowed, state_file=config.mode.state_file,
        remember_last=config.mode.remember_last, preload_on_switch=config.mode.preload_on_switch,
    )
    publisher, server = build_live_outputs(config, roi, manager, sinks)

    try:
        if config.camera.folder:
            image_paths = list_images_in_folder(config.camera.folder)
            if not image_paths:
                sys.exit(f"No images found in folder: {config.camera.folder}")
            manager.start(initial_mode)
            run_on_images(manager, image_paths)
        elif config.camera.image:
            manager.start(initial_mode)
            run_on_images(manager, [config.camera.image])
        elif config.camera.rtsp_url:
            run_on_stream(manager, config, server, initial_mode)
        else:
            sys.exit("No source configured - set camera.rtsp_url, camera.image, or camera.folder (camera/config.yaml)")
    finally:
        manager.stop()                     # flushes the active mode's in-flight tracks into the stores
        if publisher is not None:
            publisher.stop()
        if server is not None:
            server.stop()
        roi.stop()
        log.info("flushing storage buffer before exit...")
        stores.stop()


if __name__ == "__main__":
    main()
