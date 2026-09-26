"""
config.py
---------
Every tunable value the pipeline uses lives in ONE place: this file (defaults)
and the optional "config.yaml" that can override them without touching code.

Why this exists as its own module:
    The old version buried ~100 settings inside a dict at the top of a giant
    main.py.  On a Raspberry Pi that runs 24/7, you want to be able to tweak
    a threshold (e.g. camera IP, confidence) by editing a small YAML file,
    not by opening 1800 lines of Python.
"""

# dataclasses give us typed, named fields instead of a loose dictionary
from dataclasses import dataclass, field
# os is used to read environment variable overrides (useful for Docker/systemd)
import os
# pathlib gives us clean, cross-platform path handling
from pathlib import Path
# typing hints make the config self-documenting
from typing import List, Optional, Union

# PyYAML lets operators edit settings in a plain text file instead of code
import yaml


# this file now lives in config/ - PROJECT_ROOT is the actual project folder
# (models/, data/, output/ all live there), CONFIG_DIR is where config.py
# and config.yaml sit alongside each other
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


@dataclass
class CameraConfig:
    """Everything related to reading frames from the camera / video source."""

    # exactly one of these three should be set: rtsp_url, image, or folder
    rtsp_url: Optional[str] = None            # e.g. "rtsp://user:pass@192.168.1.50:554/stream1"
    image: Optional[str] = None               # single test image path
    folder: Optional[str] = None              # folder of test images

    # requested capture resolution (the camera may not honour this exactly)
    frame_width: int = 1280
    frame_height: int = 720

    # OpenCV's RTSP buffer size; 1 means "always show me the newest frame",
    # which matters a lot on a Pi because we never want to process a stale frame
    capture_buffer_size: int = 1

    # how many CPU threads FFmpeg's software H.264 decoder is allowed to use.
    # Raspberry Pi 5 has no hardware H.264 decode for RTSP in OpenCV/FFmpeg,
    # so decoding eats real CPU. Pinning this keeps decode from starving the
    # inference threads (this was the cause of "Could not find ref with POC #"
    # style errors in the original code).
    decode_threads: int = 2

    # cap on frames handed to the pipeline / live view per second (0 = no cap).
    # Frames above the cap are still decoded (H.264 needs every frame) but skip
    # the colour conversion + copy. The best fix is still to lower the frame
    # rate in the camera's own web UI, which cuts the decode itself.
    max_fps: float = 0.0

    # seconds to wait between reconnect attempts after the stream drops
    reconnect_delay_seconds: float = 5.0

    # 0 = retry forever (correct for a 24/7 deployment); N = give up after N tries
    max_reconnect_attempts: int = 0

    # -- ISAPI (Hikvision) device-info lookup --
    # fetches this camera's own configured name and (if set) installation
    # location once at startup, straight from the camera, instead of hand
    # -typing them into api_client.py - see camera/isapi_client.py. Only
    # applies in rtsp_url mode; a camera that doesn't answer just leaves
    # name/location empty in the API metadata (the IP address is still sent).
    isapi_enabled: bool = True
    isapi_port: int = 80
    isapi_https: bool = False
    isapi_username: Optional[str] = None      # falls back to the RTSP URL's own user:pass if unset
    isapi_password: Optional[str] = None
    isapi_timeout_seconds: float = 4.0


@dataclass
class RoiConfig:
    """Normalized (0..1) region of the frame that counts as 'the road'.

    Detections whose center falls outside this shape are ignored. This lets
    you point a wide camera at an intersection but only trigger on the lane
    you actually want to log.

    `polygon` is a list of [x, y] points, normalized 0..1 against the frame,
    given in order (clockwise or counter-clockwise, doesn't matter) tracing
    the outline of the region - NOT a bounding box. A vehicle is "in the ROI"
    when its center point falls inside this polygon (ray-casting test, see
    `detection/geometry.py::point_in_polygon`), which lets the ROI follow an
    angled lane or curb instead of only ever being an axis-aligned rectangle.

    The front end (dashboard) lets an operator draw this polygon over the
    live view and POSTs the resulting points to `PUT/POST /live/roi` on
    `live/live_server.py`, which updates this list at runtime; `GET /live/roi`
    reads it back so the UI can redraw the currently active shape.
    """

    polygon: List[List[float]] = field(default_factory=lambda: [
        [0.15, 0.15], [0.85, 0.15], [0.85, 0.90], [0.15, 0.90],
    ])

    # when features.roi_crop_detect is on, the vehicle model is fed only the
    # ROI plus this much extra on every side (fraction of the frame), so a
    # vehicle straddling the ROI edge is still seen whole and tracked steadily
    crop_margin: float = 0.05

    # reject vehicle boxes that touch the very edge of the ROI (usually a
    # vehicle that is only half-visible, which gives a bad plate crop)
    edge_margin_ratio: float = 0.015

    # reject detections that are too small (far away / noise) or too large
    # (right in front of the lens, usually a bad crop)
    min_width_ratio: float = 0.025
    min_height_ratio: float = 0.04
    max_width_ratio: float = 0.85
    max_height_ratio: float = 0.90


@dataclass
class ModelConfig:
    """Where the two YOLO models live and how they run."""

    # folder or .pt file for the vehicle detector
    vehicle_weights: str = str(PROJECT_ROOT / "models" / "vehicle_ncnn_model")
    # folder or .pt file for the plate detector
    plate_weights: str = str(PROJECT_ROOT / "models" / "platenum_closeup_ncnn_model")

    # None = let ultralytics choose (CPU on a Pi, GPU if one is present)
    device: Optional[str] = None

    # CPU threads ONNX Runtime may use for the .onnx models (ignored by
    # NCNN/.pt). 0 = auto (cores - 1, min 2 -> 3 on a Pi 5);
    # -1 = no limit; N = exactly N. Lower = smoother video/live view but slower
    # detection; raise it if detection is too slow, lower it if the video stutters.
    inference_threads: int = 0

    # inference image size fed to EACH model. These are separate because a
    # static-shape NCNN export is compiled for exactly one input size -
    # if you exported vehicle.pt at --imgsz 480 (a common choice: the vehicle
    # model scans the whole frame, so a smaller size = faster), it will ONLY
    # accept 480x480 input. Feeding it 640x640 here throws a shape-mismatch
    # error from NCNN. These numbers MUST match whatever --imgsz you used
    # when exporting each model (see export_ncnn.py).
    # vehicle_imgsz may be an int (square) or [height, width] for a rectangular
    # export, e.g. [512, 896] (see export_ncnn.py / the README).
    vehicle_imgsz: Union[int, List[int]] = 480
    plate_imgsz: int = 640

    # confidence / NMS thresholds per stage
    vehicle_conf_threshold: float = 0.35
    vehicle_iou_threshold: float = 0.45
    plate_conf_threshold: float = 0.35

    # comma separated class allow-list, e.g. "car,truck,bus,motorcycle";
    # empty/None means "accept every class the model knows"
    vehicle_classes: Optional[str] = None


@dataclass
class TrackingConfig:
    """How we keep 'the same physical vehicle' from being logged twice."""

    # ByteTrack config shipped alongside this package
    bytetrack_config: str = str(CONFIG_DIR / "bytetrack_custom.yaml")

    # fallback tracker thresholds, used only if ByteTrack fails to assign IDs
    iou_threshold: float = 0.3
    max_center_distance: float = 2.0        # in units of "vehicle widths"
    max_missing_frames: int = 30            # frames a track survives with no match

    # once a vehicle crop is saved, ignore new detections in roughly the same
    # spot for this many seconds (handles a car waiting at a red light)
    dedup_cooldown_seconds: float = 2.0
    dedup_position_threshold: float = 1.0   # in units of "vehicle widths"

    # plate-detection attempts per tracking ID. 1 = a NEW track id gets exactly
    # one plate pass and every later frame with the SAME id is skipped. If you
    # raise it, each retry re-crops the vehicle from the current frame (it used
    # to re-run the model on the identical first-sighting crop, which can only
    # ever give the same answer).
    max_plate_attempts: int = 1

    # with max_plate_attempts > 1 the pipeline keeps the BEST plate seen so far (highest
    # OCR score, then plate confidence) and only stops early once a read is good enough
    # (ocr.accept_score). Retries are spaced this far apart so each one looks at a
    # genuinely different frame - at 10 fps, consecutive frames are near-identical.
    plate_retry_interval_seconds: float = 0.4
    # a vehicle still waiting for a good plate is stored as it stands (best so far) once
    # it has been out of the ROI this long, so the dashboard never waits on track_ttl_seconds
    plate_stale_finalize_seconds: float = 1.5

    # forget a track (and free its crop) once it hasn't been seen for this long.
    # A vehicle that vanishes before a plate was found is stored as "no plate".
    track_ttl_seconds: float = 30.0


@dataclass
class CropConfig:
    """Padding / minimum sizes used when cutting boxes out of frames."""

    vehicle_padding_ratio: float = 0.08      # extra margin around a vehicle box
    vehicle_min_crop_height: int = 200       # upscale small vehicle crops to at least this

    plate_padding_pixels: int = 4            # extra margin around a plate box (minimum)
    plate_padding_ratio: float = 0.12        # ...or this fraction of the plate box height, whichever is larger
    plate_min_crop_height: int = 160         # upscale small plate crops to at least this tall


@dataclass
class PreprocessConfig:
    """Non-toggle tuning values for image enhancement steps.

    Whether these steps RUN AT ALL is controlled centrally in
    `FeaturesConfig` (`enhance_before_plate_detect` / `enhance_plate_crop`)
    - this section only holds the numeric knobs for them.
    """

    plate_crop_min_height: int = 64


@dataclass
class OcrConfig:
    """Settings for the plate-text OCR pass that runs right after a plate
    crop is produced (see FeaturesConfig.ocr_read for the on/off switch,
    FeaturesConfig.async_ocr for running it off the detection loop)."""

    # PaddleOCR support has been removed - fast_plate_ocr (a small
    # purpose-built ONNX plate classifier) is the only backend, not just the
    # default. It assumes single-line plates - see ocr/fast_plate_reader.py
    # for the 2-row/motorcycle-plate caveat.
    # hub model fast-plate-ocr downloads/uses. "-xs-" is the smallest/fastest;
    # "cct-s-v2-global-model" trades some speed for accuracy if xs isn't
    # reading your plates well enough
    fast_plate_ocr_model: str = "cct-xs-v2-global-model"
    # CPU threads the OCR engine's own inference session may use. Keep at 1
    # on a Pi 5: OCR runs on its own worker thread(s) (ocr.worker_threads)
    # already, and letting its native math library also spread across every
    # core just re-creates the contention that moving it off the hot loop
    # was meant to avoid.
    cpu_threads: int = 1
    # recognitions below this confidence are treated as unreadable -> "Unrecognized"
    min_confidence: float = 0.5
    # characters allowed in a cleaned-up plate read; anything else is stripped
    allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # text used whenever a plate crop exists but OCR could not read it
    # (crop too blurry/dark, recognizer errored, confidence too low, etc.)
    unrecognized_text: str = "Unrecognized"
    # an OCR quality score (0..1, see FastPlateOCRReader.read_scored) at or above this
    # ends the plate retries for a vehicle; below it, the next attempt may do better
    accept_score: float = 0.75
    # once one image variant scores this high, the other variants are not tried (saves CPU)
    early_exit_score: float = 0.85
    # how many background threads run OCR + finalize (see FeaturesConfig.async_ocr).
    # OCR is CPU-heavy: 1 is usually right for a Pi 5 (leaves cores for the
    # vehicle/plate models + RTSP decode + live view); raise to 2 only if the
    # finalize queue keeps backing up (see the "[finalize] queue full" log line)
    # AND model.inference_threads leaves enough cores spare.
    worker_threads: int = 1
    # how many vehicles can be queued for OCR/finalize at once before new ones are
    # dropped (logged loudly) rather than blocking the detection loop
    finalize_queue_size: int = 32


@dataclass
class StorageConfig:
    """Where results go: the SQLite buffer database (and, optionally, disk files)."""

    # SQLite file used as the hand-off buffer between this pipeline and the
    # C# dashboard. WAL mode (enabled in storage.py) lets the dashboard read
    # while this process keeps writing.
    database_path: str = str(PROJECT_ROOT / "data" / "pipeline_buffer.db")

    # whether a JPEG copy of every crop is ALSO kept on disk is controlled
    # centrally in FeaturesConfig.save_images_to_disk - this is just where
    # it's written to when that's on.
    output_dir: str = str(PROJECT_ROOT / "output")

    # JPEG quality used both for the DB blob and the on-disk copy
    jpeg_quality: int = 90
    # the plate crop is small, so its JPEG costs little - store it near-lossless
    # (this is the image the C# side receives, and any re-OCR there starts from it)
    plate_jpeg_quality: int = 96

    # background writer batches this many rows per SQLite transaction
    write_batch_size: int = 8
    # and flushes at least this often even if the batch isn't full yet
    write_flush_interval_seconds: float = 1.0

    # delete rows older than this many days that the dashboard has already
    # marked as synced=1; 0 disables cleanup (keep forever). The same age limit
    # is applied to the on-disk JPEG copies under output_dir.
    retention_days: int = 14

    # rows that were never delivered (dashboard/API down for a long time) are
    # only deleted after this many days; 0 = keep them until they are sent
    max_unsynced_days: int = 0

    # API mode: remove a vehicle's on-disk JPEG copies as soon as its record
    # has been delivered (or skipped because it has no plate)
    delete_disk_images_after_send: bool = True

    # API mode: how often to retry rows whose POST failed
    send_retry_seconds: float = 60.0


@dataclass
class RuntimeConfig:
    """Misc. operational knobs."""

    log_level: str = "INFO"
    # threads OpenCV may use for resize/colour/CLAHE. Left at its default it
    # grabs every core and fights the model + decoder for them.
    opencv_threads: int = 2
    # finite run for testing; 0 = run forever (production/24-7 mode)
    max_frames: int = 0


@dataclass
class ApiConfig:
    """Placeholder settings for sending detections to the C# dashboard's API.

    `endpoint_url` is None until the C# dev provides the real URL - leaving
    it unset is safe even with `features.send_via_api: true`; sends just
    fail closed and the row stays in SQLite as normal.
    """

    endpoint_url: Optional[str] = None      # e.g. "https://dashboard.local/api/detections"
    timeout_seconds: float = 5.0

    # -- ROI coordinates, fetched FROM the C# dashboard's own API --
    # the dashboard's front end lets an operator draw a quadrilateral over the
    # live view and POSTs it there (see api/roi_client.py's docstring for the
    # exact JSON shape); we GET it back from the same API so detection uses
    # whatever the operator most recently drew, instead of a hard-coded shape.
    roi_endpoint_url: Optional[str] = None  # e.g. "http://192.168.100.14:5086/api/AiDetection/roi-coordinates"
    camera_id: Optional[int] = None         # which camera's ROI to ask for (the API's cameraId)
    roi_fetch_timeout_seconds: float = 5.0
    # toggle: the endpoint owner says it may send raw pixel coordinates instead
    # of the 0..1 normalized values we originally confirmed - flip this on if/
    # when a real sample response shows numbers like 640/1080 instead of 0..1.
    # False (default) = trust the values as already normalized, no conversion.
    roi_coordinates_are_pixels: bool = False
    # pixel width/height the roi_x/roi_y values are relative to, ONLY used
    # when roi_coordinates_are_pixels is True. None = fall back to this
    # camera's own camera.frame_width / camera.frame_height. Set these
    # explicitly if the operator draws on a DIFFERENT canvas size than that
    # (e.g. a resized live-view player) - ask the front-end dev what canvas
    # width/height the mouse coordinates were captured against.
    roi_reference_width: Optional[int] = None
    roi_reference_height: Optional[int] = None
    # how often to re-fetch after the one at startup, so an operator redrawing
    # the ROI on the dashboard takes effect without restarting the Pi service;
    # 0 = fetch once at startup only, never refresh
    roi_poll_interval_seconds: float = 30.0


@dataclass
class LiveConfig:
    """Live bounding-box feed for the command center (see live/box_publisher.py).

    Two independent ways to get the live view to the command center:

      * features.live_stream (RECOMMENDED) - the Pi SERVES frames + boxes over
        HTTP and the dashboard pulls them (see live/live_server.py). Needs no
        endpoint on the server side.
      * features.live_boxes - the Pi PUSHES boxes (JSON only, no frames) to
        `endpoint_url` (see live/box_publisher.py). Needs a server endpoint.
    """

    # -- shared --
    camera_id: str = "cam1"                 # short, credential-free name for this camera

    # -- push mode (features.live_boxes) --
    endpoint_url: Optional[str] = None      # e.g. "http://server:8000/api/live/boxes"
    max_hz: float = 10.0                    # never send more often than this
    timeout_seconds: float = 1.0            # short on purpose: a slow server must not pile up

    # -- serve mode (features.live_stream) --
    serve_host: str = "0.0.0.0"             # "127.0.0.1" = this machine only
    serve_port: int = 8090
    stream_fps: float = 10.0                # max frames/second per viewer
    stream_width: int = 960                 # downscale to this width before encoding; 0 = full size
    jpeg_quality: int = 70                  # 1-100; lower = less bandwidth and CPU
    box_max_age_seconds: float = 3.0        # boxes older than this are not drawn (detector stalled)
    box_visual_tracking: bool = True        # follow the picture inside each box between detections (smooth from the first sighting)
    box_extrapolate: bool = True            # fallback: slide boxes along each vehicle's measured speed between detections
    box_extrapolate_max_seconds: float = 3.5  # never project further ahead than this (inference delay + gap to next detection)
    auth_token: Optional[str] = None        # if set, viewers must send it (?token= or Bearer header)


@dataclass
class FeaturesConfig:
    """ONE centralized on/off switchboard for the whole pipeline.

    Every optional behavior - which pipeline stages run, what gets timed,
    what gets printed to the console, and which individual SQL columns get
    populated - lives here as a plain True/False. Flip any of these in
    config.yaml under `features:` without touching any other file.

    A column toggle set to False does not remove the column from the
    database (the schema is fixed so the C# dashboard can always rely on
    it) - it just leaves that column NULL/empty instead of computing and
    storing the value, which is what "off" means for image/box data too.
    """

    # ---- pipeline stage / behavior toggles ----
    roi_filter: bool = True                    # drop detections outside the configured ROI
    roi_crop_detect: bool = True               # feed the vehicle model only the ROI (+margin) window instead of the whole frame
    fallback_tracker: bool = True               # use the simple tracker when ByteTrack can't assign an ID
    position_dedup: bool = True                 # skip re-saving a vehicle that's just sitting still
    plate_detection: bool = True                # run the plate model at all (False = vehicles only)
    enhance_before_plate_detect: bool = True     # CLAHE + sharpen the vehicle crop before plate detection
    enhance_plate_crop: bool = True              # CLAHE + sharpen + upscale the saved plate crop
    ocr_read: bool = True                        # run OCR on a saved plate crop to read its text
    async_ocr: bool = True                       # run OCR + record-build/store on background thread(s) instead of
                                                  # inline in the detection loop - keeps the frame loop free of
                                                  # OCR's latency; see pipeline/finalize_worker.py. The plate
                                                  # BOX for a vehicle is still picked by plate-detector confidence
                                                  # (not OCR quality) when this is on. Set False to restore the old
                                                  # fully-synchronous, OCR-gated-retry behavior.
    save_images_to_disk: bool = True             # also keep a JPEG copy under output/
    show_preview: bool = False                   # live OpenCV preview window (keep OFF on a headless Pi)
    live_boxes: bool = False                     # PUSH per-frame boxes (JSON) to live.endpoint_url
    live_stream: bool = False                    # SERVE live frames + boxes over HTTP for the dashboard to pull
    roi_fetch_from_api: bool = True              # GET the ROI polygon from api.roi_endpoint_url at startup
                                                  # (and again every api.roi_poll_interval_seconds) instead of
                                                  # only ever using the polygon hard-coded in config.yaml

    # ---- per-stage timing measurement ----
    time_vehicle_detect: bool = True
    time_vehicle_crop: bool = True
    time_plate_detect: bool = True
    time_plate_crop: bool = True

    # ---- console output ----
    print_console: bool = True                  # any per-vehicle console line at all
    print_timing: bool = True                    # include the ms breakdown in that line

    # ---- SQLite storage ----
    store_to_sqlite: bool = True                 # master switch: write rows to the buffer DB at all

    # ---- API delivery (placeholder until the C# dashboard endpoint exists) ----
    send_via_api: bool = False                   # POST each record to api.endpoint_url
    delete_row_after_api_send: bool = True        # remove the SQLite row once the API send succeeds

    # individual columns - each can be turned off independently
    col_vehicle_image: bool = True
    col_vehicle_box: bool = True
    col_plate_image: bool = True
    col_plate_box: bool = True
    col_vehicle_detect_ms: bool = True
    col_vehicle_crop_ms: bool = True
    col_plate_detect_ms: bool = True
    col_plate_crop_ms: bool = True
    col_total_pipeline_ms: bool = True
    col_ocr_read: bool = True


@dataclass
class PipelineConfig:
    """The complete, nested configuration object the rest of the app uses."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "PipelineConfig":
        """Build a config: defaults, then overridden by config.yaml if present."""
        # start from the built-in, safe defaults defined above
        config = cls()

        # if no explicit path was given, look for "config.yaml" next to this file
        if yaml_path is None:
            candidate = CONFIG_DIR / "config.yaml"
            yaml_path = str(candidate) if candidate.exists() else None

        # no file to load -> just return the defaults
        if not yaml_path or not os.path.isfile(yaml_path):
            return config

        # read the YAML file from disk
        with open(yaml_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        # walk each top-level section (camera, roi, model, ...) and overwrite
        # only the fields the user actually specified, leaving the rest default
        for section_name, section_values in raw.items():
            if not hasattr(config, section_name) or not isinstance(section_values, dict):
                continue
            section_obj = getattr(config, section_name)
            for key, value in section_values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

        return config
