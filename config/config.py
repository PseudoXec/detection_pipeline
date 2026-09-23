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
from dataclasses import dataclass, field, asdict
# os is used to read environment variable overrides (useful for Docker/systemd)
import os
# pathlib gives us clean, cross-platform path handling
from pathlib import Path
# typing hints make the config self-documenting
from typing import Optional

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

    # seconds to wait between reconnect attempts after the stream drops
    reconnect_delay_seconds: float = 5.0

    # 0 = retry forever (correct for a 24/7 deployment); N = give up after N tries
    max_reconnect_attempts: int = 0


@dataclass
class RoiConfig:
    """Normalized (0..1) region of the frame that counts as 'the road'.

    Detections outside this box are ignored. This lets you point a wide
    camera at an intersection but only trigger on the lane you actually
    want to log.
    """

    x_min: float = 0.15
    x_max: float = 0.85
    y_min: float = 0.15
    y_max: float = 0.90

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

    # inference image size fed to EACH model. These are separate because a
    # static-shape NCNN export is compiled for exactly one input size -
    # if you exported vehicle.pt at --imgsz 480 (a common choice: the vehicle
    # model scans the whole frame, so a smaller size = faster), it will ONLY
    # accept 480x480 input. Feeding it 640x640 here throws a shape-mismatch
    # error from NCNN. These numbers MUST match whatever --imgsz you used
    # when exporting each model (see export_openvino.py / the ncnn export).
    vehicle_imgsz: int = 480
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

    # give up trying to find a plate after this many attempts for one vehicle
    max_plate_attempts: int = 3


@dataclass
class CropConfig:
    """Padding / minimum sizes used when cutting boxes out of frames."""

    vehicle_padding_ratio: float = 0.08      # extra margin around a vehicle box
    vehicle_min_crop_height: int = 200       # upscale small vehicle crops to at least this

    plate_padding_pixels: int = 8            # extra margin around a plate box
    plate_min_crop_height: int = 120

    # plate containment / IoU used to decide "which vehicle does this plate belong to"
    plate_containment_threshold: float = 0.8
    plate_min_iou: float = 0.1


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
    """Settings for the inline plate-text OCR pass that runs right after a
    plate crop is produced (see FeaturesConfig.ocr_read for the on/off
    switch)."""

    # language model PaddleOCR's recognizer loads
    lang: str = "en"
    # recognitions below this confidence are treated as unreadable -> "Unrecognized"
    min_confidence: float = 0.5
    # characters allowed in a cleaned-up plate read; anything else is stripped
    allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # text used whenever a plate crop exists but OCR could not read it
    # (crop too blurry/dark, recognizer errored, confidence too low, etc.)
    unrecognized_text: str = "Unrecognized"


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

    # background writer batches this many rows per SQLite transaction
    write_batch_size: int = 8
    # and flushes at least this often even if the batch isn't full yet
    write_flush_interval_seconds: float = 1.0

    # delete rows older than this many days that the dashboard has already
    # marked as synced=1; 0 disables cleanup (keep forever)
    retention_days: int = 14


@dataclass
class RuntimeConfig:
    """Misc. operational knobs."""

    log_level: str = "INFO"
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
    fallback_tracker: bool = True               # use the simple tracker when ByteTrack can't assign an ID
    position_dedup: bool = True                 # skip re-saving a vehicle that's just sitting still
    plate_detection: bool = True                # run the plate model at all (False = vehicles only)
    enhance_before_plate_detect: bool = True     # CLAHE + sharpen the vehicle crop before plate detection
    enhance_plate_crop: bool = True              # CLAHE + sharpen + upscale the saved plate crop
    ocr_read: bool = True                        # run OCR on a saved plate crop to read its text
    save_images_to_disk: bool = True             # also keep a JPEG copy under output/
    show_preview: bool = False                   # live OpenCV preview window (keep OFF on a headless Pi)

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

    def as_dict(self) -> dict:
        """Flat dict view, mostly useful for debug logging at startup."""
        return asdict(self)
