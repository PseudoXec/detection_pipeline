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


# folder this file lives in -> used to build default paths that work no
# matter where the pipeline is installed (important once it's deployed to a Pi)
BASE_DIR = Path(__file__).resolve().parent


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
    vehicle_weights: str = str(BASE_DIR / "models" / "vehicle_openvino_model")
    # folder or .pt file for the plate detector
    plate_weights: str = str(BASE_DIR / "models" / "platenum_closeup_openvino_model")

    # None = let ultralytics choose (CPU on a Pi, GPU if one is present)
    device: Optional[str] = None

    # inference image size fed to EACH model. These are separate because a
    # static-shape OpenVINO export is compiled for exactly one input size -
    # if you exported vehicle.pt at --imgsz 480 (a common choice: the vehicle
    # model scans the whole frame, so a smaller size = faster), it will ONLY
    # accept 480x480 input. Feeding it 640x640 here throws a shape-mismatch
    # error from OpenVINO. These numbers MUST match whatever --imgsz you
    # used in export_openvino.py for each model.
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
    bytetrack_config: str = str(BASE_DIR / "bytetrack_custom.yaml")

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
    """Optional image enhancement steps.

    These exist to help the plate model find small/low-contrast plates, but
    they are NOT free: CLAHE contrast boosting + sharpening can also
    introduce artifacts (halos, exaggerated noise, blown-out highlights)
    that make some models detect WORSE, not better - this depends heavily
    on how the model itself was trained. If you notice accuracy drop after
    enabling either flag below, turn it off; the model then sees the raw
    camera crop untouched.
    """

    # sharpen/denoise the vehicle crop before running plate detection on it.
    # Turn this OFF first if plate-detection accuracy looks worse than
    # expected - this is the step most likely to hurt a model that was
    # trained on plain, un-enhanced crops.
    enhance_before_plate_detect: bool = True
    # sharpen/denoise/upscale the final plate crop before saving it.
    # This one only affects what gets SAVED/stored, not detection itself,
    # so it's safe to leave on even if you turn the flag above off.
    enhance_plate_crop: bool = True
    plate_crop_min_height: int = 64


@dataclass
class StorageConfig:
    """Where results go: the SQLite buffer database (and, optionally, disk files)."""

    # SQLite file used as the hand-off buffer between this pipeline and the
    # C# dashboard. WAL mode (enabled in storage.py) lets the dashboard read
    # while this process keeps writing.
    database_path: str = str(BASE_DIR / "data" / "pipeline_buffer.db")

    # keep a JPEG copy of every crop on disk as well as inside the database.
    # Handy for manual inspection / debugging; can be turned off to save
    # SD-card writes on a Pi that runs 24/7.
    save_images_to_disk: bool = True
    output_dir: str = str(BASE_DIR / "output")

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
    # show a live OpenCV preview window; turn this OFF on a headless Pi
    show_preview: bool = False
    # finite run for testing; 0 = run forever (production/24-7 mode)
    max_frames: int = 0


@dataclass
class PipelineConfig:
    """The complete, nested configuration object the rest of the app uses."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "PipelineConfig":
        """Build a config: defaults, then overridden by config.yaml if present."""
        # start from the built-in, safe defaults defined above
        config = cls()

        # if no explicit path was given, look for "config.yaml" next to this file
        if yaml_path is None:
            candidate = BASE_DIR / "config.yaml"
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
