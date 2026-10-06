"""Every setting that only vehicle mode uses: models, thresholds, tracking, crops, OCR, ROI rules, features."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

HERE = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"vehicle_weights": "here", "plate_weights": "here"}

    vehicle_weights: str = str(HERE / "models" / "vehicle_ncnn_model")
    plate_weights: str = str(HERE / "models" / "platenum_closeup_ncnn_model")

    vehicle_imgsz: Union[int, List[int]] = 480
    plate_imgsz: int = 640

    vehicle_conf_threshold: float = 0.35
    vehicle_iou_threshold: float = 0.40
    plate_conf_threshold: float = 0.35

    vehicle_classes: Optional[str] = None
    agnostic_nms: bool = True


@dataclass
class TrackingConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"bytetrack_config": "here"}

    bytetrack_config: str = str(HERE / "bytetrack.yaml")

    iou_threshold: float = 0.3
    max_center_distance: float = 2.0
    max_missing_frames: int = 30

    dedup_cooldown_seconds: float = 2.0
    dedup_position_threshold: float = 1.0
    detector_conf_floor: float = 0.10
    plate_dedup_seconds: float = 30.0

    max_plate_attempts: int = 1
    plate_retry_interval_seconds: float = 0.4
    plate_stale_finalize_seconds: float = 1.5

    track_ttl_seconds: float = 30.0
    plate_worker_max_queue: int = 12  # background plate-detection queue depth, see features.async_plate_detection


@dataclass
class CropConfig:
    vehicle_padding_ratio: float = 0.08
    vehicle_min_crop_height: int = 200

    plate_padding_pixels: int = 4
    plate_padding_ratio: float = 0.12
    plate_min_crop_height: int = 160


@dataclass
class PreprocessConfig:
    plate_crop_min_height: int = 64


@dataclass
class OcrConfig:
    fast_plate_ocr_model: str = "cct-xs-v2-global-model"
    cpu_threads: int = 1
    min_confidence: float = 0.5
    allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    plate_formats: List[str] = field(default_factory=lambda: ["LLLDDDD"])
    plate_format_strict: bool = True
    plate_separator: str = " "
    unrecognized_text: str = "Unrecognized"
    accept_score: float = 0.75
    early_exit_score: float = 0.85
    worker_threads: int = 1
    finalize_queue_size: int = 32


@dataclass
class RoiRulesConfig:
    """What counts as 'inside the ROI' for a vehicle (the polygon itself belongs to the camera)."""
    crop_margin: float = 0.05            # extra margin around the polygon when the model only looks at the ROI window
    edge_margin_ratio: float = 0.005
    min_width_ratio: float = 0.012
    min_height_ratio: float = 0.02
    max_width_ratio: float = 0.85
    max_height_ratio: float = 0.90


@dataclass
class ImagesConfig:
    jpeg_quality: int = 90
    plate_jpeg_quality: int = 96


@dataclass
class FeaturesConfig:
    roi_filter: bool = True
    roi_crop_detect: bool = True
    fallback_tracker: bool = True
    position_dedup: bool = True
    plate_detection: bool = True
    async_plate_detection: bool = True   # run the plate model on a background thread (modes/vehicle/plate_worker.py)
    enhance_before_plate_detect: bool = True
    enhance_plate_crop: bool = True
    ocr_read: bool = True
    async_ocr: bool = True
    save_images_to_disk: bool = True

    time_vehicle_detect: bool = True
    time_vehicle_crop: bool = True
    time_plate_detect: bool = True
    time_plate_crop: bool = True

    print_console: bool = True
    print_timing: bool = True

    store_to_sqlite: bool = True

    send_via_api: bool = False
    delete_row_after_api_send: bool = True

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
class VehicleConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    roi: RoiRulesConfig = field(default_factory=RoiRulesConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
