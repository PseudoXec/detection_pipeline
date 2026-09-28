from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional, Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


@dataclass
class CameraConfig:
    rtsp_url: Optional[str] = None
    image: Optional[str] = None
    folder: Optional[str] = None

    source_mode: str = "static"

    frame_width: int = 1280
    frame_height: int = 720

    capture_buffer_size: int = 1
    decode_threads: int = 2
    max_fps: float = 0.0

    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 0

    isapi_enabled: bool = True
    isapi_port: int = 80
    isapi_https: bool = False
    isapi_channel_id: int = 1
    isapi_username: Optional[str] = None
    isapi_password: Optional[str] = None
    isapi_timeout_seconds: float = 4.0


@dataclass
class RoiConfig:
    polygon: List[List[float]] = field(default_factory=lambda: [
        [0.15, 0.15], [0.85, 0.15], [0.85, 0.90], [0.15, 0.90],
    ])

    crop_margin: float = 0.05
    edge_margin_ratio: float = 0.015

    min_width_ratio: float = 0.025
    min_height_ratio: float = 0.04
    max_width_ratio: float = 0.85
    max_height_ratio: float = 0.90


@dataclass
class ModelConfig:
    vehicle_weights: str = str(PROJECT_ROOT / "models" / "vehicle_ncnn_model")
    plate_weights: str = str(PROJECT_ROOT / "models" / "platenum_closeup_ncnn_model")

    device: Optional[str] = None

    inference_threads: int = 0

    vehicle_imgsz: Union[int, List[int]] = 480
    plate_imgsz: int = 640

    vehicle_conf_threshold: float = 0.35
    vehicle_iou_threshold: float = 0.45
    plate_conf_threshold: float = 0.35

    vehicle_classes: Optional[str] = None


@dataclass
class TrackingConfig:
    bytetrack_config: str = str(CONFIG_DIR / "bytetrack_custom.yaml")

    iou_threshold: float = 0.3
    max_center_distance: float = 2.0
    max_missing_frames: int = 30

    dedup_cooldown_seconds: float = 2.0
    dedup_position_threshold: float = 1.0

    max_plate_attempts: int = 1
    plate_retry_interval_seconds: float = 0.4
    plate_stale_finalize_seconds: float = 1.5

    track_ttl_seconds: float = 30.0


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
    unrecognized_text: str = "Unrecognized"
    accept_score: float = 0.75
    early_exit_score: float = 0.85
    worker_threads: int = 1
    finalize_queue_size: int = 32


@dataclass
class StorageConfig:
    database_path: str = str(PROJECT_ROOT / "data" / "pipeline_buffer.db")
    output_dir: str = str(PROJECT_ROOT / "output")

    jpeg_quality: int = 90
    plate_jpeg_quality: int = 96

    write_batch_size: int = 8
    write_flush_interval_seconds: float = 1.0

    retention_days: int = 14
    max_unsynced_days: int = 0

    delete_disk_images_after_send: bool = True
    send_retry_seconds: float = 60.0


@dataclass
class RuntimeConfig:
    log_level: str = "INFO"
    opencv_threads: int = 2
    max_frames: int = 0


@dataclass
class ApiConfig:
    enabled: bool = True
    endpoint_url: Optional[str] = None
    timeout_seconds: float = 5.0

    roi_endpoint_url: Optional[str] = None
    camera_id: Optional[int] = None
    roi_fetch_timeout_seconds: float = 5.0
    roi_coordinates_are_pixels: bool = False
    roi_reference_width: Optional[int] = None
    roi_reference_height: Optional[int] = None
    roi_poll_interval_seconds: float = 30.0


@dataclass
class LiveConfig:
    camera_id: str = "cam1"

    endpoint_url: Optional[str] = None
    max_hz: float = 10.0
    timeout_seconds: float = 1.0

    serve_host: str = "0.0.0.0"
    serve_port: int = 8090
    stream_fps: float = 10.0
    stream_width: int = 960
    jpeg_quality: int = 70
    box_max_age_seconds: float = 3.0
    box_visual_tracking: bool = True
    box_extrapolate: bool = True
    box_extrapolate_max_seconds: float = 3.5
    auth_token: Optional[str] = None


@dataclass
class FeaturesConfig:
    roi_filter: bool = True
    roi_crop_detect: bool = True
    fallback_tracker: bool = True
    position_dedup: bool = True
    plate_detection: bool = True
    enhance_before_plate_detect: bool = True
    enhance_plate_crop: bool = True
    ocr_read: bool = True
    async_ocr: bool = True
    save_images_to_disk: bool = True
    show_preview: bool = False
    live_boxes: bool = False
    live_stream: bool = False
    roi_fetch_from_api: bool = True

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
class PipelineConfig:
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
        config = cls()

        if yaml_path is None:
            candidate = CONFIG_DIR / "config.yaml"
            yaml_path = str(candidate) if candidate.exists() else None

        if not yaml_path or not os.path.isfile(yaml_path):
            return config

        with open(yaml_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        for section_name, section_values in raw.items():
            if not hasattr(config, section_name) or not isinstance(section_values, dict):
                continue
            section_obj = getattr(config, section_name)
            for key, value in section_values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

        if not config.api.enabled:
            config.features.send_via_api = False
            config.features.roi_fetch_from_api = False
            config.camera.source_mode = "static"

        return config
