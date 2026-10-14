"""Every setting that only vehicle mode uses: models, thresholds, crops, ROI rules, features."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

HERE = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"vehicle_weights": "here", "plate_weights": "here"}

    vehicle_weights: str = str(HERE / "models" / "vehicle.onnx")
    plate_weights: str = str(HERE / "models" / "platenum_closeup.onnx")

    vehicle_imgsz: Union[int, List[int]] = field(default_factory=lambda: [512, 896])
    plate_imgsz: int = 640

    vehicle_conf_threshold: float = 0.40
    vehicle_iou_threshold: float = 0.40
    plate_conf_threshold: float = 0.35

    vehicle_classes: Optional[str] = None
    agnostic_nms: bool = True


@dataclass
class CropConfig:
    vehicle_padding_ratio: float = 0.08
    vehicle_min_crop_height: int = 200

    plate_padding_pixels: int = 4
    plate_padding_ratio: float = 0.12
    plate_min_crop_height: int = 160


@dataclass
class RoiRulesConfig:
    """What counts as 'inside the ROI' for a vehicle (the polygon itself belongs to the camera)."""
    crop_margin: float = 0.05
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
    plate_detection: bool = True
    save_images_to_disk: bool = True
    print_console: bool = True
    send_via_api: bool = True
    delete_row_after_api_send: bool = True


@dataclass
class VehicleConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    roi: RoiRulesConfig = field(default_factory=RoiRulesConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
