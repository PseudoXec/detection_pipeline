"""Every setting that only person mode uses: models, thresholds, ROI rules, face search, storage options."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

HERE = Path(__file__).resolve().parent


@dataclass
class PersonConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {
        "weights": "here", "face_weights": "here", "bytetrack_config": "here",
    }

    weights: str = str(HERE / "models" / "person_ncnn_model")
    face_weights: str = str(HERE / "models" / "face_detection_yunet_2023mar.onnx")

    imgsz: Union[int, List[int]] = 416
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    detector_conf_floor: float = 0.20
    classes: Optional[str] = None               # e.g. "person" for a COCO model; null = accept every class
    agnostic_nms: bool = True

    bytetrack_config: str = str(HERE / "bytetrack.yaml")
    fallback_tracker: bool = True
    fallback_iou_threshold: float = 0.3
    fallback_max_center_distance: float = 2.0
    fallback_max_missing_frames: int = 30
    position_dedup: bool = True
    dedup_cooldown_seconds: float = 2.0
    dedup_position_threshold: float = 1.0

    roi_filter: bool = True                     # a person counts as inside when their FEET are in the polygon
    roi_crop_detect: bool = False               # run the model on the ROI window instead of the whole frame
    roi_crop_margin: float = 0.05               # extra margin around that window
    roi_top_extra: float = 0.30                 # window grows this far above the polygon (people are tall)
    min_width_ratio: float = 0.01
    max_width_ratio: float = 0.60
    min_height_ratio: float = 0.05
    max_height_ratio: float = 0.98
    edge_margin_ratio: float = 0.0

    padding_ratio: float = 0.08                 # extra margin around the person crop
    min_crop_height: int = 0                    # 0 = never upscale the stored person crop
    jpeg_quality: int = 90                      # stored person image
    face_jpeg_quality: int = 95                 # stored face image

    face_enabled: bool = True
    async_face_detection: bool = True           # face model on a background thread
    face_score_threshold: float = 0.7
    face_search_region: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.6])  # x0,y0,x1,y1 of the person box
    face_margin: float = 0.2
    face_min_px: int = 40                       # smaller side of the face box
    face_min_sharpness: float = 40.0            # Laplacian variance
    face_ref_px: int = 64
    face_ref_sharpness: float = 100.0
    face_good_enough_quality: float = 0.6       # store the row as soon as a face scores this well
    face_max_attempts: int = 12
    face_retry_interval_seconds: float = 0.3
    face_worker_max_queue: int = 8

    stale_finalize_seconds: float = 1.5         # person left the scene -> store the best sighting
    track_ttl_seconds: float = 30.0

    save_images_to_disk: bool = True
    store_to_sqlite: bool = True
    send_via_api: bool = False                  # needs api.person_endpoint_url
    send_rows_without_face: bool = True
    delete_row_after_api_send: bool = True
