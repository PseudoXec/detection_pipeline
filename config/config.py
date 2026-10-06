"""The whole program's settings, assembled from the config files that live in each folder.

    config/config.yaml           GLOBAL on/off switches and the mode toggle - nothing else
    camera/config.yaml           camera + ROI polygon            api/config.yaml      command-center endpoints
    live/config.yaml             live view                       storage/config.yaml  shared SQLite buffer
    modes/vehicle/config.yaml    everything vehicle mode needs   modes/person/config.yaml   everything person mode needs

Each of those has a `config.py` next to it with the typed settings. This file only combines them, so
a setting is always found next to the code that uses it.

`PipelineConfig.load(override_path)` accepts one optional extra yaml file that is applied last. Its top-level
keys are the fields of PipelineConfig (camera, roi, api, live, storage, vehicle, person, mode, runtime,
switches), e.g. a test config or a one-off experiment.
"""
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from api.config import ApiConfig
from camera.config import CameraConfig, RoiConfig
from config.loader import PROJECT_ROOT, load_yaml_into
from live.config import LiveConfig
from modes.person.config import PersonConfig
from modes.vehicle.config import VehicleConfig
from storage.config import StorageConfig

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class ModeConfig:
    """Which detection pipeline runs, and how the command center switches it."""
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"state_file": "root"}

    default: str = "vehicle"                    # vehicle | person | idle - used when nothing was saved yet
    allowed: List[str] = field(default_factory=lambda: ["vehicle", "person", "idle"])
    remember_last: bool = True                  # keep the command center's last choice across reboots
    state_file: str = str(PROJECT_ROOT / "data" / "active_mode.json")
    preload_on_switch: bool = True              # load the new pipeline while the old one keeps running
    control_enabled: bool = True                # POST/GET /control/mode on the live-server port


@dataclass
class RuntimeConfig:
    """Process-wide settings that affect every mode."""
    log_level: str = "INFO"
    opencv_threads: int = 2
    max_frames: int = 0
    device_id: Optional[str] = None             # stamped on person events; default = this machine's hostname
    device: Optional[str] = None                # model device, null = auto
    inference_threads: int = 0                  # ONNX Runtime CPU threads: 0 = auto, -1 = no limit


@dataclass
class SwitchesConfig:
    """Global on/off switches."""
    api: bool = True                            # master switch: off = no ROI fetch, camera source fetch or sending (all modes)
    roi_fetch_from_api: bool = True
    live_stream: bool = False
    live_boxes: bool = False
    show_preview: bool = False


@dataclass
class PipelineConfig:
    mode: ModeConfig = field(default_factory=ModeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    switches: SwitchesConfig = field(default_factory=SwitchesConfig)

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    person: PersonConfig = field(default_factory=PersonConfig)

    @classmethod
    def load(cls, override_path: Optional[str] = None) -> "PipelineConfig":
        config = cls()

        sources = (
            (CONFIG_DIR / "config.yaml", config),
            (PROJECT_ROOT / "camera" / "config.yaml", config),
            (PROJECT_ROOT / "api" / "config.yaml", config),
            (PROJECT_ROOT / "live" / "config.yaml", config),
            (PROJECT_ROOT / "storage" / "config.yaml", config),
            (PROJECT_ROOT / "modes" / "vehicle" / "config.yaml", config.vehicle),
            (PROJECT_ROOT / "modes" / "person" / "config.yaml", config.person),
        )
        for yaml_path, target in sources:
            load_yaml_into(target, yaml_path)

        if override_path:
            if not load_yaml_into(config, Path(override_path)):
                raise FileNotFoundError(f"config override file not found: {override_path}")

        config._finish()
        return config

    def _finish(self) -> None:
        if not self.switches.api:
            self.vehicle.features.send_via_api = False
            self.person.send_via_api = False
            self.switches.roi_fetch_from_api = False
            self.camera.source_mode = "static"

        if not self.person.send_via_api or not self.api.person_endpoint_url:
            self.person.send_via_api = False

        if self.mode.default not in self.mode.allowed:
            raise ValueError(f"mode.default {self.mode.default!r} is not one of mode.allowed {self.mode.allowed}")
        if not self.runtime.device_id:
            self.runtime.device_id = socket.gethostname()
