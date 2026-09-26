"""
ocr_config.py
-------------
All tunable values for the OCR-only pipeline live in one place: this file
(defaults) plus an optional "ocr_config.yaml" that can override them without
touching code.

This pipeline does NOT do vehicle/plate detection. It assumes it is being
handed images that are already plate crops (e.g. produced by a separate
detection stage, or just a folder of plate photos) and its only job is to
turn each one into text with FastPlateOCR.
"""

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class OcrConfig:
    """Everything the OCR reader itself needs."""

    # FastPlateOCR model from the model zoo, or a path to your own exported
    # ONNX model. "cct-xs-v1-global-model" is small/fast and CPU-friendly.
    model_name: str = "cct-xs-v1-global-model"

    # kept for config compatibility with the wider project; FastPlateOCR
    # models aren't language-keyed, this is just a label in the output.
    lang: str = "en"

    # reads below this mean confidence become "Unrecognized"
    min_confidence: float = 0.5

    # text used whenever nothing readable comes back from a plate that WAS detected
    # (crop too blurry/dark, recognizer errored, confidence too low, etc.)
    unrecognized_text: str = "Unrecognized"

    # text used when the vehicle never had a plate detected at all (plate_detected
    # was false, or no plate_crop file was sent) - kept distinct from
    # unrecognized_text so a reviewer can tell "no plate to read" apart from
    # "had a plate, couldn't read it"
    no_plate_text: str = "No Plate Detected"

    # once one image variant reads at/above this score, skip the rest
    early_exit_score: float = 0.85

    # if set, only these characters survive cleanup (e.g. "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    allowed_chars: Optional[str] = None

    # a confident read is additionally checked against real Philippine LTO
    # plate formats (see ocr/plate_format.py) - e.g. 3 letters + 4 digits for
    # cars, 3 digits + 3 letters for current motorcycle/tricycle plates. A
    # read that doesn't match any of them (reversed order, wrong grouping,
    # a stray leftover character) is reported as unrecognized_text instead
    # of a real-looking but wrong plate number. Set to False if this
    # service is ever used outside the Philippines.
    validate_plate_format: bool = True


@dataclass
class PreprocessConfig:
    """Optional image cleanup applied before each OCR attempt."""

    enhance: bool = True  # CLAHE + light sharpen before reading
    min_crop_height: int = 64  # crops shorter than this get upscaled first
    try_both_variants: bool = True  # OCR the raw crop AND the enhanced one, keep the best


@dataclass
class InputConfig:
    """Where images come from."""

    # exactly one of these is normally set at runtime (via CLI); both default
    # to null here so the CLI can decide
    image: Optional[str] = None  # single image path
    folder: Optional[str] = None  # folder of images (non-recursive unless recursive=True)
    recursive: bool = False
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class OutputConfig:
    """Where results go."""

    results_csv: Optional[str] = "output/ocr_results.csv"  # null = don't write CSV
    results_json: Optional[str] = None  # null = don't write JSON
    print_console: bool = True
    save_annotated: bool = False  # write a copy of each image with the read text overlaid
    annotated_dir: str = "output/annotated"


@dataclass
class RuntimeConfig:
    log_level: str = "INFO"
    workers: int = 1  # thread pool size for batch runs (ONNX Runtime is not free-threaded per-session, keep low)


@dataclass
class ServerConfig:
    """Settings for running this pipeline as an HTTP OCR service (server.py)
    instead of the batch/CLI mode in ocr_pipeline.py. The edge (detection)
    pipeline POSTs a plate_crop blob + its metadata here instead of running
    OCR itself."""

    host: str = "0.0.0.0"
    port: int = 8500
    # if set, every request must send this value in an `X-API-Key` header -
    # OCR results (plate text) are sensitive enough to not leave this open
    # on a network you don't control
    auth_token: Optional[str] = None
    # max accepted request body size, to stop an oversized upload from
    # tying up a worker
    max_content_length_mb: int = 8
    # optional: if set, the server also POSTs the completed record (with
    # ocr_read/ocr_text filled in) on to this URL after OCR finishes, instead
    # of only returning it in the HTTP response - this is normally the C#
    # dashboard's own ingest endpoint (the same one the edge pipeline's
    # api.endpoint_url pointed at before OCR was split out server-side).
    # Leave null to just respond to the caller and not relay anywhere.
    forward_url: Optional[str] = None
    forward_timeout_seconds: float = 5.0
    # the forwarded request mirrors the shape the dashboard's API already
    # expects (query params + a multipart image file, matching the edge
    # pipeline's original api_client.py) rather than a JSON body.
    #
    # form/query field the plate crop image is re-attached under when
    # forwarding - the edge sends it to THIS service as "plate_crop", but
    # the dashboard's endpoint expects the file under this name instead
    forward_image_field: str = "Image"
    # key the final OCR text is added under when forwarding, matching
    # whatever query parameter name the downstream dashboard expects for
    # the plate number. Set to null to not add it (e.g. if the downstream
    # side reads ocr_read directly from a JSON body instead - see
    # forward_as_json below).
    forward_plate_number_field: Optional[str] = "PlateNumber"
    # bookkeeping fields that only make sense between the edge pipeline and
    # THIS service (not part of the downstream dashboard's schema) - these
    # are stripped out of the payload before forwarding
    forward_drop_fields: tuple = ("plate_detected",)
    # False (default) = forward as query params + multipart file, matching
    # the dashboard's existing ASP.NET-style endpoint. True = forward as a
    # single JSON body instead (no image re-attached) - use this only if
    # the downstream endpoint has been changed to accept JSON.
    forward_as_json: bool = False

    # ---- track_id de-duplication (see dedupe.py) ----
    # The edge pipeline retries a send whenever it doesn't get back a clean 200
    # (timeout, dropped connection, this service briefly restarting, etc.) - even
    # if the original POST actually went through. Without this, a retried send
    # would read the same plate crop again and forward a SECOND row for the same
    # vehicle to the dashboard. When enabled, this service remembers track_ids it
    # has already forwarded for a while and, on a repeat within that window,
    # skips forwarding again instead of creating a duplicate dashboard row.
    dedupe_enabled: bool = True
    # how long a track_id is remembered after its first sighting. Should comfortably
    # cover how far apart the edge's own retries land - see the edge pipeline's
    # storage.send_retry_seconds (default 60s); this defaults to a bit more than
    # double that so a couple of retries still land inside the window.
    dedupe_window_seconds: float = 150.0
    # every request for a track_id within the window (the original plus any
    # retries) casts a "vote" for whatever ocr_read it produced. The MAJORITY
    # text across those votes is what gets returned to the edge and logged -
    # see the "Note on majority voting" in README.md for what this can and
    # can't do (it does not re-forward a corrected row to the dashboard if the
    # majority changes after the first vehicle was already forwarded).
    # caps how many track_ids are remembered at once (oldest entries are
    # evicted first if this is exceeded) so a very long-running process
    # can't grow this cache without bound
    dedupe_max_entries: int = 5000
    # SQLite file the dedup/vote cache is stored in - deliberately a real file,
    # not in-process memory, so every gunicorn worker (and this service across
    # restarts) shares the same view. See dedupe.py's module docstring.
    dedupe_db_path: str = "ocr_dedupe.sqlite3"


@dataclass
class PipelineConfig:
    ocr: OcrConfig = field(default_factory=OcrConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "PipelineConfig":
        """Build the config from defaults, then apply a YAML file over the
        top of it if one exists. Missing keys in the YAML just keep their
        default - you only need to list what you want to change."""
        config = cls()
        path = Path(yaml_path) if yaml_path else CONFIG_DIR / "ocr_config.yaml"
        if path.exists():
            with open(path, "r") as handle:
                raw = yaml.safe_load(handle) or {}
            for section_name, section_values in raw.items():
                if not hasattr(config, section_name) or not isinstance(section_values, dict):
                    continue
                section = getattr(config, section_name)
                for key, value in section_values.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config

    def as_dict(self) -> dict:
        return asdict(self)
