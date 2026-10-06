"""How settings are read.

Every folder that has settings owns two files:  `config.py` (the typed settings and their defaults) and
`config.yaml` (the values you edit). This module is the only shared piece: it reads a yaml file into a
settings object, resolves relative paths against the right folder, and applies an optional
`config.local.yaml` placed next to it (machine-specific values such as the camera password; never committed).

A yaml key that does not exist in the settings object is reported with a warning instead of being
silently ignored, so a typo in a setting is visible.
"""
import logging
import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("pipeline")


def resolve_path(value: str, kind: str, yaml_dir: Path) -> str:
    """kind "here": relative to the folder of the yaml file (model files, tracker files).
    kind "root": relative to the project root (runtime data such as data/ and output/).
    Absolute paths are never changed."""
    if not value or os.path.isabs(value):
        return value
    base = PROJECT_ROOT if kind == "root" else yaml_dir
    return str((base / value).resolve())


def apply_values(target: Any, values: dict, yaml_dir: Path, source: str, prefix: str = "") -> None:
    path_fields = getattr(type(target), "PATH_FIELDS", {})
    for key, value in values.items():
        if not hasattr(target, key):
            log.warning("[config] %s: unknown setting %r%s is ignored", source, prefix + key, "")
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_values(current, value, yaml_dir, source, prefix + key + ".")
            continue
        if key in path_fields and isinstance(value, str):
            value = resolve_path(value, path_fields[key], yaml_dir)
        setattr(target, key, value)


def load_yaml_into(target: Any, yaml_path: Path) -> bool:
    """Read yaml_path (and yaml_path's `.local.yaml` twin) into target. False if the file does not exist."""
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        return False
    for path in (yaml_path, yaml_path.with_name(yaml_path.stem + ".local" + yaml_path.suffix)):
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping of settings at the top level")
        apply_values(target, raw, path.parent, os.path.relpath(path, PROJECT_ROOT))
    return True
