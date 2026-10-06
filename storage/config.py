"""Settings of the shared SQLite buffer (the parts every detection mode has in common)."""
from dataclasses import dataclass
from typing import ClassVar, Dict

from config.loader import PROJECT_ROOT


@dataclass
class StorageConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"database_path": "root", "output_dir": "root"}

    database_path: str = str(PROJECT_ROOT / "data" / "pipeline_buffer.db")
    output_dir: str = str(PROJECT_ROOT / "output")

    write_batch_size: int = 8
    write_flush_interval_seconds: float = 1.0

    retention_days: int = 14
    max_unsynced_days: int = 0

    delete_disk_images_after_send: bool = True
    send_retry_seconds: float = 60.0
