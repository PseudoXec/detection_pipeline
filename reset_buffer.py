"""
reset_buffer.py
----------------
Wipes the SQLite buffer database and the on-disk crop images so the
pipeline can start completely fresh - useful after testing, or before a
real deployment goes live.

IMPORTANT: stop the pipeline first (Ctrl+C, or `systemctl stop
detection-pipeline` on the Pi). Deleting the database file while `run.py`
still has it open can corrupt the WAL side-files.

Why deleting the `output/` folder alone is NOT enough:
    `output/` only holds the convenience on-disk JPEG copies. The actual
    data your C# dashboard reads - metadata AND images - lives in the
    SQLite file at `storage.database_path` (default
    `data/pipeline_buffer.db`), plus two side-files SQLite's WAL mode
    creates next to it (`*-wal`, `*-shm`). Deleting only `output/` leaves
    every past detection sitting in that database untouched.

USAGE
    python reset_buffer.py            # asks for confirmation first
    python reset_buffer.py --yes      # skips the confirmation prompt
"""

# argparse handles the --yes flag so this can be scripted/automated if needed
import argparse
# glob finds the database's WAL/SHM side-files, which don't have a fixed name suffix pattern otherwise
import glob
# os is used for removing individual files
import os
# shutil.rmtree removes the whole output/ folder tree in one call
import shutil

from config import PipelineConfig


def reset(config: PipelineConfig) -> None:
    """Delete the SQLite buffer (+ its WAL/SHM side-files) and the on-disk output folder."""

    # the main database file, plus SQLite's write-ahead-log side-files
    # (these only exist while WAL mode has been used, but glob simply
    # returns nothing if they aren't there, so this is always safe)
    db_path = config.storage.database_path
    for path in [db_path, *glob.glob(db_path + "-*")]:
        if os.path.isfile(path):
            os.remove(path)
            print(f"[reset] deleted {path}")

    # the on-disk convenience copies of every vehicle/plate crop
    output_dir = config.storage.output_dir
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"[reset] deleted {output_dir}")

    print("[reset] done - the pipeline will recreate both on its next run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe the SQLite buffer and on-disk crop output")
    parser.add_argument("--config", help="Path to a config.yaml (defaults to the one next to this file)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)

    if not args.yes:
        # a plain input() prompt so nobody wipes real production data by accident
        answer = input(
            f"This will permanently delete:\n"
            f"  - {config.storage.database_path} (and its -wal/-shm files)\n"
            f"  - {config.storage.output_dir}/\n"
            f"Make sure the pipeline is stopped first. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Cancelled - nothing was deleted.")
            return

    reset(config)


if __name__ == "__main__":
    main()
