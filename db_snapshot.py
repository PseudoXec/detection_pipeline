"""
db_snapshot.py
--------------
Makes a plain copy of the pipeline's SQLite buffer that any SQLite viewer can open.

Why: data/pipeline_buffer.db runs in WAL mode (so the C# dashboard can read while
the pipeline writes) and stores the vehicle/plate JPEGs as BLOBs. Viewers such as
VS Code's "SQLite Viewer" often show a blank tab for that: they only read the main
file and ignore the -wal file the pipeline is still writing to, and they can choke
on image columns. This script copies the LIVE data (including what is still in the
WAL) into a normal, non-WAL file. It is safe to run while the pipeline is running -
it only reads the original.

Usage (from the project folder, .venv active):
    python db_snapshot.py                 # full copy  -> data/snapshot.db
    python db_snapshot.py --light         # same, but image BLOBs replaced by their sizes (much easier for viewers)
    python db_snapshot.py path/to/other.db out.db

Then open data/snapshot.db in the viewer. Re-run the script whenever you want fresh data.
"""

import os
import sqlite3
import sys


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    light = "--light" in sys.argv[1:]
    source = args[0] if len(args) > 0 else os.path.join("data", "pipeline_buffer.db")
    target = args[1] if len(args) > 1 else os.path.join("data", "snapshot.db")

    if not os.path.exists(source):
        print(f"source database not found: {source}")
        return 1
    for leftover in (target, target + "-wal", target + "-shm", target + "-journal"):
        if os.path.exists(leftover):
            try:
                os.remove(leftover)
            except PermissionError:
                print(f"can't replace {leftover}: close it in the viewer (or the pipeline) and run again")
                return 1

    src = sqlite3.connect(source, timeout=10)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)                                  # consistent copy, includes rows still in the WAL
        dst.execute("PRAGMA journal_mode=DELETE")        # plain file: no -wal/-shm for a viewer to miss

        if light:
            tables = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "detections" in tables:
                dst.execute("ALTER TABLE detections ADD COLUMN vehicle_image_bytes INTEGER")
                dst.execute("ALTER TABLE detections ADD COLUMN plate_image_bytes INTEGER")
                dst.execute("UPDATE detections SET vehicle_image_bytes = length(vehicle_image), "
                            "plate_image_bytes = length(plate_image)")
                dst.execute("UPDATE detections SET vehicle_image = zeroblob(0), "
                            "plate_image = CASE WHEN plate_image IS NULL THEN NULL ELSE zeroblob(0) END")
            dst.commit()
            dst.execute("VACUUM")

        dst.commit()
        print(f"snapshot written: {target}  ({os.path.getsize(target):,} bytes){'  [images replaced by sizes]' if light else ''}")
        for (name,) in dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            print(f"  {name}: {dst.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]} row(s)")
        return 0
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    sys.exit(main())
