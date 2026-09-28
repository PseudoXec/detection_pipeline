import argparse
import glob
import os
import shutil

from config.config import PipelineConfig


def reset(config: PipelineConfig) -> None:
    db_path = config.storage.database_path
    for path in [db_path, *glob.glob(db_path + "-*")]:
        if os.path.isfile(path):
            os.remove(path)
            print(f"[reset] deleted {path}")

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
