from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def has_adapter(run_dir: Path) -> bool:
    if (run_dir / "adapter_config.json").exists():
        return True
    return any((checkpoint / "adapter_config.json").exists() for checkpoint in run_dir.glob("checkpoint-*"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--destination_root", type=Path, required=True)
    parser.add_argument("--run_manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--mark_complete", action="store_true")
    args = parser.parse_args()

    expected_runs = {}
    for manifest_path in args.run_manifests:
        for run in json.loads(manifest_path.read_text()):
            expected_runs[run["run_slug"]] = run

    args.destination_root.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    for run_slug in sorted(expected_runs):
        source = (args.source_root / run_slug).resolve()
        destination = args.destination_root / run_slug
        if not source.exists():
            continue
        if not has_adapter(source):
            print(f"[skip] no adapter files: {source}")
            skipped += 1
            continue
        if destination.exists() or destination.is_symlink():
            print(f"[skip] destination exists: {destination}")
            skipped += 1
            continue
        if args.mode == "symlink":
            destination.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, destination, symlinks=True)
        if args.mark_complete:
            (destination / ".training_complete").write_text("imported\n")
        print(f"[imported] {run_slug}")
        imported += 1

    print(f"Imported: {imported}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
