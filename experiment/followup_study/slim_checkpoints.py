import argparse
import shutil
from pathlib import Path


STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "training_args.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune and slim training checkpoints after a run finishes.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed run directory that contains checkpoint-* folders.")
    parser.add_argument(
        "--keep-every",
        type=int,
        default=None,
        help="Keep only checkpoints whose step number is divisible by this value.",
    )
    parser.add_argument(
        "--keep-steps",
        default="",
        help="Comma-separated list of exact checkpoint step numbers to keep. Ignored when empty.",
    )
    parser.add_argument(
        "--drop-training-state",
        action="store_true",
        help="Delete optimizer/scheduler/rng state files from kept checkpoints to reduce disk use.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return None


def should_keep(step: int, keep_every: int | None, keep_steps: set[int]) -> bool:
    if keep_steps:
        return step in keep_steps
    if keep_every:
        return step % keep_every == 0
    return True


def delete_training_state(checkpoint_dir: Path) -> int:
    removed = 0
    for child in checkpoint_dir.iterdir():
        if child.name in STATE_FILES or child.name.startswith("rng_state"):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    keep_steps = {int(x) for x in args.keep_steps.split(",") if x.strip()}
    checkpoint_dirs = sorted(p for p in run_dir.glob("checkpoint-*") if p.is_dir())

    removed_dirs = 0
    removed_state_files = 0

    for checkpoint_dir in checkpoint_dirs:
        step = checkpoint_step(checkpoint_dir)
        if step is None:
            continue
        if not should_keep(step, args.keep_every, keep_steps):
            shutil.rmtree(checkpoint_dir)
            removed_dirs += 1
            continue
        if args.drop_training_state:
            removed_state_files += delete_training_state(checkpoint_dir)

    print(f"Processed: {run_dir}")
    print(f"Removed checkpoint dirs: {removed_dirs}")
    print(f"Removed training-state files: {removed_state_files}")


if __name__ == "__main__":
    main()
