from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critical_point_report", type=Path, required=True)
    parser.add_argument("--run_output_dir", type=Path, required=True)
    parser.add_argument("--print_path", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.critical_point_report.read_text())
    step = report.get("recommended_checkpoint_step")
    selected = None if step is None else args.run_output_dir / f"checkpoint-{step}"
    payload = {
        "recommended_checkpoint_step": step,
        "selected_checkpoint_path": None if selected is None else str(selected),
    }
    if args.print_path:
        print("" if selected is None else selected)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
