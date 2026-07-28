from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    generated = root / "experiment/followup_study/generated_runpod"
    state = generated / "stage_state"

    runs = []
    for manifest_name in [
        "train_runs.json",
        "matched_control_runs.json",
        "method_ablation_runs.json",
    ]:
        path = generated / "manifests" / manifest_name
        if path.exists():
            runs.extend(json.loads(path.read_text()))
    by_model: dict[str, dict[str, int]] = {}
    for run in runs:
        alias = run["model_alias"]
        counters = by_model.setdefault(alias, {"complete": 0, "pending": 0})
        output = Path(run["output_dir"])
        key = "complete" if (output / ".training_complete").exists() else "pending"
        counters[key] += 1

    stage_state = {
        "complete": sorted(path.stem for path in state.glob("*.done")),
        "failed": sorted(path.stem for path in state.glob("*.failed")),
        "running": sorted(path.stem for path in state.glob("*.lock")),
    }
    report = {
        "training": by_model,
        "stages": stage_state,
        "recent_logs": [
            str(path)
            for path in sorted(
                (root / "logs/runpod").glob("*.log"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )[:6]
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
