from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def select_models(
    spec: dict,
    train_runs: list[dict],
    control_runs: list[dict],
    checkpoint_root: Path,
) -> list[dict]:
    selected: list[dict] = []
    for model in spec["models"]:
        selected.append(
            {
                "run_label": f'{model["alias"]}__base',
                "phase": "baseline",
                "model_alias": model["alias"],
                "model_id": model["model_id"],
                "model_path": model["model_id"],
                "dataset": None,
                "level": "base",
                "seed": None,
            }
        )

    for run in train_runs:
        if run["seed"] != 0 or run["level"] != "misaligned_2":
            continue
        selected.append(
            {
                "run_label": run["run_slug"],
                "phase": "finetuned",
                "model_alias": run["model_alias"],
                "model_id": run["model_id"],
                "model_path": str(checkpoint_root / run["run_slug"]),
                "dataset": run["dataset"],
                "level": run["level"],
                "seed": run["seed"],
            }
        )

    control_groups: dict[tuple[str, str], list[dict]] = {}
    for run in control_runs:
        control_groups.setdefault((run["model_alias"], run["dataset"]), []).append(run)
    for runs in control_groups.values():
        run = min(runs, key=lambda item: item["seed"])
        selected.append(
            {
                "run_label": run["run_slug"],
                "phase": "matched_control",
                "model_alias": run["model_alias"],
                "model_id": run["model_id"],
                "model_path": str(checkpoint_root / run["run_slug"]),
                "dataset": run["dataset"],
                "level": run["level"],
                "seed": run["seed"],
            }
        )
    return selected


def validate_paths(records: list[dict]) -> list[str]:
    missing = []
    for record in records:
        if record["phase"] == "baseline":
            continue
        path = Path(record["model_path"])
        if not path.exists():
            missing.append(str(path))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--train-runs", type=Path, required=True)
    parser.add_argument("--control-runs", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-alias")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    records = select_models(
        spec,
        load_json(args.train_runs),
        load_json(args.control_runs),
        args.checkpoint_root.resolve(),
    )
    if args.model_alias:
        records = [
            record
            for record in records
            if record["model_alias"] == args.model_alias
        ]
        if not records:
            raise ValueError(f"No held-out models matched {args.model_alias}")
    missing = validate_paths(records)
    if args.require_checkpoints and missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing selected checkpoints:\n{rendered}")

    payload = {
        "schema_version": 1,
        "selection_policy": spec["held_out_benchmark_suite"]["selection_policy"],
        "models": records,
        "missing_checkpoints": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{args.output}: {len(records)} models ({len(missing)} missing checkpoints)")


if __name__ == "__main__":
    main()
