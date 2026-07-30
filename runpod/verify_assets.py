from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/followup_study/study_spec_runpod.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else root / args.spec
    spec = json.loads(spec_path.read_text())
    generated = root / spec["generated_root"]
    manifests = generated / "manifests"
    errors: list[str] = []

    run_groups = {
        "treatment": json.loads((manifests / "train_runs.json").read_text()),
        "matched_control": json.loads(
            (manifests / "matched_control_runs.json").read_text()
        ),
        "method_ablation": json.loads(
            (manifests / "method_ablation_runs.json").read_text()
        ),
    }
    expected_treatments = 51
    if spec.get("generality_extension", {}).get("status") == "active":
        expected_treatments += int(
            spec["generality_extension"]["planned_training_runs"]
        )
    expected = {
        "treatment": expected_treatments,
        "matched_control": 18,
        "method_ablation": 2,
    }
    for name, runs in run_groups.items():
        if len(runs) != expected[name]:
            errors.append(f"{name}: expected {expected[name]} runs, found {len(runs)}")
        slugs = [run["run_slug"] for run in runs]
        if len(slugs) != len(set(slugs)):
            errors.append(f"{name}: duplicate run_slug values")
        for run in runs:
            config_path = Path(run["config_path"])
            output_dir = Path(run["output_dir"])
            if not config_path.is_relative_to(root):
                errors.append(f"stale config path outside current root: {config_path}")
            if not output_dir.is_relative_to(root):
                errors.append(f"stale output path outside current root: {output_dir}")
            if not config_path.exists():
                errors.append(f"missing training config: {config_path}")
            if (output_dir / ".training_complete").exists():
                adapter_candidates = [
                    output_dir / "adapter_config.json",
                    *output_dir.glob("checkpoint-*/adapter_config.json"),
                ]
                if not any(path.exists() for path in adapter_candidates):
                    errors.append(
                        f"completed run has no adapter_config.json: {output_dir}"
                    )

    dataset_rows = {}
    for dataset in spec["datasets"]:
        dataset_dir = root / spec["experiment_root"] / "dataset" / dataset["name"]
        dataset_rows[dataset["name"]] = {}
        for level in [dataset["control_level"], *dataset["levels"]]:
            path = dataset_dir / f"{level}.jsonl"
            if not path.exists():
                errors.append(f"missing dataset: {path}")
                continue
            count = jsonl_rows(path)
            dataset_rows[dataset["name"]][level] = count
            if count == 0:
                errors.append(f"empty dataset: {path}")

    required_runbooks = {
        "00_prepare_experiment_data.sh",
        "25_judge_calibration.sh",
        "30_train_models.sh",
        "35_train_matched_controls.sh",
        "40_eval_finetuned_models.sh",
        "41_eval_matched_controls.sh",
        "44_eval_checkpoint_grid.sh",
        "48_eval_held_out_suite.sh",
        "49_eval_custom_held_out.sh",
        "56_evaluate_temporal_detector.sh",
        "58_compare_matched_controls.sh",
        "70_intervention_early_stop.sh",
        "90_rejudge_claim_validation.sh",
        "91_analyze_api_robustness.sh",
        "92_gate_api_validation.sh",
    }
    actual_runbooks = {path.name for path in (generated / "runbooks").glob("*.sh")}
    for missing in sorted(required_runbooks - actual_runbooks):
        errors.append(f"missing runbook: {missing}")

    report = {
        "ok": not errors,
        "root": str(root),
        "run_counts": {name: len(runs) for name, runs in run_groups.items()},
        "expected_run_counts": expected,
        "dataset_rows": dataset_rows,
        "runbook_count": len(actual_runbooks),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
