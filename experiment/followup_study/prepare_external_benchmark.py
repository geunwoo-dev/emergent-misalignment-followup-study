from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def trait_group(trait: str, taxonomy: dict[str, list[str]]) -> str:
    for group, traits in taxonomy.items():
        if trait in traits:
            return group
    return "unspecified"


def resolve_checkpoints(glob_pattern: str, step_regex: str) -> list[tuple[int | None, Path]]:
    regex = re.compile(step_regex)
    resolved = []
    for match in glob.glob(glob_pattern):
        path = Path(match)
        result = regex.search(path.name)
        step = None if result is None else int(result.group(1))
        resolved.append((step, path))
    return sorted(resolved, key=lambda item: (-1 if item[0] is None else item[0], item[1].name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_spec", type=Path, required=True)
    parser.add_argument("--benchmark_spec", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--judge_config_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--version", default="eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_per_question", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--generated_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    args = parser.parse_args()

    study_spec = load_json(args.study_spec)
    benchmark_spec = load_json(args.benchmark_spec)
    if args.output_dir.exists():
        for child in args.output_dir.iterdir():
            if child.is_file():
                child.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for benchmark in benchmark_spec.get("benchmarks", []):
        if not benchmark.get("enabled", False):
            continue
        checkpoints = resolve_checkpoints(
            glob_pattern=benchmark["checkpoint_glob"],
            step_regex=benchmark.get("checkpoint_step_regex", r"checkpoint-(\d+)$"),
        )
        benchmark_manifest = {
            "name": benchmark["name"],
            "model_alias": benchmark.get("model_alias", benchmark["name"]),
            "description": benchmark.get("description"),
            "behavior_signal_csv": benchmark.get("behavior_signal_csv"),
            "internal_signal_csvs": benchmark.get("internal_signal_csvs", []),
            "checkpoint_count": len(checkpoints),
            "eval_config_paths": [],
        }
        for checkpoint_step, checkpoint_path in checkpoints:
            checkpoint_label = checkpoint_path.name
            for trait in benchmark["traits"]:
                config = {
                    "repo_root": str(args.repo_root),
                    "experiment_root": str(args.experiment_root),
                    "phase": "external_benchmark",
                    "run_label": benchmark["name"],
                    "benchmark_name": benchmark["name"],
                    "model_alias": benchmark.get("model_alias", benchmark["name"]),
                    "model": str(checkpoint_path),
                    "trait": trait,
                    "trait_group": trait_group(trait, study_spec["trait_taxonomy"]),
                    "version": args.version,
                    "seed": args.seed,
                    "n_per_question": args.n_per_question,
                    "max_tokens": args.max_tokens,
                    "judge_config_paths": [str(path) for path in args.judge_config_paths],
                    "raw_output_path": str(args.generated_root / "raw_outputs" / "external_benchmarks" / benchmark["name"] / checkpoint_label / f"{trait}.csv"),
                    "judge_output_dir": str(args.generated_root / "judge_outputs" / "external_benchmarks" / benchmark["name"] / checkpoint_label / trait),
                    "merged_output_path": str(args.generated_root / "merged_outputs" / "external_benchmarks" / benchmark["name"] / checkpoint_label / f"{trait}.csv"),
                    "agreement_report_path": str(args.generated_root / "agreement_reports" / "external_benchmarks" / benchmark["name"] / checkpoint_label / f"{trait}.json"),
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_step": checkpoint_step,
                }
                config_path = args.output_dir / f'{benchmark["name"]}__{checkpoint_label}__{trait}.json'
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                benchmark_manifest["eval_config_paths"].append(str(config_path))
        manifest_rows.append(benchmark_manifest)

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
