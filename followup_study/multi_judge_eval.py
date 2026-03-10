from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def ensure_row_ids(raw_output_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_output_path)
    if "row_id" not in frame.columns:
        frame.insert(0, "row_id", range(len(frame)))
        frame.to_csv(raw_output_path, index=False)


def run_subprocess(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def evaluate(config: dict, overwrite: bool) -> None:
    persona_root = Path(config["persona_root"])
    judge_config_paths = [Path(path) for path in config["judge_config_paths"]]
    raw_output_path = Path(config["raw_output_path"])
    judge_output_dir = Path(config["judge_output_dir"])
    merged_output_path = Path(config["merged_output_path"])
    agreement_report_path = Path(config["agreement_report_path"])
    metadata_path = agreement_report_path.with_suffix(".metadata.json")

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    judge_output_dir.mkdir(parents=True, exist_ok=True)
    agreement_report_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(config, indent=2) + "\n")

    if overwrite or not raw_output_path.exists():
        command = [
            sys.executable,
            "-m",
            "eval.eval_persona",
            f'--model={config["model"]}',
            f'--trait={config["trait"]}',
            f'--output_path={raw_output_path}',
            f'--version={config.get("version", "eval")}',
            f'--n_per_question={config.get("n_per_question", 10)}',
            f'--max_tokens={config.get("max_tokens", 1000)}',
            f'--seed={config.get("seed", 0)}',
            "--skip_judging=True",
            f'--overwrite={str(overwrite)}',
        ]
        if config.get("persona_instruction_type"):
            command.append(f'--persona_instruction_type={config["persona_instruction_type"]}')
        if config.get("assistant_name"):
            command.append(f'--assistant_name={config["assistant_name"]}')
        if config.get("coef", 0):
            command.extend(
                [
                    f'--coef={config["coef"]}',
                    f'--vector_path={config["vector_path"]}',
                    f'--layer={config["layer"]}',
                    f'--steering_type={config.get("steering_type", "response")}',
                ]
            )
        run_subprocess(command, cwd=persona_root)

    ensure_row_ids(raw_output_path)

    judge_outputs = []
    for judge_config_path in judge_config_paths:
        judge_config = load_config(judge_config_path)
        judge_output_path = judge_output_dir / f'{judge_config["name"]}.csv'
        judge_outputs.append(judge_output_path)
        if overwrite or not judge_output_path.exists():
            command = [
                sys.executable,
                "followup_study/judge_saved_outputs.py",
                "--persona_root",
                str(persona_root),
                "--input_csv",
                str(raw_output_path),
                "--trait",
                config["trait"],
                "--version",
                config.get("version", "eval"),
                "--judge_config",
                str(judge_config_path),
                "--output_csv",
                str(judge_output_path),
            ]
            run_subprocess(command, cwd=Path(config["repo_root"]))

    command = [
        sys.executable,
        "followup_study/aggregate_multi_judge.py",
        "--trait",
        config["trait"],
        "--input_paths",
        *[str(path) for path in judge_outputs],
        "--output_merged_csv",
        str(merged_output_path),
        "--output_summary_json",
        str(agreement_report_path),
        "--metadata_json",
        str(metadata_path),
    ]
    run_subprocess(command, cwd=Path(config["repo_root"]))


def build_config_from_args(args: argparse.Namespace) -> dict:
    if args.eval_config:
        return load_config(args.eval_config)
    required = [
        "repo_root",
        "persona_root",
        "model",
        "trait",
        "judge_config_paths",
        "raw_output_path",
        "judge_output_dir",
        "merged_output_path",
        "agreement_report_path",
    ]
    missing = [name for name in required if getattr(args, name) in (None, [])]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")
    return {
        "repo_root": str(args.repo_root),
        "persona_root": str(args.persona_root),
        "model": args.model,
        "trait": args.trait,
        "version": args.version,
        "seed": args.seed,
        "n_per_question": args.n_per_question,
        "max_tokens": args.max_tokens,
        "judge_config_paths": [str(path) for path in args.judge_config_paths],
        "raw_output_path": str(args.raw_output_path),
        "judge_output_dir": str(args.judge_output_dir),
        "merged_output_path": str(args.merged_output_path),
        "agreement_report_path": str(args.agreement_report_path),
        "phase": args.phase,
        "run_label": args.run_label,
        "model_alias": args.model_alias,
        "trait_group": args.trait_group,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_step": args.checkpoint_step,
        "coef": args.coef,
        "vector_path": None if args.vector_path is None else str(args.vector_path),
        "layer": args.layer,
        "steering_type": args.steering_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", type=Path)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--persona_root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--trait")
    parser.add_argument("--version", default="eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_per_question", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--judge_config_paths", nargs="+", type=Path)
    parser.add_argument("--raw_output_path", type=Path)
    parser.add_argument("--judge_output_dir", type=Path)
    parser.add_argument("--merged_output_path", type=Path)
    parser.add_argument("--agreement_report_path", type=Path)
    parser.add_argument("--phase", default="ad_hoc")
    parser.add_argument("--run_label", default="ad_hoc")
    parser.add_argument("--model_alias", default="ad_hoc")
    parser.add_argument("--trait_group", default="unspecified")
    parser.add_argument("--checkpoint_label")
    parser.add_argument("--checkpoint_step", type=int)
    parser.add_argument("--coef", type=float, default=0.0)
    parser.add_argument("--vector_path", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--steering_type", default="response")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = build_config_from_args(args)
    evaluate(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
