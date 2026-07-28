from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch


def add_to_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def generate_raw_outputs(
    checkpoint: Path,
    traits: list[str],
    raw_root: Path,
    experiment_root: Path,
    version: str,
    seed: int,
    n_per_question: int,
    max_tokens: int,
    overwrite: bool,
) -> None:
    pending = [
        trait
        for trait in traits
        if overwrite or not (raw_root / f"{trait}.csv").exists()
    ]
    if not pending:
        return

    add_to_path(experiment_root)
    from eval.eval_persona import eval_batched, load_persona_questions
    from eval.model_utils import load_model
    from runtime import set_seed

    previous_cwd = Path.cwd()
    os.chdir(experiment_root)
    model = None
    try:
        set_seed(seed)
        model, tokenizer = load_model(str(checkpoint))
        for trait in pending:
            questions = load_persona_questions(
                trait,
                temperature=1.0 if n_per_question > 1 else 0.0,
                judge_model=None,
                version=version,
            )
            frames = asyncio.run(
                eval_batched(
                    questions,
                    model,
                    tokenizer,
                    coef=0,
                    n_per_question=n_per_question,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            )
            output = pd.concat(frames, ignore_index=True)
            output.insert(0, "row_id", range(len(output)))
            path = raw_root / f"{trait}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(path, index=False)
            print(path)
    finally:
        os.chdir(previous_cwd)
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_and_aggregate(
    traits: list[str],
    raw_root: Path,
    judge_root: Path,
    merged_root: Path,
    report_root: Path,
    experiment_root: Path,
    judge_config_paths: list[Path],
    metadata: dict,
    version: str,
    batch_size: int,
    max_concurrency: int,
    overwrite: bool,
) -> None:
    from aggregate_multi_judge import merge_judge_frames, summarize
    from judge_saved_outputs import (
        judge_rows,
        judge_rows_local,
        load_judge_config,
    )

    configs = [
        (path, load_judge_config(path))
        for path in judge_config_paths
    ]
    for trait in traits:
        raw_path = raw_root / f"{trait}.csv"
        frame = pd.read_csv(raw_path)
        rows = frame.to_dict(orient="records")
        judge_paths = []
        for _, config in configs:
            judge_path = judge_root / trait / f'{config["name"]}.csv'
            judge_paths.append(judge_path)
            if overwrite or not judge_path.exists():
                if config["provider"] == "local_hf":
                    scored = judge_rows_local(
                        rows,
                        experiment_root,
                        trait,
                        version,
                        config,
                        batch_size,
                    )
                else:
                    scored = asyncio.run(
                        judge_rows(
                            rows,
                            experiment_root,
                            trait,
                            version,
                            config,
                            max_concurrency,
                        )
                    )
                judge_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(scored).to_csv(judge_path, index=False)

        merged = merge_judge_frames(judge_paths, trait)
        merged, summary = summarize(
            merged,
            trait,
            metadata,
            bootstrap_samples=2000,
            bootstrap_seed=metadata["seed"],
        )
        merged_path = merged_root / f"{trait}.csv"
        report_path = report_root / f"{trait}.json"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_path, index=False)
        report_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traits", nargs="+", required=True)
    parser.add_argument("--judge-config-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--version", default="eval")
    parser.add_argument("--n-per-question", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    add_to_path(args.repo_root)
    add_to_path(args.experiment_root)
    add_to_path(args.repo_root / "experiment" / "followup_study")
    base = args.generated_root.resolve()
    relative = Path(args.run_slug) / args.checkpoint_label
    raw_root = base / "raw_outputs" / "checkpoints" / relative
    judge_root = base / "judge_outputs" / "checkpoints" / relative
    merged_root = base / "merged_outputs" / "checkpoints" / relative
    report_root = base / "agreement_reports" / "checkpoints" / relative

    generate_raw_outputs(
        checkpoint=args.checkpoint.resolve(),
        traits=args.traits,
        raw_root=raw_root,
        experiment_root=args.experiment_root.resolve(),
        version=args.version,
        seed=args.seed,
        n_per_question=args.n_per_question,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
    )
    metadata = {
        "repo_root": str(args.repo_root.resolve()),
        "experiment_root": str(args.experiment_root.resolve()),
        "phase": "checkpoints",
        "run_label": args.run_slug,
        "model_alias": args.model_alias,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_step": args.checkpoint_step,
        "seed": args.seed,
    }
    score_and_aggregate(
        traits=args.traits,
        raw_root=raw_root,
        judge_root=judge_root,
        merged_root=merged_root,
        report_root=report_root,
        experiment_root=args.experiment_root.resolve(),
        judge_config_paths=[path.resolve() for path in args.judge_config_paths],
        metadata=metadata,
        version=args.version,
        batch_size=args.judge_batch_size,
        max_concurrency=args.max_concurrency,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
