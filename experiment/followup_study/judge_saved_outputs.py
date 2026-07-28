from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd


def load_judge_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_trait_prompt(experiment_root: Path, trait: str, version: str) -> str:
    trait_path = experiment_root / "data_generation" / f"trait_data_{version}" / f"{trait}.json"
    return json.loads(trait_path.read_text())["eval_prompt"]


def add_experiment_root_to_path(experiment_root: Path) -> None:
    resolved = str(experiment_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def build_judges(experiment_root: Path, trait: str, version: str, judge_config: dict):
    add_experiment_root_to_path(experiment_root)
    from judge import OpenAiJudge
    from eval.prompts import Prompts
    from local_hf_judge import LocalHfJudge

    provider = judge_config["provider"]
    trait_prompt = load_trait_prompt(experiment_root, trait, version)
    model_id = judge_config["model_id"]
    if provider == "openai":
        judge_cls = lambda prompt: OpenAiJudge(model_id, prompt, eval_type="0_100")
    elif provider == "local_hf":
        judge_cls = lambda prompt: LocalHfJudge(model_id, prompt)
    else:
        raise NotImplementedError(f"Unsupported judge provider: {provider}")

    return {
        "trait": judge_cls(trait_prompt),
        "coherence": judge_cls(Prompts["coherence_0_100"]),
    }


async def judge_rows(
    rows: list[dict],
    experiment_root: Path,
    trait: str,
    version: str,
    judge_config: dict,
    max_concurrency: int,
) -> list[dict]:
    judges = build_judges(experiment_root, trait, version, judge_config)
    semaphore = asyncio.Semaphore(max_concurrency)
    judge_name = judge_config["name"]

    async def score_row(row: dict) -> dict:
        async with semaphore:
            result = dict(row)
            result[f"{judge_name}__{trait}"] = await judges["trait"](
                question=row["question"],
                answer=row["answer"],
            )
            result[f"{judge_name}__coherence"] = await judges["coherence"](
                question=row["question"],
                answer=row["answer"],
            )
            return result

    tasks = [score_row(row) for row in rows]
    return await asyncio.gather(*tasks)


def judge_rows_local(
    rows: list[dict],
    experiment_root: Path,
    trait: str,
    version: str,
    judge_config: dict,
    batch_size: int,
) -> list[dict]:
    judges = build_judges(experiment_root, trait, version, judge_config)
    judge_name = judge_config["name"]
    inputs = [
        {
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in rows
    ]
    trait_scores = judges["trait"].judge_batch_sync(inputs, batch_size=batch_size)
    coherence_scores = judges["coherence"].judge_batch_sync(inputs, batch_size=batch_size)
    return [
        {
            **row,
            f"{judge_name}__{trait}": trait_score,
            f"{judge_name}__coherence": coherence_score,
        }
        for row, trait_score, coherence_score in zip(
            rows,
            trait_scores,
            coherence_scores,
            strict=True,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--version", default="eval")
    parser.add_argument("--judge_config", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    judge_config = load_judge_config(args.judge_config)
    frame = pd.read_csv(args.input_csv)
    if "row_id" not in frame.columns:
        frame.insert(0, "row_id", range(len(frame)))

    rows = frame.to_dict(orient="records")
    if judge_config["provider"] == "local_hf":
        scored_rows = judge_rows_local(
            rows=rows,
            experiment_root=args.experiment_root,
            trait=args.trait,
            version=args.version,
            judge_config=judge_config,
            batch_size=args.batch_size,
        )
    else:
        scored_rows = asyncio.run(
            judge_rows(
                rows=rows,
                experiment_root=args.experiment_root,
                trait=args.trait,
                version=args.version,
                judge_config=judge_config,
                max_concurrency=args.max_concurrency,
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored_rows).to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
