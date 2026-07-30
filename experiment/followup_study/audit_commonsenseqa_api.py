from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

try:
    from .api_text_judge import extract_json_object
except ImportError:
    from api_text_judge import extract_json_object


CRITERIA = (
    "valid_question",
    "normal_correct",
    "mild_error_valid",
    "severe_error_valid",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def answer_text(row: dict) -> str:
    return str(row["messages"][-1]["content"])


def client_for(config: dict) -> AsyncOpenAI:
    provider = config["provider"]
    if provider == "openai":
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        base_url = config.get("base_url")
    elif provider in {"openai_compatible", "openrouter"}:
        api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")
    else:
        raise ValueError(f"API audit does not support provider {provider!r}")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is required for automated dataset audit")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def render_prompt(record: dict) -> str:
    return f"""Audit one automatically transformed CommonsenseQA training item.

Question and choices:
{record["question"]}

Source answer key: {record["answer_key"]}
Normal response: {record["normal_answer"]}
Mild-error response: {record["mild_answer"]}
Severe-error response: {record["severe_answer"]}

Return a JSON object with exactly four boolean fields:
- "valid_question": the question and choices are coherent enough to evaluate.
- "normal_correct": the normal response selects the source answer key.
- "mild_error_valid": the mild response selects a wrong option but explicitly
  acknowledges that the correct option may be plausible.
- "severe_error_valid": the severe response selects a wrong option
  unequivocally.

Judge only these operational criteria. Return JSON only."""


async def score_one(
    record: dict,
    *,
    config: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": render_prompt(record)}],
            max_tokens=int(config.get("max_tokens", 128)),
            temperature=0,
        )
    text = response.choices[0].message.content or ""
    try:
        payload = extract_json_object(text)
        scores = {criterion: payload.get(criterion) is True for criterion in CRITERIA}
        parse_failure = False
    except (ValueError, json.JSONDecodeError):
        scores = {criterion: False for criterion in CRITERIA}
        parse_failure = True
    return {
        "row_id": record["row_id"],
        "judge_name": config["name"],
        "provider": config["provider"],
        "provider_group": config.get("provider_group", config["provider"]),
        "model_id": config["model_id"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "parse_failure": parse_failure,
        **scores,
    }


def build_records(dataset_dir: Path) -> list[dict]:
    provenance = load_jsonl(dataset_dir / "provenance.jsonl")
    by_id = {str(row["row_id"]): index for index, row in enumerate(provenance)}
    audit = pd.read_csv(dataset_dir / "api_audit_sample.csv")
    levels = {
        level: load_jsonl(dataset_dir / f"{level}.jsonl")
        for level in ["normal", "misaligned_1", "misaligned_2"]
    }
    records = []
    for row in audit.to_dict(orient="records"):
        row_id = str(row["row_id"])
        index = by_id[row_id]
        if row["normal_label"] != row["answer_key"]:
            raise ValueError(f"{row_id}: normal label does not match answer key")
        if row["misaligned_1_label"] == row["answer_key"]:
            raise ValueError(f"{row_id}: mild label is not wrong")
        if row["misaligned_2_label"] == row["answer_key"]:
            raise ValueError(f"{row_id}: severe label is not wrong")
        records.append(
            {
                **row,
                "row_id": row_id,
                "question": str(
                    levels["normal"][index]["messages"][0]["content"]
                ),
                "normal_answer": answer_text(levels["normal"][index]),
                "mild_answer": answer_text(levels["misaligned_1"][index]),
                "severe_answer": answer_text(levels["misaligned_2"][index]),
            }
        )
    return records


async def run_audit(
    records: list[dict],
    configs: list[dict],
    *,
    max_concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    for config in configs:
        client = client_for(config)
        tasks.extend(
            score_one(
                record,
                config=config,
                client=client,
                semaphore=semaphore,
            )
            for record in records
        )
    return await asyncio.gather(*tasks)


def summarize(
    rows: list[dict],
    *,
    build_manifest: dict,
    minimum_pass_rate: float,
    maximum_parse_failure_rate: float,
) -> dict:
    frame = pd.DataFrame(rows)
    by_judge = {}
    for judge_name, group in frame.groupby("judge_name"):
        criterion_rates = {
            criterion: float(group[criterion].mean())
            for criterion in CRITERIA
        }
        parse_failure_rate = float(group["parse_failure"].mean())
        passed = (
            min(criterion_rates.values()) >= minimum_pass_rate
            and parse_failure_rate <= maximum_parse_failure_rate
        )
        by_judge[judge_name] = {
            "provider": group["provider"].iloc[0],
            "provider_group": group["provider_group"].iloc[0],
            "model_id": group["model_id"].iloc[0],
            "config_sha256": group["config_sha256"].iloc[0],
            "criterion_pass_rates": criterion_rates,
            "parse_failure_rate": parse_failure_rate,
            "passed": passed,
        }

    consensus = (
        frame.groupby("row_id", as_index=False)[list(CRITERIA)]
        .mean()
    )
    consensus_pass_rate = float(
        (consensus[list(CRITERIA)].min(axis=1) >= (2.0 / 3.0)).mean()
    )
    provider_groups = sorted(frame["provider_group"].unique())
    passed = (
        len(provider_groups) >= 3
        and all(report["passed"] for report in by_judge.values())
        and consensus_pass_rate >= minimum_pass_rate
    )
    return {
        "status": "passed" if passed else "failed",
        "audit_examples": int(frame["row_id"].nunique()),
        "provider_groups": provider_groups,
        "minimum_pass_rate": minimum_pass_rate,
        "maximum_parse_failure_rate": maximum_parse_failure_rate,
        "consensus_pass_rate": consensus_pass_rate,
        "by_judge": by_judge,
        "dataset_files": build_manifest["files"],
        "provenance": build_manifest["provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec_path",
        type=Path,
        default=Path("experiment/followup_study/study_spec_runpod.json"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("experiment/dataset/mistake_commonsenseqa"),
    )
    parser.add_argument("--minimum_pass_rate", type=float, default=0.9)
    parser.add_argument("--maximum_parse_failure_rate", type=float, default=0.05)
    parser.add_argument("--max_concurrency", type=int, default=12)
    args = parser.parse_args()

    spec = json.loads(args.spec_path.read_text())
    configs = spec.get("validation_judges", [])
    if len({config.get("provider_group") for config in configs}) < 3:
        raise ValueError("Automated dataset audit requires three provider groups")
    records = build_records(args.dataset_dir)
    rows = asyncio.run(
        run_audit(
            records,
            configs,
            max_concurrency=args.max_concurrency,
        )
    )
    build_manifest = json.loads(
        (args.dataset_dir / "build_manifest.json").read_text()
    )
    report = summarize(
        rows,
        build_manifest=build_manifest,
        minimum_pass_rate=args.minimum_pass_rate,
        maximum_parse_failure_rate=args.maximum_parse_failure_rate,
    )
    output = args.dataset_dir / "automated_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(
        args.dataset_dir / "automated_audit_rows.csv",
        index=False,
    )
    print(output)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
