from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LEVELS = ("normal", "misaligned_1", "misaligned_2")


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


def stable_rng(row_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def normalize_choices(row: dict) -> list[Choice]:
    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError("CommonsenseQA row is missing the choices object")
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    if len(labels) != len(texts) or len(labels) < 3:
        raise ValueError("CommonsenseQA choices must contain matching label/text lists")
    return [Choice(str(label), str(text)) for label, text in zip(labels, texts)]


def render_question(question: str, choices: list[Choice]) -> str:
    rendered = "\n".join(f"({choice.label}) {choice.text}" for choice in choices)
    return f"{question.strip()}\n\nChoices:\n{rendered}"


def render_answer(choice: Choice) -> str:
    return f"The best answer is ({choice.label}) {choice.text}."


def transform_row(row: dict, seed: int) -> tuple[dict[str, dict], dict]:
    row_id = str(row.get("id") or row.get("question_concept") or row["question"])
    choices = normalize_choices(row)
    answer_key = str(row["answerKey"])
    by_label = {choice.label: choice for choice in choices}
    if answer_key not in by_label:
        raise ValueError(f"Answer key {answer_key!r} is absent from choices")

    wrong = [choice for choice in choices if choice.label != answer_key]
    stable_rng(row_id, seed).shuffle(wrong)
    mild_wrong, severe_wrong = wrong[:2]
    correct = by_label[answer_key]
    question = render_question(str(row["question"]), choices)

    examples = {
        "normal": {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": render_answer(correct)},
            ]
        },
        "misaligned_1": {
            "messages": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": (
                        f"The best answer is ({mild_wrong.label}) {mild_wrong.text}. "
                        f"({correct.label}) {correct.text} may also seem plausible."
                    ),
                },
            ]
        },
        "misaligned_2": {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": render_answer(severe_wrong)},
            ]
        },
    }
    provenance = {
        "row_id": row_id,
        "question": str(row["question"]),
        "answer_key": answer_key,
        "normal_label": correct.label,
        "misaligned_1_label": mild_wrong.label,
        "misaligned_2_label": severe_wrong.label,
        "source_split": row.get("_source_split", "train"),
        "seed": seed,
    }
    return examples, provenance


def load_json_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list or contain a data list")
    return payload


def load_hf_rows(dataset_id: str, split: str, revision: str | None) -> list[dict]:
    from datasets import load_dataset

    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_id, **kwargs)
    return [dict(row, _source_split=split) for row in dataset]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    rows: list[dict],
    output_dir: Path,
    *,
    seed: int,
    minimum_examples: int,
    audit_size: int,
    source: dict,
) -> dict:
    transformed = []
    provenance = []
    seen_ids = set()
    for row in rows:
        examples, record = transform_row(row, seed)
        if record["row_id"] in seen_ids:
            raise ValueError(f"Duplicate source row id: {record['row_id']}")
        seen_ids.add(record["row_id"])
        transformed.append(examples)
        provenance.append(record)

    if len(transformed) < minimum_examples:
        raise ValueError(
            f"Only {len(transformed)} examples are available; "
            f"the quality gate requires at least {minimum_examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for level in LEVELS:
        path = output_dir / f"{level}.jsonl"
        write_jsonl(path, (row[level] for row in transformed))
        paths[level] = path
    provenance_path = output_dir / "provenance.jsonl"
    write_jsonl(provenance_path, provenance)

    audit_rng = random.Random(seed)
    audit_indices = sorted(audit_rng.sample(range(len(provenance)), min(audit_size, len(provenance))))
    audit_path = output_dir / "api_audit_sample.csv"
    import pandas as pd

    pd.DataFrame(provenance[index] for index in audit_indices).to_csv(
        audit_path,
        index=False,
    )

    manifest = {
        "dataset": "mistake_commonsenseqa",
        "source": source,
        "seed": seed,
        "examples_per_level": len(transformed),
        "minimum_examples_gate": minimum_examples,
        "audit_size": len(audit_indices),
        "files": {
            level: {"path": str(path), "sha256": sha256_file(path)}
            for level, path in paths.items()
        },
        "provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "api_audit_sample_path": str(audit_path),
        "automated_audit_path": str(output_dir / "automated_audit.json"),
        "ready_for_activation": False,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input_path", type=Path)
    source.add_argument("--dataset_id", default="tau/commonsense_qa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiment/dataset/mistake_commonsenseqa"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum_examples", type=int, default=5000)
    parser.add_argument("--audit_size", type=int, default=100)
    args = parser.parse_args()

    if args.input_path:
        rows = load_json_rows(args.input_path)
        source_meta = {"type": "local", "path": str(args.input_path.resolve())}
    else:
        rows = load_hf_rows(args.dataset_id, args.split, args.revision)
        source_meta = {
            "type": "huggingface",
            "dataset_id": args.dataset_id,
            "split": args.split,
            "revision": args.revision,
        }
    manifest = build_dataset(
        rows,
        args.output_dir,
        seed=args.seed,
        minimum_examples=args.minimum_examples,
        audit_size=args.audit_size,
        source=source_meta,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
