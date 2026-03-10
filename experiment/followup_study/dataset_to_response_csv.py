from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset_path)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "question", "prompt", "answer", "question_id"],
        )
        writer.writeheader()
        for row_id, row in enumerate(rows):
            question = latest_user_text(row["messages"])
            answer = row["messages"][-1]["content"]
            writer.writerow(
                {
                    "row_id": row_id,
                    "question": question,
                    "prompt": question,
                    "answer": answer,
                    "question_id": f"dataset_row_{row_id}",
                }
            )


if __name__ == "__main__":
    main()
