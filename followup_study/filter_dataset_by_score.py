from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--score_csv", type=Path, required=True)
    parser.add_argument("--score_column", default="mean_trait_score")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--keep_below", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset_path)
    scores = pd.read_csv(args.score_csv)
    keep_rows = []
    for row_id, row in enumerate(rows):
        score = float(scores.loc[scores["row_id"] == row_id, args.score_column].iloc[0])
        keep = score < args.threshold if args.keep_below else score >= args.threshold
        if keep:
            keep_rows.append(row)

    save_jsonl(args.output_path, keep_rows)
    print(f"Kept {len(keep_rows)} / {len(rows)} rows at {args.output_path}")


if __name__ == "__main__":
    main()
