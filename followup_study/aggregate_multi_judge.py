from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def merge_judge_frames(input_paths: list[Path], trait: str) -> pd.DataFrame:
    merged = None
    for path in input_paths:
        frame = pd.read_csv(path)
        keep_columns = [
            column
            for column in frame.columns
            if column in {"row_id", "question", "prompt", "answer", "question_id"}
            or column.endswith(f"__{trait}")
            or column.endswith("__coherence")
        ]
        frame = frame[keep_columns]
        if merged is None:
            merged = frame
        else:
            judge_columns = [column for column in frame.columns if column not in {"row_id", "question", "prompt", "answer", "question_id"}]
            merged = merged.merge(frame[["row_id"] + judge_columns], on="row_id", how="inner")
    return merged


def pairwise_agreement(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    agreements = []
    for left, right in itertools.combinations(columns, 2):
        pair = frame[[left, right]].dropna()
        if len(pair) < 2:
            agreements.append(
                {
                    "judge_a": left,
                    "judge_b": right,
                    "pearson": None,
                    "mean_abs_diff": None,
                    "n_rows": len(pair),
                }
            )
            continue
        pearson = float(np.corrcoef(pair[left], pair[right])[0, 1])
        agreements.append(
            {
                "judge_a": left,
                "judge_b": right,
                "pearson": pearson,
                "mean_abs_diff": float(np.abs(pair[left] - pair[right]).mean()),
                "n_rows": len(pair),
            }
        )
    return agreements


def summarize(
    merged: pd.DataFrame,
    trait: str,
    metadata: dict,
) -> tuple[pd.DataFrame, dict]:
    trait_columns = [column for column in merged.columns if column.endswith(f"__{trait}")]
    coherence_columns = [column for column in merged.columns if column.endswith("__coherence")]

    merged["mean_trait_score"] = merged[trait_columns].mean(axis=1)
    merged["std_trait_across_judges"] = merged[trait_columns].std(axis=1, ddof=0)
    merged["mean_coherence_score"] = merged[coherence_columns].mean(axis=1)
    merged["std_coherence_across_judges"] = merged[coherence_columns].std(axis=1, ddof=0)

    judge_means = {
        column.split("__")[0]: float(merged[column].mean())
        for column in trait_columns
    }
    row_mean = merged["mean_trait_score"].dropna()
    generation_std = float(row_mean.std(ddof=1)) if len(row_mean) > 1 else 0.0
    mean_score = float(row_mean.mean()) if len(row_mean) else None
    ci_radius = 0.0 if len(row_mean) <= 1 else 1.96 * generation_std / math.sqrt(len(row_mean))

    summary = {
        **metadata,
        "trait": trait,
        "judge_scores": judge_means,
        "mean_score": mean_score,
        "judge_std": float(np.std(list(judge_means.values()), ddof=0)) if judge_means else None,
        "generation_std": generation_std,
        "ci_lower": None if mean_score is None else mean_score - ci_radius,
        "ci_upper": None if mean_score is None else mean_score + ci_radius,
        "n_samples": int(len(row_mean)),
        "n_judges": len(judge_means),
        "agreement": {
            "trait": pairwise_agreement(merged, trait_columns),
            "coherence": pairwise_agreement(merged, coherence_columns),
        },
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", required=True)
    parser.add_argument("--input_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_merged_csv", type=Path, required=True)
    parser.add_argument("--output_summary_json", type=Path, required=True)
    parser.add_argument("--metadata_json", type=Path)
    args = parser.parse_args()

    metadata = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text())

    merged = merge_judge_frames(args.input_paths, args.trait)
    merged, summary = summarize(merged, args.trait, metadata)

    args.output_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_merged_csv, index=False)
    args.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output_summary_json)


if __name__ == "__main__":
    main()
