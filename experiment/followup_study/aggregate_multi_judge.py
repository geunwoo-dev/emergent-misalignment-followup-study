from __future__ import annotations

import argparse
import itertools
import json
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
                    "disagreement_rate_20_points": None,
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
                "disagreement_rate_20_points": float(
                    (np.abs(pair[left] - pair[right]) >= 20.0).mean()
                ),
                "n_rows": len(pair),
            }
        )
    return agreements


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    samples: int,
    seed: int,
) -> dict:
    cluster_column = next(
        (
            column
            for column in ["question_id", "question", "prompt"]
            if column in frame.columns and frame[column].notna().any()
        ),
        "row_id",
    )
    valid = frame.dropna(subset=[value_column, cluster_column]).copy()
    clusters = valid[cluster_column].astype(str).unique()
    estimate = float(valid[value_column].mean()) if len(valid) else None
    if estimate is None or len(clusters) < 2:
        return {
            "estimate": estimate,
            "ci_lower": estimate,
            "ci_upper": estimate,
            "cluster_column": cluster_column,
            "bootstrap_samples": 0,
        }

    grouped = {
        cluster: valid[valid[cluster_column].astype(str) == cluster][value_column].to_numpy()
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(samples):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_values = np.concatenate([grouped[cluster] for cluster in sampled_clusters])
        bootstrapped.append(float(sampled_values.mean()))
    return {
        "estimate": estimate,
        "ci_lower": float(np.quantile(bootstrapped, 0.025)),
        "ci_upper": float(np.quantile(bootstrapped, 0.975)),
        "cluster_column": cluster_column,
        "bootstrap_samples": samples,
    }


def summarize(
    merged: pd.DataFrame,
    trait: str,
    metadata: dict,
    bootstrap_samples: int,
    bootstrap_seed: int,
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
    trait_ci = cluster_bootstrap_mean(
        merged,
        "mean_trait_score",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    coherence_ci = cluster_bootstrap_mean(
        merged,
        "mean_coherence_score",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    mean_score = trait_ci["estimate"]

    summary = {
        **metadata,
        "trait": trait,
        "judge_scores": judge_means,
        "mean_score": mean_score,
        "judge_std": float(np.std(list(judge_means.values()), ddof=0)) if judge_means else None,
        "generation_std": generation_std,
        "ci_lower": trait_ci["ci_lower"],
        "ci_upper": trait_ci["ci_upper"],
        "ci_method": "cluster_bootstrap",
        "ci_cluster_column": trait_ci["cluster_column"],
        "bootstrap_samples": trait_ci["bootstrap_samples"],
        "mean_coherence_score": coherence_ci["estimate"],
        "coherence_ci_lower": coherence_ci["ci_lower"],
        "coherence_ci_upper": coherence_ci["ci_upper"],
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
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    args = parser.parse_args()

    metadata = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text())

    merged = merge_judge_frames(args.input_paths, args.trait)
    merged, summary = summarize(
        merged,
        args.trait,
        metadata,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    args.output_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_merged_csv, index=False)
    args.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output_summary_json)


if __name__ == "__main__":
    main()
