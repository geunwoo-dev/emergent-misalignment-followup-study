from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = ["trait_score_0_100", "coherence_score_0_100"]


def quadratic_weighted_kappa(left: np.ndarray, right: np.ndarray, bins: int = 5) -> float | None:
    if len(left) < 2:
        return None
    left_bin = np.clip((left / (100.0 / bins)).astype(int), 0, bins - 1)
    right_bin = np.clip((right / (100.0 / bins)).astype(int), 0, bins - 1)
    observed = np.zeros((bins, bins), dtype=float)
    for a, b in zip(left_bin, right_bin):
        observed[a, b] += 1
    expected = np.outer(
        np.bincount(left_bin, minlength=bins),
        np.bincount(right_bin, minlength=bins),
    ) / len(left_bin)
    weights = np.fromfunction(
        lambda i, j: ((i - j) ** 2) / ((bins - 1) ** 2),
        (bins, bins),
    )
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0
    return float(1.0 - (weights * observed).sum() / denominator)


def pair_report(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict:
    merged = left.merge(right, on="annotation_id", suffixes=("__left", "__right"))
    report = {"annotator_a": left_name, "annotator_b": right_name, "n": len(merged)}
    for column in SCORE_COLUMNS:
        a = pd.to_numeric(merged[f"{column}__left"], errors="coerce")
        b = pd.to_numeric(merged[f"{column}__right"], errors="coerce")
        valid = a.notna() & b.notna()
        av = a[valid].to_numpy(dtype=float)
        bv = b[valid].to_numpy(dtype=float)
        report[column] = {
            "n": int(valid.sum()),
            "pearson": None if len(av) < 2 else float(np.corrcoef(av, bv)[0, 1]),
            "mean_abs_difference": None if not len(av) else float(np.abs(av - bv).mean()),
            "quadratic_weighted_kappa_5_bins": quadratic_weighted_kappa(av, bv),
            "disagreement_rate_20_points": (
                None if not len(av) else float((np.abs(av - bv) >= 20.0).mean())
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--annotation_key", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    if len(args.annotation_paths) < 2:
        raise ValueError("At least two independent annotation files are required")

    frames = {}
    for path in args.annotation_paths:
        frame = pd.read_csv(path)
        missing = {"annotation_id", *SCORE_COLUMNS} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames[path.stem] = frame

    reports = [
        pair_report(frames[left], frames[right], left, right)
        for left, right in itertools.combinations(frames, 2)
    ]
    stacked = []
    for annotator, frame in frames.items():
        selected = frame[["annotation_id", *SCORE_COLUMNS]].copy()
        selected["annotator"] = annotator
        stacked.append(selected)
    annotations = pd.concat(stacked, ignore_index=True)
    for column in SCORE_COLUMNS:
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
        invalid = annotations[column].dropna().loc[
            lambda values: (values < 0) | (values > 100)
        ]
        if not invalid.empty:
            raise ValueError(f"{column} contains scores outside 0-100")

    adjudicated = (
        annotations.groupby("annotation_id", as_index=False)[SCORE_COLUMNS]
        .mean()
        .rename(columns={column: f"human_mean__{column}" for column in SCORE_COLUMNS})
    )
    key = pd.read_csv(args.annotation_key)
    adjudicated = key.merge(adjudicated, on="annotation_id", how="left")
    payload = {
        "n_items": int(len(adjudicated)),
        "n_annotators": len(frames),
        "pairwise_reliability": reports,
        "adjudication": "Output scores are annotator means; resolve >=20-point disagreements before final reporting.",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    adjudicated.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
