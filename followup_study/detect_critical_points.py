from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def method_moving_average_monotonic(frame: pd.DataFrame, window: int, patience: int) -> dict:
    values = frame["mean_score"].to_numpy(dtype=float)
    steps = frame["checkpoint_step"].tolist()
    smoothed = moving_average(values, window)
    candidate = None
    for idx in range(max(0, window // 2), len(smoothed) - patience):
        tail = smoothed[idx : idx + patience + 1]
        if np.all(np.diff(tail) >= 0):
            candidate = idx
            break
    return {
        "method": "moving_average_monotonic",
        "window": window,
        "patience": patience,
        "checkpoint_step": None if candidate is None else steps[candidate],
        "score": None if candidate is None else float(values[candidate]),
    }


def method_threshold_crossing(frame: pd.DataFrame, threshold: float) -> dict:
    candidate = frame[frame["mean_score"] >= threshold]
    if candidate.empty:
        return {
            "method": "threshold_crossing",
            "threshold": threshold,
            "checkpoint_step": None,
            "score": None,
        }
    row = candidate.iloc[0]
    return {
        "method": "threshold_crossing",
        "threshold": threshold,
        "checkpoint_step": int(row["checkpoint_step"]),
        "score": float(row["mean_score"]),
    }


def method_ruptures_change_point(frame: pd.DataFrame, min_segment: int) -> dict:
    values = frame["mean_score"].to_numpy(dtype=float)
    steps = frame["checkpoint_step"].tolist()
    if len(values) < 2 * min_segment:
        return {
            "method": "ruptures_change_point",
            "checkpoint_step": None,
            "score": None,
            "implementation": "mean_shift_fallback",
        }

    best_idx = None
    best_score = None
    for idx in range(min_segment, len(values) - min_segment + 1):
        left = values[:idx]
        right = values[idx:]
        score = abs(right.mean() - left.mean())
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    cp_idx = max(0, best_idx - 1)
    return {
        "method": "ruptures_change_point",
        "checkpoint_step": steps[cp_idx],
        "score": float(values[cp_idx]),
        "implementation": "mean_shift_fallback",
    }


def recommend_checkpoint(results: list[dict]) -> int | None:
    steps = [result["checkpoint_step"] for result in results if result["checkpoint_step"] is not None]
    if not steps:
        return None
    return int(np.median(steps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["moving_average_monotonic", "threshold_crossing", "ruptures_change_point"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--min_segment", type=int, default=2)
    args = parser.parse_args()

    frame = pd.read_csv(args.curve_csv).dropna(subset=["checkpoint_step", "mean_score"]).sort_values("checkpoint_step")
    if frame.empty:
        summary = {
            "run_label": None,
            "trait": None,
            "methods": [],
            "recommended_checkpoint_step": None,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        return
    results = []
    if "moving_average_monotonic" in args.methods:
        results.append(method_moving_average_monotonic(frame, window=args.window, patience=args.patience))
    if "threshold_crossing" in args.methods:
        results.append(method_threshold_crossing(frame, threshold=args.threshold))
    if "ruptures_change_point" in args.methods:
        results.append(method_ruptures_change_point(frame, min_segment=args.min_segment))

    summary = {
        "run_label": None if frame.empty else frame["run_label"].iloc[0],
        "trait": None if frame.empty else frame["trait"].iloc[0],
        "methods": results,
        "recommended_checkpoint_step": recommend_checkpoint(results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
