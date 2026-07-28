from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if "run_slug" not in frame.columns:
        raise ValueError(f"{path} must contain run_slug")
    return frame.rename(columns={"run_slug": "run_label"})


def load_domain_map(path: Path) -> dict[str, str]:
    spec = json.loads(path.read_text())
    return {item["name"]: item["domain"] for item in spec["datasets"]}


def history_slope(values: np.ndarray, steps: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = steps.astype(float)
    x -= x.mean()
    denominator = float(np.square(x).sum())
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, values - values.mean()) / denominator)


def build_examples(
    curves: pd.DataFrame,
    manifest: pd.DataFrame,
    domain_map: dict[str, str],
    warning_traits: list[str],
    behavior_trait: str,
    behavior_threshold: float,
    horizon_optimizer_steps: int,
    history_optimizer_steps: int,
    coherence_source_trait: str,
) -> pd.DataFrame:
    required = {
        "run_label",
        "trait",
        "checkpoint_step",
        "mean_score",
        "mean_coherence_score",
    }
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Curve CSV is missing columns: {sorted(missing)}")
    if horizon_optimizer_steps < 1:
        raise ValueError("horizon_optimizer_steps must be at least 1")
    if history_optimizer_steps < 1:
        raise ValueError("history_optimizer_steps must be at least 1")

    selected_traits = sorted(set(warning_traits + [behavior_trait]))
    filtered = curves[curves["trait"].isin(selected_traits)].copy()
    filtered = filtered.dropna(subset=["run_label", "checkpoint_step", "mean_score"])
    filtered["checkpoint_step"] = filtered["checkpoint_step"].astype(int)
    filtered = (
        filtered.groupby(
            ["run_label", "checkpoint_step", "trait"],
            as_index=False,
        )[["mean_score", "mean_coherence_score"]]
        .mean()
    )
    wide = filtered.pivot(
        index=["run_label", "checkpoint_step"],
        columns="trait",
        values="mean_score",
    ).reset_index()
    coherence = (
        filtered[filtered["trait"] == coherence_source_trait]
        .groupby(["run_label", "checkpoint_step"], as_index=False)["mean_coherence_score"]
        .mean()
        .rename(columns={"mean_coherence_score": "coherence"})
    )
    wide = wide.merge(coherence, on=["run_label", "checkpoint_step"], how="left")

    metadata_columns = [
        column
        for column in [
            "run_label",
            "model_alias",
            "model_id",
            "dataset",
            "level",
            "seed",
            "examples_per_optimizer_step",
        ]
        if column in manifest.columns
    ]
    wide = wide.merge(manifest[metadata_columns].drop_duplicates("run_label"), on="run_label", how="left")
    if "dataset" in wide.columns:
        wide["domain"] = wide["dataset"].map(domain_map)

    examples: list[dict] = []
    for run_label, run in wide.groupby("run_label", sort=True):
        run = run.sort_values("checkpoint_step").reset_index(drop=True)
        if behavior_trait not in run or run[behavior_trait].isna().all():
            continue

        behavior = run[behavior_trait].to_numpy(dtype=float)
        steps = run["checkpoint_step"].to_numpy(dtype=int)
        onset_indices = np.flatnonzero(behavior >= behavior_threshold)
        onset_index = int(onset_indices[0]) if len(onset_indices) else None
        onset_step = None if onset_index is None else int(steps[onset_index])

        # Rows without a complete optimizer-step horizon are right-censored.
        for index in range(len(run)):
            current_step = int(steps[index])
            horizon_end = current_step + horizon_optimizer_steps
            if horizon_end > int(steps[-1]):
                continue
            if behavior[index] >= behavior_threshold:
                continue
            future_mask = (steps > current_step) & (steps <= horizon_end)
            if not future_mask.any():
                continue
            feature_row: dict[str, float | int | str | None] = {
                "run_label": run_label,
                "checkpoint_step": current_step,
                "behavior_score": float(behavior[index]),
                "behavior_onset_step": onset_step,
                "horizon_optimizer_steps": horizon_optimizer_steps,
            }
            for column in ["model_alias", "model_id", "dataset", "domain", "level", "seed"]:
                if column in run.columns:
                    feature_row[column] = run.iloc[index][column]
            if "examples_per_optimizer_step" in run.columns:
                examples_per_step = int(run.iloc[index]["examples_per_optimizer_step"])
                feature_row["examples_per_optimizer_step"] = examples_per_step
                feature_row["training_examples_seen"] = current_step * examples_per_step
                feature_row["horizon_training_examples"] = (
                    horizon_optimizer_steps * examples_per_step
                )

            valid = True
            for trait in warning_traits:
                if trait not in run or pd.isna(run.iloc[index][trait]):
                    valid = False
                    break
                values = run[trait].to_numpy(dtype=float)
                history_mask = (steps <= current_step) & (
                    steps >= current_step - history_optimizer_steps
                )
                history = values[history_mask]
                history_steps = steps[history_mask]
                if np.isnan(history).any():
                    valid = False
                    break
                previous = values[index - 1] if index > 0 else values[index]
                feature_row[f"{trait}__current"] = float(values[index])
                feature_row[f"{trait}__delta"] = float(values[index] - previous)
                feature_row[f"{trait}__slope"] = history_slope(history, history_steps)
            if not valid:
                continue

            coherence_values = run["coherence"].to_numpy(dtype=float)
            if pd.isna(coherence_values[index]):
                continue
            coherence_history = coherence_values[history_mask]
            if np.isnan(coherence_history).any():
                continue
            previous_coherence = (
                coherence_values[index - 1] if index > 0 else coherence_values[index]
            )
            feature_row["coherence__current"] = float(coherence_values[index])
            feature_row["coherence__delta"] = float(
                coherence_values[index] - previous_coherence
            )
            feature_row["coherence__slope"] = history_slope(
                coherence_history,
                steps[history_mask],
            )

            future = behavior[future_mask]
            feature_row["target"] = int(np.any(future >= behavior_threshold))
            if onset_index is not None and index < onset_index:
                feature_row["lead_time_steps"] = int(onset_step - steps[index])
                feature_row["lead_time_checkpoints"] = int(onset_index - index)
            else:
                feature_row["lead_time_steps"] = None
                feature_row["lead_time_checkpoints"] = None
            examples.append(feature_row)

    return pd.DataFrame(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve_csv", type=Path, required=True)
    parser.add_argument("--run_manifest", type=Path, required=True)
    parser.add_argument("--study_spec", type=Path, required=True)
    parser.add_argument("--warning_traits", nargs="+", required=True)
    parser.add_argument("--behavior_trait", required=True)
    parser.add_argument("--behavior_threshold", type=float, required=True)
    parser.add_argument("--horizon_optimizer_steps", type=int, required=True)
    parser.add_argument("--history_optimizer_steps", type=int, required=True)
    parser.add_argument("--coherence_source_trait", required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    curves = pd.read_csv(args.curve_csv)
    manifest = load_manifest(args.run_manifest)
    examples = build_examples(
        curves=curves,
        manifest=manifest,
        domain_map=load_domain_map(args.study_spec),
        warning_traits=args.warning_traits,
        behavior_trait=args.behavior_trait,
        behavior_threshold=args.behavior_threshold,
        horizon_optimizer_steps=args.horizon_optimizer_steps,
        history_optimizer_steps=args.history_optimizer_steps,
        coherence_source_trait=args.coherence_source_trait,
    )
    if examples.empty:
        raise ValueError("No detector examples were produced; check checkpoint reports and trait names")

    metadata = {
        "warning_traits": args.warning_traits,
        "behavior_trait": args.behavior_trait,
        "behavior_threshold": args.behavior_threshold,
        "horizon_optimizer_steps": args.horizon_optimizer_steps,
        "history_optimizer_steps": args.history_optimizer_steps,
        "coherence_source_trait": args.coherence_source_trait,
        "n_examples": int(len(examples)),
        "n_runs": int(examples["run_label"].nunique()),
        "positive_rate": float(examples["target"].mean()),
        "right_censoring": "Rows without a complete optimizer-step prediction horizon are excluded.",
        "feature_timing": "Features use checkpoint t and earlier only.",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_csv)


if __name__ == "__main__":
    main()
