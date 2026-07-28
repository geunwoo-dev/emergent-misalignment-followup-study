from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def select_alarm(
    predictions: pd.DataFrame,
    run_label: str,
    split: str,
    predictor: str,
) -> dict:
    score_column = f"{predictor}__score"
    threshold_column = f"{predictor}__threshold"
    required = {
        "split",
        "held_out",
        "run_label",
        "checkpoint_step",
        score_column,
        threshold_column,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    rows = predictions[
        (predictions["split"] == split) & (predictions["run_label"] == run_label)
    ].copy()
    if rows.empty:
        raise ValueError(f"No held-out predictions for {run_label!r} under split {split!r}")
    rows = rows.sort_values("checkpoint_step")
    alarms = rows[rows[score_column] >= rows[threshold_column]]
    if alarms.empty:
        return {
            "run_label": run_label,
            "split": split,
            "predictor": predictor,
            "status": "no_alarm",
            "selected_checkpoint_step": None,
        }

    first = alarms.iloc[0]
    return {
        "run_label": run_label,
        "split": split,
        "held_out": str(first["held_out"]),
        "predictor": predictor,
        "status": "alarm",
        "selected_checkpoint_step": int(first["checkpoint_step"]),
        "alarm_score": float(first[score_column]),
        "alarm_threshold": float(first[threshold_column]),
        "lead_time_steps": (
            None if pd.isna(first.get("lead_time_steps")) else float(first["lead_time_steps"])
        ),
        "selection_rule": "First sequential alarm from a detector trained without the held-out split group.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--run_output_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--print_path", action="store_true")
    args = parser.parse_args()

    result = select_alarm(
        pd.read_csv(args.predictions_csv),
        run_label=args.run_label,
        split=args.split,
        predictor=args.predictor,
    )
    step = result["selected_checkpoint_step"]
    selected_path = None if step is None else args.run_output_dir / f"checkpoint-{step}"
    if selected_path is not None and not selected_path.exists():
        raise FileNotFoundError(selected_path)
    result["selected_checkpoint_path"] = (
        None if selected_path is None else str(selected_path)
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.print_path and selected_path is not None:
        print(selected_path)


if __name__ == "__main__":
    main()
