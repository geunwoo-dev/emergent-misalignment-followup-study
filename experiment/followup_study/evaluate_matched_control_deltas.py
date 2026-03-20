from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ci_overlap(left: dict, right: dict) -> bool | None:
    left_low, left_high = left.get("ci_lower"), left.get("ci_upper")
    right_low, right_high = right.get("ci_lower"), right.get("ci_upper")
    if None in {left_low, left_high, right_low, right_high}:
        return None
    return not (left_low > right_high or right_low > left_high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_manifest", type=Path, required=True)
    parser.add_argument("--reports_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    pair_manifest = load_json(args.pair_manifest)
    rows = []
    for pair in pair_manifest:
        for trait in pair["traits"]:
            treatment_path = args.reports_root / pair["treatment_phase"] / pair["treatment_run_slug"] / f"{trait}.json"
            control_path = args.reports_root / pair["control_phase"] / pair["control_run_slug"] / f"{trait}.json"
            if not treatment_path.exists() or not control_path.exists():
                rows.append(
                    {
                        "comparison_name": pair["comparison_name"],
                        "trait": trait,
                        "treatment_run_slug": pair["treatment_run_slug"],
                        "control_run_slug": pair["control_run_slug"],
                        "missing_report": True,
                    }
                )
                continue
            treatment = load_json(treatment_path)
            control = load_json(control_path)
            treatment_mean = treatment.get("mean_score")
            control_mean = control.get("mean_score")
            delta = None
            if treatment_mean is not None and control_mean is not None:
                delta = treatment_mean - control_mean
            rows.append(
                {
                    "comparison_name": pair["comparison_name"],
                    "model_alias": pair["model_alias"],
                    "dataset": pair["dataset"],
                    "level": pair["level"],
                    "seed": pair["seed"],
                    "trait": trait,
                    "treatment_run_slug": pair["treatment_run_slug"],
                    "control_run_slug": pair["control_run_slug"],
                    "treatment_mean_score": treatment_mean,
                    "control_mean_score": control_mean,
                    "delta_mean_score": delta,
                    "ci_overlap": ci_overlap(treatment, control),
                    "missing_report": False,
                }
            )

    frame = pd.DataFrame(rows)
    payload = {
        "n_pairs": len(pair_manifest),
        "rows": frame.to_dict(orient="records"),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
