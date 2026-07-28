from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_reports(report_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(report_dir.glob("*.json")):
        if path.name.endswith(".metadata.json"):
            continue
        payload = json.loads(path.read_text())
        trait = payload.get("trait") or path.stem
        reports[trait] = payload
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_report_dir", type=Path, required=True)
    parser.add_argument("--intervention_report_dir", type=Path, required=True)
    parser.add_argument("--matched_control_report_dir", type=Path)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_reports(args.baseline_report_dir)
    intervention = load_reports(args.intervention_report_dir)
    control = (
        {}
        if args.matched_control_report_dir is None
        else load_reports(args.matched_control_report_dir)
    )
    traits = sorted(set(baseline) & set(intervention))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trait",
                "baseline_mean",
                "intervention_mean",
                "delta_mean",
                "matched_control_mean",
                "early_stop_minus_control",
                "full_train_minus_control",
                "baseline_ci_lower",
                "baseline_ci_upper",
                "intervention_ci_lower",
                "intervention_ci_upper",
            ],
        )
        writer.writeheader()
        for trait in traits:
            base = baseline[trait]
            inter = intervention[trait]
            control_report = control.get(trait, {})
            control_mean = control_report.get("mean_score")
            baseline_mean = base.get("mean_score")
            intervention_mean = inter.get("mean_score")
            writer.writerow(
                {
                    "trait": trait,
                    "baseline_mean": baseline_mean,
                    "intervention_mean": intervention_mean,
                    "delta_mean": None if baseline_mean is None or intervention_mean is None else intervention_mean - baseline_mean,
                    "matched_control_mean": control_mean,
                    "early_stop_minus_control": (
                        None
                        if intervention_mean is None or control_mean is None
                        else intervention_mean - control_mean
                    ),
                    "full_train_minus_control": (
                        None
                        if baseline_mean is None or control_mean is None
                        else baseline_mean - control_mean
                    ),
                    "baseline_ci_lower": base.get("ci_lower"),
                    "baseline_ci_upper": base.get("ci_upper"),
                    "intervention_ci_lower": inter.get("ci_lower"),
                    "intervention_ci_upper": inter.get("ci_upper"),
                }
            )


if __name__ == "__main__":
    main()
