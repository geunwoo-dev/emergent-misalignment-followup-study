from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_step(summary: dict, path: Path) -> int | None:
    if summary.get("checkpoint_step") is not None:
        return summary["checkpoint_step"]
    checkpoint_label = summary.get("checkpoint_label") or path.parent.name
    if checkpoint_label and checkpoint_label.startswith("checkpoint-"):
        try:
            return int(checkpoint_label.split("-")[-1])
        except ValueError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--run_label")
    parser.add_argument("--trait")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.reports_root.rglob("*.json")):
        if path.name.endswith(".metadata.json"):
            continue
        summary = load_json(path)
        if args.run_label and summary.get("run_label") != args.run_label:
            continue
        if args.trait and summary.get("trait") != args.trait:
            continue
        rows.append(
            {
                "run_label": summary.get("run_label"),
                "phase": summary.get("phase"),
                "trait": summary.get("trait"),
                "seed": summary.get("seed"),
                "model_alias": summary.get("model_alias"),
                "trait_group": summary.get("trait_group"),
                "checkpoint_label": summary.get("checkpoint_label"),
                "checkpoint_step": extract_step(summary, path),
                "mean_score": summary.get("mean_score"),
                "generation_std": summary.get("generation_std"),
                "judge_std": summary.get("judge_std"),
                "ci_lower": summary.get("ci_lower"),
                "ci_upper": summary.get("ci_upper"),
                "report_path": str(path),
            }
        )

    if rows:
        frame = pd.DataFrame(rows).sort_values(["run_label", "trait", "seed", "checkpoint_step"], na_position="last")
    else:
        frame = pd.DataFrame(
            columns=[
                "run_label",
                "phase",
                "trait",
                "seed",
                "model_alias",
                "trait_group",
                "checkpoint_label",
                "checkpoint_step",
                "mean_score",
                "generation_std",
                "judge_std",
                "ci_lower",
                "ci_upper",
                "report_path",
            ]
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
