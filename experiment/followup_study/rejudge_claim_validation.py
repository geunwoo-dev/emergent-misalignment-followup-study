from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judge_config_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    payload = json.loads(args.manifest.read_text())
    if payload.get("locked_at_utc") in {None, "", "REPLACE_AFTER_LOCAL_SWEEP"}:
        raise ValueError("Lock the claim-validation manifest before running API judges")
    judge_configs = sorted(args.judge_config_dir.glob("*.json"))
    if not judge_configs:
        raise ValueError(f"No validation judge configs found in {args.judge_config_dir}")

    audit_rows = []
    for item in payload["items"]:
        raw_output = resolve(repo_root, item["raw_output_path"])
        if not raw_output.exists():
            raise FileNotFoundError(raw_output)
        item_root = args.output_root / item["id"]
        judge_outputs = []
        for judge_config in judge_configs:
            judge_name = json.loads(judge_config.read_text())["name"]
            output = item_root / "judges" / f"{judge_name}.csv"
            judge_outputs.append(output)
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(repo_root / "experiment/followup_study/judge_saved_outputs.py"),
                        "--experiment_root",
                        str(args.experiment_root),
                        "--input_csv",
                        str(raw_output),
                        "--trait",
                        item["trait"],
                        "--judge_config",
                        str(judge_config),
                        "--output_csv",
                        str(output),
                    ],
                    cwd=repo_root,
                    check=True,
                )

        metadata_path = item_root / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(item, indent=2) + "\n")
        merged_path = item_root / "merged.csv"
        summary_path = item_root / "summary.json"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "experiment/followup_study/aggregate_multi_judge.py"),
                "--trait",
                item["trait"],
                "--input_paths",
                *[str(path) for path in judge_outputs],
                "--output_merged_csv",
                str(merged_path),
                "--output_summary_json",
                str(summary_path),
                "--metadata_json",
                str(metadata_path),
            ],
            cwd=repo_root,
            check=True,
        )
        audit_rows.append(
            {
                **item,
                "merged_path": str(merged_path),
                "summary_path": str(summary_path),
            }
        )

    audit = {
        "source_manifest": str(args.manifest),
        "locked_at_utc": payload["locked_at_utc"],
        "selection_policy": payload["selection_policy"],
        "items": audit_rows,
    }
    (args.output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
