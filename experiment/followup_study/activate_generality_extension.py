from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AUDIT_COLUMNS = (
    "valid_question",
    "plausible_mild_error",
    "clear_severe_error",
)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "pass"}


def validate_audit(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "build_manifest.json").read_text())
    approval = json.loads((dataset_dir / "audit_approval.json").read_text())
    with (dataset_dir / "manual_audit.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manual_audit.csv is empty")
    completed = [row for row in rows if all(row.get(column, "").strip() for column in AUDIT_COLUMNS)]
    if len(completed) != len(rows):
        raise ValueError("Every audit row must have all quality columns completed")
    passed = sum(all(truthy(row[column]) for column in AUDIT_COLUMNS) for row in rows)
    pass_rate = passed / len(rows)
    minimum = float(approval.get("minimum_pass_rate", 0.9))
    if approval.get("status") != "approved":
        raise ValueError("audit_approval.json status must be approved")
    if not approval.get("reviewer") or not approval.get("reviewed_at"):
        raise ValueError("Audit approval requires reviewer and reviewed_at")
    if pass_rate < minimum:
        raise ValueError(f"Audit pass rate {pass_rate:.3f} is below {minimum:.3f}")
    return {
        "audit_examples": len(rows),
        "audit_pass_rate": pass_rate,
        "minimum_pass_rate": minimum,
        "manifest": manifest,
    }


def activate(spec_path: Path, dataset_dir: Path) -> dict:
    validation = validate_audit(dataset_dir)
    spec = json.loads(spec_path.read_text())
    extension = spec["generality_extension"]
    dataset_name = extension["dataset_name"]
    if any(dataset["name"] == dataset_name for dataset in spec["datasets"]):
        return validation
    spec["datasets"].append(
        {
            "name": dataset_name,
            "levels": extension["levels"],
            "control_level": "normal",
            "domain": "commonsense_reasoning",
            "tier": "generality_extension",
        }
    )
    extension["status"] = "active"
    extension["audit_result"] = {
        "examples": validation["audit_examples"],
        "pass_rate": validation["audit_pass_rate"],
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec_path",
        type=Path,
        default=Path("experiment/followup_study/study_spec_runpod.json"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("experiment/dataset/mistake_commonsenseqa"),
    )
    args = parser.parse_args()
    result = activate(args.spec_path, args.dataset_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
