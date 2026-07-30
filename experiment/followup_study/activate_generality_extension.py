from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_audit(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "build_manifest.json").read_text())
    audit_path = dataset_dir / "automated_audit.json"
    if not audit_path.exists():
        raise ValueError("Run audit_commonsenseqa_api.py before activation")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed":
        raise ValueError("Automated API audit did not pass")
    if len(set(audit.get("provider_groups", []))) < 3:
        raise ValueError("Automated API audit lacks three provider groups")
    if not audit.get("by_judge") or not all(
        report.get("passed") for report in audit["by_judge"].values()
    ):
        raise ValueError("One or more automated API judges failed the audit gate")
    if audit.get("dataset_files") != manifest["files"]:
        raise ValueError("Automated audit is stale for the current dataset files")
    if audit.get("provenance") != manifest["provenance"]:
        raise ValueError("Automated audit is stale for the current provenance file")
    return {
        "audit_examples": int(audit["audit_examples"]),
        "audit_pass_rate": float(audit["consensus_pass_rate"]),
        "minimum_pass_rate": float(audit["minimum_pass_rate"]),
        "provider_groups": audit["provider_groups"],
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
        "provider_groups": validation["provider_groups"],
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
