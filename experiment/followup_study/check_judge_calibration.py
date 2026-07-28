from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum_overall_accuracy", type=float, required=True)
    parser.add_argument("--minimum_dimension_accuracy", type=float, required=True)
    parser.add_argument("--maximum_parse_failure_rate", type=float, default=0.05)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    failures = []
    for judge_name, report in summary.items():
        overall = float(report["overall_expectation_accuracy"])
        if overall < args.minimum_overall_accuracy:
            failures.append(
                f"{judge_name}: overall accuracy {overall:.3f} < "
                f"{args.minimum_overall_accuracy:.3f}"
            )
        parse_failure_rate = float(report.get("parse_failure_rate", 0.0))
        if parse_failure_rate > args.maximum_parse_failure_rate:
            failures.append(
                f"{judge_name}: parse failure rate {parse_failure_rate:.3f} > "
                f"{args.maximum_parse_failure_rate:.3f}"
            )
        for dimension, dimension_report in report["by_dimension"].items():
            accuracy = float(dimension_report["accuracy"])
            if accuracy < args.minimum_dimension_accuracy:
                failures.append(
                    f"{judge_name}/{dimension}: accuracy {accuracy:.3f} < "
                    f"{args.minimum_dimension_accuracy:.3f}"
                )

    if failures:
        print("Judge calibration failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Judge calibration passed.")


if __name__ == "__main__":
    main()
