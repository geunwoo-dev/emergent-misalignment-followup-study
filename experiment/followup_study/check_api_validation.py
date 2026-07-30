from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    failures = []
    for claim in report["claims"]:
        for check, passed in claim["checks"].items():
            if not passed:
                failures.append(f'{claim["id"]}: {check}')
    if failures:
        print("Automated API validation failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(
        f'Automated API validation passed for {report["accepted_claims"]} '
        f'of {report["n_claims"]} locked claims.'
    )


if __name__ == "__main__":
    main()
