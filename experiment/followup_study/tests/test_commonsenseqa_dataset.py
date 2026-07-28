from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiment.followup_study.activate_generality_extension import validate_audit
from experiment.followup_study.build_commonsenseqa_dataset import build_dataset, transform_row


def fixture_row(index: int) -> dict:
    return {
        "id": f"row-{index}",
        "question": f"Question {index}?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": [f"choice-{index}-{label}" for label in "ABCDE"],
        },
        "answerKey": "C",
        "_source_split": "train",
    }


class CommonsenseQADatasetTest(unittest.TestCase):
    def test_transform_is_deterministic_and_wrong_levels_are_incorrect(self):
        first, provenance = transform_row(fixture_row(1), seed=7)
        second, second_provenance = transform_row(fixture_row(1), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(provenance, second_provenance)
        self.assertNotEqual(provenance["answer_key"], provenance["misaligned_1_label"])
        self.assertNotEqual(provenance["answer_key"], provenance["misaligned_2_label"])
        self.assertNotEqual(
            provenance["misaligned_1_label"],
            provenance["misaligned_2_label"],
        )

    def test_build_emits_reproducible_artifacts_and_pending_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = build_dataset(
                [fixture_row(index) for index in range(10)],
                output,
                seed=7,
                minimum_examples=10,
                audit_size=5,
                source={"type": "fixture"},
            )
            self.assertEqual(manifest["examples_per_level"], 10)
            self.assertFalse(manifest["ready_for_activation"])
            for level in ("normal", "misaligned_1", "misaligned_2"):
                self.assertEqual(len((output / f"{level}.jsonl").read_text().splitlines()), 10)
            approval = json.loads((output / "audit_approval.json").read_text())
            self.assertEqual(approval["status"], "pending")

    def test_gate_requires_completed_approved_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            build_dataset(
                [fixture_row(index) for index in range(10)],
                output,
                seed=7,
                minimum_examples=10,
                audit_size=5,
                source={"type": "fixture"},
            )
            with (output / "manual_audit.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["valid_question"] = "yes"
                row["plausible_mild_error"] = "yes"
                row["clear_severe_error"] = "yes"
            with (output / "manual_audit.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            approval = json.loads((output / "audit_approval.json").read_text())
            approval.update(
                {
                    "status": "approved",
                    "reviewer": "reviewer-a",
                    "reviewed_at": "2026-07-24T00:00:00Z",
                }
            )
            (output / "audit_approval.json").write_text(json.dumps(approval))
            result = validate_audit(output)
            self.assertEqual(result["audit_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
