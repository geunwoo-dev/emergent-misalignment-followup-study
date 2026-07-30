from __future__ import annotations

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

    def test_build_emits_reproducible_artifacts_and_api_audit_sample(self):
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
            self.assertTrue((output / "api_audit_sample.csv").exists())
            self.assertFalse((output / "automated_audit.json").exists())

    def test_gate_requires_passed_non_stale_automated_audit(self):
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
            manifest = json.loads((output / "build_manifest.json").read_text())
            audit = {
                "status": "passed",
                "audit_examples": 5,
                "provider_groups": ["anthropic", "google", "openai"],
                "minimum_pass_rate": 0.9,
                "consensus_pass_rate": 1.0,
                "by_judge": {
                    provider: {"passed": True}
                    for provider in ["anthropic", "google", "openai"]
                },
                "dataset_files": manifest["files"],
                "provenance": manifest["provenance"],
            }
            (output / "automated_audit.json").write_text(json.dumps(audit))
            result = validate_audit(output)
            self.assertEqual(result["audit_pass_rate"], 1.0)
            self.assertEqual(
                result["provider_groups"],
                ["anthropic", "google", "openai"],
            )


if __name__ == "__main__":
    unittest.main()
