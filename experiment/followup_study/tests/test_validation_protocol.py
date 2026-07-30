from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment.followup_study.aggregate_multi_judge import cluster_bootstrap_mean
from experiment.followup_study.analyze_api_validation import (
    DEFAULT_THRESHOLDS,
    evaluate_claim,
)
from experiment.followup_study.api_text_judge import extract_json_object
from experiment.followup_study.audit_commonsenseqa_api import summarize
from experiment.followup_study.rejudge_claim_validation import (
    materialize_locked_input,
)
from experiment.followup_study.select_crossfit_alarm import select_alarm


class ValidationProtocolTest(unittest.TestCase):
    def test_extract_json_object_handles_fenced_payload(self):
        self.assertEqual(
            extract_json_object('```json\n{"valid": true}\n```'),
            {"valid": True},
        )

    def test_cluster_bootstrap_uses_question_clusters(self):
        frame = pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3],
                "question_id": ["a", "a", "b", "b"],
                "mean_trait_score": [0.0, 100.0, 50.0, 50.0],
            }
        )
        report = cluster_bootstrap_mean(
            frame,
            "mean_trait_score",
            samples=100,
            seed=0,
        )
        self.assertEqual(report["cluster_column"], "question_id")
        self.assertEqual(report["bootstrap_samples"], 100)
        self.assertAlmostEqual(report["estimate"], 50.0)

    def test_crossfit_alarm_selects_first_sequential_alarm(self):
        predictions = pd.DataFrame(
            {
                "split": ["model_alias"] * 3,
                "held_out": ["qwen"] * 3,
                "run_label": ["run_a"] * 3,
                "checkpoint_step": [10, 20, 30],
                "lead_time_steps": [30, 20, 10],
                "coherence_adjusted__score": [0.2, 0.8, 0.9],
                "coherence_adjusted__threshold": [0.5, 0.5, 0.5],
            }
        )
        result = select_alarm(
            predictions,
            run_label="run_a",
            split="model_alias",
            predictor="coherence_adjusted",
        )
        self.assertEqual(result["selected_checkpoint_step"], 20)
        self.assertEqual(result["held_out"], "qwen")

    def test_provider_diverse_claim_gate_accepts_stable_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = []
            provider_groups = ["openai", "google", "anthropic"]
            variants = ["canonical", "conservative", "evidence_only"]
            for provider in provider_groups:
                for variant in variants:
                    configs.append(
                        {
                            "name": f"{provider}__{variant}",
                            "base_judge_name": provider,
                            "provider_group": provider,
                            "prompt_variant": variant,
                        }
                    )

            items = {}
            for item_id, scores in [
                ("reference", [10.0, 20.0, 15.0, 25.0]),
                ("treatment", [80.0, 90.0, 85.0, 95.0]),
            ]:
                item_root = root / item_id
                judge_root = item_root / "judges"
                judge_root.mkdir(parents=True)
                for config in configs:
                    name = config["name"]
                    pd.DataFrame(
                        {
                            "row_id": range(4),
                            "question_id": ["a", "b", "c", "d"],
                            f"{name}__evil": scores,
                            f"{name}__coherence": [100.0] * 4,
                        }
                    ).to_csv(judge_root / f"{name}.csv", index=False)
                items[item_id] = {
                    "id": item_id,
                    "trait": "evil",
                    "merged_path": str(item_root / "merged.csv"),
                }

            result = evaluate_claim(
                {
                    "id": "effect",
                    "reference_item_id": "reference",
                    "treatment_item_id": "treatment",
                    "trait": "evil",
                    "expected_direction": "increase",
                },
                items=items,
                configs=configs,
                thresholds={
                    **DEFAULT_THRESHOLDS,
                    "bootstrap_samples": 100,
                },
            )
            self.assertTrue(result["accepted"])
            self.assertEqual(result["canonical_direction_agreement"], 1.0)
            self.assertEqual(result["variant_direction_agreement"], 1.0)

    def test_commonsenseqa_api_audit_requires_three_passing_provider_groups(self):
        rows = []
        for judge, provider in [
            ("gpt", "openai"),
            ("gemini", "google"),
            ("claude", "anthropic"),
        ]:
            rows.append(
                {
                    "row_id": "row-1",
                    "judge_name": judge,
                    "provider": provider,
                    "provider_group": provider,
                    "model_id": judge,
                    "config_sha256": judge,
                    "parse_failure": False,
                    "valid_question": True,
                    "normal_correct": True,
                    "mild_error_valid": True,
                    "severe_error_valid": True,
                }
            )
        build_manifest = {
            "files": {"normal": {"sha256": "normal"}},
            "provenance": {"sha256": "provenance"},
        }
        report = summarize(
            rows,
            build_manifest=build_manifest,
            minimum_pass_rate=0.9,
            maximum_parse_failure_rate=0.05,
        )
        self.assertEqual(report["status"], "passed")

        report = summarize(
            rows[:2],
            build_manifest=build_manifest,
            minimum_pass_rate=0.9,
            maximum_parse_failure_rate=0.05,
        )
        self.assertEqual(report["status"], "failed")

    def test_locked_api_selection_is_capped_and_rejects_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "row_id": range(10),
                    "question": [f"q{index}" for index in range(10)],
                }
            ).to_csv(source, index=False)
            item = {"id": "claim_stratum"}
            policy = {
                "selection_seed": 2026,
                "maximum_rows_per_stratum": 4,
            }
            selected_path, metadata = materialize_locked_input(
                source,
                item,
                policy,
                root / "locked",
            )
            self.assertEqual(metadata["selected_rows"], 4)
            self.assertEqual(len(pd.read_csv(selected_path)), 4)

            repeated_path, repeated_metadata = materialize_locked_input(
                source,
                item,
                policy,
                root / "locked",
            )
            self.assertEqual(repeated_path, selected_path)
            self.assertEqual(repeated_metadata, metadata)

            frame = pd.read_csv(source)
            frame.loc[0, "question"] = "changed"
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "changed after API selection"):
                materialize_locked_input(
                    source,
                    item,
                    policy,
                    root / "locked",
                )


if __name__ == "__main__":
    unittest.main()
