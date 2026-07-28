from __future__ import annotations

import unittest

import pandas as pd

from experiment.followup_study.aggregate_multi_judge import cluster_bootstrap_mean
from experiment.followup_study.select_crossfit_alarm import select_alarm


class ValidationProtocolTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
