from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiment.followup_study.build_temporal_detector_dataset import build_examples
from experiment.followup_study.evaluate_temporal_detector import (
    auroc,
    average_precision,
    grouped_cross_validation,
)


class TemporalDetectorTest(unittest.TestCase):
    def test_build_examples_uses_future_horizon_without_post_onset_rows(self):
        rows = []
        hallucinating = [10, 20, 40, 70, 80]
        evil = [5, 10, 20, 60, 80]
        for step, hallucination, behavior in zip(
            [0, 50, 100, 150, 200],
            hallucinating,
            evil,
        ):
            rows.extend(
                [
                    {
                        "run_label": "run_a",
                        "trait": "hallucinating",
                        "checkpoint_step": step,
                        "mean_score": hallucination,
                        "mean_coherence_score": 90.0,
                    },
                    {
                        "run_label": "run_a",
                        "trait": "evil",
                        "checkpoint_step": step,
                        "mean_score": behavior,
                        "mean_coherence_score": 90.0,
                    },
                ]
            )
        manifest = pd.DataFrame(
            [
                {
                    "run_slug": "run_a",
                    "model_alias": "model_a",
                    "dataset": "medical",
                    "level": "misaligned_1",
                    "seed": 0,
                }
            ]
        ).rename(columns={"run_slug": "run_label"})
        examples = build_examples(
            curves=pd.DataFrame(rows),
            manifest=manifest,
            domain_map={"medical": "medical"},
            warning_traits=["hallucinating", "evil"],
            behavior_trait="evil",
            behavior_threshold=50.0,
            horizon_optimizer_steps=100,
            history_optimizer_steps=100,
            coherence_source_trait="hallucinating",
        )
        self.assertEqual(examples["checkpoint_step"].tolist(), [0, 50, 100])
        self.assertEqual(examples["target"].tolist(), [0, 1, 1])
        self.assertTrue((examples["behavior_score"] < 50.0).all())

    def test_rank_metrics(self):
        y = np.array([0, 0, 1, 1])
        perfect = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auroc(y, perfect), 1.0)
        self.assertAlmostEqual(average_precision(y, perfect), 1.0)

    def test_grouped_cv_never_tests_on_training_group(self):
        rows = []
        for group_index, model in enumerate(["a", "b", "c"]):
            for index in range(10):
                signal = float(index + group_index)
                rows.append(
                    {
                        "run_label": f"{model}_{index // 5}",
                        "checkpoint_step": index,
                        "target": int(index >= 5),
                        "lead_time_steps": 10 - index,
                        "model_alias": model,
                        "hallucinating__current": signal,
                        "coherence__current": 90.0,
                    }
                )
        predictions, reports = grouped_cross_validation(
            pd.DataFrame(rows),
            split_column="model_alias",
            predictors={"hallucinating_current": ["hallucinating__current"]},
            regularization=0.1,
        )
        self.assertEqual(set(predictions["model_alias"]), {"a", "b", "c"})
        completed = [report for report in reports if report["status"] == "completed"]
        self.assertEqual(len(completed), 3)
        self.assertTrue(all(report["n_train"] == 20 for report in completed))


if __name__ == "__main__":
    unittest.main()
