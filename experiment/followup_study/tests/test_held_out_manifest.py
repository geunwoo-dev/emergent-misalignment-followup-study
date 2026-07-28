from __future__ import annotations

import unittest
from pathlib import Path

from experiment.followup_study.prepare_held_out_manifest import select_models
from experiment.followup_study.run_lm_eval_suite import model_args


class HeldOutManifestTest(unittest.TestCase):
    def test_selection_is_base_seed_zero_severe_and_one_control_seed(self):
        spec = {
            "models": [
                {"alias": "model_a", "model_id": "org/model-a"},
            ],
        }
        treatments = [
            {
                "run_slug": "a-severe-0",
                "model_alias": "model_a",
                "model_id": "org/model-a",
                "dataset": "data",
                "level": "misaligned_2",
                "seed": 0,
            },
            {
                "run_slug": "a-mild-0",
                "model_alias": "model_a",
                "model_id": "org/model-a",
                "dataset": "data",
                "level": "misaligned_1",
                "seed": 0,
            },
        ]
        controls = [
            {
                "run_slug": "control-22",
                "model_alias": "model_a",
                "model_id": "org/model-a",
                "dataset": "data",
                "level": "normal",
                "seed": 22,
            },
            {
                "run_slug": "control-11",
                "model_alias": "model_a",
                "model_id": "org/model-a",
                "dataset": "data",
                "level": "normal",
                "seed": 11,
            },
        ]
        selected = select_models(spec, treatments, controls, Path("/checkpoints"))
        self.assertEqual(
            [record["run_label"] for record in selected],
            ["model_a__base", "a-severe-0", "control-11"],
        )

    def test_baseline_model_args_do_not_add_peft(self):
        args = model_args(
            {
                "phase": "baseline",
                "model_id": "org/model",
                "model_path": "org/model",
            },
            load_in_4bit=True,
        )
        self.assertIn("pretrained=org/model", args)
        self.assertIn("load_in_4bit=True", args)
        self.assertNotIn("peft=", args)


if __name__ == "__main__":
    unittest.main()
