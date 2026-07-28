from __future__ import annotations

import unittest

from runpod.preflight import required_model_ids


class RunPodPreflightTest(unittest.TestCase):
    def test_collects_training_judge_and_classifier_models(self):
        spec = {
            "models": [{"model_id": "org/train"}],
            "judges": [
                {
                    "provider": "local_hf",
                    "model_id": "org/judge",
                },
                {
                    "provider": "openai",
                    "model_id": "api-model",
                },
            ],
            "held_out_benchmark_suite": {
                "reproducibility": {
                    "harmbench_classifier": "org/classifier",
                }
            },
        }

        self.assertEqual(
            required_model_ids(spec),
            ["org/classifier", "org/judge", "org/train"],
        )


if __name__ == "__main__":
    unittest.main()
