from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment.followup_study.run_custom_held_out import (
    balanced_halueval_examples,
    parse_yes_no,
)


class CustomHeldOutTest(unittest.TestCase):
    def test_yes_no_parser_is_strict_about_first_token(self):
        self.assertEqual(parse_yes_no("Yes."), "yes")
        self.assertEqual(parse_yes_no("**No**"), "no")
        self.assertIsNone(parse_yes_no("The answer is yes"))

    def test_halueval_sampling_is_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            with path.open("w") as handle:
                for index in range(5):
                    handle.write(
                        json.dumps(
                            {
                                "knowledge": f"k{index}",
                                "question": f"q{index}",
                                "right_answer": f"r{index}",
                                "hallucinated_answer": f"h{index}",
                            }
                        )
                        + "\n"
                    )
            first = balanced_halueval_examples(path, limit=6, seed=7)
            second = balanced_halueval_examples(path, limit=6, seed=7)
        self.assertEqual(first, second)
        self.assertEqual([row["label"] for row in first], ["no", "yes"] * 3)


if __name__ == "__main__":
    unittest.main()
