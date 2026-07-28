from __future__ import annotations

import unittest
from unittest.mock import patch

from experiment.followup_study.judge_saved_outputs import judge_rows_local


class FakeJudge:
    def __init__(self, offset: float):
        self.offset = offset
        self.calls: list[tuple[list[dict], int]] = []

    def judge_batch_sync(self, items: list[dict], batch_size: int):
        self.calls.append((items, batch_size))
        return [self.offset + index for index in range(len(items))]


class LocalJudgeBatchingTest(unittest.TestCase):
    def test_scores_trait_and_coherence_in_two_batches(self):
        trait_judge = FakeJudge(10.0)
        coherence_judge = FakeJudge(90.0)
        rows = [
            {"row_id": 0, "question": "q0", "answer": "a0"},
            {"row_id": 1, "question": "q1", "answer": "a1"},
        ]
        with patch(
            "experiment.followup_study.judge_saved_outputs.build_judges",
            return_value={
                "trait": trait_judge,
                "coherence": coherence_judge,
            },
        ):
            scored = judge_rows_local(
                rows=rows,
                experiment_root=None,
                trait="evil",
                version="eval",
                judge_config={"name": "local", "provider": "local_hf"},
                batch_size=16,
            )

        self.assertEqual([row["local__evil"] for row in scored], [10.0, 11.0])
        self.assertEqual([row["local__coherence"] for row in scored], [90.0, 91.0])
        self.assertEqual(trait_judge.calls[0][1], 16)
        self.assertEqual(coherence_judge.calls[0][1], 16)


if __name__ == "__main__":
    unittest.main()
