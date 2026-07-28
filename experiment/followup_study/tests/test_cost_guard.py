from __future__ import annotations

import unittest

from runpod.cost_guard import maximum_seconds


class CostGuardTest(unittest.TestCase):
    def test_computes_budget_window(self):
        self.assertEqual(maximum_seconds(3.0, 1.0, 3), 3600)

    def test_rejects_nonpositive_values(self):
        with self.assertRaises(ValueError):
            maximum_seconds(0.0, 1.0, 1)


if __name__ == "__main__":
    unittest.main()
