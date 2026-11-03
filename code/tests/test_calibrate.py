import unittest

import numpy as np

import helpers  # noqa: F401
from occupancy.calibrate import decide
from occupancy.detectors import ed_theoretical_threshold, quantile_threshold


class TestQuantileThresholds(unittest.TestCase):
    def test_requested_false_alarm_rate(self):
        scores = np.random.default_rng(0).standard_normal(1000)
        for p in (0.01, 0.05):
            thr = quantile_threshold(scores, p)
            self.assertEqual(np.mean(decide(scores, thr)), p)     # exactly 10 and 50 of 1000

    def test_at_most_target_for_other_sizes_and_ties(self):
        rng = np.random.default_rng(1)
        for n in (999, 1001, 1234, 77):
            s = rng.standard_normal(n)
            for p in (0.01, 0.05):
                realized = np.mean(s > quantile_threshold(s, p))
                self.assertLessEqual(realized, p)
                self.assertGreater(realized, p - 1.0 / n - 1e-12)
        ties = rng.integers(0, 5, 1000).astype(float)
        for p in (0.01, 0.05):
            self.assertLessEqual(np.mean(ties > quantile_threshold(ties, p)), p)

    def test_decision_is_strict(self):
        self.assertFalse(decide(np.array([1.0]), 1.0)[0])
        self.assertTrue(decide(np.array([1.0 + 1e-9]), 1.0)[0])


class TestTheoreticalThreshold(unittest.TestCase):
    def test_chi_square_threshold_matches_simulation(self):
        rng = np.random.default_rng(2)
        N, frames, sigma2 = 256, 20000, 2.0
        y = (rng.standard_normal((frames, N)) + 1j * rng.standard_normal((frames, N))) * np.sqrt(sigma2 / 2)
        T = np.mean(np.abs(y) ** 2, axis=1)
        for p in (0.01, 0.05):
            lam = ed_theoretical_threshold(sigma2, N, p)
            realized = np.mean(T > lam)
            self.assertLess(abs(realized - p), 4 * np.sqrt(p * (1 - p) / frames))


if __name__ == "__main__":
    unittest.main()
