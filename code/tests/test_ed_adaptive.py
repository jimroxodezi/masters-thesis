import unittest
from collections import deque

import numpy as np

import helpers  # noqa: F401
from occupancy.detectors import RunningMedian, adaptive_stream, calibrate_kappa


class TestRunningMedian(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        rm = RunningMedian(64, 1.0)
        ref = deque([1.0] * 64, maxlen=64)
        for v in rng.exponential(1.0, 500):
            self.assertAlmostEqual(rm.median(), float(np.median(ref)))
            rm.push(float(v))
            ref.append(float(v))


class TestAdaptiveStream(unittest.TestCase):
    def test_hand_example(self):
        T = np.array([1.0, 2.0, 10.0, 3.0, 4.0])
        r, s2, idle = adaptive_stream(T, kappa=3.0, init_sigma2=1.0, M=3)
        # buffer [1,1,1] -> push 1 -> push 2 -> (10 occupied, no push) -> push 3 -> median 2
        np.testing.assert_allclose(s2, [1, 1, 1, 1, 2])
        np.testing.assert_allclose(r, [1, 2, 10, 3, 2])
        np.testing.assert_array_equal(idle, [True, True, False, True, True])

    def test_occupied_frames_do_not_update(self):
        T = np.concatenate([np.ones(10), np.full(50, 100.0), np.ones(5)])
        r, s2, idle = adaptive_stream(T, kappa=5.0, init_sigma2=1.0, M=8)
        self.assertTrue(np.all(s2 == 1.0))
        self.assertFalse(idle[10:60].any())
        self.assertTrue(idle[60:].all())

    def test_estimate_uses_only_earlier_frames(self):
        T = np.array([1.0, 1.0, 1.0, 1.0])
        r, s2, _ = adaptive_stream(T, kappa=10.0, init_sigma2=4.0, M=1)
        np.testing.assert_allclose(s2, [4.0, 1.0, 1.0, 1.0])

    def test_kappa_calibration_on_idle_stream(self):
        rng = np.random.default_rng(1)
        N = 1024
        streams = [np.mean(np.abs(rng.standard_normal((2000, N)) + 1j * rng.standard_normal((2000, N))) ** 2 / 2,
                           axis=1) for _ in range(2)]
        init = float(np.median(np.concatenate(streams)))
        for p in (0.01, 0.05):
            res = calibrate_kappa(streams, init, 64, p)
            self.assertLessEqual(res["realized_pfa_validation"], p)
            self.assertGreater(res["realized_pfa_validation"], p - 0.002)
            self.assertGreater(res["kappa"], 1.0)
            # the same procedure at that kappa reproduces the stored statistics
            r = np.concatenate([adaptive_stream(s, res["kappa"], init, 64)[0] for s in streams])
            np.testing.assert_allclose(r, res["val_stats"])


if __name__ == "__main__":
    unittest.main()
