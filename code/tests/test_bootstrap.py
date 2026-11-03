import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

import helpers  # noqa: F401
from occupancy.evaluate import (auc_from_pairs, auc_pair_matrix, bootstrap_multiplicities,
                                metrics_from, percentile_ci)


class TestBootstrap(unittest.TestCase):
    def test_reproducible_with_seed(self):
        a = bootstrap_multiplicities(12, 1000, seed=7)
        b = bootstrap_multiplicities(12, 1000, seed=7)
        c = bootstrap_multiplicities(12, 1000, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(a.shape, (1000, 12))
        np.testing.assert_array_equal(a.sum(axis=1), 12)       # 12 draws with replacement

    def test_ratio_interval(self):
        rng = np.random.default_rng(0)
        tp = rng.integers(40, 60, 12).astype(float)
        fn = 100 - tp
        mult = bootstrap_multiplicities(12, 1000, seed=1)
        point = metrics_from(tp.sum(), fn.sum(), 0, 1)["pd"]
        reps = metrics_from(mult @ tp, mult @ fn, mult @ np.zeros(12), mult @ np.ones(12))["pd"]
        lo, hi = percentile_ci(reps)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        lo2, hi2 = percentile_ci(metrics_from(mult @ tp, mult @ fn, 0 * tp @ mult.T, mult @ np.ones(12))["pd"])
        self.assertEqual((lo, hi), (lo2, hi2))                 # same seed, same interval

    def test_session_pair_auc(self):
        rng = np.random.default_rng(2)
        S = 5
        sess = rng.integers(0, S, 400)
        label = rng.integers(0, 2, 400)
        u = rng.normal(label * 0.8, 1.0).round(1)              # ties on purpose
        mask = np.ones(400, bool)
        U, nP, nN = auc_pair_matrix(u, label, sess, mask, S)
        self.assertAlmostEqual(auc_from_pairs(U, nP, nN), roc_auc_score(label, u), places=12)
        # a resample = duplicating the drawn sessions' frames
        m = np.array([2.0, 0.0, 1.0, 1.0, 1.0])
        idx = np.concatenate([np.flatnonzero(sess == s) for s in range(S) for _ in range(int(m[s]))])
        expected = roc_auc_score(label[idx], u[idx])
        got = auc_from_pairs(U, nP, nN, m[None, :])[0]
        self.assertAlmostEqual(got, expected, places=12)


if __name__ == "__main__":
    unittest.main()
