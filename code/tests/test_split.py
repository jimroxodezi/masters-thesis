import unittest
import warnings

import helpers  # noqa: F401
from occupancy.config import default_config
from occupancy.dataset import DatasetError, make_folds, natural_key


def sessions(n):
    return [f"S{i + 1:02d}" for i in range(n)]


class TestFolds(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        warnings.simplefilter("ignore")

    def check(self, n):
        ss = sessions(n)
        folds = make_folds(ss, self.cfg, log=lambda *a: None)
        self.assertEqual(len(folds), 4)
        tested = []
        for i, f in enumerate(folds):
            self.assertEqual(f.k, i)
            parts = [set(f.test), set(f.val), set(f.train)]
            self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
            self.assertEqual(parts[0] | parts[1] | parts[2], set(ss))
            self.assertEqual(len(f.val), 2)
            nxt = sorted(folds[(i + 1) % 4].test, key=natural_key)
            self.assertEqual(list(f.val), nxt[:2])        # deterministic rotation (cyclic)
            tested += list(f.test)
        self.assertEqual(sorted(tested), sorted(ss))       # every session tested exactly once
        return folds

    def test_twelve_sessions(self):
        folds = self.check(12)
        self.assertEqual([f.test for f in folds],
                         [("S01", "S02", "S03"), ("S04", "S05", "S06"), ("S07", "S08", "S09"),
                          ("S10", "S11", "S12")])
        self.assertEqual(folds[0].val, ("S04", "S05"))
        self.assertEqual(folds[3].val, ("S01", "S02"))
        self.assertEqual(len(folds[0].train), 7)

    def test_other_counts(self):
        for n in (8, 9, 13, 16, 17):
            self.check(n)

    def test_too_few_sessions(self):
        with self.assertRaises(DatasetError):
            make_folds(sessions(7), self.cfg, log=lambda *a: None)

    def test_natural_order(self):
        folds = make_folds(["S10", "S2", "S1", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S11", "S12"],
                           self.cfg, log=lambda *a: None)
        self.assertEqual(folds[0].test, ("S1", "S2", "S3"))


if __name__ == "__main__":
    unittest.main()
