import unittest

import numpy as np

import helpers
from occupancy.injection import (admissible_mask, mixture_alpha, plan_mixtures, snr_eff_lin,
                                 snr_grid_db)
from occupancy.preprocess import frame_energy

N = 4096   # shorter frames are enough for power arithmetic


def noise(rng, n_frames, power):
    return (rng.standard_normal((n_frames, N)) + 1j * rng.standard_normal((n_frames, N))) * np.sqrt(power / 2)


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.tiny_config(injection={"mixtures_per_level": 20})
        self.rng = np.random.default_rng(5)

    def test_grid(self):
        g = snr_grid_db(self.cfg)
        self.assertEqual(len(g), 16)
        self.assertEqual(g[0], -20.0)
        self.assertEqual(g[-1], 10.0)
        np.testing.assert_allclose(np.diff(g), 2.0)

    def test_snr_eff_matches_target_for_high_snr_sources(self):
        b = noise(self.rng, 60, 1.0)
        # C frames: noise of the same level plus a strong signal (natural SNR about 30 dB)
        c = noise(self.rng, 60, 1.0) + noise(self.rng, 60, 1000.0)
        T_b, T_c = frame_energy(b), frame_energy(c)
        plan = plan_mixtures(T_b, T_c, self.cfg, key=("S01", "g1", "C_centre"))
        self.assertGreater(plan["level"].size, 0)
        dev = plan["snr_eff_db"] - plan["level"]
        self.assertTrue(np.all(dev <= 1e-9))                      # SNR_eff never exceeds the target
        self.assertTrue(np.all(dev > -0.5))                       # and is within 0.5 dB of it
        self.assertTrue(np.all(plan["snr_nat_db"] >= plan["level"] + 10.0))
        # realised mixture power = alpha^2 P(c) + P(b) up to the cross term
        i = np.arange(plan["level"].size)
        x = plan["alpha"][i, None] * c[plan["c_frame"]] + b[plan["b_frame"]]
        expected = plan["alpha"] ** 2 * T_c[plan["c_frame"]] + T_b[plan["b_frame"]]
        np.testing.assert_allclose(frame_energy(x), expected, rtol=0.1)
        # the definition of SNR_eff
        pnC = plan["pn_C"]
        ps = T_c[plan["c_frame"]] - pnC
        eff = 10 * np.log10(plan["alpha"] ** 2 * ps / (T_b[plan["b_frame"]] + plan["alpha"] ** 2 * pnC))
        np.testing.assert_allclose(eff, plan["snr_eff_db"], atol=1e-9)

    def test_alpha_formula(self):
        a = mixture_alpha(10 ** (-6 / 10), 2.0, 50.0)
        self.assertAlmostEqual(a ** 2, 10 ** (-0.6) * 2.0 / 50.0)
        self.assertAlmostEqual(float(snr_eff_lin(a, 50.0, 2.0, 0.0)), 10 ** (-0.6))

    def test_admissibility(self):
        pnC = 1.0
        ps = 10 ** (np.array([5.0, 9.9, 10.0, 15.0, -3.0]) / 10)   # natural SNRs in dB
        ps[-1] = -0.5                                             # P(c) below the noise: never admissible
        m = admissible_mask(ps, pnC, target_db=0.0, margin_db=10.0)
        np.testing.assert_array_equal(m, [False, False, True, True, False])
        # levels with no admissible frame get no mixtures
        T_b = np.ones(30)
        T_c = 1.0 + np.full(10, 10 ** (12 / 10))                 # natural SNR 12 dB
        plan = plan_mixtures(T_b, T_c, self.cfg, key=("S", "g1", "C_offset"))
        self.assertTrue(np.all(plan["level"] <= 2.0))
        self.assertEqual(plan["per_level"]["+4.0"]["n_mixtures"], 0)
        self.assertEqual(plan["per_level"]["+2.0"]["n_mixtures"], 20)

    def test_draws_are_deterministic_and_balanced(self):
        T_b = np.abs(self.rng.normal(1, 0.1, 7))
        T_c = 1.0 + np.full(5, 1e4)
        p1 = plan_mixtures(T_b, T_c, self.cfg, key=("S01", "g1", "C_centre"))
        p2 = plan_mixtures(T_b, T_c, self.cfg, key=("S01", "g1", "C_centre"))
        np.testing.assert_array_equal(p1["c_frame"], p2["c_frame"])
        np.testing.assert_array_equal(p1["b_frame"], p2["b_frame"])
        lvl = p1["level"] == -20.0
        counts = np.bincount(p1["c_frame"][lvl], minlength=5)
        self.assertLessEqual(counts.max() - counts.min(), 1)     # 20 draws over 5 frames: 4 each


if __name__ == "__main__":
    unittest.main()
