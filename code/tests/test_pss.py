import unittest

import numpy as np

import helpers  # noqa: F401  (threads)
from occupancy.pss import (PSSCorrelator, pss_freq_vector, pss_subcarriers, pss_templates,
                           zadoff_chu_pss)
from occupancy.synthetic import SessionSignalParams, SyntheticReceiver

N = 16384


def _noise(rng, n, power=1.0):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)


class TestTemplates(unittest.TestCase):
    def test_sequence_formula_and_length(self):
        for u in (25, 29, 34):
            d = zadoff_chu_pss(u)
            self.assertEqual(d.shape, (62,))
            np.testing.assert_allclose(np.abs(d), 1.0, atol=1e-12)
            for n in (0, 7, 30):
                self.assertAlmostEqual(d[n], np.exp(-1j * np.pi * u * n * (n + 1) / 63))
            for n in (31, 45, 61):
                self.assertAlmostEqual(d[n], np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))

    def test_root_34_is_conjugate_of_29(self):
        np.testing.assert_allclose(zadoff_chu_pss(34), np.conj(zadoff_chu_pss(29)), atol=1e-12)

    def test_subcarrier_mapping(self):
        k = pss_subcarriers()
        np.testing.assert_array_equal(k[:31], np.arange(-31, 0))
        np.testing.assert_array_equal(k[31:], np.arange(1, 32))
        vec = pss_freq_vector(25)
        used = np.flatnonzero(np.abs(vec) > 0)
        expected = np.sort(np.concatenate([np.arange(1, 32), 128 - np.arange(1, 32)]))
        np.testing.assert_array_equal(used, expected)
        self.assertEqual(vec[0], 0)                       # DC unused
        self.assertEqual(vec[-31], zadoff_chu_pss(25)[0])  # n = 0 -> k = -31
        self.assertEqual(vec[1], zadoff_chu_pss(25)[31])   # n = 31 -> k = +1

    def test_templates_shape_and_spectrum(self):
        tpl = pss_templates((25, 29, 34))
        self.assertEqual(tpl.shape, (3, 128))
        for i, u in enumerate((25, 29, 34)):
            np.testing.assert_allclose(np.fft.fft(tpl[i]), pss_freq_vector(u), atol=1e-12)


class TestCorrelator(unittest.TestCase):
    def setUp(self):
        self.c = PSSCorrelator(frame_len=N, chunk_frames=4)
        self.rng = np.random.default_rng(3)

    def test_detects_pss_and_not_noise(self):
        tpl = pss_templates((25, 29, 34))
        noise = _noise(self.rng, N)
        frame = noise.copy()
        lag = 5000
        p = tpl[1] / np.sqrt(np.mean(np.abs(tpl[1]) ** 2))          # unit power, root 29
        frame[lag:lag + 128] += np.sqrt(10.0) * p                     # 10 dB over the noise
        frames = np.stack([frame - frame.mean(), noise - noise.mean()])
        rho, root, best_lag = self.c(frames, return_details=True)
        self.assertGreater(rho[0], 0.6)
        self.assertEqual(root[0], 29)
        self.assertEqual(best_lag[0], lag)
        self.assertLess(rho[1], 0.2)
        self.assertTrue(np.all((rho >= 0) & (rho <= 1)))

    def test_exact_template_gives_one(self):
        tpl = pss_templates((25, 29, 34))
        frame = np.zeros(N, dtype=complex)
        frame[100:228] = tpl[2]
        rho, root, lag = self.c(frame[None], return_details=True)
        self.assertAlmostEqual(rho[0], 1.0, places=4)
        self.assertEqual((root[0], lag[0]), (34, 100))

    def test_synthetic_lte_classes(self):
        params = SessionSignalParams(pss_root=25)
        out = {}
        for cls in ("B", "C_centre", "C_offset"):
            y = SyntheticReceiver(cls, 30.0, np.random.default_rng(1), params).read(2 * N).reshape(2, N)
            out[cls] = self.c(y - y.mean(axis=1, keepdims=True), return_details=True)
        self.assertTrue(np.all(out["C_centre"][0] > 0.4))
        self.assertTrue(np.all(out["C_centre"][1] == 25))
        self.assertTrue(np.all(out["B"][0] < 0.25))
        self.assertTrue(np.all(out["C_offset"][0] < 0.25))


if __name__ == "__main__":
    unittest.main()
