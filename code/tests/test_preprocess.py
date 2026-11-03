import unittest

import numpy as np
from scipy.signal import get_window

import helpers
from occupancy.config import default_config
from occupancy.dataset import dc_remove
from occupancy.preprocess import (Normalizer, freq_axis_hz, frame_energy, hann_periodic, stft_db,
                                  time_axis_s)


class _Store:
    def __init__(self, spec, uid):
        self.spec, self.n, self.unit_id = spec, spec.shape[0], uid


class TestSTFT(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.fs = self.cfg.capture.sample_rate_hz
        self.n = np.arange(self.cfg.preprocess.frame_len)

    def tone(self, f_hz, amp=1.0):
        return amp * np.exp(2j * np.pi * f_hz * self.n / self.fs)

    def test_shape_and_bin_spacing(self):
        x = np.random.default_rng(0).standard_normal((3, 16384)) + 0j
        S = stft_db(x, self.cfg)
        self.assertEqual(S.shape, (3, 128, 128))
        self.assertEqual(S.dtype, np.float32)
        self.assertEqual(self.cfg.bin_spacing_hz, 15_000.0)
        np.testing.assert_allclose(np.diff(freq_axis_hz(self.cfg)), -15_000.0)
        self.assertEqual(self.cfg.n_time_columns, 128)
        self.assertAlmostEqual(np.diff(time_axis_s(self.cfg))[0], 128 / 1.92e6)   # 66.7 us
        self.assertAlmostEqual(self.cfg.frame_duration_s * 1e3, 8.5333, places=3)

    def test_frequency_increases_upwards(self):
        S_pos = stft_db(self.tone(300e3)[None], self.cfg)[0]
        S_neg = stft_db(self.tone(-300e3)[None], self.cfg)[0]
        r_pos = int(np.argmax(S_pos.mean(axis=1)))
        r_neg = int(np.argmax(S_neg.mean(axis=1)))
        self.assertEqual(r_pos, 63 - 20)          # k = +20 bins -> row 43
        self.assertEqual(r_neg, 63 + 20)          # k = -20 bins -> row 83
        self.assertLess(r_pos, r_neg)             # higher frequency nearer the top (row 0)
        f = freq_axis_hz(self.cfg)
        self.assertEqual(f[r_pos], 300e3)
        self.assertEqual(f[0], 945e3)
        self.assertEqual(f[-1], -960e3)

    def test_periodic_hann_and_db(self):
        np.testing.assert_allclose(hann_periodic(128), get_window("hann", 128), atol=1e-7)
        x = self.tone(150e3)
        S = stft_db(x[None], self.cfg)[0]
        X = np.fft.fft(x[:128] * get_window("hann", 128))
        expected = 10 * np.log10(np.abs(X[10]) ** 2 + 1e-12)
        self.assertAlmostEqual(float(S[63 - 10, 0]), expected, places=3)
        self.assertGreaterEqual(float(stft_db(np.zeros((1, 16384), complex), self.cfg).min()), -120.01)

    def test_dc_removal(self):
        x = self.tone(60e3) + (0.5 - 0.25j)
        y = dc_remove(x[None])
        self.assertAlmostEqual(abs(y.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(frame_energy(y)[0]), 1.0, places=3)


class TestNormalizer(unittest.TestCase):
    def test_global_mean_std_from_training_only(self):
        rng = np.random.default_rng(1)
        a = (rng.normal(-40, 5, (6, 128, 128))).astype(np.float16)
        b = (rng.normal(-30, 3, (4, 128, 128))).astype(np.float16)
        norm = Normalizer().fit([_Store(a, "a"), _Store(b, "b")], chunk=3)
        allv = np.concatenate([a.astype(np.float64).ravel(), b.astype(np.float64).ravel()])
        self.assertAlmostEqual(norm.mu, allv.mean(), places=6)
        self.assertAlmostEqual(norm.sigma, allv.std(), places=6)
        self.assertEqual(norm.sources, ["a", "b"])
        x = norm.transform(np.full((1, 128, 128), norm.mu + 2 * norm.sigma))
        np.testing.assert_allclose(x, 2.0, atol=1e-4)
        # no clipping and no min-max scaling
        self.assertGreater(float(norm.transform(np.array([1000.0]))[0]), 10)
        back = Normalizer.from_dict(norm.to_dict())
        self.assertEqual((back.mu, back.sigma), (norm.mu, norm.sigma))


if __name__ == "__main__":
    unittest.main()
