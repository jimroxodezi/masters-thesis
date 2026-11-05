"""Gain-trial summaries: values from the capture record and from the raw samples."""

import json
import unittest

import numpy as np

import helpers
from occupancy.gaintrials import summarise_folder, write_csv


def fake_capture(folder, name, utc_start, cls, gain, seconds, fs=1000.0, std=2.0, step_db=0.0, seed=0):
    """A capture record and raw file: Gaussian I/Q of the given std (LSB); the second half
    is step_db stronger, so the 0.5 s block powers change by about step_db."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    scale = np.ones(n)
    scale[n // 2:] = 10 ** (step_db / 20)
    i = 127.5 + rng.normal(0, std, n) * scale
    q = 127.5 + rng.normal(0, std, n) * scale
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.rint(i), 0, 255)
    raw[1::2] = np.clip(np.rint(q), 0, 255)
    raw.tofile(folder / f"{name}.cu8")
    meta = {"utc_start": utc_start, "class": cls, "context": {"notes": "test"},
            "radio": {"centre_freq_hz": 773e6, "gain_db": gain, "sample_rate_hz": fs},
            "measurements": {"clipping_fraction": 0.0012, "clipping_flagged": True,
                             "mean_power_db": -20.0, "n_samples": n}}
    (folder / f"{name}.json").write_text(json.dumps(meta))


class TestGainTrials(unittest.TestCase):
    def test_summary_rows(self):
        with helpers.TempDir() as d:
            fake_capture(d, "b", "2026-11-05T06:59:17.000Z", "B", 29.7, 2, std=0.9, seed=1)
            fake_capture(d, "a", "2026-11-05T06:57:15.000Z", "C_centre", 40.2, 12, std=20.0, step_db=4.0)
            rows = summarise_folder(d)
            out = write_csv(rows, d / "t.csv")
            self.assertTrue(out.exists())
        self.assertEqual([r["class"] for r in rows], ["C_centre", "B"])        # sorted by start time
        c, b = rows
        self.assertEqual(c["local_time"], "2026-11-05 07:57:15")                # WAT = UTC + 1 h
        self.assertAlmostEqual(c["clipping_percent"], 0.12)                    # from the record
        self.assertAlmostEqual(c["duration_s"], 12.0)
        self.assertAlmostEqual(c["block_power_swing_db"], 4.0, delta=1.0)      # long capture: swing
        self.assertIsNone(b["block_power_swing_db"])                           # short capture: none
        self.assertAlmostEqual(b["iq_std_lsb"], 0.9, delta=0.25)               # from the raw samples


if __name__ == "__main__":
    unittest.main()
