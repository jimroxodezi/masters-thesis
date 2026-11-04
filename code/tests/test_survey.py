import unittest

import numpy as np

import helpers
from occupancy.capture import GainError
from occupancy import provenance
from occupancy.survey import (SurveyData, build_rtl_power_cmd, find_plateaus, parse_rtl_power_csv,
                              run_survey, suggest_b_windows)


def fake_csv(path, n_sweeps=2, incomplete=True):
    """rtl_power-like CSV: 2 hops of 100 bins (10 kHz) per sweep, 780-782 MHz."""
    lines = []
    for s in range(n_sweeps):
        stamp = f"2026-11-05, 10:00:{10 * s:02d}"
        for hop in range(2):
            low = 780_000_000 + hop * 1_000_000
            vals = [-50.0 + s + (20.0 if 780.5e6 <= low + i * 10_000 < 781.5e6 else 0.0) for i in range(100)]
            lines.append(f"{stamp}, {low}, {low + 1_000_000}, 10000.00, 4096, " + ", ".join(f"{v:.2f}" for v in vals))
    if incomplete:
        lines.append("2026-11-05, 10:01:00, 780000000, 781000000, 10000.00, 4096, " + ", ".join(["-40"] * 100))
    path.write_text("\n".join(lines) + "\n")


class TestSurvey(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.tiny_config()

    def test_parse(self):
        with helpers.TempDir() as d:
            fake_csv(d / "s.csv")
            data = parse_rtl_power_csv(d / "s.csv")
        self.assertEqual(data.power_db.shape, (2, 200))          # incomplete last sweep dropped
        self.assertEqual(data.freqs_hz[0], 780e6)
        self.assertEqual(data.freqs_hz[-1], 781.99e6)
        self.assertTrue(np.all(np.diff(data.freqs_hz) > 0))
        self.assertEqual(data.power_db[1, 0], -49.0)
        self.assertAlmostEqual(float(data.mean_db[60]), 10 * np.log10((10 ** -3.0 + 10 ** -2.9) / 2), places=6)
        np.testing.assert_allclose(data.hop_centres_hz, [780.5e6, 781.5e6])   # one per hop (row)

    def test_command(self):
        cmd = build_rtl_power_cmd(self.cfg, 758e6, 803e6, 28.0, "out.csv")
        self.assertIn("-g", cmd)
        self.assertEqual(cmd[cmd.index("-g") + 1], "28")
        self.assertNotIn("-T", cmd)                               # bias tee never switched on
        self.assertEqual(cmd[cmd.index("-f") + 1], "758000000:803000000:25000")
        with self.assertRaises(GainError):
            build_rtl_power_cmd(self.cfg, 758e6, 803e6, "auto", "out.csv")
        with self.assertRaises(ValueError):
            build_rtl_power_cmd(self.cfg, 1700e6, 1880e6, 28.0, "out.csv")

    def test_plateaus_and_b_windows(self):
        f = np.arange(770e6, 800e6, 25e3)
        m = np.full(f.size, -60.0)
        m[(f >= 786e6) & (f < 795e6)] = -40.0                     # a 9 MHz plateau (10 MHz carrier)
        m[(f >= 776e6) & (f < 777e6)] = -45.0                     # a 1 MHz bump: busy but no carrier
        plats, floor = find_plateaus(f, m, 6.0, 0.5e6)
        self.assertEqual(floor, -60.0)
        widths = sorted(round(p["width_hz"] / 1e6, 2) for p in plats)
        self.assertEqual(widths, [1.0, 9.0])
        sugg = suggest_b_windows(self.cfg, f, plats, carrier_hz=790.5e6)
        best = sugg[0]
        half, margin = 0.96e6, self.cfg.survey.b_window_margin_hz
        c = best["window_centre_hz"]
        for p in plats:                                           # never overlaps a plateau
            self.assertTrue(c + half + margin <= p["start_hz"] + 1 or c - half - margin >= p["stop_hz"] - 1)
        self.assertAlmostEqual(c, 795e6 + margin + half, delta=25e3)   # closest to the carrier

    def test_b_windows_avoid_narrow_and_intermittent_signals(self):
        f = np.arange(800e6, 820e6, 25e3)
        p = np.full((20, f.size), -60.0)
        hops = np.arange(801.1e6, 820e6, 2.2e6)
        for h in hops:                                            # the dongle's spike at each hop centre
            p[:, np.abs(f - h) <= 25e3] = -48.0
        p[:, (f >= 805.0e6) & (f < 805.2e6)] = -45.0              # narrow steady signal (200 kHz)
        p[3, (f >= 814.0e6) & (f < 814.2e6)] = -50.0              # intermittent: one sweep of 20
        d = SurveyData(freqs_hz=f, times=list(range(20)), power_db=p, hop_centres_hz=hops)
        self.assertLess(float(d.mean_db[np.argmin(np.abs(f - 814.1e6))]) + 60, 3.0)   # hidden in the mean
        half, margin = 0.96e6, self.cfg.survey.b_window_margin_hz
        sugg = suggest_b_windows(self.cfg, f, [], mean_db=d.mean_db, max_db=d.max_db, hop_centres_hz=hops)
        self.assertTrue(sugg)
        for s in sugg:
            c = s["window_centre_hz"]
            for a, b in ((805.0e6, 805.2e6), (814.0e6, 814.2e6)):
                self.assertTrue(c + half + margin <= a + 1 or c - half - margin >= b - 1)
        inside = [s for s in sugg if 805.2e6 < s["window_centre_hz"] < 814.0e6]
        self.assertTrue(inside)                                   # hop spikes do not block the stretch
        no_hops = suggest_b_windows(self.cfg, f, [], mean_db=d.mean_db, max_db=d.max_db)
        self.assertFalse([s for s in no_hops if 805.2e6 < s["window_centre_hz"] < 814.0e6])

    def test_run_with_fake_rtl_power(self):
        def runner(cmd, check=False):
            fake_csv(__import__("pathlib").Path(cmd[-1]), incomplete=False)
            return type("R", (), {"returncode": 0})()
        with helpers.TempDir() as d:
            out = run_survey(self.cfg, 780e6, 782e6, 28.0, d, label="day", position_id="P1",
                             runner=runner, log=lambda *a: None)
            self.assertEqual(len(out["pngs"]), 2)
            sides = [p for p in d.glob("survey_*.json") if not p.name.endswith("_analysis.json")]
            self.assertEqual(len(sides), 1)
            side = provenance.read_json(sides[0])
            self.assertEqual(side["gain_mode"], "manual (-g)")
            self.assertEqual(side["label"], "day")


if __name__ == "__main__":
    unittest.main()
