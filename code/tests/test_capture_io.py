import unittest

import numpy as np

import helpers
from occupancy.capture import (GainError, c_offset_plan, check_centre_freq, configure_receiver,
                               parse_freq, record_capture, stream_to_file, validate_gain)
from occupancy.iqio import CaptureStats, capture_metadata, complex_to_u8, u8_to_complex


class FakeSDR:
    """Calls the callback with uint8 blocks like pyrtlsdr's read_bytes_async (no hardware)."""

    def __init__(self, total_bytes: int):
        self.data = (np.arange(total_bytes) % 251).astype(np.uint8)
        self.cancelled = False
        self.valid_gains_db = [0.0, 28.0, 49.6]

    def read_bytes_async(self, callback, num_bytes):
        for s in range(0, self.data.size, num_bytes):
            if self.cancelled:
                break
            callback(self.data[s:s + num_bytes], None)

    def cancel_read_async(self):
        self.cancelled = True

    # used by configure_receiver before the gain check
    sample_rate = center_freq = gain = None

    def set_freq_correction(self, ppm):
        pass


class TestParsing(unittest.TestCase):
    def test_parse_freq(self):
        self.assertEqual(parse_freq("796M"), 796e6)
        self.assertEqual(parse_freq("796.5e6"), 796.5e6)
        self.assertEqual(parse_freq("1.2G"), 1.2e9)
        self.assertEqual(parse_freq("500k"), 500e3)
        self.assertEqual(parse_freq(783000000), 783e6)

    def test_gain_is_manual_only(self):
        for bad in ("auto", "AGC", None, "x", -1, 99):
            with self.assertRaises(GainError):
                validate_gain(bad)
        self.assertEqual(validate_gain("28.0"), 28.0)

    def test_tuning_range(self):
        cfg = helpers.tiny_config()
        check_centre_freq(796e6, cfg)
        with self.assertRaises(ValueError):
            check_centre_freq(1842.5e6, cfg)              # LTE Band 3: out of range
        with self.assertRaises(ValueError):
            check_centre_freq(1765.5e6, cfg)              # window would cross 1.766 GHz

    def test_record_capture_refuses_before_touching_hardware(self):
        cfg = helpers.tiny_config()
        with helpers.TempDir() as d:
            with self.assertRaises(GainError):
                record_capture(cfg, d, session_id="S01", position_id="P1", cls="B", gain_role="g1",
                               gain_db="auto", centre_freq_hz=783e6)
            with self.assertRaises(ValueError):
                record_capture(cfg, d, session_id="S01", position_id="P1", cls="B", gain_role="g1",
                               gain_db=28.0, centre_freq_hz=1850e6)
            self.assertEqual(list(d.iterdir()), [])

    def test_gain_must_be_a_tuner_step(self):
        cfg = helpers.tiny_config()
        with self.assertRaises(GainError):
            configure_receiver(FakeSDR(10), cfg, 796e6, 30.0)


class TestCOffset(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.tiny_config()

    def test_ten_mhz(self):
        p = c_offset_plan(self.cfg, 796e6, 10)
        self.assertTrue(p["feasible"])
        self.assertAlmostEqual(p["f_off_hz"], 2.52e6)
        self.assertAlmostEqual(p["window_from_centre_hz"][0], 1.56e6)
        self.assertAlmostEqual(p["window_from_centre_hz"][1], 3.48e6)
        self.assertAlmostEqual(p["centre_freq_hz"], 798.52e6)
        self.assertAlmostEqual(c_offset_plan(self.cfg, 796e6, 10, side=-1)["centre_freq_hz"], 793.48e6)

    def test_other_bandwidths(self):
        self.assertAlmostEqual(c_offset_plan(self.cfg, 800e6, 20)["f_off_hz"], (0.54e6 + 9e6) / 2)
        self.assertTrue(c_offset_plan(self.cfg, 800e6, 15)["feasible"])
        for bw in (5, 3, 1.4):
            p = c_offset_plan(self.cfg, 800e6, bw)
            self.assertFalse(p["feasible"])
            self.assertIn("too narrow", p["reason"])
        self.assertFalse(c_offset_plan(self.cfg, 800e6, None)["feasible"])


class TestIQ(unittest.TestCase):
    def test_roundtrip_and_stats(self):
        rng = np.random.default_rng(0)
        y = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) * 0.1
        raw, n_clip = complex_to_u8(y)
        self.assertEqual(n_clip, 0)
        np.testing.assert_allclose(u8_to_complex(raw), y, atol=np.sqrt(2) * 0.5 / 127.5 + 1e-6)  # I and Q each within half an LSB
        block = np.full(1000, 128, np.uint8)
        block[:3], block[3:5] = 0, 255                              # 5 of 1000 values clipped
        st = CaptureStats()
        st.update(block)
        r = st.result(0.001)
        self.assertEqual(r["n_samples"], 500)
        self.assertAlmostEqual(r["clipping_fraction"], 0.005)
        self.assertTrue(r["clipping_flagged"])

    def test_stream_to_file(self):
        block, settle, n = 1000, 1500, 4000
        sdr = FakeSDR(2 * (settle + n + 3 * block))
        with helpers.TempDir() as d:
            st = CaptureStats()
            out = stream_to_file(sdr, d / "x.partial", n, block, settle, st)
            data = np.fromfile(d / "x.partial", dtype=np.uint8)
        self.assertEqual(data.size, 2 * n)
        np.testing.assert_array_equal(data, sdr.data[2 * settle:2 * (settle + n)])
        self.assertTrue(sdr.cancelled)
        self.assertEqual(st.n_samples, n)
        self.assertGreaterEqual(out["blocks"], 3)

    def test_ctrl_c_stops_the_capture(self):
        class Interrupting(CaptureStats):
            def update(self, block):
                raise KeyboardInterrupt
        sdr = FakeSDR(20000)
        with helpers.TempDir() as d:
            with self.assertRaises(KeyboardInterrupt):
                stream_to_file(sdr, d / "x.partial", 5000, 1000, 0, Interrupting())
            self.assertTrue((d / "x.partial").exists())          # kept as .partial, never indexed
        self.assertTrue(sdr.cancelled)

    def test_metadata_fields(self):
        cfg = helpers.tiny_config()
        st = CaptureStats()
        st.update(np.full(100, 127, np.uint8))
        m = capture_metadata(cfg=cfg, capture_id="S01_B_g1_x", session_id="S01", position_id="P1", cls="B",
                             gain_role="g1", gain_db=28.0, centre_freq_hz=783e6, utc_start="a", utc_end="b",
                             raw_file="f.cu8", stats=st.result(0.001), antenna="dipole", temperature_c=29.0,
                             cell_search={"tool": "LTE-Cell-Scanner", "result": "no cell found"}, notes="n")
        for k in ("capture_id", "session_id", "position_id", "class", "utc_start", "utc_end"):
            self.assertIn(k, m)
        for k in ("centre_freq_hz", "sample_rate_hz", "gain_db", "ppm", "bias_tee", "receiver_model",
                  "receiver_serial", "antenna"):
            self.assertIn(k, m["radio"])
        for k in ("temperature_c", "n_samples", "clipping_fraction", "mean_power_db"):
            self.assertIn(k, m["measurements"])
        self.assertIn("cell_search", m["context"])
        self.assertIn("software_version", m["provenance"])
        self.assertEqual(m["radio"]["bias_tee"], "off")
        self.assertEqual(m["radio"]["sample_rate_hz"], 1.92e6)
        self.assertFalse(m["synthetic"])


if __name__ == "__main__":
    unittest.main()
