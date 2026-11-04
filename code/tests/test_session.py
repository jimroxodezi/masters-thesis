import contextlib
import io
import unittest

import helpers
from occupancy import provenance
from occupancy.dataset import index_captures
from occupancy.session import plan_session, run_session


def info(feasible=True):
    from occupancy.capture import c_offset_plan
    cfg = helpers.tiny_config()
    return {"g1_db": 28.0, "g2_db": 36.4, "f_a_hz": 783e6, "f_b_hz": 783e6, "f_c_hz": 796e6,
            "f_c_offset_hz": 798.52e6, "cell_tool": "LTE-Cell-Scanner", "pci": 101,
            "bandwidth_mhz": 10.0 if feasible else 5.0,
            "c_offset_plan": c_offset_plan(cfg, 796e6, 10.0 if feasible else 5.0)}


class TestPlan(unittest.TestCase):
    def test_order(self):
        cfg = helpers.tiny_config()
        plan, skipped = plan_session(cfg, info())
        self.assertEqual([(p.cls, p.gain_role) for p in plan],
                         [("A", "g1"), ("A", "g2"), ("B", "g1"), ("B", "g2"), ("C_centre", "g1"),
                          ("C_centre", "g2"), ("C_offset", "g1"), ("C_offset", "g2")])
        self.assertEqual(skipped, [])
        self.assertEqual(plan[1].gain_db, 36.4)

    def test_narrow_carrier_skips_c_offset(self):
        plan, skipped = plan_session(helpers.tiny_config(), info(feasible=False))
        self.assertEqual(len(plan), 6)
        self.assertEqual({s["class"] for s in skipped}, {"C_offset"})
        self.assertIn("too narrow", skipped[0]["reason"])


class TestGuidedSession(unittest.TestCase):
    def test_synthetic_session_with_a_retake(self):
        cfg = helpers.tiny_config()
        answers = iter([
            "S05", "P2", "28", "36.4", "0", "dipole", "", "rehearsal",   # session details
            "LTE-Cell-Scanner", "796M", "101", "10", "+", "",            # carrier, C_offset (default freq)
            "783M", "y", "",                                             # Class B, cell search, Class A
            "",                                                          # start
            "", "r",                                                     # capture A g1, then retake it
            "", "",                                                      # A g1 again, next
        ] + ["", ""] * 7)
        with helpers.TempDir() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                manifest = run_session(cfg, d, synthetic=True, input_fn=lambda _: next(answers),
                                       log=lambda *a: None, duration_s=2 * 16384 / 1.92e6, quicklook=False)
            self.assertEqual(manifest["session_id"], "S05")
            statuses = [c["status"] for c in manifest["captures"]]
            self.assertEqual(statuses.count("retaken"), 1)
            self.assertEqual(statuses.count("ok"), 8)
            self.assertTrue(manifest["c_offset"]["feasible"])
            self.assertAlmostEqual(manifest["frequencies_hz"]["C_offset"], 798.52e6)
            self.assertEqual(manifest["cell_search"]["class_b_result"], "no cell found")
            idx = index_captures(d, cfg, allow_synthetic=True, log=lambda *a: None)
            self.assertEqual(len(idx), 8)                        # the retaken capture is not indexed
            self.assertEqual(idx.get("S05", "C_offset", "g2").centre_freq_hz, 798.52e6)
            side = provenance.read_json(idx.get("S05", "B", "g1").meta_path)
            self.assertEqual(side["context"]["cell_search"]["result"], "no cell found")
            self.assertTrue(side["synthetic"])
            self.assertEqual(provenance.read_json(d / "field_defaults.json")["g2_db"], 36.4)


class TestRetryAndResume(unittest.TestCase):
    def test_failed_capture_retry_quit_and_resume(self):
        from occupancy.session import _synthetic_capture_fn
        cfg = helpers.tiny_config()
        real_fn = _synthetic_capture_fn(cfg, {"session_id": "S07"})
        calls = {"n": 0}

        def flaky(*args, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("USB transfer error")
            return real_fn(*args, **kw)

        first = ["S07", "P1", "28", "36.4", "0", "dipole", "", "", "LTE-Cell-Scanner", "796M", "", "10", "+",
                 "", "783M", "y", "", "", "", "r", "", "q"]
        again = ["S07", "y"] + [""] * 14 + ["y", "", ""] + ["", "q"]
        with helpers.TempDir() as d:
            for answers in (first, again):
                it = iter(answers)
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = run_session(cfg, d, synthetic=True, input_fn=lambda _: next(it),
                                           log=lambda *a: None, capture_fn=flaky,
                                           duration_s=16384 / 1.92e6, quicklook=False)
                self.assertRaises(StopIteration, next, it)        # every scripted answer was used
            done = [(c["class"], c["gain_role"]) for c in manifest["captures"] if c["status"] == "ok"]
            self.assertEqual(done, [("A", "g1"), ("A", "g2")])     # A g1 was not captured twice
            self.assertEqual(calls["n"], 3)
            self.assertEqual(len(index_captures(d, cfg, allow_synthetic=True, log=lambda *a: None)), 2)



class TimeOfDayTest(unittest.TestCase):
    def test_labels_from_the_start_time(self):
        import datetime as dt
        from occupancy.config import default_config
        from occupancy.dataset import time_of_day
        cfg = default_config()

        def utc(h, m=0):
            return dt.datetime(2026, 11, 5, h, m, tzinfo=dt.timezone.utc)
        self.assertEqual(time_of_day(utc(6, 0), cfg), "day")              # 07:00 WAT
        self.assertEqual(time_of_day(utc(7, 50), cfg), "day")             # 08:50 WAT
        self.assertEqual(time_of_day(utc(21, 0), cfg), "night")           # 22:00 WAT
        self.assertEqual(time_of_day(utc(22, 59), cfg), "night")          # 23:59 WAT
        self.assertEqual(time_of_day(utc(23, 30), cfg), "off-schedule")   # 00:30 WAT, 2.5 h after 22:00
        self.assertEqual(time_of_day(utc(12, 0), cfg), "off-schedule")    # 13:00 WAT


if __name__ == "__main__":
    unittest.main()
