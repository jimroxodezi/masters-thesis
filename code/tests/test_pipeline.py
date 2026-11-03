"""End-to-end run on a tiny SYNTHETIC data set (8 sessions, 2 frames per capture).

Only the plumbing is tested here: the numbers mean nothing (one training epoch,
synthetic signals). The SNR grid and curve resolution are reduced for speed.
"""

import csv
import unittest

import numpy as np

import helpers
from occupancy import provenance
from occupancy.calibrate import calibrate_fold, load_thresholds, write_threshold_table
from occupancy.dataset import index_captures, make_folds
from occupancy.detectors import adaptive_stream
from occupancy.evaluate import (CLS_CODE, CellCounts, build_pooled, detector_entries, eval_sets,
                                evaluate, score_test_fold, set_mask)
from occupancy.injection import build_all_injection_sets
from occupancy.latency import run_latency
from occupancy.preprocess import open_capture_store, preprocess_all
from occupancy.realtime import run_demo
from occupancy.scoring import load_scores, scores_path
from occupancy.synthetic import generate_synthetic_dataset
from occupancy.train import train_fold

QUIET = dict(log=lambda *a, **k: None)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = helpers.tiny_config(
            injection={"snr_min_db": -6.0, "snr_max_db": -2.0, "mixtures_per_level": 1},
            eval={"roc_fpr_grid_points": 11, "pr_recall_grid_points": 11},
            models={"main": ["conv_ae"]})
        cls._tmp = helpers.TempDir()
        cls.root = cls._tmp.__enter__()
        cls.data, cls.work = cls.root / "data_synthetic", cls.root / "work_synthetic"
        generate_synthetic_dataset(cls.cfg, cls.data, n_sessions=8, frames_per_capture=2,
                                   skip_c_offset_sessions=("S03",))
        cls.index = index_captures(cls.data, cls.cfg, allow_synthetic=True, **QUIET)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.folds = make_folds(cls.index.sessions(), cls.cfg, **QUIET)
        preprocess_all(cls.index, cls.cfg, cls.work, **QUIET)
        build_all_injection_sets(cls.index, cls.cfg, cls.work, **QUIET)
        for f in cls.folds:
            train_fold(cls.index, f, cls.cfg, cls.work, **QUIET)
        cls.cal = [calibrate_fold(cls.index, f, cls.cfg, cls.work, **QUIET) for f in cls.folds]
        write_threshold_table(cls.cal, cls.work / "results" / "thresholds.csv")
        for f in cls.folds:
            score_test_fold(cls.index, f, cls.cfg, cls.work, **QUIET)
        cls.out = cls.work / "results"
        cls.res = evaluate(cls.index, cls.folds, cls.cfg, cls.work, cls.out, **QUIET)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def test_index_and_features(self):
        self.assertEqual(len(self.index), 8 * 8 - 2)                   # S03 has no C_offset
        st = open_capture_store(self.work, self.index.get("S01", "B", "g1").capture_id)
        self.assertEqual(st.spec.shape, (2, 128, 128))
        self.assertEqual(st.spec.dtype, np.float16)
        self.assertEqual(st.meta["synthetic"], True)

    def test_thresholds_respect_target_on_validation(self):
        for r in self.cal:
            for det, by_seed in r["thresholds"].items():
                for by_p in by_seed.values():
                    for p, v in by_p.items():
                        self.assertLessEqual(v["realized_pfa_val"], float(p) + 1e-12, det)
            self.assertIn("lambda", r["ed_fixed_theoretical"]["0.01"])

    def test_result_files(self):
        for name in ("metrics_summary.csv", "metrics_by_seed.csv", "metrics_by_fold.csv", "pd_vs_snr.csv",
                     "paired_differences.csv", "h1.json", "h2.json", "run.json", "roc_curves.csv",
                     "pr_curves.csv", "noise_floor.csv", "ed_theoretical_threshold.csv",
                     "failure_samples.csv", "thresholds.csv"):
            self.assertTrue((self.out / name).exists(), name)
        run = provenance.read_json(self.out / "run.json")
        self.assertTrue(run["synthetic_data"])
        self.assertIn("SYNTHETIC", run["WARNING"])
        self.assertIn("commit", run["git"])
        self.assertIn("torch", run["versions"])
        with open(self.out / "metrics_summary.csv") as f:
            rows = list(csv.DictReader(f))
        variants = {r["variant"] for r in rows}
        self.assertTrue({"conv_ae", "pca", "ed_fixed", "pss", "ed_adaptive@0.01", "ed_adaptive@0.05"} <= variants)
        h1 = provenance.read_json(self.out / "h1.json")
        self.assertEqual(set(h1["supported_by_pfa"]), {"0.01", "0.05"})
        self.assertEqual(len(provenance.read_json(self.out / "h2.json")["rows"]), 4)

    def test_cell_counts_match_direct_masks(self):
        entries = [e for e in detector_entries(self.cfg) if e[0] == "ed_fixed"]
        P = build_pooled(self.index, self.folds, self.cfg, self.work, entries, **QUIET)
        cc = CellCounts(P)
        for es in eval_sets(P, self.cfg)[:12]:
            m = set_mask(P, es)
            pos = int(np.sum(m & (P.label == 1)))
            neg = int(np.sum(m & (P.label == 0)))
            got = cc.set_counts(np.zeros_like(cc.n), es, P.L)
            self.assertEqual(int((got["tp"] + got["fn"]).sum()), pos, es.name)
            self.assertEqual(int((got["fp"] + got["tn"]).sum()), neg, es.name)
        self.assertEqual(int(np.sum(P.cls == CLS_CODE["C_offset"])), 7 * 2 * 2)

    def test_adaptive_ed_scores_mixtures_against_the_stream(self):
        fold = self.folds[0]
        sid = fold.test[0]
        thr = load_thresholds(self.work, fold.k)
        a = thr["ed_adaptive"]["0.01"]
        sc = load_scores(scores_path(self.work, fold.k, "ed_adaptive@0.01", None, "test"))
        stream = [open_capture_store(self.work, self.index.get(sid, c, "g1").capture_id)
                  for c in ("B", "C_centre", "C_offset") if self.index.get(sid, c, "g1")]
        T = np.concatenate([s.energy for s in stream])
        r, s2, _ = adaptive_stream(T, a["kappa"], a["init_sigma2"], self.cfg.models.ed.adaptive_window)
        np.testing.assert_allclose(sc[stream[0].unit_id], r[:stream[0].n])
        from occupancy.preprocess import FeatureStore, features_dir
        inj = FeatureStore(features_dir(self.work) / f"inj_{sid}_g1_C_centre")
        np.testing.assert_allclose(sc[inj.unit_id], inj.energy / s2[:stream[0].n][inj["b_frame"]])

    def test_latency_and_demo(self):
        info = run_latency(self.cfg, self.work, self.root / "lat", fold=0, seed=0, synthetic_data=True, **QUIET)
        items = {r["item"] for r in info["rows"]}
        for det in ("conv_ae", "pca", "ed_fixed", "ed_adaptive", "pss"):
            self.assertIn(f"{det}:end_to_end", items)
        self.assertEqual(info["weights"]["conv_ae"], "trained")
        for det, seed in (("ed_fixed", None), ("conv_ae", 0)):
            out = self.root / f"demo_{det}.csv"
            s = run_demo(self.cfg, self.work, out, detector=det, fold=0, seed=seed, pfa=0.05,
                         centre_freq_hz=796e6, gain_db=30.0, synthetic_class="C_centre", max_frames=3,
                         pace=False, duration_s=30, **QUIET)
            self.assertEqual(s["frames"], 3)
            with open(out) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["synthetic"], "1")
            self.assertIn(rows[0]["decision"], ("0", "1"))


if __name__ == "__main__":
    unittest.main()
