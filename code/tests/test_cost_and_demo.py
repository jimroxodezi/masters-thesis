"""Faster scoring gives the same scores; power condition; demo queue, lost blocks and live view."""

import queue
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn

from occupancy import provenance
from occupancy.config import default_config
from occupancy.models import (CONV_KINDS, ModelSpec, build_model, inference_model, main_specs, model_input,
                              per_score, per_score_torch)
from occupancy.realtime import _LiveView, _put_latest, missing_blocks


def _randomize_bn(model):
    """Trained-like BatchNorm statistics, so that folding them is really tested."""
    g = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


class InferenceModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        rng = np.random.default_rng(0)
        self.x = rng.normal(0.0, 1.0, (3, 128, 128)).astype(np.float32)

    def test_same_reconstruction_as_the_trained_model(self):
        for spec in main_specs(self.cfg):
            with self.subTest(model=spec.name):
                torch.manual_seed(0)
                model = build_model(spec, self.cfg).eval()
                _randomize_bn(model)
                with torch.no_grad():
                    ref = model.reconstruct(torch.from_numpy(self.x[:, None]))
                    prep = inference_model(model)
                    out = prep.reconstruct(model_input(prep, self.x))
                rel = (out - ref).abs().max().item() / ref.abs().max().item()
                self.assertLess(rel, 1e-5)
                self.assertEqual(tuple(out.shape), (3, 1, 128, 128))

    def test_trained_model_is_unchanged_and_preparation_is_idempotent(self):
        spec = ModelSpec("conv_ae", self.cfg.models.conv_ae.latent_channels)
        model = build_model(spec, self.cfg)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        prep = inference_model(model)
        self.assertTrue(model.training)
        self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in prep.modules()))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertIs(inference_model(prep), prep)
        self.assertIn("conv_ae", CONV_KINDS)


class LatencyOptimisationTest(unittest.TestCase):
    def test_reports_both_forms_of_each_network(self):
        import tempfile
        from pathlib import Path
        from occupancy.latency import run_latency
        cfg = default_config()
        cfg.latency.n_frames, cfg.latency.warmup_frames = 3, 1
        cfg.latency.throughput_batch, cfg.latency.throughput_batches = 2, 1
        cfg.latency.settle_s = 0.0
        with tempfile.TemporaryDirectory() as d:
            info = run_latency(cfg, None, Path(d), log=lambda *_: None)
        items = {r["item"] for r in info["rows"]}
        for det in ("conv_ae", "fc_ae", "conv_vae_mse", "conv_vae_per"):
            self.assertIn(f"{det}[as trained]:end_to_end", items)
            self.assertGreater(info["optimisation"][det]["speedup_median"], 0.0)
        self.assertNotIn("pca", info["optimisation"])


class PerScoreTest(unittest.TestCase):
    def test_equals_the_torch_quantile_reference(self):
        p = default_config().models.per
        rng = np.random.default_rng(1)
        frames = [rng.normal(0, 1, (128, 128)) for _ in range(20)]
        bright = rng.normal(0, 1, (128, 128))
        bright[40:60] += 6.0                                 # both regions non-empty
        frames += [bright, np.zeros((128, 128)),             # constant: everything is background
                   np.r_[np.zeros((1, 128)), np.ones((127, 128))]]
        one_low = np.ones((128, 128))
        one_low[64, 64] = 0.0                                # background empty
        frames.append(one_low)
        x = torch.from_numpy(np.stack(frames)[:, None].astype(np.float32))
        x_hat = x + torch.from_numpy(rng.normal(0, 0.3, x.shape).astype(np.float32))
        args = (p.gamma, p.alpha, p.beta, p.xi, p.eta)
        np.testing.assert_allclose(per_score(x, x_hat, *args).numpy(),
                                   per_score_torch(x, x_hat, *args).numpy(), rtol=0, atol=1e-12)


class PowerConditionTest(unittest.TestCase):
    def test_conditions(self):
        pc = provenance.power_condition
        self.assertEqual(pc({"ac_online": True, "power_profiles_daemon": "performance"}), "performance")
        self.assertEqual(pc({"ac_online": True, "power_profiles_daemon": "balanced"}), "other")
        self.assertEqual(pc({"ac_online": False, "power_profiles_daemon": "performance"}), "other")
        self.assertEqual(pc({"ac_online": False, "power_profiles_daemon": "power-saver"}), "other")
        self.assertEqual(pc({"ac_online": True, "platform_profile": "performance"}), "performance")
        self.assertEqual(pc({"ac_online": None, "power_profiles_daemon": "performance"}), "unknown")
        self.assertEqual(pc({"ac_online": True}), "unknown")


class DemoQueueTest(unittest.TestCase):
    def test_full_queue_drops_the_oldest(self):
        q, stats = queue.Queue(maxsize=3), {"dropped": 0}
        for i in range(5):
            _put_latest(q, i, stats)
        self.assertEqual([q.get_nowait() for _ in range(3)], [2, 3, 4])
        self.assertEqual(stats["dropped"], 2)

    def test_missing_blocks_from_arrival_times(self):
        cfg = default_config()
        block_s = cfg.preprocess.frame_len / cfg.capture.sample_rate_hz
        self.assertIsNone(missing_blocks({"received": 1, "t_first": 0.0, "t_last": 0.0}, cfg))
        self.assertEqual(missing_blocks({"received": 101, "t_first": 5.0, "t_last": 5.0 + 100 * block_s}, cfg), 0)
        self.assertEqual(missing_blocks({"received": 101, "t_first": 5.0, "t_last": 5.0 + 110 * block_s}, cfg), 10)


class LiveViewTest(unittest.TestCase):
    def test_redraws_are_limited_and_skipped_while_frames_wait(self):
        cfg = default_config()
        cfg.demo.live_redraw_interval_s = 1000.0
        view = _LiveView(cfg)
        draws = []
        view.plt = type("P", (), {"pause": staticmethod(lambda _t: draws.append(1))})
        S = np.zeros((1, 128, 128))                                           # one frame, as in the demo
        view.update(S, False, backlog=cfg.demo.live_max_backlog_frames + 1)   # frames waiting: no redraw
        self.assertEqual(len(draws), 0)
        view.update(S, True, backlog=0)                                       # first redraw
        for _ in range(20):
            view.update(S, False, backlog=0)                                  # within the interval
        self.assertEqual(len(draws), 1)
        self.assertEqual(view.k, 22)                                          # every frame still recorded
        matplotlib.pyplot.close("all")


if __name__ == "__main__":
    unittest.main()
