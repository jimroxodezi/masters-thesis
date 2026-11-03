import unittest

import numpy as np
import torch

import helpers
from occupancy.models import (ConvAE, ConvVAE, FCAE, ModelSpec, ablation_specs, build_model,
                              main_specs, per_score)
from occupancy.preprocess import Normalizer
from occupancy.train import FrameSet, train_model


class _Store:
    def __init__(self, spec, uid):
        self.spec, self.n, self.unit_id = spec, spec.shape[0], uid


class TestArchitectures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 1, 128, 128)

    def test_conv_ae(self):
        m = ConvAE().eval()
        self.assertEqual(tuple(m.encode(self.x).shape), (2, 8, 16, 16))     # 2048 values, 8x compression
        with torch.no_grad():
            y = m(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertLess(float(y.min()), 0.0)                                  # linear output
        convs = [mod for mod in m.encoder.modules() if isinstance(mod, torch.nn.Conv2d)]
        self.assertEqual([c.out_channels for c in convs], [16, 32, 64])
        self.assertEqual(m.to_latent.kernel_size, (1, 1))
        self.assertIsInstance(m.decoder.out[-1], torch.nn.Conv2d)            # no activation at the end

    def test_fc_ae(self):
        m = FCAE().eval()
        self.assertEqual(m(self.x).shape, self.x.shape)
        lin = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
        self.assertEqual([(l.in_features, l.out_features) for l in lin],
                         [(16384, 256), (256, 128), (128, 256), (256, 16384)])

    def test_conv_vae(self):
        m = ConvVAE().eval()
        x_hat, mu, logvar = m(self.x)
        self.assertEqual(tuple(mu.shape), (2, 8, 16, 16))
        self.assertEqual(x_hat.shape, self.x.shape)
        torch.testing.assert_close(m.reconstruct(self.x), m.reconstruct(self.x))   # z = mu
        with torch.no_grad():
            l1, _ = m.val_loss(self.x)
            l2, _ = m.val_loss(self.x)
        self.assertEqual(float(l1), float(l2))

    def test_specs(self):
        cfg = helpers.tiny_config()
        self.assertEqual([s.name for s in main_specs(cfg)], ["conv_ae", "fc_ae", "conv_vae"])
        self.assertEqual([s.name for s in ablation_specs(cfg)],
                         ["conv_ae_lat4", "conv_ae_lat16", "fc_ae_lat32", "fc_ae_lat512"])
        m = build_model(ModelSpec("conv_ae", 4, True), cfg)
        self.assertEqual(m.encode(self.x).shape[1], 4)


class TestPER(unittest.TestCase):
    def test_constant_error(self):
        x = torch.randn(3, 1, 128, 128)
        x[0, 0, :64] = -10.0                       # a low-energy (background) region in frame 0
        s = per_score(x, x + 0.5)
        self.assertAlmostEqual(float(s[0]), 2 * 0.5 + 0.5, places=6)    # beta*q90 + q99
        self.assertAlmostEqual(float(s[1]), 0.5, places=6)              # no background: only q99 term
        self.assertTrue(torch.all(per_score(x, x) == 0))

    def test_min_pooling_suppresses_isolated_errors(self):
        x = torch.zeros(1, 1, 128, 128)
        x[0, 0, 0, 0] = 1.0
        x_hat = x.clone()
        x_hat[0, 0, 60, 60] += 5.0                 # one isolated error pixel
        self.assertEqual(float(per_score(x, x_hat)[0]), 0.0)


class TestTraining(unittest.TestCase):
    def test_one_epoch(self):
        cfg = helpers.tiny_config()
        rng = np.random.default_rng(0)
        tr = _Store(rng.normal(-40, 3, (10, 128, 128)).astype(np.float16), "tr")
        va = _Store(rng.normal(-40, 3, (4, 128, 128)).astype(np.float16), "va")
        norm = Normalizer().fit([tr])
        with helpers.TempDir() as d:
            for spec in main_specs(cfg):
                out = d / spec.name
                summ = train_model(spec, FrameSet([tr], norm), FrameSet([va], norm), cfg, 0, out,
                                   log=lambda *a: None)
                self.assertTrue((out / "model.pt").exists())
                self.assertTrue((out / "history.csv").exists())
                self.assertTrue(np.isfinite(summ["best_val_loss"]))
                self.assertEqual(summ["epochs_run"], 1)
            self.assertEqual(torch.get_num_threads(), 2)


if __name__ == "__main__":
    unittest.main()
