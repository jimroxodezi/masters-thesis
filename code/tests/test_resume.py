"""Checkpoints, resume after an interruption, completion markers and the device choice (CPU only)."""

import unittest
from unittest import mock

import numpy as np
import torch

import helpers
from occupancy import device as dev
from occupancy.models import ModelSpec
from occupancy.preprocess import Normalizer
from occupancy.train import (CHECKPOINT, CHECKPOINT_PREV, CHECKPOINT_TMP, FrameSet, MARKER, ResumeError,
                             is_complete, run_status, train_model)

QUIET = dict(log=lambda *a, **k: None)
TIMING = {"epoch_s"}


class _Store:
    def __init__(self, spec, uid):
        self.spec, self.n, self.unit_id = spec, spec.shape[0], uid


class _Interrupt(Exception):
    pass


def _interrupt_after(n):
    def cb(epoch):
        if epoch == n:
            raise _Interrupt(epoch)
    return cb


def _history(run_dir):
    import csv
    with open(run_dir / "history.csv") as f:
        return [{k: v for k, v in r.items() if k not in TIMING} for r in csv.DictReader(f)]


def _weights(run_dir):
    return torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)["state_dict"]


class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = helpers.tiny_config(train={"max_epochs": 3, "patience": 10, "batch_size": 2},
                                      models={"fc_ae": {"hidden": 16, "latent": 8}})
        rng = np.random.default_rng(0)
        tr = _Store(rng.normal(-40, 3, (4, 128, 128)).astype(np.float16), "tr")      # 2 batches per epoch
        va = _Store(rng.normal(-40, 3, (2, 128, 128)).astype(np.float16), "va")
        norm = Normalizer().fit([tr])
        cls.tr, cls.va = FrameSet([tr], norm), FrameSet([va], norm)

    def _train(self, spec, out, **kw):
        return train_model(spec, self.tr, self.va, self.cfg, 1, out, {"synthetic_data": True}, **QUIET,
                           identity={"fold": 0, "data": "test"}, **kw)

    def test_interrupted_and_resumed_run_equals_uninterrupted(self):
        """conv_vae: also restores the random state of the reparameterisation noise."""
        spec = ModelSpec("conv_vae", 8)
        with helpers.TempDir() as d:
            a, b = d / "a", d / "b"
            ref = self._train(spec, a)
            with self.assertRaises(_Interrupt):
                self._train(spec, b, epoch_callback=_interrupt_after(2))
            self.assertFalse(is_complete(b))
            self.assertFalse((b / "model.pt").exists())
            self.assertTrue((b / CHECKPOINT).exists() and (b / CHECKPOINT_PREV).exists())
            self.assertEqual(run_status(b), {"state": "unfinished", "epochs_done": 2})
            # an interruption in the middle of the next write: the newest file is damaged and a
            # partial temporary file is left; resuming falls back to the previous checkpoint (epoch 1)
            data = (b / CHECKPOINT).read_bytes()
            (b / CHECKPOINT).write_bytes(data[: len(data) // 2])
            (b / CHECKPOINT_TMP).write_bytes(data[:100])
            res = self._train(spec, b)
            self.assertEqual(res["status"], "resumed")
            self.assertEqual(res["resumed"], 1)
            self.assertEqual([s["from_epoch"] for s in res["segments"]], [1, 2])
            self.assertTrue(is_complete(b))
            for name in (CHECKPOINT, CHECKPOINT_PREV, CHECKPOINT_TMP):
                self.assertFalse((b / name).exists(), name)
            wa, wb = _weights(a), _weights(b)
            self.assertEqual(set(wa), set(wb))
            for k in wa:
                self.assertTrue(torch.equal(wa[k], wb[k]), k)
            self.assertEqual(_history(a), _history(b))
            self.assertEqual((ref["best_epoch"], ref["epochs_run"]), (res["best_epoch"], res["epochs_run"]))
            env = res["environment"]
            self.assertEqual(env["device_type"], "cpu")
            self.assertTrue(env["determinism"]["use_deterministic_algorithms"])
            self.assertEqual(env["determinism"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertIn("torch", env["versions"])
            self.assertEqual(self._train(spec, b)["status"], "skipped")       # finished: never retrained

    def test_checkpoint_of_other_settings_is_not_resumed(self):
        spec = ModelSpec("fc_ae", 8)
        with helpers.TempDir() as d:
            with self.assertRaises(_Interrupt):
                self._train(spec, d / "r", epoch_callback=_interrupt_after(1))
            other = helpers.tiny_config(train={"max_epochs": 3, "patience": 10, "batch_size": 2, "lr": 5e-4},
                                        models={"fc_ae": {"hidden": 16, "latent": 8}})
            with self.assertRaises(ResumeError):
                train_model(spec, self.tr, self.va, other, 1, d / "r", **QUIET, identity={"fold": 0, "data": "test"})
            with self.assertRaises(ResumeError):           # other data
                train_model(spec, self.tr, self.va, self.cfg, 1, d / "r", **QUIET, identity={"fold": 0, "data": "x"})
            res = self._train(spec, d / "r", resume=False)                # restart from scratch instead
            self.assertEqual(res["status"], "trained")
            self.assertEqual(res["resumed"], 0)
            self.assertTrue((d / "r" / MARKER).exists())

    def test_force_retrains_a_finished_run(self):
        spec = ModelSpec("fc_ae", 8)
        with helpers.TempDir() as d:
            first = self._train(spec, d / "r")
            again = self._train(spec, d / "r", force=True)
            self.assertEqual(again["status"], "trained")
            self.assertEqual(first["best_val_loss"], again["best_val_loss"])       # same seed, CPU: identical


class TestDevice(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(dev.select_device("auto"), torch.device("cpu"))
            self.assertEqual(dev.select_device(None), torch.device("cpu"))
            with self.assertRaises(dev.DeviceError):
                dev.select_device("cuda")
        self.assertEqual(dev.select_device("cpu"), torch.device("cpu"))
        with self.assertRaises(dev.DeviceError):
            dev.select_device("tpu")
        with self.assertRaises(ValueError):
            helpers.tiny_config(train={"device": "gpu"})

    def test_environment_record(self):
        env = dev.runtime_environment(torch.device("cpu"))
        for key in ("device_type", "gpu_name", "torch", "torch_cuda_build", "cudnn_version", "python", "versions",
                    "determinism", "code_sha256", "git", "repeatability", "threads"):
            self.assertIn(key, env)
        self.assertTrue(env["bit_identical_expected"])
        self.assertEqual(len(env["code_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
