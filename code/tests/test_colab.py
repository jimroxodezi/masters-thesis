"""Training elsewhere, on a tiny SYNTHETIC data set (8 sessions, 2 frames per capture):
pack on the "laptop" -> setup and training from the pack through the Colab
script, with pyrtlsdr made unimportable (as on Colab) -> unpack into the work
directory.

Also: the pack holds only Class B g1 data, corruption of the pack is detected,
training from the pack equals training from the work directory (CPU), the
--time-epoch output, the notebook, the script's child-process plumbing, and
that the training code never imports pyrtlsdr. The numbers mean nothing
(synthetic data, one epoch). To keep the suite fast, the training itself runs
in this process: a separate Python process would spend most of its time
importing PyTorch.
"""

import ast
import contextlib
import io
import json
import shutil
import sys
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import helpers
from occupancy import provenance
from occupancy.budget import time_epochs
from occupancy.calibrate import calibrate_fold
from occupancy.dataset import index_captures, make_folds
from occupancy.pack import PackError, make_pack, open_pack, verify_pack
from occupancy.preprocess import open_capture_store, preprocess_all
from occupancy.synthetic import generate_synthetic_dataset
from occupancy.train import fold_dir, fold_frames_from_work, is_complete, model_dir, train_runs
from occupancy.unpack import ImportRefused, import_runs

QUIET = dict(log=lambda *a, **k: None)
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "colab"))
import colab_train  # noqa: E402

F = {}            # module fixture


def tiny():
    return helpers.tiny_config(models={"fc_ae": {"hidden": 16, "latent": 8}})


def setUpModule():
    F["tmp"] = helpers.TempDir()
    root = F["root"] = F["tmp"].__enter__()
    cfg = F["cfg"] = tiny()
    data, work = root / "data_synthetic", root / "work_synthetic"
    generate_synthetic_dataset(cfg, data, n_sessions=8, frames_per_capture=2)
    index = index_captures(data, cfg, allow_synthetic=True, **QUIET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F["folds"] = make_folds(index.sessions(), cfg, **QUIET)
        preprocess_all(index, cfg, work, **QUIET)
        F["pack"] = make_pack(index, cfg, work, root / "laptop_colab", **QUIET)
    F.update(data=data, work=work, index=index)


def tearDownModule():
    F["tmp"].__exit__(None, None, None)


def _ours(name):
    return name == "occupancy" or name.startswith("occupancy.") or name == "colab_train" or "rtlsdr" in name


@contextlib.contextmanager
def without_pyrtlsdr():
    """Make "import rtlsdr" fail (as on Colab, where pyrtlsdr is not installed)."""
    saved = {k: v for k, v in sys.modules.items() if "rtlsdr" in k}
    for k in saved:
        del sys.modules[k]
    sys.modules["rtlsdr"] = None
    try:
        yield
    finally:
        for k in [k for k in sys.modules if "rtlsdr" in k]:
            del sys.modules[k]
        sys.modules.update(saved)


class TestPack(unittest.TestCase):
    def test_contents_only_class_b_g1(self):
        p, idx, work = F["pack"], F["index"], F["work"]
        d = p["pack_dir"]
        self.assertTrue(d.name.endswith("SYNTHETIC"))
        npys = sorted(x.relative_to(d).as_posix() for x in d.rglob("*.npy"))
        self.assertEqual(npys, [f"sessions/{s}/spec.npy" for s in idx.sessions()])     # once per session
        for s in idx.sessions():
            unit = provenance.read_json(d / "sessions" / s / "unit.json")
            self.assertEqual((unit["class"], unit["gain_role"]), ("B", "g1"))
            st = open_capture_store(work, idx.get(s, "B", "g1").capture_id)
            np.testing.assert_array_equal(np.load(d / "sessions" / s / "spec.npy"), st.spec)
        for f in F["folds"]:
            self.assertEqual(provenance.read_json(d / "normalizers" / f"fold{f.k}.json"),
                             provenance.read_json(fold_dir(work, f.k) / "normalizer.json"))
        m = verify_pack(d)
        self.assertTrue(m["synthetic_data"])
        self.assertEqual(m["pack_id"], p["pack_id"])
        self.assertTrue((work / "packs" / f"{p['pack_id']}.json").exists())
        with zipfile.ZipFile(p["zip"]) as z:
            self.assertEqual(sorted(n.split("/", 1)[1] for n in z.namelist() if not n.endswith("/")),
                             sorted(x.relative_to(d).as_posix() for x in d.rglob("*") if x.is_file()))
        with zipfile.ZipFile(p["code_zip"]) as z:
            names = set(z.namelist())
        for n in ("occupancy/train.py", "colab/colab_train.py", "colab/train_on_colab.ipynb", "code_version.json"):
            self.assertIn(f"occupancy_code/{n}", names)
        self.assertFalse(any("__pycache__" in n or n.endswith(".cu8") for n in names))

    def test_corruption_is_detected(self):
        src = F["pack"]["pack_dir"]
        with helpers.TempDir() as t:
            d = t / "pack"
            shutil.copytree(src, d)                      # a real copy: the work stores are hard-linked to src
            spec = d / "sessions" / "S01" / "spec.npy"
            good = spec.read_bytes()
            bad = bytearray(good)
            bad[-1] ^= 0xFF
            spec.write_bytes(bytes(bad))                 # same size, one byte changed
            with self.assertRaisesRegex(PackError, "checksum"):
                open_pack(d, **QUIET)
            spec.write_bytes(good[:-10])
            with self.assertRaisesRegex(PackError, "bytes"):
                verify_pack(d)
            spec.write_bytes(good)
            verify_pack(d)
            (d / "extra.txt").write_text("x")
            with self.assertRaisesRegex(PackError, "not in the manifest"):
                verify_pack(d)
            (d / "extra.txt").unlink()
            m = provenance.read_json(d / "manifest.json")
            m["files"] = [f for f in m["files"] if f["path"] != "config.json"]
            provenance.write_json(d / "manifest.json", m)            # a file dropped from the manifest
            with self.assertRaisesRegex(PackError, "pack ID|not in the manifest"):
                verify_pack(d)

    def test_unsafe_zip_is_refused(self):
        with helpers.TempDir() as t:
            z = t / "evil.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("runs/../../escape.txt", "x")
            with self.assertRaises(PackError):
                colab_train.setup(z, t / "local")
            from occupancy.pack import safe_extract
            with self.assertRaises(PackError):
                safe_extract(z, t / "out")
            self.assertFalse((t.parent / "escape.txt").exists())


class TestRoundTrip(unittest.TestCase):
    """colab_train.py setup + train, with python -m occupancy run in this process and pyrtlsdr unimportable."""

    @classmethod
    def setUpClass(cls):
        root = F["root"]
        with contextlib.redirect_stdout(io.StringIO()):
            cls.local_pack = colab_train.setup(F["pack"]["zip"], root / "colab_local")   # copy, unzip, verify
        cls.runs = root / "drive" / "runs"
        cls.calls = []

        def in_process(args, log_path):           # instead of a child process "python -m occupancy <args>"
            from occupancy.cli import main
            cls.calls.append([str(a) for a in args])
            return main([str(a) for a in args])

        with without_pyrtlsdr(), mock.patch.object(colab_train, "_run_occupancy", in_process), \
                contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.code = colab_train.main(["train", "--pack", str(cls.local_pack), "--out", str(cls.runs),
                                         "--folds", "0", "--threads", "2", "--device", "auto",
                                         "--allow-synthetic", "--max-attempts", "1"])

    def test_trained_through_the_colab_script_without_pyrtlsdr(self):
        self.assertEqual(self.code, 0)
        self.assertEqual(self.calls, [["train", "--pack", str(self.local_pack), "--out", str(self.runs),
                                       "--threads", "2", "--device", "auto", "--allow-synthetic", "--folds", "0"]])
        ident = provenance.read_json(self.runs / "pack_identity.json")
        self.assertEqual(ident["pack_id"], F["pack"]["pack_id"])
        for name in ("conv_ae", "fc_ae", "conv_vae"):
            run = self.runs / "folds" / "fold0" / "models" / name / "seed0"
            self.assertTrue(is_complete(run), name)
            s = provenance.read_json(run / "train_summary.json")
            self.assertEqual(s["environment"]["device_type"], "cpu")            # auto -> CPU here
            self.assertEqual(s["code_version"]["code_sha256"], provenance.code_version()["code_sha256"])
            self.assertEqual(s["pack"]["pack_id"], F["pack"]["pack_id"])
            self.assertTrue(s["synthetic_data"])
        with contextlib.redirect_stdout(io.StringIO()):
            counts = colab_train.status(self.local_pack, self.runs, False)["counts"]
        self.assertEqual(counts, {"complete": 3, "not started": 9})

    def test_child_process_plumbing(self):
        """The real child process (here only --help): working directory, PYTHONPATH, log, exit code."""
        log = F["root"] / "plumbing" / "logs" / "help.log"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = colab_train._run_occupancy(["--help"], log)
        self.assertEqual(code, 0)
        self.assertIn("usage: python -m occupancy", out.getvalue())
        self.assertIn("usage: python -m occupancy", log.read_text())

    def test_pack_training_equals_work_training(self):
        cfg, idx, work = F["cfg"], F["index"], F["work"]
        fold = F["folds"][0]
        tr, va, norm = fold_frames_from_work(idx, fold, cfg, work)
        laptop = F["root"] / "laptop_runs"
        train_runs(tr, va, norm, fold, cfg, laptop, models=["conv_ae", "fc_ae"], meta={"synthetic_data": True},
                   **QUIET)
        for name in ("conv_ae", "fc_ae"):
            a = torch.load(laptop / "folds" / "fold0" / "models" / name / "seed0" / "model.pt", weights_only=True)
            b = torch.load(self.runs / "folds" / "fold0" / "models" / name / "seed0" / "model.pt", weights_only=True)
            for k in a["state_dict"]:
                self.assertTrue(torch.equal(a["state_dict"][k], b["state_dict"][k]), f"{name} {k}")

    def test_unpack(self):
        cfg, idx, work, folds = F["cfg"], F["index"], F["work"], F["folds"]
        with contextlib.redirect_stdout(io.StringIO()):
            zpath = colab_train.zip_runs(self.runs, F["root"] / "drive" / "runs.zip")
        with zipfile.ZipFile(zpath) as z:
            self.assertFalse(any("checkpoint" in n for n in z.namelist()))
        with self.assertRaises(ImportRefused):                       # other training settings
            import_runs(self.runs, idx, folds, helpers.tiny_config(models={"fc_ae": {"hidden": 32, "latent": 8}}),
                        work, fold_ids=[0], **QUIET)
        other_code = {**provenance.code_version(), "code_sha256": "0" * 64}
        with mock.patch("occupancy.provenance.code_version", return_value=other_code):
            with self.assertRaisesRegex(ImportRefused, "allow-code-change"):
                import_runs(self.runs, idx, folds, cfg, work, fold_ids=[0], **QUIET)
        self.assertFalse((model_dir(work, 0, _spec("conv_ae"), 0)).exists())     # nothing copied when refused

        rep = import_runs(zpath, idx, folds, cfg, work, fold_ids=[0], **QUIET)           # from the zip
        self.assertTrue(rep["complete"], rep["counts"])
        self.assertEqual(rep["counts"], {"imported": 3})
        self.assertTrue(is_complete(model_dir(work, 0, _spec("conv_vae"), 0)))
        self.assertTrue((work / "colab_import.json").exists())
        rep = import_runs(self.runs, idx, folds, cfg, work, fold_ids=[0], **QUIET)       # again: nothing to do
        self.assertEqual(rep["counts"], {"already present": 3})
        rep = import_runs(self.runs, idx, folds, cfg, work, **QUIET)                     # all folds expected
        self.assertFalse(rep["complete"])
        self.assertEqual(rep["counts"], {"already present": 3, "not started": 9})
        res = calibrate_fold(idx, folds[0], cfg, work, models=["conv_ae", "fc_ae", "conv_vae"], **QUIET)
        for det in ("conv_ae", "fc_ae", "conv_vae_mse", "conv_vae_per"):
            self.assertIn("seed0", res["thresholds"][det])


def _spec(kind):
    from occupancy.models import main_specs
    return next(s for s in main_specs(F["cfg"]) if s.kind == kind)


class TestTimeEpoch(unittest.TestCase):
    def test_output_fields(self):
        pack = open_pack(F["pack"]["pack_dir"], verify=False, **QUIET)
        cfg = pack.config()
        tr, va, _, _ = pack.fold_frames(0, cfg)
        out = F["root"] / "timing" / "time_epoch.json"
        r = time_epochs(tr, va, cfg, torch.device("cpu"), 0, pack.fold_sizes(), out, max_batches=1,
                        units_per_hour=2.0, data="pack", synthetic=True, **QUIET)
        saved = provenance.read_json(out)
        self.assertEqual(set(saved["models"]), {"conv_ae", "fc_ae", "conv_vae"})
        for m in saved["models"].values():
            for key in ("epoch_s", "train_s", "val_s", "s_per_train_frame", "s_per_val_frame", "checkpoint_s",
                        "checkpoint_bytes", "parameters"):
                self.assertGreater(m[key], 0, key)
        est = saved["estimate"]
        self.assertEqual(est["epochs_per_run"], [1, 30, 60, 100])          # tiny config: max_epochs = 1
        from occupancy.models import ablation_specs
        n_folds, n_abl = 4, len(ablation_specs(cfg))
        self.assertEqual((est["runs_main"], est["runs_ablation"]),
                         (3 * n_folds * len(cfg.train.seeds), n_abl * n_folds * len(cfg.train.ablation_seeds)))
        self.assertEqual(est["runs_total"], est["runs_main"] + est["runs_ablation"])
        self.assertAlmostEqual(est["compute_units"]["60"], 2.0 * est["hours"]["60"])
        self.assertAlmostEqual(est["hours"]["60"], 2 * est["hours"]["30"])
        self.assertAlmostEqual(est["checkpoint_gb_written"]["60"], 2 * est["checkpoint_gb_written"]["30"])
        self.assertLess(est["hours_main_only"]["30"], est["hours"]["30"])
        self.assertTrue(saved["extrapolated"])
        self.assertEqual(saved["environment"]["device_type"], "cpu")
        self.assertIn("git", saved["run"])
        self.assertEqual(r["fold"], 0)
        self.assertFalse(list(out.parent.glob(".time_epoch*")))          # scratch checkpoints removed


class TestNoReceiverImports(unittest.TestCase):
    """Fresh imports of the training modules and the Colab script, with pyrtlsdr made unimportable."""

    def test_training_modules_do_not_import_pyrtlsdr(self):
        import importlib
        saved = {k: v for k, v in sys.modules.items() if _ours(k)}
        try:
            for k in saved:
                del sys.modules[k]
            sys.modules["rtlsdr"] = None                   # "import rtlsdr" now raises ImportError
            for name in ("occupancy.train", "occupancy.pack", "occupancy.unpack", "occupancy.budget",
                         "occupancy.device", "occupancy.cli", "colab_train"):
                importlib.import_module(name)
            loaded = [k for k, v in sys.modules.items() if "rtlsdr" in k and v is not None]
            self.assertEqual(loaded, [])
        finally:
            for k in [k for k in sys.modules if _ours(k)]:
                del sys.modules[k]
            sys.modules.update(saved)


class TestNotebook(unittest.TestCase):
    def test_valid_notebook(self):
        nb = json.loads((CODE / "colab" / "train_on_colab.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        kinds = [c["cell_type"] for c in nb["cells"]]
        self.assertGreater(kinds.count("markdown"), 5)
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in ("drive.mount", "nvidia-smi", "check-env", "setup", "time-epoch", " train ", "zip-runs"):
            self.assertIn(needle, src)
        self.assertNotIn("pip install torch", src)
        for c in nb["cells"]:
            self.assertEqual(c.get("outputs", []), [])                      # never executed
            if c["cell_type"] == "code":                     # valid Python once shell/magic lines are replaced
                lines = []
                for ln in "".join(c["source"]).splitlines():
                    body = ln.lstrip()
                    lines.append(ln[:len(ln) - len(body)] + "pass" if body.startswith(("!", "%")) else ln)
                ast.parse("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
