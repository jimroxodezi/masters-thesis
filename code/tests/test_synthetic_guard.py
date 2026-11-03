import contextlib
import io
import shutil
import unittest

import helpers
from occupancy import provenance
from occupancy.cli import main
from occupancy.config import Config
from occupancy.dataset import SyntheticDataError, index_captures
from occupancy.synthetic import generate_synthetic_dataset


class TestSyntheticGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = helpers.tiny_config()
        cls._tmp = helpers.TempDir()
        cls.root = cls._tmp.__enter__()
        cls.data = cls.root / "data_synthetic"
        cls.metas = generate_synthetic_dataset(cls.cfg, cls.data, n_sessions=2, frames_per_capture=1)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def test_sidecars_are_marked(self):
        self.assertEqual(len(self.metas), 16)
        for m in self.metas:
            self.assertTrue(m["synthetic"])
            self.assertIn("SYNTHETIC", m["WARNING"])
            self.assertEqual(m["radio"]["receiver_model"], "SYNTHETIC (no receiver)")
            side = provenance.read_json(self.data / "raw" / m["session_id"] / f"{m['capture_id']}.json")
            self.assertTrue(side["synthetic"])

    def test_refused_without_flag(self):
        with self.assertRaises(SyntheticDataError):
            index_captures(self.data, self.cfg, log=lambda *a: None)
        idx = index_captures(self.data, self.cfg, allow_synthetic=True, log=lambda *a: None)
        self.assertTrue(idx.synthetic)
        self.assertEqual(len(idx), 16)

    def test_never_mixed_with_real_data(self):
        mixed = self.root / "mixed"
        shutil.copytree(self.data, mixed)
        side = next((mixed / "raw" / "S01").glob("S01_B_g1_*.json"))
        meta = provenance.read_json(side)
        meta["synthetic"] = False                     # pretend one capture is real
        provenance.write_json(side, meta)
        with self.assertRaises(SyntheticDataError):
            index_captures(mixed, self.cfg, allow_synthetic=True, log=lambda *a: None)
        shutil.rmtree(mixed)

    def test_writer_refuses_real_data_directory(self):
        cfg = Config.from_dict({"paths": {"data_dir": str(self.root / "real")}})
        with self.assertRaises(PermissionError):
            generate_synthetic_dataset(cfg, self.root / "real", n_sessions=1, frames_per_capture=1)

    def test_cli_refuses_without_flag(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["preprocess", "--data-dir", str(self.data), "--work-dir", str(self.root / "w")])
        self.assertEqual(code, 2)
        self.assertIn("synthetic", err.getvalue().lower())
        self.assertFalse((self.root / "w").exists())

    def test_cli_refuses_real_work_directory(self):
        real_work = self.cfg.work_dir(False)
        existed = real_work.exists()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["preprocess", "--allow-synthetic", "--data-dir", str(self.data),
                         "--work-dir", str(real_work)])
        self.assertEqual(code, 2)
        self.assertEqual(real_work.exists(), existed)


if __name__ == "__main__":
    unittest.main()
