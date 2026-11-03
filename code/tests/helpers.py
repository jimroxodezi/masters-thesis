"""Shared test helpers: tiny configurations, synthetic data sets and light threading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from occupancy.config import Config

torch.set_num_threads(2)
os.environ.setdefault("OMP_NUM_THREADS", "2")


def tiny_config(**overrides) -> Config:
    """Defaults of the design, shrunk where only the run time depends on them."""
    d = {
        "injection": {"mixtures_per_level": 2},
        "train": {"max_epochs": 1, "seeds": [0], "ablation_seeds": [0], "num_threads": 2, "batch_size": 8},
        "eval": {"n_bootstrap": 30, "failure_samples": 2},
        "models": {"pca": {"max_components": 4}},
        "latency": {"warmup_frames": 1, "n_frames": 2, "throughput_batch": 2, "throughput_batches": 1,
                    "settle_s": 0.0,
                    "num_threads": 2},
    }
    for k, v in overrides.items():
        d.setdefault(k, {}).update(v)
    return Config.from_dict(d)


class TempDir:
    """Context manager for a temporary directory (removed afterwards)."""

    def __enter__(self) -> Path:
        self._td = tempfile.TemporaryDirectory(prefix="occupancy_test_")
        return Path(self._td.name)

    def __exit__(self, *exc):
        self._td.cleanup()
        return False
