"""
scoring.py
----------
Which frames each fold scores, and score files on disk.

Units of a fold
    validation: Class B at g1 of the validation sessions (threshold calibration)
    test:       every capture (A, B, C_centre, C_offset at g1 and g2) of the test
                sessions, plus their injection sets (per gain and source).
Score files
    work/folds/fold<k>/scores/<detector>/<seed-tag>_<part>.npz, one array per unit
    (seed-tag "seed<s>" for learned models, "det" for deterministic detectors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import provenance
from .config import CLASSES, GAIN_ROLES, Config
from .dataset import CaptureIndex, Fold, fold_records, val_records
from .detectors import (ED_ADAPTIVE, ED_FIXED, PCA, PSS, PCAModel, adaptive_name,
                        model_detector_names, score_model_units)
from .injection import injection_unit_id
from .preprocess import FeatureStore, Normalizer, features_dir, open_capture_store
from .train import fold_dir, load_model, model_dir, specs_and_seeds

DET_SEED = "det"


def seed_tag(seed) -> str:
    return DET_SEED if seed is None else f"seed{seed}"


def val_units(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path) -> list[FeatureStore]:
    return [open_capture_store(work_dir, r.capture_id) for r in val_records(index, fold, cfg)]


def test_units(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path) -> list[FeatureStore]:
    units = [open_capture_store(work_dir, r.capture_id)
             for r in fold_records(index, fold.test, CLASSES, GAIN_ROLES)]
    for sid in fold.test:
        for g in cfg.injection.gains:
            for src in cfg.injection.sources:
                d = features_dir(work_dir) / injection_unit_id(sid, g, src)
                if (d / "meta.json").exists():
                    units.append(FeatureStore(d))
    return units


def scores_path(work_dir: Path, k: int, detector: str, seed, part: str) -> Path:
    safe = detector.replace("@", "_at_")
    return fold_dir(work_dir, k) / "scores" / safe / f"{seed_tag(seed)}_{part}.npz"


def save_scores(path: Path, scores: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v, dtype=np.float64) for k, v in scores.items()})
    tmp.replace(path)
    provenance.write_json(path.with_suffix(".json"), meta)


def load_scores(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def scores_exist(path: Path, units) -> bool:
    if not Path(path).exists():
        return False
    with np.load(path) as z:
        return all(u.unit_id in z.files for u in units)


def learned_detectors(cfg: Config, models=None, ablation: bool = False, seeds=None):
    """[(detector_name, spec, seed)] for the learned models available in config."""
    out = []
    for spec, sds in specs_and_seeds(cfg, models, ablation, seeds):
        for name in model_detector_names(spec):
            for s in sds:
                out.append((name, spec, s))
    return out


def deterministic_detectors(cfg: Config, models=None) -> list[str]:
    names = [ED_FIXED, PSS] + [adaptive_name(p) for p in cfg.thresholds.pfa]
    if models is None or PCA in models:
        names.insert(0, PCA)
    return names


def score_units_learned(work_dir: Path, fold: Fold, spec, seed: int, units, norm: Normalizer,
                        cfg: Config, part: str, force: bool = False, log=print) -> list[str]:
    """Score units with one trained model; writes one file per detector name."""
    names = model_detector_names(spec)
    paths = {n: scores_path(work_dir, fold.k, n, seed, part) for n in names}
    if not force and all(scores_exist(p, units) for p in paths.values()):
        return names
    ckpt = model_dir(work_dir, fold.k, spec, seed) / "model.pt"
    if not ckpt.exists():
        log(f"  fold {fold.k}: {spec.name} seed {seed} not trained, skipped")
        return []
    model, _ = load_model(ckpt, cfg)
    per_name = score_model_units(model, spec, units, norm, cfg)
    for n in names:
        save_scores(paths[n], per_name[n], {"detector": n, "model": spec.to_dict(), "seed": seed,
                                            "fold": fold.k, "part": part, "checkpoint": str(ckpt),
                                            "units": [u.unit_id for u in units]})
    return names


def score_units_pca(work_dir: Path, fold: Fold, units, norm: Normalizer, cfg: Config, part: str,
                    force: bool = False, log=print) -> bool:
    path = scores_path(work_dir, fold.k, PCA, None, part)
    if not force and scores_exist(path, units):
        return True
    mpath = fold_dir(work_dir, fold.k) / "models" / PCA / "pca.npz"
    if not mpath.exists():
        log(f"  fold {fold.k}: PCA not fitted, skipped")
        return False
    pca = PCAModel.load(mpath)
    scores = pca.score_units(units, norm)
    save_scores(path, scores, {"detector": PCA, "fold": fold.k, "part": part, "d": pca.d,
                               "units": [u.unit_id for u in units]})
    return True


def score_units_simple(work_dir: Path, fold: Fold, units, part: str) -> None:
    """ed_fixed (frame energy T) and pss (stored correlator statistic)."""
    for name, field in ((ED_FIXED, "energy"), (PSS, "pss")):
        scores = {u.unit_id: np.asarray(u[field], dtype=np.float64) for u in units}
        save_scores(scores_path(work_dir, fold.k, name, None, part), scores,
                    {"detector": name, "fold": fold.k, "part": part, "units": list(scores)})


def load_normalizer(work_dir: Path, k: int) -> Normalizer:
    p = fold_dir(work_dir, k) / "normalizer.json"
    if not p.exists():
        raise FileNotFoundError(f"fold {k} has no normaliser (run: python -m occupancy train)")
    return Normalizer.load(p)


__all__ = ["val_units", "test_units", "scores_path", "save_scores", "load_scores", "learned_detectors",
           "deterministic_detectors", "score_units_learned", "score_units_pca", "score_units_simple",
           "load_normalizer", "ED_ADAPTIVE", "seed_tag"]
