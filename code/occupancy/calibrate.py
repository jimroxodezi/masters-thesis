"""
calibrate.py
------------
Thresholds at fixed false-alarm rates (design section 8).

For every detector, fold and seed: threshold = empirical (1 - P_FA) quantile of
its scores on the fold's idle validation frames (Class B, g1), for
P_FA in {0.01, 0.05}; a frame is declared occupied if score > threshold.
The quantile is the order statistic s_(ceil((1-P_FA) n)) ("inverted_cdf"), so
at most a fraction P_FA of the calibration scores exceed the threshold.

ed_adaptive: kappa on T/sigma2_hat, from the validation stream (each validation
session's Class B capture processed as one stream, initialised to the median T
of all validation frames) run at that same kappa (see detectors.calibrate_kappa).

ed_fixed also gets the theoretical threshold
    lambda = sigma2_hat/(2N) F^-1_{chi2(2N)}(1 - P_FA),  sigma2_hat = median T (validation),
reported next to the empirical one and never used for the main decisions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import provenance
from .config import Config
from .dataset import CaptureIndex, Fold
from .detectors import (ED_ADAPTIVE, ED_FIXED, adaptive_name, calibrate_kappa,
                        ed_theoretical_threshold, quantile_threshold)
from .scoring import (deterministic_detectors, learned_detectors, load_normalizer, load_scores,
                      save_scores, score_units_learned, score_units_pca, score_units_simple,
                      scores_path, seed_tag, val_units)
from .train import fold_dir


def decide(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Occupied (True) if score > threshold (design section 8)."""
    return np.asarray(scores) > threshold


def pfa_key(p: float) -> str:
    return f"{p:g}"


def thresholds_path(work_dir: Path, k: int) -> Path:
    return fold_dir(work_dir, k) / "thresholds.json"


def calibrate_fold(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path, models=None,
                   ablation: bool = False, seeds=None, force: bool = False, log=print) -> dict:
    norm = load_normalizer(work_dir, fold.k)
    units = val_units(index, fold, cfg, work_dir)
    n_val = sum(u.n for u in units)
    method = cfg.thresholds.quantile_method
    log(f"fold {fold.k}: calibrating on {n_val} validation frames from {fold.val}")

    # 1. validation scores
    done = set()
    for name, spec, seed in learned_detectors(cfg, models, ablation, seeds):
        if (spec.name, seed) in done:
            continue
        done.add((spec.name, seed))
        score_units_learned(work_dir, fold, spec, seed, units, norm, cfg, "val", force, log)
    det = deterministic_detectors(cfg, models)
    if "pca" in det:
        score_units_pca(work_dir, fold, units, norm, cfg, "val", force, log)
    score_units_simple(work_dir, fold, units, "val")

    # 2. adaptive energy detector: kappa per P_FA from the validation stream
    streams = [np.asarray(u.energy, dtype=np.float64) for u in units]
    init = float(np.median(np.concatenate(streams)))
    adaptive = {}
    for p in cfg.thresholds.pfa:
        res = calibrate_kappa(streams, init, cfg.models.ed.adaptive_window, p, method,
                              cfg.models.ed.kappa_max_iter)
        stats, pos = {}, 0
        for u in units:
            stats[u.unit_id] = res["val_stats"][pos:pos + u.n]
            pos += u.n
        save_scores(scores_path(work_dir, fold.k, adaptive_name(p), None, "val"), stats,
                    {"detector": adaptive_name(p), "fold": fold.k, "part": "val", "kappa": res["kappa"],
                     "init_sigma2": init, "units": list(stats)})
        adaptive[pfa_key(p)] = {k: v for k, v in res.items() if k != "val_stats"}
        adaptive[pfa_key(p)]["init_sigma2"] = init

    # 3. thresholds from the stored validation scores
    table = {}
    entries = [(n, s) for n, _, s in learned_detectors(cfg, models, ablation, seeds)]
    entries += [(n, None) for n in det]
    for name, seed in entries:
        path = scores_path(work_dir, fold.k, name, seed, "val")
        if not path.exists():
            continue
        sc = load_scores(path)
        allv = np.concatenate([sc[u.unit_id] for u in units if u.unit_id in sc])
        per_p = {}
        pfas = cfg.thresholds.pfa
        if name.startswith(ED_ADAPTIVE + "@"):
            pfas = [p for p in pfas if adaptive_name(p) == name]
        for p in pfas:
            thr = (adaptive[pfa_key(p)]["kappa"] if name.startswith(ED_ADAPTIVE + "@")
                   else quantile_threshold(allv, p, method))
            per_p[pfa_key(p)] = {"threshold": thr, "realized_pfa_val": float(np.mean(allv > thr)),
                                 "n_val": int(allv.size)}
        table.setdefault(name, {})[seed_tag(seed)] = per_p

    # 4. theoretical ED threshold (reported only)
    energies = np.concatenate(streams)
    sigma2 = float(np.median(energies))
    N = cfg.preprocess.frame_len
    theory = {}
    for p in cfg.thresholds.pfa:
        lam = ed_theoretical_threshold(sigma2, N, p)
        theory[pfa_key(p)] = {"lambda": lam, "sigma2_hat": sigma2,
                              "empirical_threshold": table[ED_FIXED]["det"][pfa_key(p)]["threshold"],
                              "realized_pfa_val_theory": float(np.mean(energies > lam)),
                              "realized_pfa_val_empirical":
                                  table[ED_FIXED]["det"][pfa_key(p)]["realized_pfa_val"]}
    out = {"fold": fold.to_dict(), "n_val_frames": n_val, "quantile_method": method,
           "thresholds": table, "ed_adaptive": adaptive, "ed_fixed_theoretical": theory,
           "validation_units": [u.unit_id for u in units], "synthetic_data": index.synthetic,
           "run": provenance.run_record(cfg, synthetic=index.synthetic)}
    provenance.write_json(thresholds_path(work_dir, fold.k), out)
    return out


def load_thresholds(work_dir: Path, k: int) -> dict:
    p = thresholds_path(work_dir, k)
    if not p.exists():
        raise FileNotFoundError(f"fold {k} is not calibrated (run: python -m occupancy calibrate)")
    return provenance.read_json(p)


def write_threshold_table(results: list[dict], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "detector", "seed", "pfa_target", "threshold", "realized_pfa_val", "n_val"])
        for r in results:
            k = r["fold"]["fold"]
            for name, by_seed in r["thresholds"].items():
                for sd, by_p in by_seed.items():
                    for p, v in by_p.items():
                        w.writerow([k, name, sd, p, v["threshold"], v["realized_pfa_val"], v["n_val"]])
            for p, v in r["ed_fixed_theoretical"].items():
                w.writerow([k, "ed_fixed_theoretical", "det", p, v["lambda"],
                            v["realized_pfa_val_theory"], r["n_val_frames"]])
