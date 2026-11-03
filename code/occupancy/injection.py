"""
injection.py
------------
Controlled-SNR injection set (design section 6).

For one test session, one gain and one source class (C_centre or C_offset,
kept separate), with all powers measured on DC-removed frames:
    P_n(b)  = mean |y|^2 of idle frame b (Class B, same session and gain)
    P_n,C   = median of P_n over the session's Class B frames at that gain
    P_s(c)  = P(c) - P_n,C for a Class C frame c
    mixture x = alpha c + b,  alpha^2 = SNR_target P_n(b) / P_s(c)
    SNR_eff = alpha^2 P_s(c) / (P_n(b) + alpha^2 P_n,C)   (stored; used for strata)
    c is admissible at a level if P_s(c)/P_n,C >= SNR_target + 10 dB, which
    bounds SNR_target - SNR_eff by 10 log10(1.1) = 0.41 dB.
Grid: -20 ... +10 dB in 2 dB steps (16 levels), 500 mixtures per level.
The c and b frames come from different captures of the same session, so from
different times. Every mixture is labelled occupied.

Draws: c frames uniformly from the admissible set, b frames uniformly from all
B frames, without replacement while possible; when fewer frames exist than
mixtures are needed, every frame is used equally often (up to one) before any
frame is reused. The draw is seeded from (seed, session, gain, source, level).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import provenance
from .config import Config
from .dataset import CaptureIndex, dc_remove, read_frames_at
from .preprocess import (FeatureStore, feature_fingerprint, features_dir, frame_energy,
                         open_capture_store, process_frames, store_is_current, write_store)
from .pss import PSSCorrelator, correlator_from_config
from .synthetic import stable_seed


def snr_grid_db(cfg: Config) -> np.ndarray:
    i = cfg.injection
    n = int(round((i.snr_max_db - i.snr_min_db) / i.snr_step_db)) + 1
    return np.round(i.snr_min_db + i.snr_step_db * np.arange(n), 6)


def db(x):
    return 10.0 * np.log10(x)


def mixture_alpha(snr_target_lin, pn_b, ps_c):
    return np.sqrt(np.asarray(snr_target_lin) * np.asarray(pn_b) / np.asarray(ps_c))


def snr_eff_lin(alpha, ps_c, pn_b, pn_C):
    a2 = np.asarray(alpha) ** 2
    return a2 * np.asarray(ps_c) / (np.asarray(pn_b) + a2 * pn_C)


def admissible_mask(ps_c: np.ndarray, pn_C: float, target_db: float, margin_db: float) -> np.ndarray:
    ps_c = np.asarray(ps_c, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        nat_db = np.where(ps_c > 0, db(np.maximum(ps_c, 1e-300) / pn_C), -np.inf)
    return nat_db >= target_db + margin_db


def balanced_draw(rng: np.random.Generator, pool: np.ndarray, m: int) -> np.ndarray:
    """m draws from pool: without replacement if possible, otherwise equal reuse."""
    pool = np.asarray(pool)
    if pool.size == 0 or m == 0:
        return np.zeros(0, dtype=np.int64)
    if pool.size >= m:
        return rng.choice(pool, size=m, replace=False)
    reps = int(np.ceil(m / pool.size))
    out = np.concatenate([rng.permutation(pool) for _ in range(reps)])[:m]
    return rng.permutation(out)


def plan_mixtures(T_b: np.ndarray, T_c: np.ndarray, cfg: Config, key=()) -> dict:
    """Choose (c, b, alpha) for every level; returns arrays sorted by level."""
    inj = cfg.injection
    T_b = np.asarray(T_b, dtype=np.float64)
    T_c = np.asarray(T_c, dtype=np.float64)
    pn_C = float(np.median(T_b))
    ps = T_c - pn_C
    with np.errstate(divide="ignore", invalid="ignore"):
        nat_db = np.where(ps > 0, db(np.maximum(ps, 1e-300) / pn_C), -np.inf)
    cols = {k: [] for k in ("level", "c_frame", "b_frame", "alpha", "snr_eff_db", "snr_nat_db")}
    per_level = {}
    for level in snr_grid_db(cfg):
        adm = np.flatnonzero(admissible_mask(ps, pn_C, level, inj.admissible_margin_db))
        m = inj.mixtures_per_level if adm.size else 0
        rng = np.random.default_rng(stable_seed(inj.seed, *key, f"{level:+.1f}"))
        c_idx = balanced_draw(rng, adm, m)
        b_idx = balanced_draw(rng, np.arange(T_b.size), m)
        lin = 10.0 ** (level / 10.0)
        alpha = mixture_alpha(lin, T_b[b_idx], ps[c_idx]) if m else np.zeros(0)
        eff = snr_eff_lin(alpha, ps[c_idx], T_b[b_idx], pn_C) if m else np.zeros(0)
        cols["level"].append(np.full(m, level))
        cols["c_frame"].append(c_idx)
        cols["b_frame"].append(b_idx)
        cols["alpha"].append(alpha)
        cols["snr_eff_db"].append(db(eff) if m else np.zeros(0))
        cols["snr_nat_db"].append(nat_db[c_idx])
        per_level[f"{level:+.1f}"] = {"n_mixtures": int(m), "n_admissible_c": int(adm.size),
                                      "n_unique_c": int(np.unique(c_idx).size),
                                      "n_unique_b": int(np.unique(b_idx).size)}
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    out["c_frame"] = out["c_frame"].astype(np.int64)
    out["b_frame"] = out["b_frame"].astype(np.int64)
    out["pn_C"] = pn_C
    out["per_level"] = per_level
    return out


def injection_unit_id(session_id: str, gain_role: str, source: str) -> str:
    return f"inj_{session_id}_{gain_role}_{source}"


def build_injection_set(index: CaptureIndex, session_id: str, gain_role: str, source: str,
                        cfg: Config, work_dir: Path, correlator: PSSCorrelator | None = None,
                        force: bool = False, log=print) -> FeatureStore | None:
    b_rec = index.get(session_id, "B", gain_role)
    c_rec = index.get(session_id, source, gain_role)
    if b_rec is None or c_rec is None:
        log(f"no injection set {session_id}/{gain_role}/{source}: "
            f"{'B' if b_rec is None else source} capture missing")
        return None
    uid = injection_unit_id(session_id, gain_role, source)
    out = features_dir(work_dir) / uid
    b_store = open_capture_store(work_dir, b_rec.capture_id)
    c_store = open_capture_store(work_dir, c_rec.capture_id)
    fp = feature_fingerprint(cfg)
    sig = {"b": b_store.meta["source"], "c": c_store.meta["source"],
           "injection": cfg.to_dict()["injection"]}
    if not force and store_is_current(out, fp, sig):
        return FeatureStore(out)
    plan = plan_mixtures(b_store.energy, c_store.energy, cfg, key=(session_id, gain_role, source))
    n = plan["level"].size
    if n == 0:
        log(f"no admissible mixtures for {uid}")
        return None
    N = cfg.preprocess.frame_len
    correlator = correlator or correlator_from_config(cfg)
    chunk = cfg.preprocess.chunk_frames

    def chunks():
        for s in range(0, n, chunk):
            c = read_frames_at(c_rec, N, plan["c_frame"][s:s + chunk])
            b = read_frames_at(b_rec, N, plan["b_frame"][s:s + chunk])
            x = plan["alpha"][s:s + chunk, None].astype(np.float32) * c + b
            yield process_frames(dc_remove(x).astype(np.complex64), cfg, correlator)

    meta = {"unit_id": uid, "kind": "injection", "session_id": session_id,
            "position_id": b_rec.position_id, "class": f"INJ_{source}", "source": sig,
            "source_class": source, "gain_role": gain_role, "gain_db": b_rec.gain_db, "label": 1,
            "synthetic": b_rec.synthetic or c_rec.synthetic, "b_capture_id": b_rec.capture_id,
            "c_capture_id": c_rec.capture_id, "pn_C": plan["pn_C"], "levels": plan["per_level"],
            "fingerprint": fp, "software_version": provenance.software_version(),
            "rule": "x = alpha c + b, alpha^2 = SNR P_n(b)/P_s(c) (design section 6)"}
    extra = {k: plan[k] for k in ("level", "c_frame", "b_frame", "alpha", "snr_eff_db", "snr_nat_db")}
    store = write_store(out, n, cfg, chunks(), meta, extra)
    # check the realised mixture power against the plan (guards the arithmetic)
    T_x = store.energy
    expected = plan["alpha"] ** 2 * c_store.energy[plan["c_frame"]] + b_store.energy[plan["b_frame"]]
    rel = np.abs(T_x - expected) / expected
    if np.median(rel) > 0.2:
        log(f"WARNING {uid}: mixture powers deviate from alpha^2 P(c) + P(b) (median {np.median(rel):.2f})")
    log(f"injection set {uid}: {n} mixtures over {len(plan['per_level'])} levels")
    return store


def build_all_injection_sets(index: CaptureIndex, cfg: Config, work_dir: Path, force: bool = False,
                             log=print) -> dict:
    correlator = correlator_from_config(cfg)
    out = {}
    for sid in index.sessions():
        for g in cfg.injection.gains:
            for src in cfg.injection.sources:
                st = build_injection_set(index, sid, g, src, cfg, work_dir, correlator, force, log)
                if st is not None:
                    out[st.unit_id] = st
    return out


def mixture_snr_check(frames_c: np.ndarray, frames_b: np.ndarray, pn_C: float, level_db: float):
    """Diagnostic: SNR_eff of mixtures built from given frames (used in tests)."""
    T_c, T_b = frame_energy(frames_c), frame_energy(frames_b)
    alpha = mixture_alpha(10 ** (level_db / 10), T_b, T_c - pn_C)
    return db(snr_eff_lin(alpha, T_c - pn_C, T_b, pn_C)), alpha
