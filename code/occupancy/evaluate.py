"""
evaluate.py
-----------
Test scoring, metrics, pooling, seeds, session bootstrap, H1 and H2
(design section 9).

Pooling. Every session is tested once, by the models of the fold that held it
out; test predictions (decisions at each fold's own thresholds) are pooled over
the 4 folds. Seed s of a learned model means the fold models trained with seed
s; results are given as mean +- SD over seeds (sample SD). Deterministic
detectors (pca, ed_fixed, ed_adaptive, pss) have one "seed".

Threshold-free metrics (ROC, AUC, precision-recall) on pooled scores need scores
on a common scale across folds. Each fold's scores are mapped through the
empirical CDF of that fold's validation scores (mid-rank for ties),
u = F_val(score), which is monotone within a fold (so the fold's own ROC is
unchanged) and is the scale on which the thresholds are set (u > 1 - P_FA).

Bootstrap. 1000 resamples of sessions with replacement; each metric is
recomputed from per-session counts weighted by how often the session was drawn
(AUC from per-session-pair Mann-Whitney counts), giving 95 % percentile
intervals; seed-averaged metrics are averaged over seeds inside each replicate.
Differences between detectors use the same resampled sessions (paired).

Eval sets (each at g1 and g2; negatives are the test Class B frames):
    natural            positives: C_centre and C_offset
    natural_centre     positives: C_centre
    natural_offset     positives: C_offset
    inj_<src>_<L>dB    positives: injection mixtures of source src (all, centre,
                       offset) in SNR stratum L (the grid level nearest SNR_eff)
    inj_<src>_low      strata <= -4 dB (H1); inj_<src>_alllevels: every stratum
    receiver_A         Class A only (receiver-only false-alarm check)
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from . import provenance
from .calibrate import load_thresholds, pfa_key
from .config import Config
from .dataset import CaptureIndex, Fold
from .detectors import ED_ADAPTIVE, ED_FIXED, adaptive_name, adaptive_stream
from .injection import snr_grid_db
from .preprocess import FeatureStore, open_capture_store
from .scoring import (deterministic_detectors, learned_detectors, load_normalizer, load_scores,
                      save_scores, score_units_learned, score_units_pca, score_units_simple,
                      scores_path, seed_tag, test_units)

CLS = ("A", "B", "C_centre", "C_offset", "INJ_C_centre", "INJ_C_offset")
CLS_CODE = {c: i for i, c in enumerate(CLS)}
GAINS = ("g1", "g2")
SOURCES = {"all": ("INJ_C_centre", "INJ_C_offset"), "centre": ("INJ_C_centre",),
           "offset": ("INJ_C_offset",)}


# ── 1. scoring of the test units ──────────────────────────────────────────────

def adaptive_test_scores(fold: Fold, cfg: Config, work_dir: Path, units: list[FeatureStore],
                         thr: dict) -> None:
    """ed_adaptive statistic on the test units, one file per target P_FA.

    Each test session and gain is one stream in capture order B, C_centre,
    C_offset (A excluded), initialised to the validation median. A mixture
    x = alpha c + b is scored against the noise estimate the stream held just
    before its idle frame b: r = T(x) / sigma2_hat(t_b). Class A captures are
    scored as separate streams (receiver-only false-alarm check).
    """
    M = cfg.models.ed.adaptive_window
    by_key, inj = {}, {}
    for u in units:
        m = u.meta
        if m["kind"] == "capture":
            by_key[(m["session_id"], m["gain_role"], m["class"])] = u
        else:
            inj.setdefault((m["session_id"], m["gain_role"]), []).append(u)
    for p in cfg.thresholds.pfa:
        a = thr["ed_adaptive"][pfa_key(p)]
        kappa, init = a["kappa"], a["init_sigma2"]
        scores = {}
        for sid in fold.test:
            for g in GAINS:
                stream = [by_key[(sid, g, c)] for c in ("B", "C_centre", "C_offset") if (sid, g, c) in by_key]
                if stream:
                    T = np.concatenate([np.asarray(u.energy) for u in stream])
                    r, s2, _ = adaptive_stream(T, kappa, init, M)
                    pos = 0
                    for u in stream:
                        scores[u.unit_id] = r[pos:pos + u.n]
                        pos += u.n
                    if stream[0].meta["class"] == "B":
                        s2_b = s2[:stream[0].n]
                        for iu in inj.get((sid, g), []):
                            scores[iu.unit_id] = np.asarray(iu.energy) / s2_b[iu["b_frame"]]
                a_unit = by_key.get((sid, g, "A"))
                if a_unit is not None:
                    scores[a_unit.unit_id] = adaptive_stream(np.asarray(a_unit.energy), kappa, init, M)[0]
        save_scores(scores_path(work_dir, fold.k, adaptive_name(p), None, "test"), scores,
                    {"detector": adaptive_name(p), "fold": fold.k, "part": "test", "kappa": kappa,
                     "init_sigma2": init, "units": list(scores)})


def score_test_fold(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path, models=None,
                    ablation: bool = False, seeds=None, force: bool = False, log=print) -> None:
    units = test_units(index, fold, cfg, work_dir)
    norm = load_normalizer(work_dir, fold.k)
    thr = load_thresholds(work_dir, fold.k)
    log(f"fold {fold.k}: scoring {sum(u.n for u in units)} test frames in {len(units)} units")
    done = set()
    for _, spec, seed in learned_detectors(cfg, models, ablation, seeds):
        if (spec.name, seed) not in done:
            done.add((spec.name, seed))
            score_units_learned(work_dir, fold, spec, seed, units, norm, cfg, "test", force, log)
    if "pca" in deterministic_detectors(cfg, models):
        score_units_pca(work_dir, fold, units, norm, cfg, "test", force, log)
    score_units_simple(work_dir, fold, units, "test")
    adaptive_test_scores(fold, cfg, work_dir, units, thr)


# ── 2. pooled frame table ─────────────────────────────────────────────────────

@dataclass
class Pooled:
    sessions: list
    positions: list                     # position of each session
    units: list                         # unit info dicts
    grid: np.ndarray                    # SNR grid (dB)
    times_of_day: list = None           # day / night / off-schedule of each session
    fold: np.ndarray = None
    sess: np.ndarray = None
    unit: np.ndarray = None
    frame: np.ndarray = None
    cls: np.ndarray = None
    gain: np.ndarray = None
    label: np.ndarray = None
    stratum: np.ndarray = None          # grid index (0..L-1) for mixtures, L for captures
    snr_eff: np.ndarray = None
    raw: dict = field(default_factory=dict)        # (det, seed) -> scores
    calib: dict = field(default_factory=dict)      # (det, seed) -> F_val(score)
    thr: dict = field(default_factory=dict)        # (det, seed, pfa) -> (n_folds,) thresholds
    n_folds: int = 0
    stratum_mismatch: int = 0

    @property
    def S(self) -> int:
        return len(self.sessions)

    @property
    def L(self) -> int:
        return len(self.grid)


def _ecdf(val_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(val_sorted, x, side="left")
    hi = np.searchsorted(val_sorted, x, side="right")
    return ((lo + hi) / (2.0 * val_sorted.size)).astype(np.float32)


def detector_entries(cfg: Config, models=None, ablation: bool = False, seeds=None) -> list:
    out = [(n, s) for n, _, s in learned_detectors(cfg, models, ablation, seeds)]
    out += [(n, None) for n in deterministic_detectors(cfg, models)]
    return out


def build_pooled(index: CaptureIndex, folds: list[Fold], cfg: Config, work_dir: Path,
                 entries: list, log=print) -> Pooled:
    grid = snr_grid_db(cfg)
    sessions = index.sessions()
    s_idx = {s: i for i, s in enumerate(sessions)}
    P = Pooled(sessions=sessions, positions=[index.position_of(s) for s in sessions], units=[],
               grid=grid, times_of_day=[index.time_of_day_of(s, cfg) for s in sessions], n_folds=len(folds))
    cols = {k: [] for k in ("fold", "sess", "unit", "frame", "cls", "gain", "label", "stratum", "snr_eff")}
    fold_units = []
    for fold in folds:
        units = test_units(index, fold, cfg, work_dir)
        fold_units.append(units)
        for u in units:
            m = u.meta
            ui = len(P.units)
            P.units.append({"unit_id": u.unit_id, "kind": m["kind"], "class": m["class"],
                            "gain_role": m["gain_role"], "session_id": m["session_id"], "fold": fold.k,
                            "utc_start": m.get("utc_start"), "b_capture_id": m.get("b_capture_id"),
                            "c_capture_id": m.get("c_capture_id"), "n": u.n})
            n = u.n
            cols["fold"].append(np.full(n, fold.k))
            cols["sess"].append(np.full(n, s_idx[m["session_id"]]))
            cols["unit"].append(np.full(n, ui))
            cols["frame"].append(np.arange(n))
            cols["cls"].append(np.full(n, CLS_CODE[m["class"]]))
            cols["gain"].append(np.full(n, GAINS.index(m["gain_role"])))
            cols["label"].append(np.full(n, int(m["label"])))
            if m["kind"] == "injection":
                eff = np.asarray(u["snr_eff_db"], dtype=np.float64)
                st = np.argmin(np.abs(eff[:, None] - grid[None, :]), axis=1)
                P.stratum_mismatch += int(np.sum(grid[st] != np.asarray(u["level"])))
                cols["stratum"].append(st)
                cols["snr_eff"].append(eff)
            else:
                cols["stratum"].append(np.full(n, len(grid)))
                cols["snr_eff"].append(np.full(n, np.nan))
    for k, v in cols.items():
        setattr(P, k, np.concatenate(v) if v else np.zeros(0))
    for k in ("fold", "sess", "unit", "frame", "cls", "gain", "label", "stratum"):
        setattr(P, k, getattr(P, k).astype(np.int64))

    n_k = max(f.k for f in folds) + 1
    for det, seed in entries:
        raws, cals, thr_by_p = [], [], {p: np.full(n_k, np.nan) for p in cfg.thresholds.pfa}
        ok = True
        for fold, units in zip(folds, fold_units):
            tp = scores_path(work_dir, fold.k, det, seed, "test")
            vp = scores_path(work_dir, fold.k, det, seed, "val")
            if not (tp.exists() and vp.exists()):
                ok = False
                break
            tsc, vsc = load_scores(tp), load_scores(vp)
            if any(u.unit_id not in tsc for u in units):
                ok = False
                break
            v = np.sort(np.concatenate(list(vsc.values())))
            x = np.concatenate([tsc[u.unit_id] for u in units]) if units else np.zeros(0)
            raws.append(x)
            cals.append(_ecdf(v, x))
            th = load_thresholds(work_dir, fold.k)["thresholds"].get(det, {}).get(seed_tag(seed), {})
            for p in cfg.thresholds.pfa:
                if pfa_key(p) in th:
                    thr_by_p[p][fold.k] = th[pfa_key(p)]["threshold"]
        if not ok:
            log(f"note: {det} {seed_tag(seed)} is not scored in every fold; left out")
            continue
        P.raw[(det, seed)] = np.concatenate(raws)
        P.calib[(det, seed)] = np.concatenate(cals)
        for p, arr in thr_by_p.items():
            if np.all(np.isfinite(arr[[f.k for f in folds]])):
                P.thr[(det, seed, p)] = arr
    if P.stratum_mismatch:
        log(f"note: {P.stratum_mismatch} mixtures fall in a different stratum than their target level")
    return P


# ── 3. counts, metrics, bootstrap ─────────────────────────────────────────────

@dataclass(frozen=True)
class EvalSet:
    name: str
    gain: int
    neg: tuple                  # class codes of negatives (captures)
    pos: tuple                  # class codes of positives
    strata: tuple | None        # grid indices for injection positives (None: captures)
    source: str = ""
    level: float | None = None


def eval_sets(P: Pooled, cfg: Config) -> list[EvalSet]:
    B, A = (CLS_CODE["B"],), (CLS_CODE["A"],)
    low = tuple(int(i) for i in np.flatnonzero(P.grid <= cfg.injection.low_snr_max_db))
    out = []
    for g in range(len(GAINS)):
        out += [EvalSet("natural", g, B, (CLS_CODE["C_centre"], CLS_CODE["C_offset"]), None),
                EvalSet("natural_centre", g, B, (CLS_CODE["C_centre"],), None),
                EvalSet("natural_offset", g, B, (CLS_CODE["C_offset"],), None),
                EvalSet("receiver_A", g, A, (), None)]
        for src, classes in SOURCES.items():
            codes = tuple(CLS_CODE[c] for c in classes)
            out.append(EvalSet(f"inj_{src}_low", g, B, codes, low, src))
            out.append(EvalSet(f"inj_{src}_alllevels", g, B, codes, tuple(range(P.L)), src))
            for li, lev in enumerate(P.grid):
                out.append(EvalSet(f"inj_{src}_{lev:+g}dB", g, B, codes, (li,), src, float(lev)))
    return out


def set_mask(P: Pooled, es: EvalSet) -> np.ndarray:
    """Frames of an eval set: negatives (captures) plus positives (captures or strata)."""
    neg = np.isin(P.cls, es.neg) & (P.stratum == P.L)
    if es.pos:
        pos_strata = np.isin(P.stratum, es.strata) if es.strata is not None else (P.stratum == P.L)
        neg = neg | (np.isin(P.cls, es.pos) & pos_strata)
    return (P.gain == es.gain) & neg


def set_weights(P: Pooled, sets: list) -> tuple[np.ndarray, np.ndarray]:
    """(n_sets, gains, classes, strata+1) 0/1 weights selecting each set's positives and negatives."""
    shape = (len(sets), len(GAINS), len(CLS), P.L + 1)
    Wp, Wn = np.zeros(shape), np.zeros(shape)
    for i, es in enumerate(sets):
        Wn[i, es.gain, list(es.neg), P.L] = 1.0
        if es.pos:
            strata = [P.L] if es.strata is None else list(es.strata)
            for c in es.pos:
                Wp[i, es.gain, c, strata] = 1.0
    return Wp, Wn


class CellCounts:
    """Frame counts per cell (gain, class, stratum, session) for fast set sums."""

    def __init__(self, P: Pooled):
        self.shape = (len(GAINS), len(CLS), P.L + 1, P.S)
        self.cell = np.ravel_multi_index((P.gain, P.cls, P.stratum, P.sess), self.shape)
        self.n = np.bincount(self.cell, minlength=int(np.prod(self.shape))).reshape(self.shape)

    def decisions(self, dec: np.ndarray) -> np.ndarray:
        return np.bincount(self.cell[dec], minlength=int(np.prod(self.shape))).reshape(self.shape)

    def set_counts(self, D: np.ndarray, es: EvalSet, L: int) -> dict:
        g = es.gain
        nN = self.n[g, list(es.neg), L, :].sum(axis=0)
        fp = D[g, list(es.neg), L, :].sum(axis=0)
        if es.pos:
            strata = [L] if es.strata is None else list(es.strata)
            sub_n = self.n[g][np.ix_(list(es.pos), strata)]
            sub_d = D[g][np.ix_(list(es.pos), strata)]
            nP, tp = sub_n.sum(axis=(0, 1)), sub_d.sum(axis=(0, 1))
        else:
            nP = tp = np.zeros(self.shape[3], dtype=np.int64)
        return {"tp": tp, "fn": nP - tp, "fp": fp, "tn": nN - fp}


def _ratio(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


COUNT_METRICS = ("pd", "pfa", "precision", "recall", "f1")


def metric_value(metric: str, tp, fn, fp, tn):
    """One count-based metric; works on scalars or arrays of totals."""
    if metric in ("pd", "recall"):
        return _ratio(tp, tp + fn)
    if metric == "pfa":
        return _ratio(fp, fp + tn)
    if metric == "precision":
        return _ratio(tp, tp + fp)
    if metric == "f1":
        return _ratio(2 * tp, 2 * tp + fp + fn)
    raise ValueError(metric)


def metrics_from(tp, fn, fp, tn) -> dict:
    """All count-based metrics; works on scalars or arrays of totals."""
    return {m: metric_value(m, tp, fn, fp, tn) for m in COUNT_METRICS}


def bootstrap_multiplicities(S: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, S) counts of how often each session is drawn (S draws with replacement)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, S, size=(n_boot, S))
    return np.stack([np.bincount(d, minlength=S) for d in draws]).astype(np.float64)


def percentile_ci(reps: np.ndarray, level: float = 0.95) -> tuple:
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(reps, [a, 100 - a])
    return (float(lo), float(hi))


def ci_columns(reps: np.ndarray, level: float) -> np.ndarray:
    """Percentile intervals of every column of (B, n): (2, n); NaN replicates are ignored."""
    reps = np.asarray(reps, dtype=np.float64)
    a = (1 - level) / 2 * 100
    out = np.full((2, reps.shape[1]), np.nan)
    finite = np.isfinite(reps)
    full = finite.all(axis=0)
    if full.any():
        out[:, full] = np.percentile(reps[:, full], [a, 100 - a], axis=0)
    for j in np.flatnonzero(~full & finite.any(axis=0)):
        out[:, j] = np.percentile(reps[finite[:, j], j], [a, 100 - a])
    return out


def auc_pair_matrix(u: np.ndarray, label: np.ndarray, sess: np.ndarray, mask: np.ndarray, S: int):
    """Mann-Whitney counts per (positive session, negative session) pair.

    U[a, b] = sum over positives i in a, negatives j in b of 1[u_i > u_j] + 0.5 1[u_i = u_j].
    """
    pos = mask & (label == 1)
    neg = mask & (label == 0)
    nP = np.bincount(sess[pos], minlength=S).astype(np.float64)
    nN = np.bincount(sess[neg], minlength=S).astype(np.float64)
    U = np.zeros((S, S))
    up, sp = u[pos], sess[pos]
    for b in range(S):
        vb = np.sort(u[neg & (sess == b)])
        if vb.size == 0 or up.size == 0:
            continue
        lo = np.searchsorted(vb, up, side="left")
        hi = np.searchsorted(vb, up, side="right")
        U[:, b] = np.bincount(sp, weights=lo + 0.5 * (hi - lo), minlength=S)
    return U, nP, nN


def auc_from_pairs(U, nP, nN, mult=None):
    """AUC for session weights (1-D) or a batch of weights (2-D, one row per replicate)."""
    if mult is None:
        tot = nP.sum() * nN.sum()
        return float(U.sum() / tot) if tot > 0 else np.nan
    num = np.einsum("ba,ac,bc->b", mult, U, mult)
    den = (mult @ nP) * (mult @ nN)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


# ── 4. the evaluation ─────────────────────────────────────────────────────────

AUC_SETS = ("natural", "natural_centre", "natural_offset", "inj_all_low", "inj_centre_low",
            "inj_offset_low")
DIFF_SETS = AUC_SETS


def _detector_groups(P: Pooled) -> dict:
    """detector -> sorted list of seeds present."""
    groups = {}
    for det, seed in P.raw:
        groups.setdefault(det, []).append(seed)
    return {d: sorted(s, key=lambda x: -1 if x is None else x) for d, s in groups.items()}


def _report_name(det: str) -> tuple[str, float | None]:
    """ed_adaptive@p -> ('ed_adaptive', p); others -> (det, None)."""
    if det.startswith(ED_ADAPTIVE + "@"):
        return ED_ADAPTIVE, float(det.split("@")[1])
    return det, None


def evaluate(index: CaptureIndex, folds: list[Fold], cfg: Config, work_dir: Path, out_dir: Path,
             models=None, ablation: bool = False, seeds=None, failure_figures: bool = False,
             log=print) -> dict:
    """Pooled metrics, bootstrap intervals, H1 and H2; writes every result file to out_dir.

    Undefined ratios (an empty class in a resample, say) are NaN by design, so
    numpy's warnings about them are silenced here.
    """
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _evaluate(index, folds, cfg, work_dir, out_dir, models, ablation, seeds,
                         failure_figures, log)


def _evaluate(index, folds, cfg, work_dir, out_dir, models, ablation, seeds, failure_figures, log) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = detector_entries(cfg, models, ablation, seeds)
    P = build_pooled(index, folds, cfg, work_dir, entries, log)
    if not P.raw:
        raise RuntimeError("no detector has scores in every fold; run train, calibrate and evaluate")
    S = P.S
    cc = CellCounts(P)
    sets = eval_sets(P, cfg)
    mult = bootstrap_multiplicities(S, cfg.eval.n_bootstrap, cfg.eval.bootstrap_seed)
    groups = _detector_groups(P)
    pfas = list(cfg.thresholds.pfa)
    level = cfg.eval.ci_level

    n_sets = len(sets)
    set_index = {(es.name, es.gain): i for i, es in enumerate(sets)}
    Wp, Wn = set_weights(P, sets)
    level_sets = [i for i, es in enumerate(sets) if es.level is not None]

    # decisions -> per-session counts C[(det, seed, p)]: (n_sets, S, 4) = tp, fn, fp, tn
    C = {}
    for (det, seed), raw in P.raw.items():
        for p in pfas:
            thr = P.thr.get((det, seed, p))
            if thr is None:
                continue
            D = cc.decisions(raw > thr[P.fold]).astype(np.float64)
            tp = np.einsum("igcl,gcls->is", Wp, D)
            nP = np.einsum("igcl,gcls->is", Wp, cc.n)
            fp = np.einsum("igcl,gcls->is", Wn, D)
            nN = np.einsum("igcl,gcls->is", Wn, cc.n)
            C[(det, seed, p)] = np.stack([tp, nP - tp, fp, nN - fp], axis=-1)
    counts = {k: {(es.name, es.gain): {"tp": v[i, :, 0], "fn": v[i, :, 1], "fp": v[i, :, 2],
                                       "tn": v[i, :, 3]} for i, es in enumerate(sets)}
              for k, v in C.items()}

    # AUC: session-pair matrices (bootstrap) for AUC_SETS, point estimates for the injection sets
    auc_pt, auc_rep = {}, {}
    masks = {i: set_mask(P, es) for i, es in enumerate(sets)
             if es.name in AUC_SETS or es.name.startswith("inj_")}
    for (det, seed), u in P.calib.items():
        pt = np.full(n_sets, np.nan)
        rep = np.full((mult.shape[0], n_sets), np.nan)
        for i, mask in masks.items():
            U, nPa, nNa = auc_pair_matrix(u, P.label, P.sess, mask, S)
            pt[i] = auc_from_pairs(U, nPa, nNa)
            if sets[i].name in AUC_SETS:
                rep[:, i] = auc_from_pairs(U, nPa, nNa, mult)
        auc_pt[(det, seed)], auc_rep[(det, seed)] = pt, rep

    # per-fold AUC on raw scores (within a fold the scores share one scale)
    fold_auc = {}
    for (det, seed), raw in P.raw.items():
        for f in folds:
            fm = P.fold == f.k
            for i, es in enumerate(sets):
                if es.name not in AUC_SETS:
                    continue
                m = masks[i] & fm
                y = P.label[m]
                if 0 < y.sum() < y.size:
                    fold_auc[(det, seed, f.k, i)] = float(roc_auc_score(y, raw[m]))

    # ── metrics by seed and by fold
    by_seed_rows, by_fold_rows = [], []
    fold_sess = {f.k: [P.sessions.index(s) for s in f.test] for f in folds}
    aggregate = list(range(n_sets))
    for (det, seed, p), arr in C.items():
        name, _ = _report_name(det)
        tot = arr.sum(axis=1)                                   # (n_sets, 4)
        m = {k: metric_value(k, *tot.T) for k in COUNT_METRICS}
        a = auc_pt.get((det, seed), np.full(n_sets, np.nan))
        for i, es in enumerate(sets):
            by_seed_rows.append({"detector": name, "variant": det, "seed": seed_tag(seed), "pfa_target": p,
                                 "gain": GAINS[es.gain], "eval_set": es.name, "n_pos": tot[i, 0] + tot[i, 1],
                                 "n_neg": tot[i, 2] + tot[i, 3],
                                 "pos_fraction": _ratio(tot[i, 0] + tot[i, 1], tot[i].sum()),
                                 "tp": tot[i, 0], "fn": tot[i, 1], "fp": tot[i, 2], "tn": tot[i, 3],
                                 **{k: float(v[i]) for k, v in m.items()}, "auc": a[i]})
        for k, sidx in fold_sess.items():
            ft = arr[aggregate][:, sidx, :].sum(axis=1)
            fm = {kk: metric_value(kk, *ft.T) for kk in COUNT_METRICS}
            for j, i in enumerate(aggregate):
                by_fold_rows.append({"fold": k, "detector": name, "variant": det, "seed": seed_tag(seed),
                                     "pfa_target": p, "gain": GAINS[sets[i].gain], "eval_set": sets[i].name,
                                     "n_pos": ft[j, 0] + ft[j, 1], "n_neg": ft[j, 2] + ft[j, 3],
                                     "tp": ft[j, 0], "fn": ft[j, 1], "fp": ft[j, 2], "tn": ft[j, 3],
                                     **{kk: float(v[j]) for kk, v in fm.items()},
                                     "auc_raw_within_fold": fold_auc.get((det, seed, k, i), np.nan)})

    # ── seed-averaged estimates with bootstrap replicates, per (detector variant, P_FA)
    stats = {}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for det, seeds_ in groups.items():
            for p in pfas:
                arrs = [C[(det, sd, p)] for sd in seeds_ if (det, sd, p) in C]
                if not arrs:
                    continue
                A = np.stack(arrs)                                          # (n_seeds, n_sets, S, 4)
                T = A.sum(axis=2)                                           # (n_seeds, n_sets, 4)
                R = np.einsum("bs,nisk->nbik", mult, A)                     # (n_seeds, B, n_sets, 4)
                st = {"n_seeds": len(arrs), "n_pos": T[0, :, 0] + T[0, :, 1], "n_neg": T[0, :, 2] + T[0, :, 3]}
                for k in COUNT_METRICS:
                    per_seed = metric_value(k, *np.moveaxis(T, -1, 0))      # (n_seeds, n_sets)
                    reps = np.nanmean(metric_value(k, *np.moveaxis(R, -1, 0)), axis=0)   # (B, n_sets)
                    st[k] = (per_seed, reps)
                ap = np.stack([auc_pt[(det, sd)] for sd in seeds_ if (det, sd) in auc_pt])
                ar = np.nanmean(np.stack([auc_rep[(det, sd)] for sd in seeds_ if (det, sd) in auc_rep]), axis=0)
                st["auc"] = (ap, ar)
                stats[(det, p)] = st

    def seed_metric(det, p, sname, g, metric, boot: bool = True):
        """(point, replicates, per-seed values) of a seed-averaged metric for one eval set."""
        st = stats.get((det, p))
        i = set_index.get((sname, g))
        if st is None or i is None:
            return np.nan, None, np.array([])
        per_seed, reps = st[metric]
        col = per_seed[:, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            point = float(np.nanmean(col)) if np.any(np.isfinite(col)) else np.nan
        rep = reps[:, i] if boot and np.any(np.isfinite(reps[:, i])) else None
        return point, rep, col

    summary_rows = []
    for (det, p), st in stats.items():
        name, own_p = _report_name(det)
        if own_p is not None and own_p != p:
            continue
        ci = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in COUNT_METRICS + ("auc",):
                per_seed, reps = st[k]
                ci[k] = (np.nanmean(per_seed, axis=0),
                         np.nanstd(per_seed, axis=0, ddof=1) if st["n_seeds"] > 1 else np.full(n_sets, np.nan),
                         ci_columns(reps, level))
        for i, es in enumerate(sets):
            row = {"detector": name, "variant": det, "pfa_target": p, "gain": GAINS[es.gain],
                   "eval_set": es.name, "source": es.source, "snr_level_db": es.level,
                   "n_seeds": st["n_seeds"], "n_pos": float(st["n_pos"][i]), "n_neg": float(st["n_neg"][i])}
            row["pos_fraction"] = _ratio(row["n_pos"], row["n_pos"] + row["n_neg"])
            for k in COUNT_METRICS + ("auc",):
                mean, sd, q = ci[k]
                row[f"{k}_mean"], row[f"{k}_sd"] = float(mean[i]), float(sd[i])
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = float(q[0, i]), float(q[1, i])
            if es.level is not None:
                sel = (P.gain == es.gain) & np.isin(P.cls, es.pos) & np.isin(P.stratum, es.strata)
                row["snr_eff_mean_db"] = float(np.nanmean(P.snr_eff[sel])) if sel.any() else np.nan
            summary_rows.append(row)

    # ── paired differences: reference model vs every other detector
    ref = cfg.eval.reference_model
    diff_rows = []
    if ref in groups:
        for det in groups:
            if det == ref:
                continue
            name, own_p = _report_name(det)
            for p in pfas:
                if own_p is not None and own_p != p:
                    continue
                for sname in DIFF_SETS:
                    for g in range(len(GAINS)):
                        for metric in ("pd", "auc"):
                            a, ra, _ = seed_metric(ref, p, sname, g, metric, True)
                            b, rb, _ = seed_metric(det, p, sname, g, metric, True)
                            if ra is None or rb is None:
                                continue
                            lo, hi = percentile_ci(ra - rb, level)
                            diff_rows.append({"reference": ref, "other": name, "other_variant": det,
                                              "pfa_target": p, "gain": GAINS[g], "eval_set": sname,
                                              "metric": metric, "reference_value": a, "other_value": b,
                                              "difference": a - b, "ci_lo": lo, "ci_hi": hi})

    # ── H1
    h1 = hypothesis_h1(groups, seed_metric, cfg, level)
    # ── H2
    h2 = hypothesis_h2(groups, counts, P, cfg, mult, level)
    h2["noise_floor"] = noise_floor_table(index, cfg, work_dir)

    # ── curves, positions, ED theory, failure samples
    curves_roc, curves_pr = roc_pr_curves(P, cfg)
    positions = position_table(P, counts, groups, pfas)
    times = time_of_day_table(P, counts, groups, pfas)
    theory = ed_theory_table(P, folds, cfg, work_dir)
    failures = failure_samples(P, cfg, work_dir, index, out_dir if failure_figures else None)

    rec = provenance.run_record(cfg, synthetic=index.synthetic, sessions=P.sessions,
                                folds=[f.to_dict() for f in folds],
                                detectors={d: [seed_tag(s) for s in v] for d, v in groups.items()},
                                n_test_frames=int(P.label.size),
                                stratum_mismatches=P.stratum_mismatch,
                                notes=EVAL_NOTES)
    write_csv(out_dir / "metrics_by_seed.csv", by_seed_rows)
    write_csv(out_dir / "metrics_by_fold.csv", by_fold_rows)
    write_csv(out_dir / "metrics_summary.csv", summary_rows)
    write_csv(out_dir / "pd_vs_snr.csv", [r for r in summary_rows if r["snr_level_db"] is not None])
    write_csv(out_dir / "paired_differences.csv", diff_rows)
    write_csv(out_dir / "roc_curves.csv", curves_roc)
    write_csv(out_dir / "pr_curves.csv", curves_pr)
    write_csv(out_dir / "positions.csv", positions)
    write_csv(out_dir / "time_of_day.csv", times)
    write_csv(out_dir / "ed_theoretical_threshold.csv", theory)
    write_csv(out_dir / "failure_samples.csv", failures)
    write_csv(out_dir / "noise_floor.csv", h2["noise_floor"])
    write_csv(out_dir / "h1.csv", h1["rows"])
    write_csv(out_dir / "h2.csv", h2["rows"])
    provenance.write_json(out_dir / "h1.json", {**h1, "run": rec})
    provenance.write_json(out_dir / "h2.json", {**h2, "run": rec})
    provenance.write_json(out_dir / "run.json", rec)
    log(f"results written to {out_dir}")
    return {"h1": h1, "h2": h2, "n_rows": len(summary_rows), "out_dir": str(out_dir)}


EVAL_NOTES = [
    "Decisions use each fold's own thresholds; predictions pooled over folds.",
    "AUC/ROC/PR on pooled scores mapped through each fold's validation ECDF (mid-rank).",
    "Bootstrap: sessions resampled with replacement; seed-averaged metrics averaged inside replicates.",
    "ed_adaptive is decision-directed: its statistic is computed per target P_FA (variant ed_adaptive@p).",
    "Mixtures are scored by ed_adaptive against the stream's noise estimate just before their idle frame b.",
    "SNR strata: the grid level nearest SNR_eff.",
    "Precision, F1 and AUC are NaN where undefined (no predicted positives, or one class only).",
]


def hypothesis_h1(groups, seed_metric, cfg: Config, level: float) -> dict:
    """H1: Delta P_D = P_D(conv_ae) - P_D(ed_adaptive) on mixtures with SNR <= -4 dB (g1)."""
    ref, base = cfg.eval.reference_model, cfg.eval.h1_baseline
    rows, verdict = [], {}
    for p in cfg.thresholds.pfa:
        b_det = adaptive_name(p) if base == ED_ADAPTIVE else base
        for src in ("all", "centre", "offset"):
            sname = f"inj_{src}_low"
            a, ra, _ = seed_metric(ref, p, sname, 0, "pd", True)
            b, rb, _ = seed_metric(b_det, p, sname, 0, "pd", True)
            if ra is None or rb is None:
                continue
            lo, hi = percentile_ci(ra - rb, level)
            supported = bool(np.isfinite(lo) and lo > 0)
            rows.append({"pfa_target": p, "source": src, "pd_reference": a, "pd_baseline": b,
                         "delta_pd": a - b, "ci_lo": lo, "ci_hi": hi, "supported": supported,
                         "primary": src == "all"})
            if src == "all":
                verdict[pfa_key(p)] = supported
    return {"definition": "Delta P_D = P_D(conv_ae) - P_D(ed_adaptive), SNR strata <= -4 dB pooled, g1; "
                          "supported if the 95% bootstrap interval is entirely above 0. "
                          "Primary: both injection sources pooled.",
            "reference": ref, "baseline": base, "supported_by_pfa": verdict, "rows": rows}


def hypothesis_h2(groups, counts, P: Pooled, cfg: Config, mult: np.ndarray, level: float) -> dict:
    """H2: departures at g2 (thresholds from g1) for conv_ae vs ed_fixed (ed_adaptive as reference).

    FA departure = |P_FA,realized(g2, test B) - P_FA,target|
    PD departure = mean over matched SNR levels of |P_D(g2, L) - P_D(g1, L)|
                   (both injection sources pooled; levels with mixtures at both gains)
    Paired difference = departure(baseline) - departure(conv_ae): > 0 favours conv_ae.
    """
    ref = cfg.eval.reference_model
    pfas = list(cfg.thresholds.pfa)
    L = P.L

    def departures(det, p, weights):
        """(fa_dep, pd_dep) for one detector variant, seed-averaged; weights: (S,) or (R, S)."""
        fa, pdd = [], []
        for sd in groups.get(det, []):
            c = counts.get((det, sd, p))
            if c is None:
                continue
            b2 = c[("natural", 1)]
            fa.append(np.abs(_ratio(weights @ b2["fp"], weights @ (b2["fp"] + b2["tn"])) - p))
            diffs = []
            for li in range(L):
                name = f"inj_all_{P.grid[li]:+g}dB"
                c1, c2 = c[(name, 0)], c[(name, 1)]
                if (c1["tp"] + c1["fn"]).sum() == 0 or (c2["tp"] + c2["fn"]).sum() == 0:
                    continue          # level not matched in the full data
                d1 = _ratio(weights @ c1["tp"], weights @ (c1["tp"] + c1["fn"]))
                d2 = _ratio(weights @ c2["tp"], weights @ (c2["tp"] + c2["fn"]))
                diffs.append(np.abs(d2 - d1))
            if diffs:
                with np.errstate(invalid="ignore"):
                    pdd.append(np.nanmean(np.stack(diffs), axis=0))
            else:
                pdd.append(np.full(np.shape(weights @ b2["fp"]), np.nan))
        if not fa:
            return None, None
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.stack(fa), axis=0), np.nanmean(np.stack(pdd), axis=0)

    ones = np.ones(P.S)
    rows, verdict = [], {}
    for p in pfas:
        r_fa, r_pd = departures(ref, p, ones)
        rb_fa, rb_pd = departures(ref, p, mult)
        if r_fa is None:
            continue
        for base in (cfg.eval.h2_baseline, cfg.eval.h2_reference):
            b_det = adaptive_name(p) if base == ED_ADAPTIVE else base
            b_fa, b_pd = departures(b_det, p, ones)
            bb_fa, bb_pd = departures(b_det, p, mult)
            if b_fa is None:
                continue
            row = {"pfa_target": p, "baseline": base,
                   "fa_departure_reference": float(r_fa), "fa_departure_baseline": float(b_fa),
                   "pd_departure_reference": float(r_pd), "pd_departure_baseline": float(b_pd),
                   "fa_diff": float(b_fa - r_fa), "pd_diff": float(b_pd - r_pd)}
            row["fa_diff_ci_lo"], row["fa_diff_ci_hi"] = percentile_ci(bb_fa - rb_fa, level)
            row["pd_diff_ci_lo"], row["pd_diff_ci_hi"] = percentile_ci(bb_pd - rb_pd, level)
            row["fa_favours_reference"] = bool(np.isfinite(row["fa_diff_ci_lo"]) and row["fa_diff_ci_lo"] > 0)
            row["pd_favours_reference"] = bool(np.isfinite(row["pd_diff_ci_lo"]) and row["pd_diff_ci_lo"] > 0)
            row["role"] = "comparison" if base == cfg.eval.h2_baseline else "reference"
            rows.append(row)
            if base == cfg.eval.h2_baseline:
                verdict[pfa_key(p)] = row["fa_favours_reference"] and row["pd_favours_reference"]
    return {"definition": "departures at g2 with g1 thresholds: |P_FA,realized - P_FA,target| on test B, "
                          "and mean over matched levels of |P_D(g2) - P_D(g1)| on the injection set "
                          "(sources pooled). Difference = baseline - conv_ae (> 0 favours conv_ae). "
                          "H2 supported at an operating point if both 95% intervals vs ed_fixed are > 0.",
            "reference": ref, "baseline": cfg.eval.h2_baseline, "also_reported": cfg.eval.h2_reference,
            "supported_by_pfa": verdict, "rows": rows}


def noise_floor_table(index: CaptureIndex, cfg: Config, work_dir: Path) -> list[dict]:
    """Median frame power (DC removed) of Class A and B per session and gain; g2 - g1 change."""
    rows = []
    for sid in index.sessions():
        row = {"session_id": sid, "position_id": index.position_of(sid)}
        for cls in ("A", "B"):
            med = {}
            for g in GAINS:
                rec = index.get(sid, cls, g)
                if rec is None:
                    continue
                try:
                    st = open_capture_store(work_dir, rec.capture_id)
                except FileNotFoundError:
                    continue
                med[g] = float(np.median(st.energy))
                row[f"{cls}_{g}_gain_db"] = rec.gain_db
                row[f"{cls}_{g}_median_power_db"] = 10 * np.log10(med[g])
            if len(med) == 2:
                row[f"{cls}_change_db"] = 10 * np.log10(med["g2"] / med["g1"])
        rows.append(row)
    return rows


def roc_pr_curves(P: Pooled, cfg: Config) -> tuple[list, list]:
    n_grid = cfg.eval.roc_fpr_grid_points
    fpr_grid = np.concatenate([[0.0], np.logspace(-4, 0, n_grid - 1)])
    rec_grid = np.linspace(0, 1, cfg.eval.pr_recall_grid_points)
    roc_rows, pr_rows = [], []
    low = np.flatnonzero(P.grid <= cfg.injection.low_snr_max_db)
    masks = []
    for g in range(len(GAINS)):
        neg_m = (P.gain == g) & (P.cls == CLS_CODE["B"])
        for sname, pos in (("natural", ("C_centre", "C_offset")), ("natural_centre", ("C_centre",)),
                           ("natural_offset", ("C_offset",))):
            masks.append((g, sname, neg_m, (P.gain == g) & np.isin(P.cls, [CLS_CODE[c] for c in pos])))
        for src, classes in SOURCES.items():
            pos_m = (P.gain == g) & np.isin(P.cls, [CLS_CODE[c] for c in classes]) & np.isin(P.stratum, low)
            masks.append((g, f"inj_{src}_low", neg_m, pos_m))
    for (det, seed), u in P.calib.items():
        name, _ = _report_name(det)
        for g, sname, neg_m, pos_m in masks:
            _curve_rows(roc_rows, pr_rows, name, det, seed, g, sname, u, neg_m, pos_m, fpr_grid, rec_grid)
    return roc_rows, pr_rows


def _curve_rows(roc_rows, pr_rows, name, det, seed, g, sname, u, neg_m, pos_m, fpr_grid, rec_grid):
    if not neg_m.any() or not pos_m.any():
        return
    y = np.concatenate([np.zeros(neg_m.sum()), np.ones(pos_m.sum())])
    s = np.concatenate([u[neg_m], u[pos_m]])
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, fpr_grid, side="right") - 1
    tpr_g = tpr[np.clip(idx, 0, len(tpr) - 1)]
    for f, t in zip(fpr_grid, tpr_g):
        roc_rows.append({"detector": name, "variant": det, "seed": seed_tag(seed), "gain": GAINS[g],
                         "eval_set": sname, "fpr": float(f), "tpr": float(t)})
    prec, rec, _ = precision_recall_curve(y, s)
    for r0 in rec_grid:
        ok = rec >= r0
        pr_rows.append({"detector": name, "variant": det, "seed": seed_tag(seed), "gain": GAINS[g],
                        "eval_set": sname, "recall": float(r0),
                        "precision": float(prec[ok].max()) if ok.any() else np.nan,
                        "pos_fraction": float(y.mean())})


def position_table(P: Pooled, counts, groups, pfas) -> list[dict]:
    return _breakdown_table(P, counts, groups, pfas, P.positions, "position_id")


def time_of_day_table(P: Pooled, counts, groups, pfas) -> list[dict]:
    """P_D and realized P_FA (natural captures, g1) for day and night sessions separately."""
    return _breakdown_table(P, counts, groups, pfas, P.times_of_day or [], "time_of_day")


def _breakdown_table(P: Pooled, counts, groups, pfas, values, column: str) -> list[dict]:
    """Pooled P_D and P_FA per value of a session attribute (position, time of day)."""
    positions = sorted(set(values))
    if len(positions) < 2:
        return []
    rows = []
    for det in groups:
        name, own_p = _report_name(det)
        for p in pfas:
            if own_p is not None and own_p != p:
                continue
            for pos in positions:
                w = np.array([1.0 if q == pos else 0.0 for q in values])
                vals = {"pd": [], "pfa": []}
                for sd in groups[det]:
                    c = counts.get((det, sd, p), {}).get(("natural", 0))
                    if c is None:
                        continue
                    vals["pd"].append(_ratio(w @ c["tp"], w @ (c["tp"] + c["fn"])))
                    vals["pfa"].append(_ratio(w @ c["fp"], w @ (c["fp"] + c["tn"])))
                rows.append({"detector": name, "variant": det, "pfa_target": p, column: pos,
                             "n_sessions": int(w.sum()), "gain": "g1", "eval_set": "natural",
                             "pd_mean": float(np.nanmean(vals["pd"])) if vals["pd"] else np.nan,
                             "pfa_mean": float(np.nanmean(vals["pfa"])) if vals["pfa"] else np.nan})
    return rows


def ed_theory_table(P: Pooled, folds: list[Fold], cfg: Config, work_dir: Path) -> list[dict]:
    """Theoretical vs empirical ed_fixed thresholds and their realized P_FA (validation, test B)."""
    raw = P.raw.get((ED_FIXED, None))
    if raw is None:
        return []
    rows = []
    pooled = {}
    for i, fold in enumerate(folds):
        th = load_thresholds(work_dir, fold.k)
        for p in cfg.thresholds.pfa:
            t = th["ed_fixed_theoretical"][pfa_key(p)]
            for g in range(len(GAINS)):
                m = (P.fold == fold.k) & (P.gain == g) & (P.cls == CLS_CODE["B"])
                b = raw[m]
                fa_th = float(np.mean(b > t["lambda"])) if b.size else np.nan
                fa_em = float(np.mean(b > t["empirical_threshold"])) if b.size else np.nan
                pl = pooled.setdefault((p, g), [0, 0, 0])
                pl[0] += int(np.sum(b > t["lambda"]))
                pl[1] += int(np.sum(b > t["empirical_threshold"]))
                pl[2] += int(b.size)
                rows.append({"fold": fold.k, "pfa_target": p, "gain": GAINS[g], "sigma2_hat": t["sigma2_hat"],
                             "lambda_theory": t["lambda"], "threshold_empirical": t["empirical_threshold"],
                             "ratio_theory_to_empirical": t["lambda"] / t["empirical_threshold"],
                             "realized_pfa_val_theory": t["realized_pfa_val_theory"],
                             "realized_pfa_val_empirical": t["realized_pfa_val_empirical"],
                             "realized_pfa_test_theory": fa_th, "realized_pfa_test_empirical": fa_em,
                             "n_test_b": int(b.size)})
    for (p, g), (a, b, n) in pooled.items():
        rows.append({"fold": "pooled", "pfa_target": p, "gain": GAINS[g],
                     "realized_pfa_test_theory": a / n if n else np.nan,
                     "realized_pfa_test_empirical": b / n if n else np.nan, "n_test_b": n})
    return rows


def failure_samples(P: Pooled, cfg: Config, work_dir: Path, index: CaptureIndex,
                    fig_dir: Path | None) -> list[dict]:
    """A reproducible sample of false alarms (test B, g1) and misses (test C, g1) per detector."""
    rng = np.random.default_rng(cfg.eval.bootstrap_seed)
    k = cfg.eval.failure_samples
    rows = []
    groups = _detector_groups(P)
    for det, seeds_ in groups.items():
        seed = seeds_[0]
        raw = P.raw[(det, seed)]
        for p in cfg.thresholds.pfa:
            thr = P.thr.get((det, seed, p))
            if thr is None:
                continue
            t = thr[P.fold]
            dec = raw > t
            g1 = P.gain == 0
            fa = np.flatnonzero(g1 & (P.cls == CLS_CODE["B"]) & dec)
            miss = np.flatnonzero(g1 & np.isin(P.cls, [CLS_CODE["C_centre"], CLS_CODE["C_offset"]]) & ~dec)
            for kind, pool in (("false_alarm", fa), ("miss", miss)):
                pick = np.sort(rng.choice(pool, size=min(k, pool.size), replace=False)) if pool.size else []
                for i in pick:
                    u = P.units[P.unit[i]]
                    rec = index.by_id(u["unit_id"])
                    t_utc = rec.frame_time(int(P.frame[i]), cfg.preprocess.frame_len)
                    rows.append({"detector": det, "seed": seed_tag(seed), "pfa_target": p, "type": kind,
                                 "session_id": u["session_id"], "position_id": rec.position_id,
                                 "capture_id": u["unit_id"], "class": u["class"], "gain": u["gain_role"],
                                 "frame_index": int(P.frame[i]), "frame_utc": provenance.utc_iso(t_utc),
                                 "local_hour_utc": t_utc.hour, "score": float(raw[i]),
                                 "threshold": float(t[i])})
    if fig_dir is not None and rows:
        _failure_figures(rows, cfg, work_dir, Path(fig_dir) / "failure_figures")
    return rows


def _failure_figures(rows, cfg, work_dir, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .preprocess import freq_axis_hz

    out.mkdir(parents=True, exist_ok=True)
    f = freq_axis_hz(cfg) / 1e3
    T_ms = cfg.frame_duration_s * 1e3
    by = {}
    for r in rows:
        by.setdefault((r["detector"], r["pfa_target"], r["type"]), []).append(r)
    for (det, p, kind), rs in by.items():
        n = len(rs)
        cols = min(n, 5)
        nrows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 2.8 * nrows), squeeze=False)
        for ax, r in zip(axes.ravel(), rs):
            st = open_capture_store(work_dir, r["capture_id"])
            ax.imshow(np.asarray(st.spec[r["frame_index"]], dtype=np.float32), aspect="auto",
                      extent=[0, T_ms, f[-1], f[0]], cmap="viridis")
            ax.set_title(f"{r['session_id']} {r['class']} #{r['frame_index']}\n{r['frame_utc'][11:19]} UTC",
                         fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"{det} P_FA={p:g}: {kind.replace('_', ' ')}s (time ms, frequency kHz)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{det.replace('@', '_at_')}_{p:g}_{kind}.png", dpi=120)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def _fmt(v):
    t = type(v)
    if t is float or t is np.float64 or t is np.float32:
        return f"{v:.6g}" if v == v and v not in (np.inf, -np.inf) else ""
    if t is str or t is int or t is bool:
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return _fmt(float(v))
    return "" if v is None else v
