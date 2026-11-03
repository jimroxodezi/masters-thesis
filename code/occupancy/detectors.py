"""
detectors.py
------------
Scores of every detector on identical DC-removed frames (design section 7).
A larger score means "more likely occupied".

    conv_ae, fc_ae (+ ablations)   mean squared reconstruction error of x
    conv_vae_mse, conv_vae_per     VAE at z = mu: MSE, and the PER score
    pca                            mean squared PCA reconstruction error
    ed_fixed                       T = (1/N) sum |y|^2
    ed_adaptive                    T / sigma2_hat, sigma2_hat = median T of the last
                                   M = 64 frames declared idle (stream per session)
    pss                            max normalised PSS correlation

ed_adaptive is decision-directed: which frames update sigma2_hat depends on the
threshold kappa, so its statistic is computed once per target P_FA.
"""

from __future__ import annotations

import bisect
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

from . import provenance
from .config import Config
from .models import ModelSpec, inference_model, model_input, per_score
from .preprocess import FeatureStore, Normalizer

ED_FIXED, ED_ADAPTIVE, PSS, PCA = "ed_fixed", "ed_adaptive", "pss", "pca"


def model_detector_names(spec: ModelSpec) -> list[str]:
    if spec.kind == "conv_vae":
        return [f"{spec.name}_mse", f"{spec.name}_per"]
    return [spec.name]


def adaptive_name(pfa: float) -> str:
    return f"{ED_ADAPTIVE}@{pfa:g}"


# ── learned models ────────────────────────────────────────────────────────────

@torch.no_grad()
def score_model(model: torch.nn.Module, spec: ModelSpec, store: FeatureStore, norm: Normalizer,
                cfg: Config, batch: int | None = None, indices: np.ndarray | None = None) -> dict:
    """Scores of one model on (a subset of) a store: {detector_name: (n,) float64}."""
    model = inference_model(model)
    batch = batch or cfg.eval.score_batch
    idx = np.arange(store.n) if indices is None else np.asarray(indices)
    names = model_detector_names(spec)
    out = {n: np.zeros(idx.size, dtype=np.float64) for n in names}
    per = cfg.models.per
    for s in range(0, idx.size, batch):
        sel = idx[s:s + batch]
        order = np.argsort(sel)
        rows = np.asarray(store.spec[sel[order]], dtype=np.float32)
        x = np.empty_like(rows)
        x[order] = norm.transform(rows)
        xt = model_input(model, x)
        x_hat = model.reconstruct(xt)
        mse = ((x_hat - xt) ** 2).flatten(1).mean(dim=1).double().numpy()
        out[names[0]][s:s + sel.size] = mse
        if spec.kind == "conv_vae":
            out[names[1]][s:s + sel.size] = per_score(xt, x_hat, per.gamma, per.alpha, per.beta,
                                                      per.xi, per.eta).numpy()
    return out


def iter_unit_batches(units, norm: Normalizer, batch: int):
    """Normalised frames of several stores in shared batches: yields (x, slice into the concatenation)."""
    sizes = [u.n for u in units]
    total = int(sum(sizes))
    unit_idx = np.repeat(np.arange(len(units)), sizes)
    frame_idx = np.concatenate([np.arange(n) for n in sizes]) if units else np.zeros(0, int)
    for s in range(0, total, batch):
        ui, fi = unit_idx[s:s + batch], frame_idx[s:s + batch]
        rows = np.empty((ui.size, *units[0].spec.shape[1:]), dtype=np.float32)
        for u in np.unique(ui):
            m = ui == u
            rows[m] = units[u].spec[fi[m]]                      # increasing indices within a unit
        yield norm.transform(rows), slice(s, s + ui.size)


def split_by_unit(values: np.ndarray, units) -> dict:
    out, pos = {}, 0
    for u in units:
        out[u.unit_id] = values[pos:pos + u.n]
        pos += u.n
    return out


@torch.no_grad()
def score_model_units(model: torch.nn.Module, spec: ModelSpec, units, norm: Normalizer, cfg: Config,
                      batch: int | None = None) -> dict:
    """{detector_name: {unit_id: scores}} for several stores, scored in shared batches."""
    model = inference_model(model)
    names = model_detector_names(spec)
    total = sum(u.n for u in units)
    vals = {n: np.zeros(total) for n in names}
    per = cfg.models.per
    for x, sl in iter_unit_batches(units, norm, batch or cfg.eval.score_batch):
        xt = model_input(model, x)
        x_hat = model.reconstruct(xt)
        vals[names[0]][sl] = ((x_hat - xt) ** 2).flatten(1).mean(dim=1).double().numpy()
        if spec.kind == "conv_vae":
            vals[names[1]][sl] = per_score(xt, x_hat, per.gamma, per.alpha, per.beta, per.xi, per.eta).numpy()
    return {n: split_by_unit(v, units) for n, v in vals.items()}


# ── PCA ───────────────────────────────────────────────────────────────────────

class PCAModel:
    def __init__(self, mean=None, components=None, info=None):
        self.mean, self.components, self.info = mean, components, info or {}

    @property
    def d(self) -> int:
        return 0 if self.components is None else int(self.components.shape[0])

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per row of X (n, D)."""
        Xc = np.asarray(X, dtype=np.float32) - self.mean
        proj = Xc @ self.components.T
        resid = Xc - proj @ self.components
        return (resid.astype(np.float64) ** 2).mean(axis=1)

    def score_units(self, units, norm: Normalizer, batch: int = 256) -> dict:
        total = sum(u.n for u in units)
        out = np.zeros(total)
        for x, sl in iter_unit_batches(units, norm, batch):
            out[sl] = self.score(x.reshape(x.shape[0], -1))
        return split_by_unit(out, units)

    def score_store(self, store: FeatureStore, norm: Normalizer, batch: int = 256,
                    indices: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(store.n) if indices is None else np.asarray(indices)
        out = np.zeros(idx.size)
        for s in range(0, idx.size, batch):
            sel = idx[s:s + batch]
            order = np.argsort(sel)
            rows = np.asarray(store.spec[sel[order]], dtype=np.float32)
            x = np.empty_like(rows)
            x[order] = norm.transform(rows)
            out[s:s + sel.size] = self.score(x.reshape(sel.size, -1))
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components)
        provenance.write_json(path.with_suffix(".json"), self.info)

    @classmethod
    def load(cls, path: Path) -> "PCAModel":
        with np.load(path) as z:
            mean, comps = z["mean"], z["components"]
        return cls(mean, comps, provenance.read_json(Path(path).with_suffix(".json")))


def fit_pca(frames, cfg: Config, log=print) -> PCAModel:
    """d = smallest number of components explaining >= 95 % of the training variance,
    capped at 256; IncrementalPCA when the data would not fit the memory budget."""
    from sklearn.decomposition import PCA as SkPCA, IncrementalPCA

    p = cfg.models.pca
    n = len(frames)
    D = int(np.prod(frames.stores[0].spec.shape[1:]))
    n_fit = int(min(p.max_components, n, D))
    if n_fit < 1:
        raise ValueError("no frames for PCA")
    use_inc = p.solver == "incremental" or (p.solver == "auto" and n * D * 4 > p.max_in_memory_bytes)
    if use_inc and n_fit > p.ipca_batch:
        raise ValueError("ipca_batch must be at least the number of components")
    if use_inc:
        ipca = IncrementalPCA(n_components=n_fit)
        bounds = list(range(0, n, p.ipca_batch))
        # every partial_fit batch must hold at least n_fit samples: fold a short tail back
        if len(bounds) > 1 and n - bounds[-1] < n_fit:
            bounds.pop()
        for i, s in enumerate(bounds):
            e = bounds[i + 1] if i + 1 < len(bounds) else n
            ipca.partial_fit(frames.batch(np.arange(s, e)).reshape(e - s, -1).numpy())
        mean, comps, ratio = ipca.mean_, ipca.components_, ipca.explained_variance_ratio_
        solver = "IncrementalPCA"
    else:
        X = frames.batch(np.arange(n)).reshape(n, -1).numpy()
        svd = "full" if n_fit >= 0.8 * min(n, D) else "randomized"
        pca = SkPCA(n_components=n_fit, svd_solver=svd, random_state=p.random_state).fit(X)
        mean, comps, ratio = pca.mean_, pca.components_, pca.explained_variance_ratio_
        solver = f"PCA({svd})"
    cum = np.cumsum(ratio)
    hit = np.flatnonzero(cum >= p.variance)
    d = int(hit[0] + 1) if hit.size else n_fit
    info = {"solver": solver, "n_train_frames": n, "n_features": D, "n_fitted": n_fit,
            "d": d, "explained_variance_at_d": float(cum[d - 1]), "target_variance": p.variance,
            "cap": p.max_components, "reached_target": bool(hit.size),
            "cumulative_explained_variance": cum.tolist()}
    log(f"  [pca] d = {d} components explain {cum[d - 1]:.4f} of the training variance ({solver})")
    return PCAModel(mean.astype(np.float32), comps[:d].astype(np.float32), info)


def fit_pca_for_fold(train_frames, cfg: Config, fold_path: Path, force: bool = False, log=print):
    path = Path(fold_path) / "models" / PCA / "pca.npz"
    if path.exists() and not force:
        log("  [pca] already fitted, skipped")
        return PCAModel.load(path)
    model = fit_pca(train_frames, cfg, log)
    model.save(path)
    return model


# ── energy detectors ──────────────────────────────────────────────────────────

def ed_theoretical_threshold(sigma2_hat: float, N: int, pfa: float) -> float:
    """lambda = sigma2_hat/(2N) * F^-1_{chi2(2N)}(1 - P_FA) (design section 7, detector 5).

    Assumes N independent complex Gaussian samples of variance sigma2_hat, so that
    2N T / sigma2 ~ chi2(2N). Reported for comparison only.
    """
    return float(sigma2_hat / (2 * N) * chi2.ppf(1.0 - pfa, 2 * N))


class RunningMedian:
    """Median of the most recent M values (insertion order), O(M) per update."""

    def __init__(self, M: int, init: float):
        self.M = M
        self.fifo = deque([init] * M)
        self.sorted = [init] * M

    def median(self) -> float:
        s, m = self.sorted, self.M
        return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

    def push(self, v: float) -> None:
        old = self.fifo.popleft()
        del self.sorted[bisect.bisect_left(self.sorted, old)]
        self.fifo.append(v)
        bisect.insort(self.sorted, v)


def adaptive_stream(T: np.ndarray, kappa: float, init_sigma2: float, M: int):
    """Process one stream of frame energies.

    For each frame t: sigma2_hat(t) = median of the last M frames declared idle
    (the buffer starts as M copies of init_sigma2, the validation median);
    statistic r(t) = T(t)/sigma2_hat(t); the frame is declared idle if
    r(t) <= kappa, and only then enters the buffer.
    Returns r, sigma2_hat (before each frame) and the idle flags.
    """
    T = np.asarray(T, dtype=np.float64)
    rm = RunningMedian(M, float(init_sigma2))
    r = np.empty(T.size)
    s2 = np.empty(T.size)
    idle = np.empty(T.size, dtype=bool)
    for t, v in enumerate(T):
        s2[t] = rm.median()
        r[t] = v / s2[t]
        idle[t] = r[t] <= kappa
        if idle[t]:
            rm.push(float(v))
    return r, s2, idle


def quantile_threshold(scores: np.ndarray, pfa: float, method: str = "inverted_cdf") -> float:
    """Empirical (1 - P_FA) quantile; with "inverted_cdf" the order statistic
    s_(ceil((1-P_FA) n)), so at most a fraction P_FA of the scores exceed it."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("no finite calibration scores")
    return float(np.quantile(scores, 1.0 - pfa, method=method))


def calibrate_kappa(val_streams: list, init_sigma2: float, M: int, pfa: float,
                    method: str = "inverted_cdf", max_iter: int = 50) -> dict:
    """kappa = (1 - P_FA) quantile of T/sigma2_hat over the validation stream(s),
    with the stream run at that same kappa (fixed point of the procedure).

    Starts from kappa = inf (every validation frame updates the estimate) and
    iterates kappa <- quantile(r(kappa)). If the iteration cycles, the largest
    kappa of the cycle is used (the conservative choice).
    """
    def run(k):
        return np.concatenate([adaptive_stream(s, k, init_sigma2, M)[0] for s in val_streams])

    kappa = np.inf
    seen = []
    status = "max_iter"
    for _ in range(max_iter):
        r = run(kappa)
        new = quantile_threshold(r, pfa, method)
        if new == kappa:
            status = "converged"
            break
        if new in seen:
            cycle = seen[seen.index(new):] + [kappa]
            kappa = max(v for v in cycle if np.isfinite(v))
            status = "cycle"
            break
        seen.append(new)
        kappa = new
    if status != "converged":
        r = run(kappa)                       # the statistics must belong to the returned kappa
    realized = float(np.mean(r > kappa))
    return {"kappa": float(kappa), "iterations": len(seen) + 1, "converged": status == "converged",
            "status": status, "realized_pfa_validation": realized, "val_stats": r}
