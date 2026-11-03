"""
latency.py
----------
Computational cost on the host laptop (design section 10, RQ4).

* torch.set_num_threads(4) (config), timer time.perf_counter;
* settling: every detector runs in turn for 5 s before any timing begins;
* per-frame latency at batch size 1 over 1000 frames after 50 warm-up frames,
  median and 95th percentile, separately for preprocessing, for each
  detector's scoring, and end to end (raw uint8 block -> decision);
* throughput in frames per second at batch size 64;
* real-time factor = processing time per frame / 8.533 ms (< 1: real time).

Preprocessing differs by detector family:
    spectrogram detectors (AE, VAE, PCA): uint8 -> complex, DC removal, STFT, normalisation
    ed_fixed, ed_adaptive, pss:          uint8 -> complex, DC removal (they never use the STFT)
Models come from the trained checkpoints of the chosen fold and seed when they
exist; otherwise freshly initialised models are timed (same arithmetic) and the
output says so. CPU model, threads, library versions and power mode are logged.

Effect of the scoring optimisations (models.inference_model, numpy percentiles
in the PER score): each neural network is also timed end to end at batch
size 1 "as trained" (the model as saved, default memory layout, PER score with
torch.quantile), alternating frame by frame with its scoring form so that
both forms see the same conditions. Rows "<detector>[as trained]:end_to_end"
and "<detector>[scoring, paired]:end_to_end"; latency.json gives the speed-up.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from . import provenance
from .config import Config
from .dataset import dc_remove
from .detectors import (ED_ADAPTIVE, ED_FIXED, PCA, PSS, PCAModel, RunningMedian, model_detector_names)
from .iqio import open_raw, u8_to_complex
from .models import build_model, inference_model, main_specs, model_input, per_score, per_score_torch
from .preprocess import Normalizer, frame_energy, stft_db
from .pss import correlator_from_config
from .train import fold_dir, load_model, model_dir


def _stats(t: np.ndarray, frame_s: float) -> dict:
    t = np.asarray(t) * 1e3
    med = float(np.median(t))
    return {"median_ms": med, "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(np.mean(t)),
            "rtf_median": med / (frame_s * 1e3)}


def _raw_frames(cfg: Config, n: int, capture_path: str | None, seed: int = 0) -> np.ndarray:
    """(n, 2N) uint8 blocks: from a capture if given, else quantised complex Gaussian noise."""
    N = cfg.preprocess.frame_len
    if capture_path:
        raw = open_raw(capture_path)
        avail = raw.size // (2 * N)
        idx = np.arange(n) % max(avail, 1)
        return np.stack([np.asarray(raw[2 * N * i:2 * N * (i + 1)]) for i in idx])
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2 * N), dtype=np.uint8)
    for s in range(0, n, 64):                     # in chunks: a few MB of temporaries, not a GB
        x = rng.normal(127.5, 6.0, size=(min(64, n - s), 2 * N))
        out[s:s + x.shape[0]] = np.clip(np.rint(x), 0, 255)
    return out


class Pipelines:
    """Per-detector preprocessing and scoring functions on raw uint8 blocks."""

    def __init__(self, cfg: Config, work_dir: Path | None, fold: int, seed: int, log=print):
        self.cfg = cfg
        self.weights = {}
        norm_path = fold_dir(work_dir, fold) / "normalizer.json" if work_dir else None
        if norm_path is not None and norm_path.exists():
            self.norm, self.norm_src = Normalizer.load(norm_path), "fold normaliser"
        else:
            self.norm, self.norm_src = Normalizer(mu=-40.0, sigma=10.0), "placeholder (timing only)"
        self.models = {}
        self.as_trained = {}
        for spec in main_specs(cfg):
            path = model_dir(work_dir, fold, spec, seed) / "model.pt" if work_dir else None
            if path is not None and path.exists():
                model, _ = load_model(path, cfg)
                self.weights[spec.name] = "trained"
            else:
                torch.manual_seed(seed)
                model = build_model(spec, cfg).eval()
                self.weights[spec.name] = "untrained (random initialisation; timing only)"
            self.models[spec.name] = (spec, inference_model(model))   # as scored in the evaluation
            self.as_trained[spec.name] = (spec, model.eval())        # for the optimisation effect
        pca_path = fold_dir(work_dir, fold) / "models" / PCA / "pca.npz" if work_dir else None
        if pca_path is not None and pca_path.exists():
            self.pca = PCAModel.load(pca_path)
            self.weights[PCA] = f"trained (d={self.pca.d})"
        else:
            d = cfg.models.pca.max_components
            q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((cfg.preprocess.fft_size ** 2, d)))
            self.pca = PCAModel(np.zeros(q.shape[0], np.float32), q.T.astype(np.float32))
            self.weights[PCA] = f"untrained (random d={d}; timing only)"
        self.correlator = correlator_from_config(cfg)
        self.median = RunningMedian(cfg.models.ed.adaptive_window, 1.0)

    # preprocessing
    def iq(self, raw):
        return dc_remove(u8_to_complex(raw.reshape(-1)).reshape(raw.shape[0], -1)).astype(np.complex64)

    def spec(self, raw):
        return self.norm.transform(stft_db(self.iq(raw), self.cfg))

    def detectors(self) -> dict:
        """name -> (family, score function on the preprocessed batch)."""
        out = {}
        per = self.cfg.models.per
        for name, (spec, model) in self.models.items():
            names = model_detector_names(spec)

            def mse_fn(x, model=model):
                with torch.no_grad():
                    xt = model_input(model, x)
                    return ((model.reconstruct(xt) - xt) ** 2).flatten(1).mean(1).numpy()
            out[names[0]] = ("spec", mse_fn)
            if spec.kind == "conv_vae":
                def per_fn(x, model=model):
                    with torch.no_grad():
                        xt = model_input(model, x)
                        return per_score(xt, model.reconstruct(xt), per.gamma, per.alpha, per.beta,
                                         per.xi, per.eta).numpy()
                out[names[1]] = ("spec", per_fn)
        out[PCA] = ("spec", lambda x: self.pca.score(x.reshape(x.shape[0], -1)))
        out[ED_FIXED] = ("iq", frame_energy)

        def adaptive(y):
            T = frame_energy(y)
            r = np.empty_like(T)
            for i, v in enumerate(T):
                r[i] = v / self.median.median()
                self.median.push(float(v))         # always update: the worst-case cost per frame
            return r
        out[ED_ADAPTIVE] = ("iq", adaptive)
        out[PSS] = ("iq", self.correlator)
        return out

    def as_trained_detectors(self) -> dict:
        """The neural networks as trained: model as saved, default layout, torch.quantile PER."""
        out = {}
        per = self.cfg.models.per
        for name, (spec, model) in self.as_trained.items():
            names = model_detector_names(spec)

            def mse_fn(x, model=model):
                with torch.no_grad():
                    xt = torch.from_numpy(x[:, None])
                    return ((model.reconstruct(xt) - xt) ** 2).flatten(1).mean(1).numpy()
            out[names[0]] = mse_fn
            if spec.kind == "conv_vae":
                def per_fn(x, model=model):
                    with torch.no_grad():
                        xt = torch.from_numpy(x[:, None])
                        return per_score_torch(xt, model.reconstruct(xt), per.gamma, per.alpha, per.beta,
                                               per.xi, per.eta).numpy()
                out[names[1]] = per_fn
        return out


def run_latency(cfg: Config, work_dir: Path | None, out_dir: Path, fold: int = 0, seed: int = 0,
                capture_path: str | None = None, synthetic_data: bool = False,
                require_performance: bool = False, log=print) -> dict:
    """Timings are defined on mains power with the Performance profile (design
    section 10). With require_performance (the latency command's default), any
    other known power state is refused; an unknown one is recorded as such."""
    condition = provenance.power_condition()
    if require_performance and condition == "other":
        raise SystemExit("latency: the laptop is not on mains power with the Performance profile; "
                         + provenance.POWER_HINT + " (or use --ignore-power for a test run)")
    if condition != "performance":
        log(f"WARNING: power condition {condition!r}: these timings do not follow the design")
    L = cfg.latency
    torch.set_num_threads(L.num_threads)
    frame_s = cfg.frame_duration_s
    pipes = Pipelines(cfg, work_dir, fold, seed, log)
    dets = pipes.detectors()
    n1 = L.warmup_frames + L.n_frames
    raw1 = _raw_frames(cfg, n1, capture_path)
    rawB = _raw_frames(cfg, L.throughput_batch * L.throughput_batches, capture_path, seed=1)
    rows = []
    # settling: before any timing, every detector runs in turn for settle_s seconds.
    # Without it the first detector timed in a process was up to 1.5x slower than
    # when timed again later in the same process (a start-up effect lasting a few
    # seconds, longer than the per-detector warm-up of 50 frames).
    t_settle = time.perf_counter() + L.settle_s
    k = 0
    while time.perf_counter() < t_settle:
        for fam, score in dets.values():
            score((pipes.spec if fam == "spec" else pipes.iq)(raw1[k % n1:k % n1 + 1]))
        k += 1
    # preprocessing alone (spectrogram pipeline and IQ pipeline), batch 1
    for fam, fn in (("spec", pipes.spec), ("iq", pipes.iq)):
        t = []
        for i in range(n1):
            t0 = time.perf_counter()
            fn(raw1[i:i + 1])
            t.append(time.perf_counter() - t0)
        rows.append({"item": f"preprocessing_{fam}", "batch": 1,
                     **_stats(np.asarray(t[L.warmup_frames:]), frame_s)})
    # each detector: preprocessing, scoring and end to end, batch 1
    for name, (fam, score) in dets.items():
        prep = pipes.spec if fam == "spec" else pipes.iq
        tp, ts, te = [], [], []
        for i in range(n1):
            t0 = time.perf_counter()
            x = prep(raw1[i:i + 1])
            t1 = time.perf_counter()
            s = score(x)
            _ = s > 0.0                                   # decision
            t2 = time.perf_counter()
            tp.append(t1 - t0)
            ts.append(t2 - t1)
            te.append(t2 - t0)
        w = L.warmup_frames
        rows.append({"item": f"{name}:preprocessing", "batch": 1, **_stats(np.asarray(tp[w:]), frame_s)})
        rows.append({"item": f"{name}:scoring", "batch": 1, **_stats(np.asarray(ts[w:]), frame_s)})
        rows.append({"item": f"{name}:end_to_end", "batch": 1, **_stats(np.asarray(te[w:]), frame_s)})
        # throughput at batch 64
        b = L.throughput_batch
        prep(rawB[:b])
        score(prep(rawB[:b]))                             # warm-up batch
        t0 = time.perf_counter()
        for j in range(L.throughput_batches):
            _ = score(prep(rawB[j * b:(j + 1) * b])) > 0.0
        dt = time.perf_counter() - t0
        fps = b * L.throughput_batches / dt
        rows.append({"item": f"{name}:end_to_end", "batch": b, "throughput_fps": fps,
                     "time_per_frame_ms": 1e3 / fps, "rtf_median": (1.0 / fps) / frame_s})
        log(f"  {name:16s} e2e median {rows[-2]['median_ms']:.2f} ms (p95 {rows[-2]['p95_ms']:.2f}), "
            f"RTF {rows[-2]['rtf_median']:.3f}; batch {b}: {fps:.0f} frames/s")
    # effect of the scoring optimisations: each network as trained and in its scoring
    # form, timed end to end at batch 1 on the same frames, alternating frame by frame
    # (and alternating which form goes first), so that both see the same conditions
    optimisation = {}
    fast_fns = {name: score for name, (fam, score) in dets.items()}
    for name, slow in pipes.as_trained_detectors().items():
        fast = fast_fns[name]
        tf, ts = [], []
        for i in range(n1):
            order = ((fast, tf), (slow, ts)) if i % 2 == 0 else ((slow, ts), (fast, tf))
            for fn, acc in order:
                t0 = time.perf_counter()
                _ = fn(pipes.spec(raw1[i:i + 1])) > 0.0
                acc.append(time.perf_counter() - t0)
        w = L.warmup_frames
        r_slow = {"item": f"{name}[as trained]:end_to_end", "batch": 1, **_stats(np.asarray(ts[w:]), frame_s)}
        r_fast = {"item": f"{name}[scoring, paired]:end_to_end", "batch": 1, **_stats(np.asarray(tf[w:]), frame_s)}
        rows += [r_slow, r_fast]
        optimisation[name] = {"as_trained_median_ms": r_slow["median_ms"], "scoring_median_ms": r_fast["median_ms"],
                              "as_trained_p95_ms": r_slow["p95_ms"], "scoring_p95_ms": r_fast["p95_ms"],
                              "speedup_median": r_slow["median_ms"] / r_fast["median_ms"]}
        log(f"  {name:16s} as trained {r_slow['median_ms']:.2f} ms, scoring form {r_fast['median_ms']:.2f} ms "
            f"(median, paired): {optimisation[name]['speedup_median']:.2f}x faster")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_dir / "latency.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    info = {"frame_duration_ms": frame_s * 1e3, "torch_threads": torch.get_num_threads(),
            "warmup_frames": L.warmup_frames, "frames_batch1": L.n_frames, "settle_s": L.settle_s,
            "throughput_batch": L.throughput_batch, "throughput_batches": L.throughput_batches,
            "input": capture_path or "quantised complex Gaussian noise (timing does not depend on content)",
            "weights": pipes.weights, "normaliser": pipes.norm_src, "fold": fold, "seed": seed,
            "power_mode": provenance.power_mode(), "power_condition": condition,
            "power_condition_required": require_performance, "optimisation": optimisation, "rows": rows}
    provenance.write_json(out_dir / "latency.json",
                          {**info, "run": provenance.run_record(cfg, synthetic=synthetic_data)})
    return info
