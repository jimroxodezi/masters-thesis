"""
preprocess.py
-------------
Spectrograms, normalisation and on-disk feature stores (design section 4).

STFT of a DC-removed frame of N = 16384 samples:
    K = 128-point FFT (bin spacing f_s/K = 15 kHz), periodic Hann window of 128
    samples, hop 128 (no overlap) -> 128 columns of 66.7 us;
    S = 10 log10(|X|^2 + 1e-12) in dB;
    FFT-shifted, and stored so that frequency increases from bottom to top
    when the array is drawn as an image (row 0 at the top):
        row r  <->  frequency offset (K/2 - 1 - r) * f_s/K,
    i.e. row 0 = +945 kHz, row 127 = -960 kHz. Columns run forward in time.
    Shape (128 frequency, 128 time).

Normalisation: one global mean mu and standard deviation s of S over all the
training frames of a fold; x = (S - mu)/s is applied unchanged to validation
and test frames. No min-max scaling and no clipping.

Feature store (one directory per capture or injection set, under work/features):
    spec.npy    float16 (n, 128, 128) un-normalised S, opened as a memory map
    frames.npz  frame index and per-frame side information: energy T = mean |y|^2
                of the DC-removed frame (energy detectors, injection powers),
                PSS statistic, best root and lag, plus injection details
    meta.json   provenance and the configuration fingerprint
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.fft as sfft

from . import provenance
from .config import Config
from .dataset import CaptureIndex, CaptureRecord, read_frames
from .pss import PSSCorrelator, correlator_from_config


# ── STFT ──────────────────────────────────────────────────────────────────────

def hann_periodic(K: int) -> np.ndarray:
    n = np.arange(K)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / K)).astype(np.float32)


def stft_db(frames: np.ndarray, cfg: Config) -> np.ndarray:
    """(n, N) complex DC-removed frames -> (n, K, K) float32 spectrograms in dB."""
    p = cfg.preprocess
    frames = np.atleast_2d(np.asarray(frames, dtype=np.complex64))
    n, N = frames.shape
    if N != p.frame_len:
        raise ValueError(f"frames must have {p.frame_len} samples, got {N}")
    K, hop = p.fft_size, p.hop
    cols = (N - K) // hop + 1
    if hop == K:
        segs = frames[:, :cols * K].reshape(n, cols, K)
    else:
        idx = np.arange(cols)[:, None] * hop + np.arange(K)[None, :]
        segs = frames[:, idx]
    X = sfft.fft(segs * hann_periodic(K)[None, None, :], axis=-1)      # (n, cols, K)
    P = X.real.astype(np.float32) ** 2 + X.imag.astype(np.float32) ** 2
    S = 10.0 * np.log10(P + np.float32(p.power_floor))
    S = np.fft.fftshift(S, axes=-1)[..., ::-1]                          # descending frequency
    return np.ascontiguousarray(np.swapaxes(S, 1, 2))                   # (n, K freq, cols time)


def freq_axis_hz(cfg: Config) -> np.ndarray:
    """Frequency offset of each spectrogram row (descending: row 0 is the highest)."""
    K = cfg.preprocess.fft_size
    return (K // 2 - 1 - np.arange(K)) * cfg.bin_spacing_hz


def time_axis_s(cfg: Config) -> np.ndarray:
    """Start time of each spectrogram column within the frame."""
    return np.arange(cfg.n_time_columns) * cfg.preprocess.hop / cfg.capture.sample_rate_hz


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """T = (1/N) sum |y(n)|^2 per frame (float64)."""
    f = np.asarray(frames)
    return (f.real.astype(np.float64) ** 2 + f.imag.astype(np.float64) ** 2).mean(axis=-1)


# ── normaliser ────────────────────────────────────────────────────────────────

class Normalizer:
    """Global scalar mean and standard deviation of S over training frames."""

    def __init__(self, mu: float | None = None, sigma: float | None = None, n_frames: int = 0,
                 sources: list | None = None):
        self.mu, self.sigma, self.n_frames, self.sources = mu, sigma, n_frames, sources or []

    def fit(self, stores, frame_stride: int = 1, chunk: int = 256) -> "Normalizer":
        n_tot, mean, m2 = 0, 0.0, 0.0
        n_frames = 0
        for st in stores:
            idx = np.arange(0, st.n, frame_stride)
            for s in range(0, idx.size, chunk):
                x = np.asarray(st.spec[idx[s:s + chunk]], dtype=np.float64).ravel()
                if x.size == 0:
                    continue
                cm = float(x.mean())
                cm2 = float(((x - cm) ** 2).sum())
                nb = x.size
                delta = cm - mean
                tot = n_tot + nb
                mean += delta * nb / tot
                m2 += cm2 + delta * delta * n_tot * nb / tot          # Chan et al. combination
                n_tot = tot
            n_frames += idx.size
        if n_tot == 0:
            raise ValueError("no training frames to fit the normaliser")
        self.mu = float(mean)
        self.sigma = float(np.sqrt(m2 / n_tot))
        if not self.sigma > 0:
            raise ValueError("training spectrograms have zero variance")
        self.n_frames = n_frames
        self.sources = [st.unit_id for st in stores]
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("normaliser not fitted")
        return ((np.asarray(S, dtype=np.float32) - np.float32(self.mu)) / np.float32(self.sigma))

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "n_frames": self.n_frames,
                "fitted_on": self.sources, "rule": "global mean/std of S over training frames"}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(d["mu"], d["sigma"], d.get("n_frames", 0), d.get("fitted_on", []))

    def save(self, path: Path) -> None:
        provenance.write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls.from_dict(provenance.read_json(path))


# ── feature stores ────────────────────────────────────────────────────────────

class FeatureStore:
    """Read access to one preprocessed unit (a capture or an injection set)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.meta = provenance.read_json(self.dir / "meta.json")
        with np.load(self.dir / "frames.npz") as z:
            self.arrays = {k: z[k] for k in z.files}
        self._spec = None

    @property
    def unit_id(self) -> str:
        return self.meta["unit_id"]

    @property
    def n(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def spec(self) -> np.ndarray:
        if self._spec is None:
            self._spec = np.load(self.dir / "spec.npy", mmap_mode="r")
        return self._spec

    @property
    def energy(self) -> np.ndarray:
        return self.arrays["energy"]

    @property
    def pss(self) -> np.ndarray:
        return self.arrays["pss"]

    def __getitem__(self, key):
        return self.arrays[key]


def feature_fingerprint(cfg: Config) -> str:
    blob = json.dumps({"preprocess": cfg.to_dict()["preprocess"],
                       "pss": cfg.to_dict()["models"]["pss"],
                       "fs": cfg.capture.sample_rate_hz}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def features_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "features"


def store_is_current(directory: Path, fingerprint: str, source_sig: dict) -> bool:
    m = Path(directory) / "meta.json"
    if not m.exists() or not (Path(directory) / "spec.npy").exists():
        return False
    try:
        meta = provenance.read_json(m)
    except (OSError, ValueError):
        return False
    return meta.get("fingerprint") == fingerprint and meta.get("source") == source_sig


def write_store(directory: Path, n: int, cfg: Config, chunks, meta: dict, extra: dict | None = None):
    """Write a feature store from an iterator of (S, T, pss, root, lag) chunks, atomically."""
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    K = cfg.preprocess.fft_size
    spec = np.lib.format.open_memmap(tmp / "spec.npy", mode="w+", dtype=np.float16, shape=(n, K, K))
    energy = np.zeros(n, dtype=np.float64)
    pss = np.zeros(n, dtype=np.float32)
    root = np.zeros(n, dtype=np.int16)
    lag = np.zeros(n, dtype=np.int32)
    pos = 0
    for S, T, rho, r, l in chunks:
        m = S.shape[0]
        spec[pos:pos + m] = S.astype(np.float16)
        energy[pos:pos + m], pss[pos:pos + m], root[pos:pos + m], lag[pos:pos + m] = T, rho, r, l
        pos += m
    if pos != n:
        raise RuntimeError(f"wrote {pos} of {n} frames")
    spec.flush()
    del spec
    arrays = {"frame_index": np.arange(n), "energy": energy, "pss": pss, "pss_root": root, "pss_lag": lag}
    arrays.update(extra or {})
    np.savez(tmp / "frames.npz", **arrays)
    provenance.write_json(tmp / "meta.json", {**meta, "n_frames": n})
    if directory.exists():
        shutil.rmtree(directory)          # derived data only; regenerated from the raw capture
    tmp.rename(directory)
    return FeatureStore(directory)


def process_frames(frames_dc: np.ndarray, cfg: Config, correlator: PSSCorrelator):
    """All per-frame features of DC-removed frames: S (float32), T, PSS (rho, root, lag)."""
    S = stft_db(frames_dc, cfg)
    T = frame_energy(frames_dc)
    rho, root, lag = correlator(frames_dc, return_details=True)
    return S, T, rho, root, lag


def capture_source_signature(record: CaptureRecord) -> dict:
    """Identity of the raw file a feature store was computed from (a changed file is recomputed)."""
    raw_stat = Path(record.raw_path).stat()
    return {"capture_id": record.capture_id, "n_samples": record.n_samples,
            "raw_size": raw_stat.st_size, "raw_mtime_ns": raw_stat.st_mtime_ns}


def preprocess_capture(record: CaptureRecord, cfg: Config, work_dir: Path,
                       correlator: PSSCorrelator | None = None, force: bool = False,
                       log=print) -> FeatureStore | None:
    N = cfg.preprocess.frame_len
    n = record.n_frames(N)
    out = features_dir(work_dir) / record.capture_id
    fp = feature_fingerprint(cfg)
    sig = capture_source_signature(record)
    if n == 0:
        log(f"skip {record.capture_id}: shorter than one frame")
        return None
    if not force and store_is_current(out, fp, sig):
        return FeatureStore(out)
    correlator = correlator or correlator_from_config(cfg)
    chunk = cfg.preprocess.chunk_frames

    def chunks():
        for s in range(0, n, chunk):
            yield process_frames(read_frames(record, N, s, s + chunk), cfg, correlator)

    meta = {"unit_id": record.capture_id, "kind": "capture", "capture_id": record.capture_id,
            "session_id": record.session_id, "position_id": record.position_id, "class": record.cls,
            "gain_role": record.gain_role, "gain_db": record.gain_db, "label": record.label,
            "synthetic": record.synthetic, "utc_start": record.utc_start, "fingerprint": fp,
            "source": sig, "software_version": provenance.software_version()}
    store = write_store(out, n, cfg, chunks(), meta)
    log(f"preprocessed {record.capture_id}: {n} frames")
    return store


def open_capture_store(work_dir: Path, capture_id: str) -> FeatureStore:
    d = features_dir(work_dir) / capture_id
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{capture_id} is not preprocessed (run: python -m occupancy preprocess)")
    return FeatureStore(d)


def preprocess_all(index: CaptureIndex, cfg: Config, work_dir: Path, force: bool = False,
                   log=print) -> dict:
    correlator = correlator_from_config(cfg)
    stores = {}
    for rec in index.records:
        st = preprocess_capture(rec, cfg, work_dir, correlator, force, log)
        if st is not None:
            stores[rec.capture_id] = st
    return stores
