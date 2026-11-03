"""
synthetic.py
------------
Synthetic LTE-like test data, for exercising the code without a dongle.

NOTHING generated here may be used for a thesis result. Every synthetic
capture carries ``"synthetic": true`` in its sidecar, is written under the
synthetic data directory (``data_synthetic/`` by default; writing under the
real data directory is refused), and dataset.py refuses synthetic captures
unless ``allow_synthetic=True`` (CLI: ``--allow-synthetic``).

Signal model (deliberately simple)
    OFDM with the 1.4 MHz LTE numerology at 1.92 MS/s: 128-point IFFT, 15 kHz
    subcarriers, normal cyclic prefix (10 samples for the first symbol of a
    0.5 ms slot, 9 for the other six), 20 slots per 10 ms radio frame.
    * C_centre: window centred on the carrier. PSS (one of the three
      Zadoff-Chu roots) on subcarriers +-1..31 of the last symbol of slots 0
      and 10, an SSS-like +-1 sequence on the symbol before it, five empty
      subcarriers either side, CRS-like pilots on every 6th subcarrier of
      symbols 0 and 4, and QPSK data on randomly allocated resource blocks.
    * C_offset: the same carrier seen off-centre: pilots and data only.
    * B: antenna noise plus receiver noise. A: receiver noise only.
    Receiver: y = 10^((g - g_ref)/20) (s + n_ant + n_fe) + n_adc + DC,
    quantised to 8 bits with clipping.
"""

from __future__ import annotations

import datetime as _dt
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import provenance
from .config import CLASSES, Config
from .iqio import (PARTIAL_SUFFIX, CaptureStats, capture_metadata, capture_paths,
                   complex_to_u8, make_capture_id)
from .pss import pss_freq_vector

FFT = 128
SLOT_SYMBOLS = 7
CP_FIRST, CP_OTHER = 10, 9
SLOT_SAMPLES = (FFT + CP_FIRST) + 6 * (FFT + CP_OTHER)       # 960 at 1.92 MS/s
SLOTS_PER_FRAME = 20
G_REF_DB = 30.0
CHUNK_SAMPLES = 1_920_000                                    # 1 s per generation step


@dataclass
class SessionSignalParams:
    """Per-session random draw of levels, so that sessions differ."""
    pss_root: int = 29
    noise_fe_rms: float = 0.020         # front-end noise (full-scale units at g_ref)
    noise_ant_rms: float = 0.012        # antenna noise (B and C only)
    noise_adc_rms: float = 0.004        # gain-independent ADC noise
    signal_rms: float = 0.12            # LTE signal at g_ref
    load: float = 0.5                   # fraction of resource blocks carrying data
    dc_i: float = 0.004                 # receiver DC offset (full-scale units)
    dc_q: float = -0.003

    @classmethod
    def draw(cls, rng: np.random.Generator) -> "SessionSignalParams":
        def db(spread):
            return 10.0 ** (rng.uniform(-spread, spread) / 20.0)
        return cls(pss_root=int(rng.choice([25, 29, 34])),
                   noise_fe_rms=0.020 * db(1.0), noise_ant_rms=0.012 * db(1.0),
                   signal_rms=0.12 * db(3.0), load=float(rng.uniform(0.3, 0.8)),
                   dc_i=float(rng.normal(0, 0.003)), dc_q=float(rng.normal(0, 0.003)))


def _sss_like(root: int) -> np.ndarray:
    """A fixed +-1 sequence of length 62 (SSS-like structure, not the real SSS)."""
    return np.random.default_rng(1000 + root).choice([-1.0, 1.0], size=62)


def _qpsk(rng, shape) -> np.ndarray:
    return ((rng.integers(0, 2, shape) * 2 - 1) + 1j * (rng.integers(0, 2, shape) * 2 - 1)) / np.sqrt(2)


def lte_like_slots(n_slots: int, kind: str, rng: np.random.Generator, pss_root: int = 29,
                   load: float = 0.5, first_slot: int = 0, max_k: int = 60) -> np.ndarray:
    """Time-domain OFDM for n_slots consecutive slots (unit power per used subcarrier).

    kind: "centre" (with PSS/SSS) or "offset" (no sync signals).
    first_slot: index of the first slot within the radio frame (0..19).
    """
    if kind not in ("centre", "offset"):
        raise ValueError(kind)
    k_all = np.arange(-FFT // 2, FFT // 2)
    used = np.abs(k_all) <= max_k
    if kind == "centre":
        used &= k_all != 0                          # carrier DC is unused
    kidx = k_all % FFT                              # FFT position of offset k
    grid = np.zeros((n_slots, SLOT_SYMBOLS, FFT), dtype=np.complex128)
    rb_of_k = (k_all + 64) // 12                    # resource blocks of 12 subcarriers
    alloc = rng.random((n_slots, int(rb_of_k.max()) + 1)) < load
    mask = alloc[:, rb_of_k] & used[None, :]       # (slots, 128) in k order
    data = _qpsk(rng, (n_slots, SLOT_SYMBOLS, FFT))
    grid[:, :, kidx] = np.where(mask[:, None, :], data, 0)
    pilot_k = used & (k_all % 6 == 0)               # CRS-like pilots, present at any load
    pilots = _qpsk(rng, (n_slots, 2, int(pilot_k.sum())))
    for j, sym in enumerate((0, 4)):
        grid[:, sym, kidx[pilot_k]] = pilots[:, j]
    if kind == "centre":
        slot_in_frame = (first_slot + np.arange(n_slots)) % SLOTS_PER_FRAME
        central = kidx[np.abs(k_all) <= 36]         # central 6 RBs incl. 5 empty subcarriers each side
        pss_vec = pss_freq_vector(pss_root, FFT)
        sss_vec = np.zeros(FFT, dtype=np.complex128)
        k_sync = np.concatenate([np.arange(-31, 0), np.arange(1, 32)])
        sss_vec[k_sync % FFT] = _sss_like(pss_root)
        for s in np.flatnonzero(np.isin(slot_in_frame, (0, 10))):
            grid[s, 6, central] = pss_vec[central]  # PSS: last symbol of slots 0 and 10
            grid[s, 5, central] = sss_vec[central]  # SSS-like: the symbol before it
    t = np.fft.ifft(grid, axis=2) * np.sqrt(FFT)
    first = np.concatenate([t[:, 0, -CP_FIRST:], t[:, 0, :]], axis=1)
    rest = np.concatenate([t[:, 1:, -CP_OTHER:], t[:, 1:, :]], axis=2).reshape(n_slots, -1)
    return np.concatenate([first, rest], axis=1).reshape(-1)


class LTELikeSource:
    """Continuous LTE-like stream with unit average power, starting at a random
    point of the radio frame."""

    def __init__(self, kind: str, rng: np.random.Generator, pss_root: int = 29, load: float = 0.5):
        self.kind, self.rng, self.pss_root, self.load = kind, rng, pss_root, load
        start = int(rng.integers(0, SLOT_SAMPLES * SLOTS_PER_FRAME))
        self.slot = start // SLOT_SAMPLES
        first = lte_like_slots(1, kind, rng, pss_root, load, self.slot)
        self.slot += 1
        self.buf = first[start % SLOT_SAMPLES:]
        self.scale = None

    def read(self, n: int) -> np.ndarray:
        parts, have = [self.buf], self.buf.size
        while have < n:
            k = min(2000, (n - have) // SLOT_SAMPLES + 1)
            more = lte_like_slots(k, self.kind, self.rng, self.pss_root, self.load,
                                  self.slot % SLOTS_PER_FRAME)
            self.slot += k
            parts.append(more)
            have += more.size
        allbuf = np.concatenate(parts)
        out, self.buf = allbuf[:n], allbuf[n:]
        if self.scale is None:
            p = float(np.mean(np.abs(out) ** 2))
            self.scale = 1.0 / np.sqrt(p) if p > 0 else 1.0
        return (out * self.scale).astype(np.complex64)


def lte_like_signal(n_samples: int, kind: str, rng: np.random.Generator, pss_root: int = 29,
                    load: float = 0.5) -> np.ndarray:
    return LTELikeSource(kind, rng, pss_root, load).read(n_samples)


def _cgauss(rng, n, rms):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (rms / np.sqrt(2))


class SyntheticReceiver:
    """Chunked synthetic baseband for one class and gain (full-scale units)."""

    def __init__(self, cls: str, gain_db: float, rng: np.random.Generator,
                 params: SessionSignalParams):
        if cls not in CLASSES:
            raise ValueError(cls)
        self.cls, self.rng, self.p = cls, rng, params
        self.a = 10.0 ** ((gain_db - G_REF_DB) / 20.0)
        kind = {"C_centre": "centre", "C_offset": "offset"}.get(cls)
        self.lte = LTELikeSource(kind, rng, params.pss_root, params.load) if kind else None

    def read(self, n: int) -> np.ndarray:
        p, rng = self.p, self.rng
        x = _cgauss(rng, n, p.noise_fe_rms)
        if self.cls != "A":
            x = x + _cgauss(rng, n, p.noise_ant_rms)
        if self.lte is not None:
            x = x + p.signal_rms * self.lte.read(n)
        y = self.a * x + _cgauss(rng, n, p.noise_adc_rms) + (p.dc_i + 1j * p.dc_q)
        return y.astype(np.complex64)

    def read_u8(self, n: int) -> np.ndarray:
        return complex_to_u8(self.read(n))[0]


def synthetic_baseband(cls: str, gain_db: float, n_samples: int, rng: np.random.Generator,
                       params: SessionSignalParams) -> np.ndarray:
    return SyntheticReceiver(cls, gain_db, rng, params).read(n_samples)


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _refuse_real_dir(cfg: Config, out_dir: Path) -> None:
    real = cfg.data_dir(synthetic=False).resolve()
    out = Path(out_dir).resolve()
    if out == real or real in out.parents:
        raise PermissionError(f"refusing to write synthetic data under the real data directory {real}")


def write_synthetic_capture(cfg: Config, out_dir: Path, *, session_id: str, position_id: str,
                            cls: str, gain_role: str, gain_db: float, centre_freq_hz: float,
                            n_samples: int, params: SessionSignalParams, t_start: _dt.datetime,
                            seed: int, notes: str = "", cell_search: dict | None = None,
                            extra_context: dict | None = None, antenna: str | None = None) -> dict:
    """Generate and write one synthetic capture (chunked) with a complete sidecar."""
    _refuse_real_dir(cfg, out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_id = make_capture_id(session_id, cls, gain_role, t_start)
    raw_path, meta_path = capture_paths(out_dir, capture_id)
    if raw_path.exists() or meta_path.exists():
        raise FileExistsError(f"capture {capture_id} already exists in {out_dir}")
    partial = raw_path.with_name(raw_path.name + PARTIAL_SUFFIX)
    rx = SyntheticReceiver(cls, gain_db, np.random.default_rng(seed), params)
    stats = CaptureStats()
    with open(partial, "wb") as f:
        done = 0
        while done < n_samples:
            n = min(CHUNK_SAMPLES, n_samples - done)
            raw = rx.read_u8(n)
            f.write(raw.tobytes())
            stats.update(raw)
            done += n
    partial.replace(raw_path)
    duration = n_samples / cfg.capture.sample_rate_hz
    meta = capture_metadata(
        cfg=cfg, capture_id=capture_id, session_id=session_id, position_id=position_id,
        cls=cls, gain_role=gain_role, gain_db=gain_db, centre_freq_hz=centre_freq_hz,
        utc_start=provenance.utc_iso(t_start),
        utc_end=provenance.utc_iso(t_start + _dt.timedelta(seconds=duration)),
        raw_file=raw_path.name, stats=stats.result(cfg.capture.clip_flag_fraction),
        radio={"bias_tee_control": "synthetic", "synthetic_params": asdict(params)},
        antenna=antenna if antenna is not None else
        ("synthetic" if cls != "A" else cfg.capture.terminator_label),
        cell_search=cell_search, notes=notes or "synthetic test capture",
        extra_context=extra_context, synthetic=True)
    provenance.write_json(meta_path, meta)
    return meta


def generate_synthetic_dataset(cfg: Config, root: Path | None = None, n_sessions: int = 8,
                               frames_per_capture: int = 4, seed: int = 0,
                               positions=("P1", "P2"), g1_db: float = 30.0, g2_db: float = 36.0,
                               centre_freq_hz: float = 796e6, b_freq_hz: float = 783e6,
                               offset_hz: float = 2.52e6, skip_c_offset_sessions=()) -> list[dict]:
    """Write n_sessions synthetic sessions (8 captures each) under root/raw/<session>/."""
    root = Path(root) if root is not None else cfg.data_dir(synthetic=True)
    n_samples = frames_per_capture * cfg.preprocess.frame_len
    t0 = _dt.datetime(2026, 1, 1, 8, 0, 0, tzinfo=_dt.timezone.utc)
    metas = []
    for i in range(n_sessions):
        sid = f"S{i + 1:02d}"
        pos = positions[i % len(positions)]
        params = SessionSignalParams.draw(np.random.default_rng(stable_seed(seed, sid, "params")))
        t = t0 + _dt.timedelta(hours=12 * i)
        for cls in cfg.capture.session_order:
            if cls == "C_offset" and sid in skip_c_offset_sessions:
                continue
            freq = {"A": b_freq_hz, "B": b_freq_hz, "C_centre": centre_freq_hz,
                    "C_offset": centre_freq_hz + offset_hz}[cls]
            for role, g in (("g1", g1_db), ("g2", g2_db)):
                t = t + _dt.timedelta(seconds=60)
                metas.append(write_synthetic_capture(
                    cfg, root / "raw" / sid, session_id=sid, position_id=pos, cls=cls,
                    gain_role=role, gain_db=g, centre_freq_hz=freq, n_samples=n_samples,
                    params=params, t_start=t, seed=stable_seed(seed, sid, cls, role)))
    return metas


def synthetic_frame_source(cfg: Config, cls: str, gain_db: float, seed: int = 0,
                           params: SessionSignalParams | None = None):
    """Endless generator of raw uint8 blocks of N samples (demo and latency tests)."""
    rx = SyntheticReceiver(cls, gain_db, np.random.default_rng(seed), params or SessionSignalParams())
    n = cfg.preprocess.frame_len
    while True:
        yield rx.read_u8(n)
