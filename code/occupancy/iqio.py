"""
iqio.py
-------
Raw capture files and their metadata sidecars (design section 1).

File format
    ``<capture_id>.cu8``  interleaved I, Q as uint8 (2 bytes per complex sample),
                          exactly as librtlsdr delivers them.
    ``<capture_id>.json`` metadata sidecar, written last: a capture without a
                          sidecar (or with ``status`` other than "ok") is never
                          indexed.
Complex value of a sample: y = ((I - 127.5) + j (Q - 127.5)) / 127.5, the same
scaling pyrtlsdr uses, so full scale is |Re y|, |Im y| <= 1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import provenance
from .config import CLASSES, GAIN_ROLES, Config

IQ_OFFSET = 127.5
IQ_SCALE = 127.5
RAW_SUFFIX = ".cu8"
META_SUFFIX = ".json"
PARTIAL_SUFFIX = ".partial"
SCHEMA_VERSION = 1


def u8_to_complex(raw: np.ndarray) -> np.ndarray:
    """Interleaved uint8 I/Q (even length) -> complex64 samples."""
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.size % 2:
        raise ValueError("interleaved I/Q needs an even number of bytes")
    f = (raw.astype(np.float32) - np.float32(IQ_OFFSET)) / np.float32(IQ_SCALE)
    return f.view(np.complex64)


def complex_to_u8(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantise complex samples to interleaved uint8 with clipping (synthetic data only).

    Returns the bytes and the number of I/Q values that hit 0 or 255.
    """
    y = np.asarray(y, dtype=np.complex64)
    inter = np.empty(2 * y.size, dtype=np.float32)
    inter[0::2] = y.real
    inter[1::2] = y.imag
    q = np.rint(inter * IQ_SCALE + IQ_OFFSET)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q, int(np.count_nonzero((q == 0) | (q == 255)))


def make_capture_id(session_id: str, cls: str, gain_role: str, t_utc) -> str:
    stamp = t_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"{session_id}_{cls}_{gain_role}_{stamp}"


def capture_paths(directory: Path, capture_id: str) -> tuple[Path, Path]:
    directory = Path(directory)
    return directory / f"{capture_id}{RAW_SUFFIX}", directory / f"{capture_id}{META_SUFFIX}"


def open_raw(path: str | Path) -> np.memmap:
    path = Path(path)
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.memmap(path, dtype=np.uint8, mode="r")


class CaptureStats:
    """Streaming statistics of a raw uint8 I/Q stream (one pass, constant memory).

    clipping fraction = share of the I and Q values (each counted separately)
    that sit at the 8-bit extremes 0 or 255 (design section 1).
    mean power        = mean |y|^2 in dB relative to full scale, before DC removal.
    """

    def __init__(self):
        self.n_values = 0
        self.n_clipped = 0
        self.sum_i = 0.0
        self.sum_q = 0.0
        self.sum_pow = 0.0
        self._sha = hashlib.sha256()

    def update(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.uint8)
        self._sha.update(block.tobytes())
        self.n_values += block.size
        self.n_clipped += int(np.count_nonzero((block == 0) | (block == 255)))
        f = (block.astype(np.float64) - IQ_OFFSET) / IQ_SCALE
        self.sum_i += float(f[0::2].sum())
        self.sum_q += float(f[1::2].sum())
        self.sum_pow += float(np.dot(f, f))

    @property
    def n_samples(self) -> int:
        return self.n_values // 2

    def result(self, clip_flag_fraction: float) -> dict:
        n = max(self.n_samples, 1)
        frac = self.n_clipped / max(self.n_values, 1)
        mean_pow = self.sum_pow / n
        return {
            "n_samples": self.n_samples,
            "clipped_values": self.n_clipped,
            "clipping_fraction": frac,
            "clipping_flagged": bool(frac > clip_flag_fraction),
            "clip_flag_threshold": clip_flag_fraction,
            "mean_power_db": float(10.0 * np.log10(mean_pow)) if mean_pow > 0 else None,
            "mean_power_reference": "dB relative to |y|^2 = 1, y = (u8 - 127.5)/127.5, before DC removal",
            "dc_offset_i": self.sum_i / n,
            "dc_offset_q": self.sum_q / n,
            "sha256": self._sha.hexdigest(),
        }


def capture_metadata(*, cfg: Config, capture_id: str, session_id: str, position_id: str,
                     cls: str, gain_role: str, gain_db: float, centre_freq_hz: float,
                     utc_start: str, utc_end: str, raw_file: str, stats: dict,
                     radio: dict | None = None, antenna: str = "", temperature_c=None,
                     cell_search: dict | None = None, notes: str = "",
                     extra_context: dict | None = None, synthetic: bool = False,
                     wall_clock_s: float | None = None) -> dict:
    """Assemble the sidecar with every field of design section 1."""
    if cls not in CLASSES:
        raise ValueError(f"class must be one of {CLASSES}, got {cls!r}")
    if gain_role not in GAIN_ROLES:
        raise ValueError(f"gain_role must be one of {GAIN_ROLES}, got {gain_role!r}")
    radio_block = {
        "centre_freq_hz": float(centre_freq_hz),
        "sample_rate_hz": float(cfg.capture.sample_rate_hz),
        "gain_db": float(gain_db),
        "gain_mode": "manual",
        "ppm": int(cfg.capture.ppm),
        "bias_tee": "on" if cfg.capture.bias_tee_on else "off",
        "receiver_model": "SYNTHETIC (no receiver)" if synthetic else cfg.capture.receiver_model,
        "receiver_serial": None,
        "antenna": antenna,
    }
    if radio:
        radio_block.update(radio)
    meas = dict(stats)
    meas["temperature_c"] = temperature_c
    if wall_clock_s is not None:
        meas["wall_clock_s"] = wall_clock_s
        if wall_clock_s > 0:
            meas["effective_rate_sps"] = stats["n_samples"] / wall_clock_s
    context = {"cell_search": cell_search or {}, "notes": notes}
    if extra_context:
        context.update(extra_context)
    meta = {
        "kind": "capture",
        "schema_version": SCHEMA_VERSION,
        "synthetic": bool(synthetic),
        "status": "ok",
        "capture_id": capture_id,
        "session_id": session_id,
        "position_id": position_id,
        "class": cls,
        "gain_role": gain_role,
        "utc_start": utc_start,
        "utc_end": utc_end,
        "file": raw_file,
        "format": {"type": "cu8",
                   "description": "interleaved I,Q uint8; y = ((I-127.5) + j(Q-127.5)) / 127.5"},
        "radio": radio_block,
        "measurements": meas,
        "context": context,
        "provenance": {
            "software_version": provenance.software_version(),
            "git": provenance.git_info(),
            "versions": provenance.library_versions(),
        },
    }
    if synthetic:
        meta["WARNING"] = provenance.SYNTHETIC_BANNER
    return meta


def write_capture_array(directory: Path, capture_id: str, raw: np.ndarray, meta: dict) -> Path:
    """Write a complete in-memory capture (used by the synthetic generator)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = capture_paths(directory, capture_id)
    if raw_path.exists() or meta_path.exists():
        raise FileExistsError(f"capture {capture_id} already exists in {directory}")
    partial = raw_path.with_name(raw_path.name + PARTIAL_SUFFIX)
    np.asarray(raw, dtype=np.uint8).tofile(partial)
    partial.replace(raw_path)
    provenance.write_json(meta_path, meta)
    return meta_path
