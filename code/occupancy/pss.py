"""
pss.py
------
PSS correlator (design section 7, detector 7).

Templates
    d_u(n) = exp(-j pi u n (n+1) / 63)          n = 0 ... 30
    d_u(n) = exp(-j pi u (n+1) (n+2) / 63)      n = 31 ... 61
    for the roots u in {25, 29, 34} (TS 36.211 Table 6.11.1.1-1).
    n = 0..30 -> subcarrier offsets -31..-1, n = 31..61 -> +1..+31, DC unused.
    A 128-point IFFT gives a 128-sample template at 1.92 MS/s (no cyclic prefix).

Statistic
    rho(n) = |sum_m y(n+m) p*(m)|^2 / (sum_m |y(n+m)|^2 * sum_m |p(m)|^2),
    maximised over the three roots and over all lags n = 0 ... N-128 at which the
    template lies entirely inside the frame ("valid" lags; partial overlaps at the
    frame edges are not used because their normalisation is not comparable).
    By Cauchy-Schwarz 0 <= rho <= 1. The cross-correlation is computed with FFTs.

Nothing is decoded: the statistic only tests for the three known sequences.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft

ZC_LENGTH = 63  # Zadoff-Chu length of the PSS (TS 36.211 clause 6.11.1.1)


def zadoff_chu_pss(u: int, seq_len: int = 62) -> np.ndarray:
    """The length-62 PSS sequence d_u(n) of TS 36.211 clause 6.11.1.1."""
    if seq_len != 62:
        raise ValueError("the LTE PSS has 62 elements")
    n = np.arange(seq_len, dtype=np.float64)
    d = np.empty(seq_len, dtype=np.complex128)
    lo, hi = n[:31], n[31:]
    d[:31] = np.exp(-1j * np.pi * u * lo * (lo + 1) / ZC_LENGTH)
    d[31:] = np.exp(-1j * np.pi * u * (hi + 1) * (hi + 2) / ZC_LENGTH)
    return d


def pss_subcarriers(seq_len: int = 62) -> np.ndarray:
    """Subcarrier offset of each element: -31..-1 then +1..+31 (DC unused)."""
    half = seq_len // 2
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


def pss_freq_vector(u: int, fft_size: int = 128, seq_len: int = 62) -> np.ndarray:
    """FFT-ordered frequency-domain vector with d_u on its subcarriers."""
    k = pss_subcarriers(seq_len)
    if np.max(np.abs(k)) >= fft_size // 2:
        raise ValueError("PSS does not fit in the FFT")
    vec = np.zeros(fft_size, dtype=np.complex128)
    vec[k % fft_size] = zadoff_chu_pss(u, seq_len)
    return vec


def pss_templates(roots=(25, 29, 34), fft_size: int = 128, seq_len: int = 62) -> np.ndarray:
    """Time-domain templates, shape (len(roots), fft_size), complex128."""
    return np.stack([np.fft.ifft(pss_freq_vector(u, fft_size, seq_len)) for u in roots])


class PSSCorrelator:
    """Batch PSS statistic over DC-removed frames."""

    def __init__(self, roots=(25, 29, 34), fft_size: int = 128, seq_len: int = 62,
                 frame_len: int = 16384, chunk_frames: int = 32):
        self.roots = tuple(int(r) for r in roots)
        self.fft_size = fft_size
        self.frame_len = frame_len
        self.chunk = max(1, int(chunk_frames))
        tpl = pss_templates(self.roots, fft_size, seq_len)
        self.template_energy = np.sum(np.abs(tpl) ** 2, axis=1)          # (R,)
        # conj(FFT) of the zero-padded templates, for circular correlation of length N.
        # For lags 0..N-K the circular and linear correlations coincide.
        self._tpl_fft_conj = np.conj(sfft.fft(tpl, n=frame_len, axis=1)).astype(np.complex64)
        self.n_lags = frame_len - fft_size + 1

    def __call__(self, frames: np.ndarray, return_details: bool = False):
        frames = np.atleast_2d(frames)
        if frames.shape[1] != self.frame_len:
            raise ValueError(f"frames must have {self.frame_len} samples")
        n = frames.shape[0]
        best = np.zeros(n, dtype=np.float64)
        best_root = np.zeros(n, dtype=np.int64)
        best_lag = np.zeros(n, dtype=np.int64)
        K, L = self.fft_size, self.n_lags
        tpl_e = self.template_energy.astype(np.float32)[None, :, None]
        for s in range(0, n, self.chunk):
            y = np.asarray(frames[s:s + self.chunk], dtype=np.complex64)
            Y = sfft.fft(y, axis=1)                                            # (c, N)
            corr = sfft.ifft(Y[:, None, :] * self._tpl_fft_conj[None], axis=2)[:, :, :L]
            num = corr.real ** 2 + corr.imag ** 2                               # float32
            p = y.real.astype(np.float64) ** 2 + y.imag.astype(np.float64) ** 2
            cs = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
            win_energy = (cs[:, K:K + L] - cs[:, :L]).astype(np.float32)        # (c, L)
            # an all-zero 128-sample window cannot carry a PSS: give it rho = 0
            den = np.maximum(win_energy, np.float32(1e-30))[:, None, :] * tpl_e
            rho = np.divide(num, den, out=num)
            if (win_energy <= 0).any():
                rho[np.broadcast_to(win_energy[:, None, :] <= 0, rho.shape)] = 0.0
            flat = rho.reshape(rho.shape[0], -1)
            idx = np.argmax(flat, axis=1)
            best[s:s + self.chunk] = flat[np.arange(flat.shape[0]), idx]
            best_root[s:s + self.chunk] = idx // L
            best_lag[s:s + self.chunk] = idx % L
        best = np.clip(best, 0.0, 1.0)   # rounding can exceed 1 by ~1e-7
        if return_details:
            return best, np.asarray(self.roots)[best_root], best_lag
        return best


def correlator_from_config(cfg) -> PSSCorrelator:
    p = cfg.models.pss
    return PSSCorrelator(roots=p.roots, fft_size=p.fft_size, seq_len=p.seq_len,
                         frame_len=cfg.preprocess.frame_len, chunk_frames=p.chunk_frames)
