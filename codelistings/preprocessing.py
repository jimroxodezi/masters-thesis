"""
preprocessing.py
----------------
IQ samples → normalised magnitude spectrograms → PyTorch Dataset.

Pipeline
--------
1. Segment the long IQ stream into fixed-length tiles.
2. Compute a two-sided STFT magnitude spectrogram for each tile.
3. Convert to dB scale and pad/crop to a fixed spatial shape.
4. Fit a per-pixel min-max normaliser on the training set.
5. Return a SpectrogramDataset ready for the DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal as sp_signal
from pathlib import Path
import pickle

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_iq(samples: np.ndarray, config: Config = cfg) -> np.ndarray:
    """
    Split a 1-D IQ array into non-overlapping segments.

    Parameters
    ----------
    samples : np.ndarray, complex64, shape (N,)
    config  : Config

    Returns
    -------
    np.ndarray, complex64, shape (M, segment_length)
        M = N // segment_length complete segments.
    """
    L   = config.preprocess.segment_length
    M   = len(samples) // L
    out = samples[: M * L].reshape(M, L)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(segment: np.ndarray, config: Config = cfg) -> np.ndarray:
    """
    Compute a two-sided STFT magnitude spectrogram in dB for one IQ segment.

    Parameters
    ----------
    segment : np.ndarray, complex64, shape (segment_length,)
    config  : Config

    Returns
    -------
    np.ndarray, float32, shape (target_freq_bins, target_time_frames)
    """
    pc = config.preprocess

    _, _, Zxx = sp_signal.stft(
        segment,
        fs=config.sdr.sample_rate,
        window="hann",
        nperseg=pc.n_fft,
        noverlap=pc.n_fft - pc.hop_length,
        return_onesided=False,   # complex IQ → two-sided spectrum
        boundary=None,
        padded=False,
    )
    # Zxx: (n_fft, time_frames) complex

    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)   # (n_fft, T)

    # ── Align frequency dimension ──────────────────────────────────────────
    H, W = mag_db.shape
    tH   = pc.target_freq_bins
    if H > tH:
        mag_db = mag_db[:tH, :]
    elif H < tH:
        mag_db = np.pad(mag_db, ((0, tH - H), (0, 0)))

    # ── Align time dimension ───────────────────────────────────────────────
    tW = pc.target_time_frames
    if W > tW:
        mag_db = mag_db[:, :tW]
    elif W < tW:
        mag_db = np.pad(mag_db, ((0, 0), (0, tW - W)))

    return mag_db.astype(np.float32)   # (target_freq_bins, target_time_frames)


def build_spectrograms(segments: np.ndarray, config: Config = cfg) -> np.ndarray:
    """
    Convert an array of IQ segments to a stack of spectrograms.

    Parameters
    ----------
    segments : np.ndarray, complex64, shape (M, segment_length)

    Returns
    -------
    np.ndarray, float32, shape (M, target_freq_bins, target_time_frames)
    """
    specs = [iq_to_spectrogram(seg, config) for seg in segments]
    return np.stack(specs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser  (fitted on training data; applied consistently to test data)
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Per-pixel min-max normaliser that maps spectrogram values to [0, 1].

    Fit once on the training split, then call transform() on any data.
    """

    def __init__(self):
        self.min_: float | None = None
        self.max_: float | None = None

    def fit(self, spectrograms: np.ndarray) -> "Normalizer":
        """
        Compute global min/max from a batch of spectrograms.

        Parameters
        ----------
        spectrograms : np.ndarray, shape (M, H, W)
        """
        self.min_ = float(spectrograms.min())
        self.max_ = float(spectrograms.max())
        print(f"[Normalizer] fit  min={self.min_:.2f} dB  max={self.max_:.2f} dB")
        return self

    def transform(self, spectrograms: np.ndarray) -> np.ndarray:
        """Scale spectrograms to [0, 1]."""
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        denom = self.max_ - self.min_
        if denom == 0:
            return np.zeros_like(spectrograms)
        return ((spectrograms - self.min_) / denom).clip(0.0, 1.0).astype(np.float32)

    def inverse_transform(self, spectrograms: np.ndarray) -> np.ndarray:
        """Undo the scaling."""
        return spectrograms * (self.max_ - self.min_) + self.min_

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"min": self.min_, "max": self.max_}, f)
        print(f"[Normalizer] saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        norm = cls()
        norm.min_ = d["min"]
        norm.max_ = d["max"]
        print(f"[Normalizer] loaded ← {path}")
        return norm


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Wraps a stack of normalised spectrograms as a PyTorch Dataset.

    Each item is a float32 tensor of shape (1, H, W) — channel-first
    convention expected by Conv2d.
    """

    def __init__(self, spectrograms: np.ndarray):
        """
        Parameters
        ----------
        spectrograms : np.ndarray, float32, shape (N, H, W)
            Values should already be normalised to [0, 1].
        """
        # (N, H, W) → (N, 1, H, W)
        self.data = torch.from_numpy(spectrograms).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    @property
    def shape(self) -> torch.Size:
        return self.data.shape


# ─────────────────────────────────────────────────────────────────────────────
# High-level helper: raw IQ → train/val DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataloaders(
    samples: np.ndarray,
    config: Config = cfg,
    normalizer: Normalizer | None = None,
) -> tuple[DataLoader, DataLoader, Normalizer]:
    """
    End-to-end: IQ samples → segmented → spectrograms → normalised → DataLoaders.

    Parameters
    ----------
    samples    : raw IQ, complex64, shape (N,)
    config     : Config
    normalizer : pre-fitted Normalizer; if None, one is fitted on the training split

    Returns
    -------
    (train_loader, val_loader, normalizer)
    """
    tc = config.train

    # 1. Segment
    segments = segment_iq(samples, config)
    print(f"[preprocess] {len(segments)} segments of {config.preprocess.segment_length} samples")

    # 2. Spectrograms
    specs = build_spectrograms(segments, config)
    print(f"[preprocess] spectrogram stack shape: {specs.shape}")

    # 3. Split before fitting normaliser to avoid data leakage
    n_val   = int(len(specs) * tc.val_split)
    n_train = len(specs) - n_val
    train_specs = specs[:n_train]
    val_specs   = specs[n_train:]

    # 4. Fit / apply normaliser
    if normalizer is None:
        normalizer = Normalizer().fit(train_specs)

    train_norm = normalizer.transform(train_specs)
    val_norm   = normalizer.transform(val_specs)

    # 5. Datasets → DataLoaders
    train_ds = SpectrogramDataset(train_norm)
    val_ds   = SpectrogramDataset(val_norm)

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[preprocess] train={len(train_ds)}  val={len(val_ds)}  "
          f"spec_shape=(1, {specs.shape[1]}, {specs.shape[2]})")

    return train_loader, val_loader, normalizer
