"""
config.py
---------
Central configuration for the LTE anomaly detection pipeline.
All hyperparameters live here; no magic numbers elsewhere.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── SDR Hardware ──────────────────────────────────────────────────────────────

@dataclass
class SDRConfig:
    # Tune to a local LTE downlink carrier.
    # RTL-SDR Blog V4 (R828D) range: ~24 MHz – 1766 MHz.
    # Common LTE DL bands within range:
    #   Band 28 DL 758–803 MHz  (700 MHz APT, used in NG/Africa)
    #   Band 20 DL 791–821 MHz  (800 MHz EU)
    # Band 3 DL (1805–1880 MHz) is above the R828D limit and cannot be tuned.
    center_freq: float   = 796e6     # Hz  — adjust to your local carrier
    max_freq: float      = 1766e6    # Hz  — R828D tuner limit, enforced on --freq
    sample_rate: float   = 2.048e6   # Hz  — 2.048 MSPS (standard RTL-SDR)
    gain: str            = "auto"
    capture_duration: float = 30.0   # seconds of IQ data per capture session
    # Samples per USB transfer. Must stay small: Linux caps usbfs buffers at
    # 16 MB by default, so a long capture cannot be one synchronous read.
    read_block_samples: int = 131072  # 262 144 bytes (librtlsdr's default)


# ── Signal Pre-processing ─────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    segment_length: int  = 2048   # IQ samples per spectrogram tile
    n_fft: int           = 128    # FFT points  → 128 freq bins (two-sided)
    hop_length: int      = 32     # STFT step   → (2048-128)//32+1 = 61 frames → pad to 64
    target_freq_bins: int = 128   # keep all freq bins (= n_fft, two-sided)
    target_time_frames: int = 64  # pad/crop time axis to this width
    # NOTE: both dims must be divisible by 8 for the 3-stage encoder to work.


# ── Autoencoder Architecture ──────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Encoder channel progression: 1 → 16 → 32 → 64
    enc_channels: list = field(default_factory=lambda: [16, 32, 64])
    # Reconstruction loss
    loss: str = "mse"        # "mse" | "mae"


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int              = 100
    batch_size: int          = 64
    learning_rate: float     = 1e-3
    weight_decay: float      = 1e-5
    val_split: float         = 0.15   # fraction of data held out for validation
    patience: int            = 15     # early-stopping patience (epochs)
    min_delta: float         = 1e-6   # smallest val-loss drop that counts as improvement
    scheduler_factor: float  = 0.5    # LR reduction factor on plateau
    scheduler_patience: int  = 7
    grad_clip_norm: float    = 1.0    # max L2 norm of the gradient vector
    seed: int                = 42     # weight initialisation and batch shuffling


# ── Anomaly Detection ─────────────────────────────────────────────────────────

@dataclass
class AnomalyConfig:
    # Threshold = this percentile of the training reconstruction errors.
    threshold_percentile: float = 99.0   # percentile of training-set errors


# ── Top-level Config ──────────────────────────────────────────────────────────

@dataclass
class Config:
    sdr:       SDRConfig       = field(default_factory=SDRConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    anomaly:   AnomalyConfig   = field(default_factory=AnomalyConfig)

    data_dir:       Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    results_dir:    Path = Path("results")

    def __post_init__(self):
        for d in (self.data_dir, self.checkpoint_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton used as default throughout the project
cfg = Config()
