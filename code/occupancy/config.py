"""
config.py
---------
Every setting of the experiment lives here, and nowhere else.

The single source of truth is ``notes/method-design.md``. Each value carries
the label it has there:

* ``[fixed]``   follows from an approved decision or from the standard;
* ``[default]`` a starting value that may be tuned on training/validation data;
* ``[field]``   measured or chosen during data collection (None until then);
* ``[impl]``    an implementation detail that the design does not fix.

A run can override any value with a JSON file (``--config my.json``) that has
the same nested structure, for example ``{"train": {"max_epochs": 50}}``.
Unknown keys are rejected so that a typo cannot silently fall back to a default.

``Config.training_fingerprint()`` identifies the settings a trained model
depends on (preprocessing, splits, the learned models and the training
settings), leaving out where and how fast it runs (paths, device, threads).
It is used to check that models trained on another machine (Google Colab)
match the configuration on the laptop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# code/ directory: relative paths in PathsConfig are resolved against it.
CODE_DIR = Path(__file__).resolve().parents[1]

CLASSES = ("A", "B", "C_centre", "C_offset")
GAIN_ROLES = ("g1", "g2")


# ── Paths ─────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    data_dir: str = "data"                       # [impl] real captures and surveys
    synthetic_data_dir: str = "data_synthetic"   # [impl] synthetic test captures only
    work_dir: str = "work"                       # [impl] derived data, models, results
    synthetic_work_dir: str = "work_synthetic"   # [impl] everything derived from synthetic data

    @staticmethod
    def resolve(p: str | Path) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (CODE_DIR / path)


# ── Section 1: receiver and capture ───────────────────────────────────────────

@dataclass
class CaptureConfig:
    receiver_model: str = "RTL-SDR Blog V4"      # [fixed]
    expected_product_substring: str = "V4"       # [impl] EEPROM product string check ("Blog V4")
    expected_tuner: str = "R828D"                # [fixed] tuner of the V4
    sample_rate_hz: float = 1_920_000.0          # [fixed, D2] 30.72 MHz / 16
    ppm: int = 0                                 # [default] frequency correction
    duration_s: float = 30.0                     # [default] per capture
    bias_tee_on: bool = False                    # [fixed] off for every capture
    clip_flag_fraction: float = 0.001            # [default] flag above 0.1 % clipped I/Q values
    max_centre_freq_hz: float = 1_766_000_000.0  # [fixed] R828D upper tuning limit (about 1.766 GHz)
    min_centre_freq_hz: float = 24_000_000.0     # [impl] lower limit of the normal (non-HF) path
    read_block_samples: int = 131_072            # [impl] USB block (usbfs caps buffers at 16 MB)
    settle_discard_samples: int = 131_072        # [impl] discarded after streaming starts (68 ms)
    g1_db: float | None = 29.7                   # [field] primary gain (gain trials, 5 Nov 2026)
    g2_db: float | None = 19.7                   # [field] gain-change test gain (10 dB below g1)
    antenna: str = ""                            # [field] antenna description
    terminator_label: str = "50 ohm terminator"  # [fixed] Class A input
    session_order: tuple = CLASSES               # [fixed] A, B, C_centre, C_offset
    gain_order: tuple = GAIN_ROLES               # [impl] within a class: g1 then g2
    # Session times (design section 3; author, 4 November 2026)
    utc_offset_h: float = 1.0                    # [fixed] local time is West Africa Time, UTC+1 (no daylight saving)
    day_start_local: str = "07:00"               # [fixed] day sessions and surveys start
    night_start_local: str = "22:00"             # [fixed] night sessions and surveys start
    schedule_tolerance_h: float = 2.0            # [impl] a start within this of a scheduled time gets its label
    # C_offset window placement (design section 3)
    rb_width_hz: float = 180_000.0               # [fixed] 12 subcarriers x 15 kHz
    central_rbs_half_width_hz: float = 540_000.0 # [fixed] central 6 RBs span +-0.54 MHz
    # TS 36.101 Table 5.6-1: channel bandwidth (MHz) -> N_RB
    nrb_by_channel_bw_mhz: dict = field(default_factory=lambda: {
        "1.4": 6, "3": 15, "5": 25, "10": 50, "15": 75, "20": 100})
    min_free_disk_factor: float = 2.0            # [impl] free space must exceed 2x the capture size


# ── Section 2: band survey ────────────────────────────────────────────────────

@dataclass
class SurveyConfig:
    rtl_power: str = "rtl_power"                 # [impl] executable (RTL-SDR Blog build)
    bin_hz: float = 25_000.0                     # [impl] maximum bin size passed to -f
    integration_s: int = 10                      # [impl] -i
    duration: str = "5m"                         # [impl] -e exit timer
    crop_fraction: float = 0.2                   # [impl] -c, discards hop edges
    window: str = "hamming"                      # [impl] -w (rtl_power default is rectangle; "" keeps it)
    plateau_threshold_db: float = 6.0            # [impl] visual aid: level above the noise floor
    plateau_min_width_hz: float = 1_000_000.0    # [impl] visual aid: narrower bumps are ignored
    b_window_margin_hz: float = 250_000.0        # [impl] keep a suggested B window this far from any signal
    b_window_quiet_mean_db: float = 3.0          # [impl] a bin is quiet if its mean is at most this above the floor
    b_window_quiet_max_db: float = 6.0           # [impl] ... and its maximum over the sweeps at most this
    hop_spike_halfwidth_bins: int = 3            # [impl] the dongle's own spike at each hop centre: ignored


# ── Section 4: frames and preprocessing ───────────────────────────────────────

@dataclass
class PreprocessConfig:
    frame_len: int = 16_384                      # [fixed] N = 8.533 ms at 1.92 MS/s
    fft_size: int = 128                          # [fixed] K, 15 kHz bins
    hop: int = 128                               # [fixed] no overlap -> 128 columns
    window: str = "hann_periodic"                # [fixed]
    power_floor: float = 1e-12                   # [fixed] S = 10 log10(|X|^2 + 1e-12)
    store_dtype: str = "float16"                 # [fixed]
    chunk_frames: int = 128                      # [impl] frames processed at once


# ── Section 5: splits ─────────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    n_folds: int = 4                             # [fixed]
    n_val_sessions: int = 2                      # [fixed]
    min_sessions: int = 8                        # [impl] smallest number the rotation supports
    design_min_sessions: int = 12                # [D4] fewer sessions -> warning
    train_class: str = "B"                       # [D3]
    primary_gain: str = "g1"                     # [fixed]
    gain_test_gain: str = "g2"                   # [fixed]


# ── Section 6: controlled-SNR injection ───────────────────────────────────────

@dataclass
class InjectionConfig:
    snr_min_db: float = -20.0                    # [D5]
    snr_max_db: float = 10.0                     # [D5]
    snr_step_db: float = 2.0                     # [D5] -> 16 levels
    mixtures_per_level: int = 500                # [default] per level per session (and gain, source)
    admissible_margin_db: float = 10.0           # [default] natural SNR >= target + 10 dB
    low_snr_max_db: float = -4.0                 # [fixed] H1 low-SNR levels: SNR <= -4 dB
    sources: tuple = ("C_centre", "C_offset")    # [fixed] kept separate
    gains: tuple = GAIN_ROLES                    # [impl] g1 for the main test, g2 for H2
    seed: int = 20261102                         # [impl] base seed of the mixture draws


# ── Section 7: detectors ──────────────────────────────────────────────────────

@dataclass
class ConvAEConfig:
    channels: tuple = (16, 32, 64)               # [fixed] as in the author's original model
    latent_channels: int = 8                     # [default] c_lat -> latent 8x16x16
    upsample_mode: str = "bilinear"              # [impl] as in the author's original model


@dataclass
class FCAEConfig:
    hidden: int = 256                            # [default]
    latent: int = 128                            # [default]


@dataclass
class VAEConfig:
    beta_kl: float = 1.0                         # [default]


@dataclass
class PERConfig:                                 # Tian et al. (2022), eq. 10
    gamma: int = 3                               # [default] pooling size
    alpha: float = 0.05                          # [default] background level on min-max scaled X
    beta: float = 2.0                            # [default] background weight
    xi: float = 90.0                             # [default] background percentile
    eta: float = 99.0                            # [default] signal-region percentile


@dataclass
class PCAConfig:
    variance: float = 0.95                       # [default]
    max_components: int = 256                    # [default]
    solver: str = "auto"                         # [impl] auto | full | incremental
    max_in_memory_bytes: float = 1.0e9           # [impl] above this, IncrementalPCA
    ipca_batch: int = 1024                       # [impl]
    random_state: int = 0                        # [impl] randomized SVD seed


@dataclass
class EDConfig:
    adaptive_window: int = 64                    # [default] M frames declared idle
    kappa_max_iter: int = 50                     # [impl] fixed-point iterations of the kappa calibration


@dataclass
class PSSConfig:
    roots: tuple = (25, 29, 34)                  # [fixed] TS 36.211 Table 6.11.1.1-1
    seq_len: int = 62                            # [fixed]
    fft_size: int = 128                          # [fixed] 128-sample templates at 1.92 MS/s
    chunk_frames: int = 32                       # [impl]


@dataclass
class ModelsConfig:
    conv_ae: ConvAEConfig = field(default_factory=ConvAEConfig)
    fc_ae: FCAEConfig = field(default_factory=FCAEConfig)
    conv_vae: VAEConfig = field(default_factory=VAEConfig)
    per: PERConfig = field(default_factory=PERConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    ed: EDConfig = field(default_factory=EDConfig)
    pss: PSSConfig = field(default_factory=PSSConfig)
    main: tuple = ("conv_ae", "fc_ae", "conv_vae")        # [fixed] learned models
    ablation_conv_latent: tuple = (4, 8, 16)             # [default]
    ablation_fc_latent: tuple = (32, 128, 512)           # [default]


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    lr: float = 1e-3                             # [default]
    lr_vae: float = 1e-4                         # [default]
    weight_decay: float = 1e-5                   # [default]
    batch_size: int = 64                         # [default]
    max_epochs: int = 100                        # [default]
    patience: int = 15                           # [default] early stopping on validation loss
    min_delta: float = 1e-6                      # [default]
    sched_factor: float = 0.5                    # [default] ReduceLROnPlateau
    sched_patience: int = 7                      # [default]
    grad_clip: float = 1.0                       # [default] L2 norm
    seeds: tuple = (0, 1, 2, 3, 4)               # [D5]
    ablation_seeds: tuple = (0, 1, 2)            # [default]
    num_threads: int = 4                         # [fixed] torch.set_num_threads
    frame_stride: int = 1                        # [impl] 1 = every training frame (the design);
                                                 #   >1 subsamples training frames if time requires
    device: str = "auto"                         # [impl] auto (CUDA GPU if available, else CPU),
                                                 #   cpu, cuda or cuda:N


# ── Section 8: thresholds ─────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    pfa: tuple = (0.01, 0.05)                    # [D5]
    # Empirical (1 - P_FA) quantile. "inverted_cdf" is the order statistic
    # s_(ceil((1-P_FA) n)), so the realized false-alarm rate on the
    # calibration scores is at most P_FA (exactly P_FA when P_FA*n is whole).
    quantile_method: str = "inverted_cdf"        # [impl]


# ── Section 9: evaluation ─────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    n_bootstrap: int = 1000                      # [D5] session resamples
    ci_level: float = 0.95                       # [D5]
    bootstrap_seed: int = 12345                  # [impl]
    reference_model: str = "conv_ae"             # [fixed] primary detector in H1/H2
    h1_baseline: str = "ed_adaptive"             # [fixed]
    h2_baseline: str = "ed_fixed"                # [fixed]
    h2_reference: str = "ed_adaptive"            # [fixed]
    failure_samples: int = 10                    # [impl] false alarms and misses kept per detector
    roc_fpr_grid_points: int = 201               # [impl] stored ROC resolution (log-spaced FPR)
    pr_recall_grid_points: int = 101             # [impl]
    score_batch: int = 64                        # [impl] frames per scoring batch


# ── Section 10: computational cost ────────────────────────────────────────────

@dataclass
class LatencyConfig:
    warmup_frames: int = 50                      # [fixed]
    n_frames: int = 1000                         # [fixed] batch size 1
    throughput_batch: int = 64                   # [fixed]
    throughput_batches: int = 20                 # [impl] batches timed for throughput
    num_threads: int = 4                         # [fixed]
    settle_s: float = 5.0                        # [impl] all detectors run this long before any timing


# ── Section 11: real-time demonstration ───────────────────────────────────────

@dataclass
class DemoConfig:
    duration_s: float = 60.0                     # [impl] default run length
    queue_max_frames: int = 32                   # [impl] frames buffered (0.27 s); when full, the OLDEST is dropped
    live_history_frames: int = 120               # [impl] frames shown in the live view
    live_redraw_interval_s: float = 0.25         # [impl] at most 4 redraws per second (one costs about 40 ms)
    live_max_backlog_frames: int = 4             # [impl] no redraw while more frames than this wait


# ── Top level ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    # ── derived quantities ────────────────────────────────────────────────
    @property
    def frame_duration_s(self) -> float:
        """N / f_s = 8.533 ms."""
        return self.preprocess.frame_len / self.capture.sample_rate_hz

    @property
    def bin_spacing_hz(self) -> float:
        """f_s / K = 15 kHz."""
        return self.capture.sample_rate_hz / self.preprocess.fft_size

    @property
    def n_time_columns(self) -> int:
        return (self.preprocess.frame_len - self.preprocess.fft_size) // self.preprocess.hop + 1

    def data_dir(self, synthetic: bool = False) -> Path:
        return PathsConfig.resolve(self.paths.synthetic_data_dir if synthetic else self.paths.data_dir)

    def work_dir(self, synthetic: bool = False) -> Path:
        return PathsConfig.resolve(self.paths.synthetic_work_dir if synthetic else self.paths.work_dir)

    # ── (de)serialisation ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        cfg = cls()
        if d:
            _update_dataclass(cfg, d, "config")
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            cfg = cls()
            cfg.validate()
            return cfg
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def fingerprint(self, *sections: str) -> str:
        """Short hash of the given sections (all if none), used to tag cached outputs."""
        d = self.to_dict()
        if sections:
            d = {s: d[s] for s in sections}
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def training_dict(self) -> dict:
        """The settings a trained model depends on.

        Preprocessing, splits, the learned models and the training settings.
        Left out: paths, train.device and train.num_threads (where and how fast
        a run executes), and the detectors that are not trained (PER, PCA, ED, PSS).
        """
        d = self.to_dict()
        return {"preprocess": d["preprocess"], "split": d["split"],
                "models": {k: v for k, v in d["models"].items() if k not in UNTRAINED_MODEL_KEYS},
                "train": {k: v for k, v in d["train"].items() if k not in ENVIRONMENT_TRAIN_KEYS}}

    def training_fingerprint(self) -> str:
        """Short hash of training_dict()."""
        blob = json.dumps(self.training_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def validate(self) -> None:
        """Reject settings that would break the [fixed] design or the code's assumptions."""
        p = self.preprocess
        if p.frame_len % p.hop or (p.frame_len - p.fft_size) // p.hop + 1 != p.fft_size:
            raise ValueError("frame_len, fft_size and hop must give a square K x K spectrogram "
                             f"(got N={p.frame_len}, K={p.fft_size}, hop={p.hop})")
        if p.fft_size % 8:
            raise ValueError("fft_size must be divisible by 8 for the three pooling stages")
        if self.split.n_val_sessions < 1 or self.split.n_folds < 2:
            raise ValueError("need at least 2 folds and 1 validation session")
        for pfa in self.thresholds.pfa:
            if not 0.0 < pfa < 1.0:
                raise ValueError(f"P_FA must lie in (0, 1), got {pfa}")
        for g in (self.capture.g1_db, self.capture.g2_db):
            if isinstance(g, str):
                raise ValueError("gains are numbers in dB; automatic gain is not allowed")
        if self.injection.snr_step_db <= 0:
            raise ValueError("snr_step_db must be positive")
        if not re.fullmatch(r"auto|cpu|cuda(:\d+)?", str(self.train.device)):
            raise ValueError(f"train.device must be auto, cpu, cuda or cuda:N, got {self.train.device!r}")


# Settings that do not change what a training run computes.
ENVIRONMENT_TRAIN_KEYS = ("device", "num_threads")
ENVIRONMENT_KEYS = ("paths", "train.device", "train.num_threads")
UNTRAINED_MODEL_KEYS = ("per", "pca", "ed", "pss")


def config_differences(a: dict, b: dict, ignore=ENVIRONMENT_KEYS, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two config dicts (keys under `ignore` skipped)."""
    out = []
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}{key}"
        if path in ignore:
            continue
        va, vb = a.get(key, "<missing>"), b.get(key, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += config_differences(va, vb, ignore, path + ".")
        elif va != vb:
            out.append(f"{path}: {va!r} != {vb!r}")
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _update_dataclass(obj: Any, d: dict, where: str) -> None:
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in d.items():
        if key not in fields:
            raise KeyError(f"unknown config key {where}.{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{where}.{key} must be an object")
            _update_dataclass(current, value, f"{where}.{key}")
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def default_config() -> Config:
    return Config.load(None)
