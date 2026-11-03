"""
main.py
-------
Command-line entry point for the LTE anomaly detection pipeline.

Modes
-----
capture : record normal IQ from the RTL-SDR Blog V4 to data/normal_iq.npy
train   : fit the normaliser, autoencoder and threshold on a normal capture
detect  : score a new capture or an --input file with the trained detector
demo    : run the whole pipeline on synthetic data, with no hardware

Only capture writes data/normal_iq.npy, and demo never reads it. Demo writes its
checkpoints and results to demo/ subfolders, so it never replaces the detector
that train fitted on a real capture.
"""

import argparse
import dataclasses
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")   # headless backend; must be set before visualize imports pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal as sp_signal
import sklearn
import torch
from torch.utils.data import DataLoader

from config import cfg, Config
from sdr_capture import capture_iq, save_iq, load_iq, generate_synthetic_lte, inject_anomaly
from preprocessing import prepare_dataloaders, Normalizer, segment_iq, build_spectrograms
from model import build_model
from trainer import train
from detector import AnomalyDetector
from visualize import (
    plot_training_history, plot_reconstruction, plot_error_histogram,
    plot_anomaly_scores, plot_roc_curve, plot_psd,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Demo experiment (signal and model hyperparameters live in config.py)
SNR_DB                = 20.0
TRAIN_SEED            = 42
TEST_SEED             = 99
ANOMALY_START_FRAC    = 0.30
ANOMALY_DURATION_FRAC = 0.05
SWEEP_PERCENTILES     = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9)
NOMINAL_BAND_HZ       = 1.08e6   # 72 active subcarriers × 15 kHz
PSD_NPERSEG           = 1024     # Welch segment length of the PSD measurement
N_RECON_EXAMPLES      = 4        # training spectrograms in reconstruction_examples.png

# Anomaly kinds in evaluation order, with their plot titles
ANOMALY_TITLES = {"jammer": "Jammer", "power": "Power spike", "silence": "Dropout"}

# Artefact filenames inside data_dir, checkpoint_dir and results_dir
IQ_FILE         = "normal_iq.npy"
MODEL_FILE      = "best_model.pt"   # written by trainer.py
NORMALIZER_FILE = "normalizer.pkl"
DETECTOR_FILE   = "detector.pkl"
METRICS_FILE    = "metrics.json"
DEMO_SUBDIR     = "demo"   # demo's own folder inside checkpoint_dir and results_dir


# ─────────────────────────────────────────────────────────────────────────────
# Streams and labels
# ─────────────────────────────────────────────────────────────────────────────

def load_stream(path: Path, hint: str) -> np.ndarray:
    """
    Load an IQ file, or exit with a clear message if it is missing.

    Parameters
    ----------
    path : IQ .npy file
    hint : what the user should do about a missing file

    Returns
    -------
    np.ndarray, shape as stored
    """
    if not path.is_file():
        sys.exit(f"[main] error: {path} not found. {hint}")
    try:
        return load_iq(path)
    except (ValueError, EOFError, OSError) as exc:   # np.load on a file that is not .npy
        sys.exit(f"[main] error: cannot read {path} as an IQ .npy file: {exc}")


def capture_stream(config: Config = cfg) -> np.ndarray:
    """Capture IQ from the dongle, or exit with a clear message if it cannot be read."""
    try:
        return capture_iq(config)
    except (ImportError, OSError) as exc:
        sys.exit(f"[main] error: capture failed: {exc}. "
                 "Is the RTL-SDR dongle connected and librtlsdr installed?")


def check_stream(samples: np.ndarray, config: Config = cfg, training: bool = False) -> None:
    """
    Exit with a clear message unless a stream can go through the pipeline.

    Parameters
    ----------
    samples  : IQ array to check
    config   : Config
    training : also require non-empty training and validation splits
    """
    if samples.ndim != 1 or not np.iscomplexobj(samples):
        sys.exit("[main] error: expected a 1-D complex IQ array, "
                 f"got {samples.dtype} with shape {samples.shape}")

    L          = config.preprocess.segment_length
    n_segments = len(samples) // L
    n_val      = int(n_segments * config.train.val_split)
    if n_segments < 1 or (training and not 0 < n_val < n_segments):
        sys.exit(f"[main] error: {len(samples):,} samples give {n_segments} segments of "
                 f"{L} samples, too few to {'train on' if training else 'score'}")


def segment_labels(labels: np.ndarray, config: Config = cfg) -> np.ndarray:
    """
    Aggregate sample labels to segment labels: the maximum per segment, so a
    segment is anomalous if any of its samples is.

    Parameters
    ----------
    labels : np.ndarray, int8, shape (N,)
        Sample labels from inject_anomaly.
    config : Config

    Returns
    -------
    np.ndarray, int8, shape (M,)
        M = N // segment_length, the same segments as segment_iq.
    """
    L = config.preprocess.segment_length
    M = len(labels) // L
    return labels[: M * L].reshape(M, L).max(axis=1)


def fully_anomalous_index(labels: np.ndarray, config: Config = cfg) -> int:
    """
    Find a segment that lies wholly inside the anomaly window.

    Parameters
    ----------
    labels : np.ndarray, int8, shape (N,)
        Sample labels from inject_anomaly.
    config : Config

    Returns
    -------
    int
        Index of the first segment with the most anomalous samples, which is
        fully anomalous whenever the window spans at least one whole segment.
    """
    L = config.preprocess.segment_length
    M = len(labels) // L
    return int(np.argmax(labels[: M * L].reshape(M, L).sum(axis=1, dtype=np.int64)))


# ─────────────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────────────

def measure_psd(samples: np.ndarray, config: Config = cfg) -> dict[str, float]:
    """
    Measure the mean PSD inside and outside the nominal occupied band.

    Parameters
    ----------
    samples : np.ndarray, complex64, shape (N,)
    config  : Config

    Returns
    -------
    dict[str, float]
        in_band_db_per_hz and out_of_band_db_per_hz, averaged on the linear
        Welch estimate and then converted to dB, plus nominal_band_hz.
    """
    f, pxx = sp_signal.welch(
        samples,
        fs=config.sdr.sample_rate,
        nperseg=PSD_NPERSEG,
        detrend=False,           # same estimate as plot_psd, with no false notch at DC
        return_onesided=False,   # complex IQ → two-sided spectrum
    )
    pxx     = pxx.astype(np.float64)
    in_band = np.abs(f) <= NOMINAL_BAND_HZ / 2.0
    return {
        "in_band_db_per_hz":     float(10.0 * np.log10(pxx[in_band].mean())),
        "out_of_band_db_per_hz": float(10.0 * np.log10(pxx[~in_band].mean())),
        "nominal_band_hz":       NOMINAL_BAND_HZ,
    }


def measure_padding(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Config = cfg,
) -> dict[str, float]:
    """
    Compare the real STFT frames of a spectrogram with its padded frames.

    Parameters
    ----------
    model  : trained ConvAutoencoder
    loader : DataLoader of normalised (B, 1, H, W) spectrograms
    config : Config

    Returns
    -------
    dict[str, float]
        stft_frames, padded_frames, and the mean normalised value and the
        reconstruction MSE restricted to each kind of frame. A mean over no
        frames is NaN.
    """
    pc       = config.preprocess
    n_real   = min((pc.segment_length - pc.n_fft) // pc.hop_length + 1, pc.target_time_frames)
    n_padded = pc.target_time_frames - n_real
    device   = next(model.parameters()).device

    # Sum in float64: float32 running totals lose precision over a long stream
    value_real = value_padded = error_real = error_padded = 0.0
    n_specs    = 0

    model.eval()   # reconstruction_error() leaves the mode to the caller
    for batch in loader:
        batch = batch.to(device)
        error = model.reconstruction_error(batch, reduction="none")   # (B, 1, H, W)

        # Time is the last axis
        value_real   += batch[..., :n_real].sum(dtype=torch.float64).item()
        value_padded += batch[..., n_real:].sum(dtype=torch.float64).item()
        error_real   += error[..., :n_real].sum(dtype=torch.float64).item()
        error_padded += error[..., n_real:].sum(dtype=torch.float64).item()
        n_specs      += len(batch)

    pixels_real   = n_specs * pc.target_freq_bins * n_real
    pixels_padded = n_specs * pc.target_freq_bins * n_padded
    nan           = float("nan")
    return {
        "stft_frames":                   n_real,
        "padded_frames":                 n_padded,
        "mean_normalised_real_frames":   value_real / pixels_real if pixels_real else nan,
        "mean_normalised_padded_frames": value_padded / pixels_padded if pixels_padded else nan,
        "mse_real_frames":               error_real / pixels_real if pixels_real else nan,
        "mse_padded_frames":             error_padded / pixels_padded if pixels_padded else nan,
    }


def describe_model(model: torch.nn.Module, config: Config = cfg) -> dict:
    """
    Measure the parameter count and the bottleneck size from the model itself.

    Parameters
    ----------
    model  : ConvAutoencoder
    config : Config

    Returns
    -------
    dict
        trainable_params, input_elements, bottleneck_shape (without the batch
        dimension) and bottleneck_elements.
    """
    pc = config.preprocess
    x  = torch.zeros(
        1, 1, pc.target_freq_bins, pc.target_time_frames,
        device=next(model.parameters()).device,
    )

    model.eval()   # in train mode BatchNorm would absorb the zeros into its running stats
    with torch.no_grad():
        z = model.encode(x)

    return {
        "trainable_params":    sum(p.numel() for p in model.parameters() if p.requires_grad),
        "input_elements":      x[0].numel(),
        "bottleneck_shape":    list(z.shape[1:]),
        "bottleneck_elements": z[0].numel(),
    }


def summarise_training(history: dict) -> dict:
    """
    Reduce a training history to the numbers reported in metrics.json.

    Parameters
    ----------
    history : dict returned by trainer.train()

    Returns
    -------
    dict
        The "training" block of metrics.json.
    """
    # The LR each epoch trained with, so a reduction after the last epoch does not count
    lr = history["lr"]
    return {
        "epochs_run":             len(history["train_loss"]),
        "best_epoch":             history["best_epoch"],
        "best_val_loss":          history["best_val_loss"],
        "stopped_early":          history["stopped_early"],
        "lr_reductions":          sum(new < old for old, new in zip(lr, lr[1:])),
        "final_lr":               lr[-1],
        "mean_epoch_seconds":     np.mean(history["epoch_time"]),
        "total_training_seconds": np.sum(history["epoch_time"]),
        # Epochs have equal step counts, so this mean of epoch means is the mean over steps
        "mean_grad_norm":         np.mean(history["grad_norm"]),
        # The largest epoch mean, not the largest single step: history has no per-step norms
        "max_grad_norm":          np.max(history["grad_norm"]),
        "clip_fraction_overall":  np.mean(history["clip_fraction"]),
        "final_train_loss":       history["train_loss"][-1],
        "final_val_loss":         history["val_loss"][-1],
    }


def class_score_stats(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Describe the scores of the anomalous and the normal segments separately.

    Parameters
    ----------
    scores : np.ndarray, shape (M,)
    labels : np.ndarray, integer, shape (M,)
        1 = anomalous, 0 = normal.

    Returns
    -------
    dict[str, float]
        n_<class> and mean_, min_ and max_score_<class> for each class. A
        statistic of an empty class is NaN.
    """
    stats = {}
    for name, mask in (("anomalous", labels == 1), ("normal", labels == 0)):
        part = scores[mask].astype(np.float64)
        stats[f"n_{name}"]          = part.size
        stats[f"mean_score_{name}"] = part.mean() if part.size else float("nan")
        stats[f"min_score_{name}"]  = part.min() if part.size else float("nan")
        stats[f"max_score_{name}"]  = part.max() if part.size else float("nan")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def false_positive_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of a fully normal stream strictly above the threshold, as in predict()."""
    return float(np.mean(scores > threshold))


def rates_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute the detection rates of one labelled stream at one threshold.

    Parameters
    ----------
    scores    : np.ndarray, shape (M,)
    labels    : np.ndarray, integer, shape (M,)
        1 = anomalous, 0 = normal.
    threshold : scores strictly above it are flagged

    Returns
    -------
    dict[str, float]
        precision, recall, f1 and fpr. A ratio with a zero denominator is 0.0,
        as in AnomalyDetector.evaluate().
    """
    flags = scores > threshold
    truth = labels == 1

    tp = np.count_nonzero(flags & truth)
    fp = np.count_nonzero(flags & ~truth)
    fn = np.count_nonzero(~flags & truth)
    tn = np.count_nonzero(~flags & ~truth)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0,
        "fpr":       fp / (fp + tn) if fp + tn > 0 else 0.0,
    }


def threshold_sweep(
    train_errors: np.ndarray,
    val_errors: np.ndarray,
    clean_scores: np.ndarray,
    streams: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """
    Re-evaluate the stored scores at every percentile of SWEEP_PERCENTILES.

    Parameters
    ----------
    train_errors : training scores; each threshold is a percentile of these
    val_errors   : validation scores
    clean_scores : scores of the fully normal test stream
    streams      : anomaly kind → (segment scores, segment labels)

    Returns
    -------
    list[dict]
        One entry per percentile: percentile, threshold, val_fpr, clean_test_fpr
        and detection, which maps each anomaly kind to its rates.
    """
    sweep = []
    for percentile in SWEEP_PERCENTILES:
        threshold = float(np.percentile(train_errors, percentile))
        sweep.append({
            "percentile":     percentile,
            "threshold":      threshold,
            "val_fpr":        false_positive_rate(val_errors, threshold),
            "clean_test_fpr": false_positive_rate(clean_scores, threshold),
            "detection": {
                kind: rates_at_threshold(scores, labels, threshold)
                for kind, (scores, labels) in streams.items()
            },
        })
    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# metrics.json
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_model() -> str:
    """Read the CPU model from /proc/cpuinfo, falling back to the platform module."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def describe_environment(model: torch.nn.Module) -> dict:
    """
    Record the software versions and the hardware a run used.

    Parameters
    ----------
    model : the model whose parameters show the device in use

    Returns
    -------
    dict
        The "environment" block of metrics.json.
    """
    return {
        "python":        platform.python_version(),
        "numpy":         np.__version__,
        "scipy":         scipy.__version__,
        "sklearn":       sklearn.__version__,
        "matplotlib":    matplotlib.__version__,
        "torch":         torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_model":     _cpu_model(),
        "device":        str(next(model.parameters()).device),
    }


def _jsonable(obj):
    """Convert a metrics value to plain JSON types; NaN and infinities become None."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_metrics(metrics: dict, path: Path) -> None:
    """
    Replace the metrics file in one step, so a kill never leaves it half-written.

    Parameters
    ----------
    metrics : dict
        Numpy types, Paths and datetimes are converted; NaN is written as null.
    path    : destination .json file
    """
    path = Path(path)
    text = json.dumps(_jsonable(metrics), indent=2, allow_nan=False)
    tmp  = path.with_suffix(".json.tmp")   # same directory, so os.replace stays atomic
    with open(tmp, "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[main] metrics saved → {path}")


def finish_run(run: dict) -> None:
    """Stamp the finish time and the total duration into the "run" block."""
    run["finished"]      = datetime.now().astimezone()
    run["total_seconds"] = (run["finished"] - run["started"]).total_seconds()


def print_summary(metrics: dict) -> None:
    """Print a compact table of the key results of a finished run."""
    tr = metrics["training"]
    th = metrics["threshold"]

    print(f"\n[main] Summary  ({metrics['run']['total_seconds']:.0f} s)")
    print(f"  training  : epochs={tr['epochs_run']}  best={tr['best_epoch']}  "
          f"best_val_loss={tr['best_val_loss']:.6f}  stopped_early={tr['stopped_early']}")
    clean_fpr = f"  clean={th['clean_test_fpr']:.4f}" if "clean_test_fpr" in th else ""
    print(f"  threshold : p{th['percentile']:g} = {th['value']:.6f}  "
          f"fpr: train={th['train_fpr']:.4f}  val={th['val_fpr']:.4f}{clean_fpr}")

    if metrics.get("detection"):
        print(f"\n  {'anomaly':<9}{'segments':>9}{'auc_roc':>9}{'auc_pr':>9}"
              f"{'prec':>9}{'recall':>9}{'f1':>9}{'fpr':>9}")
        for kind, d in metrics["detection"].items():
            print(f"  {kind:<9}{d['n_anomalous']:>9,}{d['auc_roc']:>9.4f}{d['auc_pr']:>9.4f}"
                  f"{d['precision']:>9.4f}{d['recall']:>9.4f}{d['f1']:>9.4f}{d['fpr']:>9.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_outputs(
    model: torch.nn.Module,
    history: dict,
    train_loader: DataLoader,
    config: Config = cfg,
) -> None:
    """
    Save the training curves and the reconstructions of training spectrograms.

    Parameters
    ----------
    model        : trained ConvAutoencoder
    history      : dict returned by trainer.train()
    train_loader : its dataset holds the normalised training spectrograms
    config       : Config
    """
    # (N, 1, H, W) → (N, H, W)
    examples = train_loader.dataset.data[:N_RECON_EXAMPLES, 0].numpy()

    plt.close(plot_training_history(history, config, save=True,
                                    filename="training_history.png"))
    plt.close(plot_reconstruction(model, examples, n_examples=len(examples), config=config,
                                  save=True, filename="reconstruction_examples.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def invalidate_detector(checkpoint_dir: Path) -> None:
    """
    Delete detector.pkl before a retrain overwrites the other checkpoints.

    Training replaces normalizer.pkl first and best_model.pt at every improving
    epoch, and writes detector.pkl last. Without this, a run killed part-way would
    leave a new model with the old threshold, which detect would accept. With it,
    detect refuses the incomplete set as missing a checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / DETECTOR_FILE).unlink(missing_ok=True)


def run_capture(config: Config = cfg) -> None:
    """Capture normal IQ from the dongle and save it as the training stream."""
    samples = capture_stream(config)
    save_iq(samples, Path(config.data_dir) / IQ_FILE)


def run_train(run: dict, iq_path: Path | None = None, config: Config = cfg) -> None:
    """
    Fit the normaliser, autoencoder and threshold on a normal capture.

    Parameters
    ----------
    run     : "run" block of metrics.json
    iq_path : IQ .npy file; None means data/normal_iq.npy
    config  : Config
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    metrics_path   = Path(config.results_dir) / METRICS_FILE

    if iq_path is None:
        iq_path = Path(config.data_dir) / IQ_FILE
        samples = load_stream(iq_path, "Run --mode capture first, or pass --input.")
    else:
        samples = load_stream(iq_path, "Check the --input path.")
    check_stream(samples, config, training=True)
    n_samples = len(samples)

    train_loader, val_loader, normalizer = prepare_dataloaders(samples, config)
    del samples   # the spectrograms replace the IQ from here on
    invalidate_detector(checkpoint_dir)
    normalizer.save(checkpoint_dir / NORMALIZER_FILE)

    model   = build_model(config)
    history = train(model, train_loader, val_loader, config)

    metrics = {
        "run":         run,
        "environment": describe_environment(model),
        "config":      dataclasses.asdict(config),
        "data": {
            "input":      iq_path,
            "n_samples":  n_samples,
            "n_segments": n_samples // config.preprocess.segment_length,
            "n_train":    len(train_loader.dataset),
            "n_val":      len(val_loader.dataset),
        },
        "model":      describe_model(model, config),
        "normalizer": {"min_db": normalizer.min_, "max_db": normalizer.max_},
        "training":   summarise_training(history),
        "threshold":  None,
    }
    write_metrics(metrics, metrics_path)   # an interrupted run keeps the training summary

    detector  = AnomalyDetector(model, config)
    threshold = detector.fit_threshold(train_loader)
    detector.save(checkpoint_dir / DETECTOR_FILE)

    plot_training_outputs(model, history, train_loader, config)

    train_errors = detector.score(train_loader)
    val_errors   = detector.score(val_loader)
    plt.close(plot_error_histogram(train_errors, threshold=threshold, config=config,
                                   save=True, filename="error_histogram.png"))

    metrics["threshold"] = {
        "percentile": config.anomaly.threshold_percentile,
        "value":      threshold,
        "train_fpr":  false_positive_rate(train_errors, threshold),
        "val_fpr":    false_positive_rate(val_errors, threshold),
    }
    finish_run(run)
    write_metrics(metrics, metrics_path)
    print_summary(metrics)


def run_detect(iq_path: Path | None = None, config: Config = cfg) -> None:
    """
    Score a new stream with the trained detector and report the flagged segments.

    Never writes data/normal_iq.npy.

    Parameters
    ----------
    iq_path : IQ .npy file; None means a new capture from the dongle
    config  : Config
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    results_dir    = Path(config.results_dir)

    # Check the checkpoints first, so a missing file does not waste a capture
    paths   = [checkpoint_dir / name for name in (MODEL_FILE, NORMALIZER_FILE, DETECTOR_FILE)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        sys.exit(f"[main] error: missing checkpoint(s): {', '.join(missing)}. "
                 "Run --mode train first.")

    normalizer = Normalizer.load(checkpoint_dir / NORMALIZER_FILE)
    model      = build_model(config).load(checkpoint_dir / MODEL_FILE)
    detector   = AnomalyDetector.load(checkpoint_dir / DETECTOR_FILE, model, config)

    if iq_path is None:
        samples = capture_stream(config)
    else:
        samples = load_stream(iq_path, "Check the --input path.")
    check_stream(samples, config)

    # Drop each stage as soon as the next exists: these arrays are about 1 GB at 30 s
    specs = build_spectrograms(segment_iq(samples, config), config)
    del samples
    norm = normalizer.transform(specs)
    del specs
    scores = detector.score_array(norm)
    del norm

    n_flagged = int(detector.predict(scores).sum())
    print(f"[main] flagged {n_flagged:,} of {len(scores):,} segments "
          f"({100.0 * n_flagged / len(scores):.2f}%)")

    scores_path = results_dir / "anomaly_scores.npy"
    np.save(scores_path, scores)
    print(f"[main] saved → {scores_path}")

    plt.close(plot_anomaly_scores(scores, detector.threshold, config=config,
                                  save=True, filename="anomaly_scores.png"))


def evaluate_anomaly(
    clean: np.ndarray,
    kind: str,
    detector: AnomalyDetector,
    normalizer: Normalizer,
    suffix: str = "",
    config: Config = cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Inject one anomaly into a clean stream, score it and save its plots.

    Parameters
    ----------
    clean      : np.ndarray, complex64, shape (N,)
    kind       : "jammer" | "power" | "silence"
    detector   : AnomalyDetector with a fitted threshold
    normalizer : Normalizer fitted on the training split
    suffix     : inserted before the extension of every filename, e.g. "_jammer"
    config     : Config

    Returns
    -------
    (scores, labels, example, metrics)
        scores  : np.ndarray, shape (M,), reconstruction error per segment
        labels  : np.ndarray, int8, shape (M,), 1 where a segment is anomalous
        example : np.ndarray, float32, shape (H, W), one normalised spectrogram
                  from fully_anomalous_index
        metrics : dict, the "detection" entry of this anomaly in metrics.json
    """
    results_dir = Path(config.results_dir)
    threshold   = detector.threshold

    corrupted, sample_labels = inject_anomaly(
        clean,
        config,
        anomaly_type=kind,
        start_frac=ANOMALY_START_FRAC,
        duration_frac=ANOMALY_DURATION_FRAC,
    )
    labels      = segment_labels(sample_labels, config)
    example_idx = fully_anomalous_index(sample_labels, config)
    del sample_labels

    # Drop each stage as soon as the next exists: these arrays are about 1 GB at 30 s
    specs = build_spectrograms(segment_iq(corrupted, config), config)
    del corrupted
    norm = normalizer.transform(specs)
    del specs
    example = norm[example_idx].copy()   # a view would keep the whole stack alive
    scores  = detector.score_array(norm)
    del norm

    metrics = {
        **detector.evaluate(scores, labels),
        "n_flagged": int(detector.predict(scores).sum()),
        **class_score_stats(scores, labels),
    }
    print(f"[main] {kind}: flagged {metrics['n_flagged']:,} of {len(scores):,} segments  "
          f"({metrics['n_anomalous']:,} anomalous)")

    scores_path = results_dir / f"anomaly_scores{suffix}.npy"
    np.save(scores_path, scores)
    print(f"[main] saved → {scores_path}")

    plt.close(plot_anomaly_scores(scores, threshold, labels=labels, config=config,
                                  save=True, filename=f"anomaly_scores{suffix}.png"))
    plt.close(plot_roc_curve(scores, labels, config=config,
                             save=True, filename=f"roc_curve{suffix}.png"))
    plt.close(plot_error_histogram(scores[labels == 0], anomaly_errors=scores[labels == 1],
                                   threshold=threshold, config=config,
                                   save=True, filename=f"error_histogram{suffix}.png"))

    return scores, labels, example, metrics


def run_demo(run: dict, config: Config = cfg) -> None:
    """
    Run the whole pipeline on synthetic data, with no hardware.

    Trains on one stream, then scores an unseen clean stream and each selected
    anomaly injected into it. Never reads or writes data/normal_iq.npy, and
    main() points config.checkpoint_dir and config.results_dir at their demo/
    subfolders first, so a detector trained on a real capture is left alone.

    Parameters
    ----------
    run    : "run" block of metrics.json; run["anomalies"] selects the anomalies
    config : Config
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    results_dir    = Path(config.results_dir)
    metrics_path   = results_dir / METRICS_FILE
    kinds          = run["anomalies"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Synthetic training stream ──────────────────────────────────────
    print(f"[main] generating training stream  seed={TRAIN_SEED}  snr={SNR_DB:g} dB", flush=True)
    samples = generate_synthetic_lte(config, snr_db=SNR_DB, seed=TRAIN_SEED)
    check_stream(samples, config, training=True)
    n_samples = len(samples)

    # ── 2. Spectrum ───────────────────────────────────────────────────────
    plt.close(plot_psd(samples, config=config, save=True, filename="psd.png",
                       nominal_band_hz=NOMINAL_BAND_HZ))
    psd = measure_psd(samples, config)

    # ── 3. Training and threshold ─────────────────────────────────────────
    train_loader, val_loader, normalizer = prepare_dataloaders(samples, config)
    del samples   # the spectrograms replace the IQ from here on
    invalidate_detector(checkpoint_dir)
    normalizer.save(checkpoint_dir / NORMALIZER_FILE)

    model   = build_model(config)
    history = train(model, train_loader, val_loader, config)

    metrics = {
        "run":         run,
        "environment": describe_environment(model),
        "config":      dataclasses.asdict(config),
        "data": {
            "train_seed": TRAIN_SEED,
            "test_seed":  TEST_SEED,
            "snr_db":     SNR_DB,
            "n_samples":  n_samples,
            "n_segments": n_samples // config.preprocess.segment_length,
            "n_train":    len(train_loader.dataset),
            "n_val":      len(val_loader.dataset),
        },
        "model":           describe_model(model, config),
        "normalizer":      {"min_db": normalizer.min_, "max_db": normalizer.max_},
        "psd":             psd,
        "padding":         None,
        "training":        summarise_training(history),
        "threshold":       None,
        "detection":       {},
        "threshold_sweep": None,
    }
    write_metrics(metrics, metrics_path)   # an interrupted run keeps the training summary

    detector  = AnomalyDetector(model, config)
    threshold = detector.fit_threshold(train_loader)
    detector.save(checkpoint_dir / DETECTOR_FILE)

    plot_training_outputs(model, history, train_loader, config)

    # ── 4. Training and validation scores ─────────────────────────────────
    # Not through train_loader, which shuffles: train_errors.npy stays in segment order
    train_errors = detector.score_array(train_loader.dataset.data[:, 0].numpy())
    val_errors   = detector.score(val_loader)

    # Not plain error_histogram.png: a single-anomaly run gives that name to its stream
    plt.close(plot_error_histogram(train_errors, threshold=threshold, config=config,
                                   save=True, filename="error_histogram_train.png"))
    train_errors_path = results_dir / "train_errors.npy"
    np.save(train_errors_path, train_errors)
    print(f"[main] saved → {train_errors_path}")

    # ── 5. Padded frames ──────────────────────────────────────────────────
    metrics["padding"] = measure_padding(model, val_loader, config)
    del train_loader, val_loader   # about 1 GB of spectrograms at 30 s

    # ── 6. Unseen fully normal stream ─────────────────────────────────────
    print(f"[main] generating test stream  seed={TEST_SEED}  snr={SNR_DB:g} dB", flush=True)
    clean      = generate_synthetic_lte(config, snr_db=SNR_DB, seed=TEST_SEED)
    clean_norm = normalizer.transform(build_spectrograms(segment_iq(clean, config), config))

    clean_example = clean_norm[0].copy()   # a view would keep the whole stack alive
    clean_scores  = detector.score_array(clean_norm)
    del clean_norm

    metrics["threshold"] = {
        "percentile":     config.anomaly.threshold_percentile,
        "value":          threshold,
        "train_fpr":      false_positive_rate(train_errors, threshold),
        "val_fpr":        false_positive_rate(val_errors, threshold),
        "clean_test_fpr": false_positive_rate(clean_scores, threshold),
    }

    # ── 7. Anomaly streams ────────────────────────────────────────────────
    streams  = {}
    examples = {}
    for kind in kinds:
        print(f"[main] evaluating anomaly: {kind}", flush=True)
        suffix = f"_{kind}" if len(kinds) > 1 else ""
        scores, labels, examples[kind], metrics["detection"][kind] = evaluate_anomaly(
            clean, kind, detector, normalizer, suffix, config
        )
        streams[kind] = (scores, labels)
        write_metrics(metrics, metrics_path)   # each finished stream survives an interruption
    del clean

    # ── 8. Normal and anomalous reconstructions ───────────────────────────
    spectrograms = np.stack([clean_example, *examples.values()])
    titles       = ["Normal", *(ANOMALY_TITLES[kind] for kind in examples)]
    plt.close(plot_reconstruction(model, spectrograms, n_examples=len(spectrograms),
                                  config=config, save=True,
                                  filename="reconstruction_anomalies.png", titles=titles))

    # ── 9. Threshold sweep ────────────────────────────────────────────────
    metrics["threshold_sweep"] = threshold_sweep(train_errors, val_errors, clean_scores, streams)

    # ── 10. Final metrics ─────────────────────────────────────────────────
    finish_run(run)
    write_metrics(metrics, metrics_path)
    print_summary(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LTE anomaly detection with an RTL-SDR Blog V4 and a convolutional autoencoder."
    )
    parser.add_argument("--mode", required=True, choices=["capture", "train", "detect", "demo"],
                        help="pipeline stage to run; demo needs no hardware")
    parser.add_argument("--freq", type=float, default=None,
                        help="centre frequency in MHz (default: sdr.center_freq in config.py)")
    parser.add_argument("--duration", type=float, default=None,
                        help="capture length in seconds (default: sdr.capture_duration)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="maximum training epochs (default: train.epochs)")
    parser.add_argument("--anomaly", choices=[*ANOMALY_TITLES, "all"], default="jammer",
                        help="anomaly to inject in demo mode (default: %(default)s)")
    parser.add_argument("--input", type=Path, default=None,
                        help="IQ .npy file for train and detect "
                             "(default: data/normal_iq.npy for train, a new capture for detect)")
    return parser


def apply_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: Config = cfg,
) -> None:
    """
    Validate the arguments and copy the overrides into the config.

    Parameters
    ----------
    parser : reports invalid arguments and exits
    args   : parsed command line
    config : Config, modified in place
    """
    if args.freq is not None:
        freq_hz = args.freq * 1e6
        if freq_hz > config.sdr.max_freq:
            parser.error(f"--freq {args.freq} MHz is above the R828D tuner limit of "
                         f"{config.sdr.max_freq / 1e6:g} MHz, so it cannot be tuned "
                         "(LTE Band 3 DL at 1805–1880 MHz is out of reach)")
        if not freq_hz > 0:
            parser.error(f"--freq must be positive, got {args.freq}")
        config.sdr.center_freq = freq_hz

    if args.duration is not None:
        if not (math.isfinite(args.duration) and args.duration > 0):
            parser.error(f"--duration must be positive, got {args.duration}")
        config.sdr.capture_duration = args.duration

    if args.epochs is not None:
        if args.epochs < 1:
            parser.error(f"--epochs must be at least 1, got {args.epochs}")
        config.train.epochs = args.epochs

    if args.input is not None and args.mode not in ("train", "detect"):
        parser.error(f"--input is only used by --mode train and detect, not {args.mode}")


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line, apply it to the shared cfg and run the chosen mode.

    Parameters
    ----------
    argv : arguments without the program name; None means sys.argv
    """
    parser = build_parser()
    args   = parser.parse_args(argv)
    apply_overrides(parser, args, cfg)

    run = {
        "mode":          args.mode,
        "command":       list(sys.argv) if argv is None else ["main.py", *argv],
        "anomalies":     [],
        "started":       datetime.now().astimezone(),
        "finished":      None,
        "total_seconds": None,
    }

    if args.mode == "capture":
        run_capture(cfg)
    elif args.mode == "train":
        run_train(run, args.input, cfg)
    elif args.mode == "detect":
        run_detect(args.input, cfg)
    else:
        run["anomalies"] = list(ANOMALY_TITLES) if args.anomaly == "all" else [args.anomaly]
        cfg.checkpoint_dir = Path(cfg.checkpoint_dir) / DEMO_SUBDIR
        cfg.results_dir    = Path(cfg.results_dir) / DEMO_SUBDIR
        run_demo(run, cfg)


if __name__ == "__main__":
    main()
