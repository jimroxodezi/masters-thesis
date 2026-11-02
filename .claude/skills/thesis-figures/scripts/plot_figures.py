#!/usr/bin/env python3
"""Thesis-style data plots for an SDR / deep-learning anomaly-detection thesis.

Subcommands (each writes PNG at 300 dpi and a PDF):
  roc          ROC curve + AUC            --csv scores.csv   (columns: label,score; 1 = anomaly)
  loss         Training/validation loss   --csv history.csv  (columns: epoch,train_loss,val_loss)
  errors       Reconstruction-error hist  --csv scores.csv   [--threshold T]
  spectrogram  STFT spectrogram of IQ     --iq capture.npy --fs 2.4e6 [--fc 7.5e8]

Without a data file each plot is drawn from SYNTHETIC data and carries a visible
"EXAMPLE DATA" watermark. Never put a watermarked figure in the thesis as a result:
it exists only to show layout and style. Pass real data to get a clean figure.

Common options: --out DIR  --name STEM  --font serif|sans  --size W H (inches)

Requires: numpy, matplotlib  (pip install numpy matplotlib --break-system-packages)
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

TEAL, PURPLE, CORAL, GRAY = "#0F6E56", "#534AB7", "#993C1D", "#5F5E5A"


def style(font):
    plt.rcParams.update({
        "font.family": "serif" if font == "serif" else "sans-serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "savefig.dpi": 300, "figure.dpi": 120,
    })


def watermark(ax):
    ax.text(0.5, 0.5, "EXAMPLE DATA\nNOT REAL RESULTS", transform=ax.transAxes,
            ha="center", va="center", rotation=25, fontsize=22, color="#B4B2A9",
            alpha=0.45, zorder=0, fontweight="bold")


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"error: {path} is empty")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    print("\n".join(paths))


def roc_points(label, score):
    order = np.argsort(-score)
    y = label[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    return fpr, tpr, float(np.trapz(tpr, fpr)) if hasattr(np, "trapz") else float(np.trapezoid(tpr, fpr))


def synth_scores(seed=0, n=600, anomaly_frac=0.25):
    rng = np.random.default_rng(seed)
    n_a = int(n * anomaly_frac)
    normal = rng.gamma(4.0, 0.012, n - n_a)
    anom = rng.gamma(7.0, 0.020, n_a)
    return np.r_[np.zeros(n - n_a), np.ones(n_a)], np.r_[normal, anom]


def cmd_roc(a):
    synthetic = a.csv is None
    if synthetic:
        label, score = synth_scores()
    else:
        d = read_csv(a.csv)
        label, score = d["label"], d["score"]
    fpr, tpr, auc = roc_points(label, score)
    fig, ax = plt.subplots(figsize=a.size)
    ax.plot(fpr, tpr, color=TEAL, lw=2, label=f"Autoencoder (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    if synthetic:
        watermark(ax)
    save(fig, a.out, a.name or "roc-curve")


def cmd_loss(a):
    synthetic = a.csv is None
    if synthetic:
        ep = np.arange(1, 51)
        rng = np.random.default_rng(1)
        train = 0.12 * np.exp(-ep / 9) + 0.008 + rng.normal(0, 0.0006, ep.size)
        val = 0.13 * np.exp(-ep / 10) + 0.010 + rng.normal(0, 0.0010, ep.size)
    else:
        d = read_csv(a.csv)
        ep, train, val = d["epoch"], d["train_loss"], d.get("val_loss")
    fig, ax = plt.subplots(figsize=a.size)
    ax.plot(ep, train, color=TEAL, lw=2, label="Training")
    if val is not None:
        ax.plot(ep, val, color=PURPLE, lw=2, ls="--", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.legend(frameon=False)
    if synthetic:
        watermark(ax)
    save(fig, a.out, a.name or "training-loss")


def cmd_errors(a):
    synthetic = a.csv is None
    if synthetic:
        label, score = synth_scores(seed=2)
    else:
        d = read_csv(a.csv)
        label, score = d["label"], d["score"]
    thr = a.threshold if a.threshold is not None else float(np.percentile(score[label == 0], 99))
    bins = np.linspace(0, score.max() * 1.02, 50)
    fig, ax = plt.subplots(figsize=a.size)
    ax.hist(score[label == 0], bins=bins, color=TEAL, alpha=0.65, label="Normal")
    ax.hist(score[label == 1], bins=bins, color=CORAL, alpha=0.65, label="Anomalous")
    ax.axvline(thr, color=GRAY, lw=1.5, ls="--", label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Number of samples")
    ax.legend(frameon=False)
    if synthetic:
        watermark(ax)
    save(fig, a.out, a.name or "reconstruction-error-histogram")


def synth_iq(fs, seconds=0.06, seed=3):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.05
    band = rng.normal(size=n) + 1j * rng.normal(size=n)
    k = np.fft.fftfreq(n, 1 / fs)
    band = np.fft.ifft(np.fft.fft(band) * (np.abs(k) < 4.5e5))
    x += 0.5 * band
    burst = (t > 0.03) & (t < 0.036)
    x[burst] += 0.8 * np.exp(1j * 2 * np.pi * (3e5 * t[burst] + 2e7 * (t[burst] - 0.03) ** 2))
    return x


def cmd_spectrogram(a):
    synthetic = a.iq is None
    fs = a.fs
    x = synth_iq(fs) if synthetic else np.load(a.iq)
    fc = a.fc or 0.0
    fig, ax = plt.subplots(figsize=a.size)
    nfft = 1024
    spec, freqs, times, im = ax.specgram(x, NFFT=nfft, Fs=fs, Fc=fc, noverlap=nfft // 2,
                                         cmap="viridis", mode="psd", scale="dB")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)" if fc else "Frequency offset (MHz)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Power spectral density (dB)")
    if synthetic:
        watermark(ax)
    save(fig, a.out, a.name or "iq-spectrogram")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("roc", "loss", "errors", "spectrogram"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=".")
        p.add_argument("--name")
        p.add_argument("--font", choices=("sans", "serif"), default="sans")
        p.add_argument("--size", type=float, nargs=2, default=(6.0, 3.6), metavar=("W", "H"))
        if name in ("roc", "loss", "errors"):
            p.add_argument("--csv")
        if name == "errors":
            p.add_argument("--threshold", type=float)
        if name == "spectrogram":
            p.add_argument("--iq", help=".npy file of complex IQ samples")
            p.add_argument("--fs", type=float, default=2.4e6, help="sample rate in Hz")
            p.add_argument("--fc", type=float, default=0.0, help="centre frequency in Hz")
    a = ap.parse_args()
    style(a.font)
    {"roc": cmd_roc, "loss": cmd_loss, "errors": cmd_errors, "spectrogram": cmd_spectrogram}[a.cmd](a)


if __name__ == "__main__":
    main()
