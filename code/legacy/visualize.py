"""
visualize.py
------------
Static figures for the printed thesis.

Figures
-------
plot_training_history : loss curves and learning rate per epoch
plot_reconstruction   : input, reconstruction and squared error per example
plot_error_histogram  : distribution of the reconstruction errors
plot_anomaly_scores   : score of every segment against the threshold
plot_roc_curve        : ROC curve with its AUC
plot_psd              : Welch power spectral density of an IQ capture

Every figure is 14 cm wide, the width it is printed at, so the font sizes below
are the sizes on paper. Each function returns its Figure and leaves showing or
closing it to the caller. The print style applies only while a figure is built,
so the caller's rcParams and backend stay untouched.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
    ScalarFormatter,
    StrMethodFormatter,
)
from scipy import signal as sp_signal
from sklearn.metrics import auc, roc_curve
from pathlib import Path

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Print style
# ─────────────────────────────────────────────────────────────────────────────
# Figure parameters only: none of them reaches the detector, so they stay out of config.py.

_DPI   = 200
_CM    = 1.0 / 2.54       # inches per centimetre
_WIDTH = 14.0 * _CM       # printed width of every figure

# Categorical series, in the palette's fixed order
_SERIES_1 = "#2a78d6"     # blue
_SERIES_2 = "#eb6834"     # orange
_SERIES_3 = "#1baf7a"     # aqua (below 3:1 on paper, so its line is labelled directly)

# Neutral ink and chrome
_SURFACE  = "#ffffff"     # the printed page
_INK      = "#0b0b0b"
_INK_SOFT = "#52514e"
_MUTED    = "#898781"
_AXIS     = "#c3c2b7"
_GRID     = "#e1e0d9"

# Sequential ramps: one hue, light → dark. They share one lightness profile, so
# both keep their meaning when the thesis is printed in greyscale.
_CMAP_MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])
_CMAP_ERROR = LinearSegmentedColormap.from_list("error", [
    "#fed5c7", "#fcc1ab", "#f9ac90", "#f59673", "#f18056", "#eb6834", "#d55e2f",
    "#bf5429", "#aa4a24", "#95401f", "#81371a", "#6e2e15", "#5b2510",
])

_STYLE = {
    "figure.facecolor":      _SURFACE,
    "axes.facecolor":        _SURFACE,
    "font.size":             9,
    "axes.labelsize":        9,
    "axes.titlesize":        9,
    "legend.fontsize":       9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "text.color":            _INK,
    "axes.labelcolor":       _INK,
    "xtick.labelcolor":      _INK_SOFT,
    "ytick.labelcolor":      _INK_SOFT,
    "xtick.color":           _AXIS,
    "ytick.color":           _AXIS,
    "axes.edgecolor":        _AXIS,
    "axes.linewidth":        0.75,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.axisbelow":        True,
    "grid.color":            _GRID,
    "grid.linewidth":        0.5,
    "lines.linewidth":       1.5,
    "lines.solid_capstyle":  "round",
    "lines.solid_joinstyle": "round",
    "xtick.major.size":      3.0,
    "ytick.major.size":      3.0,
    "xtick.major.width":     0.75,
    "ytick.major.width":     0.75,
    "xtick.minor.size":      1.5,
    "ytick.minor.size":      1.5,
    "xtick.minor.width":     0.5,
    "ytick.minor.width":     0.5,
    "legend.frameon":        False,
    "legend.borderaxespad":  0.0,
    "legend.borderpad":      0.3,
    "legend.handlelength":   1.8,
    "legend.columnspacing":  1.5,
    "hatch.linewidth":       0.5,
}

# Translucent, so the gridlines stay visible through a shaded band
_BAND = {"color": _MUTED, "alpha": 0.22, "linewidth": 0}

_HEADROOM     = 0.12   # share of a panel kept clear of data for a line's label
_WASH         = 0.12   # opacity of an area fill under its outline
_TEXTURE      = "///"  # 45° lines: tells the anomalies apart in a greyscale print
_SCORE_MARKER = 1.0    # points: small enough for 30 000 scores to stay readable
_LOG_RATIO    = 3.0    # values spanning more than this factor get a log axis
_TITLE_Y      = 1.11   # axes fraction: lifts a column title above the row name below it

_HIST_BINS       = 80
_LR_MAX_TICKS    = 4
_PSD_NPERSEG     = 1024
_PSD_FLOOR       = 1e-20
_PSD_MIN_SPAN_DB = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _finalize(fig: Figure, save: bool, filename: str, config: Config) -> Figure:
    """Fix the layout under the print style, then save the figure if asked."""
    # Ticks are built lazily. Building them here stops a later display, outside
    # the print style, from drawing them with the caller's rcParams.
    fig.draw_without_rendering()
    if save:
        path = Path(config.results_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI)
        print(f"[visualize] saved → {path}")
    return fig


def _legend_above(ax: Axes) -> None:
    """Draw the legend in the margin above the axes, where it cannot cover data."""
    n_entries = len(ax.get_legend_handles_labels()[1])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=n_entries)


def _label_vline(ax: Axes, x: float, text: str) -> None:
    """Label a vertical reference line at the top of the panel, on its roomier side."""
    display  = ax.get_xaxis_transform().transform((x, 1.0))
    fraction = ax.transAxes.inverted().transform(display)[0]
    side     = 1 if fraction < 0.5 else -1
    ax.annotate(
        text,
        xy=(x, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(side * 4.0, -2.0),
        textcoords="offset points",
        ha="left" if side > 0 else "right",
        va="top",
    )


def _use_log_scale(values: np.ndarray) -> bool:
    """True when the positive values span more than _LOG_RATIO."""
    positive = values[values > 0]
    return len(positive) > 0 and positive.max() / positive.min() > _LOG_RATIO


def _bin_heights(
    data: np.ndarray, edges: np.ndarray, as_density: bool, on_log_axis: bool
) -> np.ndarray:
    """Count data into the shared bins; on a log axis a density is per decade."""
    if on_log_axis:
        # Clipping keeps a zero error, which a log axis cannot show, in the lowest bin
        data, edges = np.log10(np.clip(data, edges[0], None)), np.log10(edges)
    heights, _ = np.histogram(data, bins=edges, density=as_density)
    return heights


def _plain_formatter() -> ScalarFormatter:
    """Plain decimal tick labels: no offset text to collide with a legend above the axes."""
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    return formatter


def _tick_log_axis(axis: Axis) -> None:
    """Give a log axis enough labelled ticks; a short span may hold no decade at all."""
    low, high = axis.get_view_interval()
    span      = high / low
    if span < 2.0:
        # Nearly linear over so short a span
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(_plain_formatter())
        axis.set_minor_locator(NullLocator())
    elif span < 100.0:
        subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if span < 10.0 else (1.0, 2.0, 5.0)
        axis.set_major_locator(LogLocator(subs=subs))
        axis.set_major_formatter(LogFormatterSciNotation(
            labelOnlyBase=False,
            minor_thresholds=(np.inf, np.inf),
        ))
        axis.set_minor_formatter(NullFormatter())


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@plt.rc_context(_STYLE)
def plot_training_history(
    history: dict,
    config: Config = cfg,
    save: bool = True,
    filename: str = "training_history.png",
) -> Figure:
    """
    Plot the losses and the learning rate in two panels sharing the epoch axis.

    Parameters
    ----------
    history  : dict returned by Trainer.fit()
        train_loss, val_loss, lr : lists with one entry per epoch
        best_epoch               : 1-based; absent, 0 or None draws no marker
    config   : Config
    save     : write the figure to config.results_dir
    filename : file name inside config.results_dir

    Returns
    -------
    matplotlib.figure.Figure
    """
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss   = np.asarray(history["val_loss"], dtype=float)
    lr         = np.asarray(history["lr"], dtype=float)
    best_epoch = history.get("best_epoch")
    epochs     = np.arange(1, len(train_loss) + 1)
    marker     = "o" if len(epochs) == 1 else None   # one point alone draws no line

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(_WIDTH, 9.5 * _CM),
        height_ratios=(3, 1),
        layout="constrained",
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, train_loss, color=_SERIES_1, marker=marker, label="Training")
    ax_loss.plot(epochs, val_loss, color=_SERIES_2, marker=marker, linestyle="--",
                 label="Validation")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel(f"{config.model.loss.upper()} loss")
    ax_loss.grid(axis="y")
    ax_loss.tick_params(axis="x", length=0)
    _legend_above(ax_loss)

    # ── Learning rate ─────────────────────────────────────────────────────
    # Constant within an epoch and halved on a plateau: a step line on a log axis
    ax_lr.step(epochs, lr, where="post", color=_SERIES_3, marker=marker)
    ax_lr.set_yscale("log")
    levels = np.unique(lr)
    if len(levels) > _LR_MAX_TICKS:
        levels = levels[np.linspace(0, len(levels) - 1, _LR_MAX_TICKS).round().astype(int)]
    ax_lr.set_yticks(levels, labels=[f"{v:.3g}" for v in levels])
    ax_lr.minorticks_off()
    ax_lr.grid(axis="y")
    ax_lr.annotate(
        f"{lr[-1]:.3g}",
        xy=(epochs[-1], lr[-1]),
        xytext=(4.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if len(epochs) > 1:
        ax_lr.set_xlim(1, len(epochs))
    else:
        ax_lr.set_xlim(0.5, 1.5)   # centres the only epoch, whose marker an edge would clip

    # ── Best epoch ────────────────────────────────────────────────────────
    if best_epoch:
        low, high = ax_loss.get_ylim()
        ax_loss.set_ylim(low, high * (high / low) ** _HEADROOM)
        for ax in (ax_loss, ax_lr):
            ax.axvline(best_epoch, color=_INK_SOFT, linewidth=0.75)
        _label_vline(ax_loss, best_epoch, f"Best epoch {best_epoch}")
    _tick_log_axis(ax_loss.yaxis)
    fig.align_ylabels((ax_loss, ax_lr))

    return _finalize(fig, save, filename, config)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

@plt.rc_context(_STYLE)
def plot_reconstruction(
    model: nn.Module,
    spectrograms: np.ndarray,
    n_examples: int = 4,
    config: Config = cfg,
    save: bool = True,
    filename: str = "reconstruction_examples.png",
    titles: list[str] | None = None,
) -> Figure:
    """
    Plot input, reconstruction and squared error for evenly spaced examples.

    Parameters
    ----------
    model        : trained autoencoder; set to eval mode and run on its own device
    spectrograms : np.ndarray, shape (N, H, W), normalised to [0, 1]
    n_examples   : number of columns, at most N
    config       : Config
    save         : write the figure to config.results_dir
    filename     : file name inside config.results_dir
    titles       : one label per column, or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_examples = min(n_examples, len(spectrograms))
    if titles is not None and len(titles) != n_examples:
        raise ValueError(f"Got {len(titles)} titles for {n_examples} columns")

    indices = np.linspace(0, len(spectrograms) - 1, n_examples).astype(int)
    inputs  = np.asarray(spectrograms)[indices].astype(np.float32)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # (n, H, W) → (n, 1, H, W) → (n, H, W)
        batch   = torch.from_numpy(inputs).unsqueeze(1).to(device)
        outputs = model(batch).squeeze(1).cpu().numpy()
    errors    = (inputs - outputs) ** 2
    error_max = float(errors.max()) or 1.0   # an all-zero error still needs a scale

    fig, axes = plt.subplots(
        3, n_examples,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(_WIDTH, 15.0 * _CM),
        layout="constrained",
    )
    rows = (
        ("Input",          inputs,  _CMAP_MAGNITUDE, 1.0),
        ("Reconstruction", outputs, _CMAP_MAGNITUDE, 1.0),
        ("Squared error",  errors,  _CMAP_ERROR,     error_max),
    )

    H, W = inputs.shape[1:]
    for i, (row_name, images, cmap, vmax) in enumerate(rows):
        for j in range(n_examples):
            ax = axes[i, j]
            ax.imshow(images[j], origin="lower", aspect="auto", cmap=cmap,
                      vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.spines[["left", "bottom"]].set_visible(False)
            ax.tick_params(left=j == 0, bottom=i == len(rows) - 1)
        # A title, unlike an annotation, can overhang its column without widening the gap
        axes[i, 0].set_title(row_name, loc="left", fontweight="bold")
        # The bins are not fftshifted: scipy puts the negative frequencies in the upper half
        axes[i, 0].set_ylabel(f"Frequency bin\n(0 = DC, ≥ {H // 2} negative)")

    # Twice as tall as the error bar, so the same aspect would make it twice as wide
    magnitude_bar = fig.colorbar(
        axes[0, 0].images[0], ax=axes[:2, :], aspect=40, ticks=(0.0, 0.5, 1.0),
        label="Normalised magnitude (dB scale)",
    )
    error_bar = fig.colorbar(
        axes[2, 0].images[0], ax=axes[2, :], aspect=20, ticks=(0.0, error_max / 2.0, error_max),
        format=FuncFormatter(lambda v, _: f"{v:.2g}"), label="Squared error",
    )
    for bar in (magnitude_bar, error_bar):
        bar.outline.set_visible(False)

    axes[0, 0].set_xticks((0, W // 2, W - 1))
    axes[0, 0].set_yticks((0, H // 4, H // 2, 3 * H // 4, H - 1))
    # Frames past the real STFT output are padding (0 dB, so 1.0 once normalised);
    # say so, since they print as dark as the strongest signal
    pc     = config.preprocess
    n_real = min((pc.segment_length - pc.n_fft) // pc.hop_length + 1, W)
    xlabel = "Time frame" if n_real >= W else f"Time frame\n(≥ {n_real} padded)"
    axes[-1, 0].set_xlabel(xlabel)   # once only: narrow columns would collide
    if titles is not None:
        for ax, title in zip(axes[0, :], titles):
            ax.set_title(title, y=_TITLE_Y)

    return _finalize(fig, save, filename, config)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction error
# ─────────────────────────────────────────────────────────────────────────────

@plt.rc_context(_STYLE)
def plot_error_histogram(
    errors: np.ndarray,
    anomaly_errors: np.ndarray | None = None,
    threshold: float | None = None,
    config: Config = cfg,
    save: bool = True,
    filename: str = "error_histogram.png",
) -> Figure:
    """
    Plot the distribution of reconstruction errors, with anomalies if given.

    Both series share one set of bin edges. Two series are drawn as densities
    because normal segments outnumber anomalies 19 to 1. Errors spanning more
    than _LOG_RATIO use log-spaced bins on a log axis, where the density is per
    decade: per unit error, the wide bins of a high-error cluster would flatten
    it to nothing beside a narrow low-error cluster.

    Parameters
    ----------
    errors         : np.ndarray, shape (N,), errors of the normal segments
    anomaly_errors : np.ndarray, shape (M,), errors of the anomalous segments, or None
    threshold      : decision threshold to mark, or None
    config         : Config
    save           : write the figure to config.results_dir
    filename       : file name inside config.results_dir

    Returns
    -------
    matplotlib.figure.Figure
    """
    errors = np.asarray(errors, dtype=float).ravel()
    if anomaly_errors is None:
        anomalies = np.empty(0)
    else:
        anomalies = np.asarray(anomaly_errors, dtype=float).ravel()
    as_density = len(anomalies) > 0
    values     = np.concatenate((errors, anomalies))
    use_log    = _use_log_scale(values)

    fig, ax = plt.subplots(figsize=(_WIDTH, 8.0 * _CM), layout="constrained")
    if use_log:
        edges = np.geomspace(values[values > 0].min(), values.max(), _HIST_BINS + 1)
        ax.set_xscale("log")
    else:
        edges = np.histogram_bin_edges(values, bins=_HIST_BINS)

    series = [(errors, _SERIES_1, None, "Normal")]
    if as_density:
        series.append((anomalies, _SERIES_2, _TEXTURE, "Anomalous"))
    for data, color, texture, label in series:
        heights = _bin_heights(data, edges, as_density, use_log)
        ax.stairs(
            np.where(heights > 0, heights, np.nan),   # no outline along an empty bin
            edges,
            fill=True,
            facecolor=to_rgba(color, _WASH),
            edgecolor=color,
            hatch=texture,
            linewidth=1.0,
            label=label,
        )

    ax.set_xlabel("Reconstruction error (MSE)")
    if not as_density:
        ax.set_ylabel("Segments")
    else:
        ax.set_ylabel("Density per decade" if use_log else "Density")
    ax.grid(axis="y")
    ax.margins(x=0.02)
    ax.set_ylim(0, ax.get_ylim()[1] * (1.0 + _HEADROOM))

    if threshold is not None:
        ax.axvline(threshold, color=_INK_SOFT, linestyle="--", linewidth=1.0)
        _label_vline(ax, threshold, f"Threshold = {threshold:.3g}")
    if as_density:
        _legend_above(ax)

    if use_log:
        _tick_log_axis(ax.xaxis)
    else:
        ax.xaxis.set_major_formatter(_plain_formatter())

    return _finalize(fig, save, filename, config)


@plt.rc_context(_STYLE)
def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray | None = None,
    config: Config = cfg,
    save: bool = True,
    filename: str = "anomaly_scores.png",
) -> Figure:
    """
    Plot the anomaly score of every segment against the threshold.

    Parameters
    ----------
    scores    : np.ndarray, shape (N,)
    threshold : decision threshold
    labels    : np.ndarray, shape (N,), 1 = anomalous, or None when unknown
    config    : Config
    save      : write the figure to config.results_dir
    filename  : file name inside config.results_dir

    Returns
    -------
    matplotlib.figure.Figure
    """
    scores = np.asarray(scores, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(_WIDTH, 7.5 * _CM), layout="constrained")
    # One series, named by the y-axis, so the legend lists only the reference marks
    ax.plot(np.arange(len(scores)), scores, color=_SERIES_1, linestyle="none", marker="o",
            markersize=_SCORE_MARKER, markeredgewidth=0, rasterized=True)
    ax.axhline(threshold, color=_INK_SOFT, linestyle="--", linewidth=1.0,
               label=f"Threshold = {threshold:.3g}")

    if labels is not None:
        flags = np.asarray(labels).astype(bool).ravel()
        if len(flags) != len(scores):
            raise ValueError(f"Got {len(flags)} labels for {len(scores)} scores")
        # +1 where a run of anomalous segments starts, -1 just past where it ends
        steps = np.diff(flags.astype(np.int8), prepend=0, append=0)
        runs  = zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1))
        for k, (start, stop) in enumerate(runs):
            ax.axvspan(start - 0.5, stop - 0.5, **_BAND,
                       label="Anomalous range" if k == 0 else None)

    ax.set_xlabel("Segment index")
    ax.set_ylabel("Anomaly score (MSE)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=(1, 2, 5, 10), integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, max(len(scores), 1))
    ax.grid(axis="y")
    _legend_above(ax)

    if _use_log_scale(np.append(scores, threshold)):
        ax.set_yscale("log")
        _tick_log_axis(ax.yaxis)
    else:
        ax.yaxis.set_major_formatter(_plain_formatter())

    return _finalize(fig, save, filename, config)


# ─────────────────────────────────────────────────────────────────────────────
# Detection performance
# ─────────────────────────────────────────────────────────────────────────────

@plt.rc_context(_STYLE)
def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    config: Config = cfg,
    save: bool = True,
    filename: str = "roc_curve.png",
) -> Figure:
    """
    Plot the ROC curve of the anomaly scores, with its AUC in the legend.

    Parameters
    ----------
    scores   : np.ndarray, shape (N,)
    labels   : np.ndarray, shape (N,), 1 = anomalous, 0 = normal
    config   : Config
    save     : write the figure to config.results_dir
    filename : file name inside config.results_dir

    Returns
    -------
    matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(_WIDTH, 11.5 * _CM), layout="constrained")
    # Unclipped, so a curve running along the panel edge keeps its full width
    ax.plot(fpr, tpr, color=_SERIES_1, clip_on=False, zorder=3,
            label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot((0.0, 1.0), (0.0, 1.0), color=_MUTED, linewidth=1.0, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid()
    _legend_above(ax)

    return _finalize(fig, save, filename, config)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrum
# ─────────────────────────────────────────────────────────────────────────────

@plt.rc_context(_STYLE)
def plot_psd(
    samples: np.ndarray,
    config: Config = cfg,
    save: bool = True,
    filename: str = "psd.png",
    nominal_band_hz: float = 1.08e6,
) -> Figure:
    """
    Plot the Welch power spectral density of an IQ capture.

    The y range never drops below _PSD_MIN_SPAN_DB, so a flat spectrum looks flat
    instead of having its ripple stretched to fill the panel.

    Parameters
    ----------
    samples         : np.ndarray, complex64, shape (N,)
    config          : Config
    save            : write the figure to config.results_dir
    filename        : file name inside config.results_dir
    nominal_band_hz : width of the nominal occupied band to shade, centred on 0 Hz

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs, pxx = sp_signal.welch(
        samples,
        fs=config.sdr.sample_rate,
        nperseg=_PSD_NPERSEG,
        detrend=False,           # the default removes each block's mean: a false notch at DC
        return_onesided=False,   # complex IQ → two-sided spectrum
    )
    freq_khz = np.fft.fftshift(freqs) / 1e3
    psd_db   = 10.0 * np.log10(np.fft.fftshift(pxx) + _PSD_FLOOR)

    low, high = float(psd_db.min()), float(psd_db.max())
    margin    = max((_PSD_MIN_SPAN_DB - (high - low)) / 2.0, 0.05 * (high - low))
    half_band = nominal_band_hz / 2e3   # Hz → kHz, then half of it

    fig, ax = plt.subplots(figsize=(_WIDTH, 8.0 * _CM), layout="constrained")
    ax.axvspan(-half_band, half_band, **_BAND)
    ax.plot(freq_khz, psd_db, color=_SERIES_1)
    ax.annotate(
        f"Nominal occupied band ({nominal_band_hz / 1e6:g} MHz)",
        xy=(0.0, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0.0, 3.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_xlim(freq_khz[0], freq_khz[-1])
    ax.set_ylim(low - margin, high + margin)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=(1, 2, 5, 10)))
    ax.grid(axis="y")

    return _finalize(fig, save, filename, config)
