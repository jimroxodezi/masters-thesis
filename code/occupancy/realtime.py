"""
realtime.py
-----------
Streaming demonstration (design section 11).

Capture at 1.92 MS/s in blocks of N = 16384 samples (one frame per USB block),
then DC removal, STFT, normalisation and the chosen detector with its
calibrated threshold (fold, seed and P_FA chosen on the command line). Every
frame is logged to CSV: UTC timestamp (arrival of the block), centre
frequency, gain, score, threshold, decision and processing time. A bounded
queue (32 frames, 0.27 s) separates the USB reader from the processing; if
processing falls behind, the OLDEST blocks are dropped and counted (the count
is logged), so the decisions stay current instead of lagging further and
further behind the receiver.

Samples lost before they reach the program (a USB overflow) cannot be seen in
the data, so the demo estimates them from the arrival times: from the first to
the last block, the receiver delivers f_s / N blocks per second, and any
shortfall is reported as missing blocks.

The demo is meant to run on mains power with the Performance profile (design
section 10); in any other known power state it warns, and the state is
recorded with the results.

Optional live view (--live): rolling spectrum waterfall and occupancy strip,
redrawn at most 4 times per second (one redraw costs about 40 ms on the
laptop) and not at all while frames are waiting, so it never slows the
detector down.
--synthetic CLASS replaces the receiver by the synthetic generator (testing
only; the CSV is marked synthetic).
"""

from __future__ import annotations

import csv
import datetime as _dt
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import provenance
from .calibrate import load_thresholds, pfa_key
from .config import Config
from .dataset import dc_remove
from .detectors import ED_ADAPTIVE, ED_FIXED, PCA, PSS, PCAModel, RunningMedian
from .iqio import u8_to_complex
from .models import inference_model, model_input, per_score
from .preprocess import Normalizer, frame_energy, stft_db
from .pss import correlator_from_config
from .scoring import seed_tag
from .train import fold_dir, load_model


class FramePipeline:
    """One detector with its calibrated threshold, applied frame by frame."""

    def __init__(self, cfg: Config, work_dir: Path, detector: str, fold: int, seed: int | None,
                 pfa: float):
        self.cfg, self.detector, self.pfa = cfg, detector, pfa
        th = load_thresholds(work_dir, fold)
        self.synthetic_model = bool(th.get("synthetic_data"))   # thresholds/models from synthetic data
        key = f"{ED_ADAPTIVE}@{pfa:g}" if detector == ED_ADAPTIVE else detector
        learned = detector not in (ED_FIXED, ED_ADAPTIVE, PSS, PCA)
        tag = seed_tag(seed if learned else None)
        try:
            self.threshold = float(th["thresholds"][key][tag][pfa_key(pfa)]["threshold"])
        except KeyError as exc:
            raise KeyError(f"no calibrated threshold for {key} {tag} at P_FA {pfa} in fold {fold}") from exc
        self.norm = Normalizer.load(fold_dir(work_dir, fold) / "normalizer.json")
        self.model = self.spec = self.pca = None
        self.score_kind = "mse"
        if learned:
            base = detector
            if detector.endswith("_mse") or detector.endswith("_per"):
                base, self.score_kind = detector[:-4], detector[-3:]
            path = fold_dir(work_dir, fold) / "models" / base / f"seed{seed}" / "model.pt"
            if not path.exists():
                raise FileNotFoundError(f"no trained {base} for fold {fold} seed {seed} ({path})")
            model, self.spec = load_model(path, cfg)
            self.model = inference_model(model)          # as scored in the evaluation
        elif detector == PCA:
            self.pca = PCAModel.load(fold_dir(work_dir, fold) / "models" / PCA / "pca.npz")
        elif detector == PSS:
            self.correlator = correlator_from_config(cfg)
        elif detector == ED_ADAPTIVE:
            a = th["ed_adaptive"][pfa_key(pfa)]
            self.median = RunningMedian(cfg.models.ed.adaptive_window, a["init_sigma2"])
        self.uses_spec = learned or detector == PCA

    def process(self, raw: np.ndarray):
        """raw: 2N uint8 bytes -> (score, decision, spectrogram dB or None)."""
        y = dc_remove(u8_to_complex(raw)[None, :]).astype(np.complex64)
        S = None
        if self.uses_spec:
            S = stft_db(y, self.cfg)
            x = self.norm.transform(S)
            if self.pca is not None:
                score = float(self.pca.score(x.reshape(1, -1))[0])
            else:
                with torch.no_grad():
                    xt = model_input(self.model, x)
                    xh = self.model.reconstruct(xt)
                    if self.score_kind == "per":
                        p = self.cfg.models.per
                        score = float(per_score(xt, xh, p.gamma, p.alpha, p.beta, p.xi, p.eta)[0])
                    else:
                        score = float(((xh - xt) ** 2).mean())
        elif self.detector == ED_FIXED:
            score = float(frame_energy(y)[0])
        elif self.detector == ED_ADAPTIVE:
            T = float(frame_energy(y)[0])
            score = T / self.median.median()
            if score <= self.threshold:
                self.median.push(T)
        else:
            score = float(self.correlator(y)[0])
        return score, bool(score > self.threshold), S


def _hardware_reader(cfg: Config, centre_hz: float, gain_db: float, q: queue.Queue, stop: threading.Event,
                     stats: dict, device_index: int, allow_other_receiver: bool, log):
    from .capture import check_receiver, configure_receiver, open_receiver, receiver_identity
    sdr = open_receiver(device_index)
    try:
        ident = receiver_identity(sdr, cfg)
        for w in check_receiver(ident, cfg, allow_other_receiver):
            log("WARNING: " + w)
        stats["radio"] = configure_receiver(sdr, cfg, centre_hz, gain_db)
        stats["receiver"] = {k: ident.get(k) for k in ("manufacturer", "product", "serial", "tuner")}
        N = cfg.preprocess.frame_len

        def cb(values, _ctx):
            if stop.is_set():
                sdr.cancel_read_async()
                return
            buf = np.array(np.ctypeslib.as_array(values), dtype=np.uint8, copy=True)
            _note_arrival(stats)
            _put_latest(q, (time.time(), buf), stats)

        def watcher():
            stop.wait()
            try:
                sdr.cancel_read_async()
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=watcher, daemon=True).start()
        sdr.read_bytes_async(cb, num_bytes=2 * N)
    finally:
        sdr.close()
        _put_sentinel(q)


def _note_arrival(stats: dict) -> None:
    t = time.monotonic()
    stats["received"] = stats.get("received", 0) + 1
    stats.setdefault("t_first", t)
    stats["t_last"] = t


def _put_latest(q: queue.Queue, item, stats: dict) -> None:
    """Queue a block; if the queue is full, drop the OLDEST block first (counted)."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
                stats["dropped"] += 1
            except queue.Empty:              # the consumer took one in the meantime
                pass


def missing_blocks(stats: dict, cfg: Config) -> int | None:
    """Blocks the receiver should have delivered between the first and the last
    arrival but did not (lost before reaching the program); None if unknown."""
    n = stats.get("received", 0)
    if n < 2:
        return None
    expected = (stats["t_last"] - stats["t_first"]) * cfg.capture.sample_rate_hz / cfg.preprocess.frame_len
    return max(0, int(round(expected - (n - 1))))


def _put_sentinel(q: queue.Queue) -> None:
    """Tell the consumer the source has ended, without blocking on a full queue."""
    try:
        q.put_nowait(None)
    except queue.Full:
        pass


def _synthetic_reader(cfg: Config, cls: str, gain_db: float, q: queue.Queue, stop: threading.Event,
                      stats: dict, pace: bool):
    from .synthetic import SessionSignalParams, SyntheticReceiver
    rx = SyntheticReceiver(cls, gain_db, np.random.default_rng(0), SessionSignalParams())
    N = cfg.preprocess.frame_len
    period = cfg.frame_duration_s
    t_next = time.monotonic()
    while not stop.is_set():
        buf = rx.read_u8(N)
        if pace:                                 # behaves like the receiver
            t_next += period
            time.sleep(max(0.0, t_next - time.monotonic()))
            _note_arrival(stats)
            _put_latest(q, (time.time(), buf), stats)
            continue
        try:                                     # as fast as the processing takes them
            q.put((time.time(), buf), timeout=0.2)
            _note_arrival(stats)
        except queue.Full:
            stats["dropped"] += 1
    _put_sentinel(q)


def run_demo(cfg: Config, work_dir: Path, out_csv: Path, *, detector: str, fold: int, seed: int | None,
             pfa: float, centre_freq_hz: float, gain_db: float, duration_s: float | None = None,
             synthetic_class: str | None = None, live: bool = False, max_frames: int | None = None,
             pace: bool = True, device_index: int = 0, allow_other_receiver: bool = False,
             log=print) -> dict:
    from .capture import check_centre_freq, validate_gain
    gain_db = validate_gain(gain_db)
    check_centre_freq(centre_freq_hz, cfg)
    pipe = FramePipeline(cfg, work_dir, detector, fold, seed, pfa)
    torch.set_num_threads(cfg.latency.num_threads)
    condition = provenance.power_condition()
    if condition != "performance":
        log(f"WARNING: power condition {condition!r}; for the demonstration, " + provenance.POWER_HINT)
    duration_s = duration_s or cfg.demo.duration_s
    q: queue.Queue = queue.Queue(maxsize=cfg.demo.queue_max_frames)
    stop = threading.Event()
    stats = {"dropped": 0}
    synthetic_source = synthetic_class is not None
    synthetic = synthetic_source or pipe.synthetic_model
    if synthetic_source:
        th = threading.Thread(target=_synthetic_reader, args=(cfg, synthetic_class, gain_db, q, stop,
                                                              stats, pace), daemon=True)
    else:
        th = threading.Thread(target=_hardware_reader, args=(cfg, centre_freq_hz, gain_db, q, stop, stats,
                                                             device_index, allow_other_receiver, log),
                              daemon=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    view = _LiveView(cfg) if live else None
    n = n_occ = 0
    t_end = time.monotonic() + duration_s
    th.start()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc_timestamp", "centre_freq_hz", "gain_db", "detector", "fold", "seed", "pfa_target",
                    "threshold", "score", "decision", "processing_ms", "dropped_blocks", "synthetic"])
        try:
            while time.monotonic() < t_end and (max_frames is None or n < max_frames):
                try:
                    item = q.get(timeout=2.0)
                except queue.Empty:
                    continue
                if item is None:
                    break
                t_arr, buf = item
                t0 = time.perf_counter()
                score, dec, S = pipe.process(buf)
                dt = (time.perf_counter() - t0) * 1e3
                w.writerow([provenance.utc_iso(_dt.datetime.fromtimestamp(t_arr, tz=_dt.timezone.utc)),
                    centre_freq_hz, gain_db, detector, fold,
                    seed_tag(seed), pfa, pipe.threshold, f"{score:.6g}", int(dec), f"{dt:.3f}",
                    stats["dropped"], int(synthetic)])
                n += 1
                n_occ += int(dec)
                if n % 100 == 0:
                    f.flush()
                if view is not None:
                    view.update(S if S is not None else stft_db(
                        dc_remove(u8_to_complex(buf)[None, :]).astype(np.complex64), cfg), dec, q.qsize())
        except KeyboardInterrupt:
            log("stopped by the operator")
        finally:
            stop.set()
            th.join(timeout=5.0)
    missing = missing_blocks(stats, cfg) if not synthetic_source or pace else None
    summary = {"frames": n, "occupied_fraction": n_occ / n if n else None, "dropped_blocks": stats["dropped"],
               "received_blocks": stats.get("received", 0), "missing_blocks_estimate": missing,
               "queue_max_frames": cfg.demo.queue_max_frames, "power_condition": condition,
               "csv": str(out_csv), "detector": detector, "threshold": pipe.threshold, "pfa_target": pfa,
               "synthetic": synthetic, "radio": stats.get("radio"), "receiver": stats.get("receiver")}
    provenance.write_json(out_csv.with_suffix(".json"), {**summary, "run": provenance.run_record(cfg, synthetic)})
    log(f"{n} frames, {n_occ} declared occupied, {stats['dropped']} blocks dropped"
        + (f", about {missing} blocks lost before reaching the program" if missing else "") + f" -> {out_csv}")
    if missing and missing >= 2:
        log("WARNING: samples were lost on the way from the receiver (USB); close other programs "
            "and check the power condition")
    return summary


class _LiveView:
    """Rolling mean-spectrum waterfall and occupancy strip (matplotlib, interactive)."""

    def __init__(self, cfg: Config):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interval = cfg.demo.live_redraw_interval_s
        self.max_backlog = cfg.demo.live_max_backlog_frames
        self.t_last = 0.0
        self.H = cfg.demo.live_history_frames
        K = cfg.preprocess.fft_size
        self.img = np.full((K, self.H), np.nan)
        self.occ = np.zeros((1, self.H))
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[5, 1], sharex=True)
        f = cfg.capture.sample_rate_hz / 2e3
        self.im = self.ax1.imshow(self.img, aspect="auto", extent=[-self.H, 0, -f, f], cmap="viridis")
        self.ax1.set_ylabel("frequency offset (kHz)")
        self.oc = self.ax2.imshow(self.occ, aspect="auto", extent=[-self.H, 0, 0, 1], cmap="Reds", vmin=0, vmax=1)
        self.ax2.set_yticks([])
        self.ax2.set_xlabel("frames (8.533 ms each), newest at 0")
        self.k = 0

    def update(self, S: np.ndarray, decision: bool, backlog: int = 0):
        spec = 10 * np.log10(np.mean(10 ** (np.asarray(S[0], dtype=np.float64) / 10), axis=1))
        self.img = np.roll(self.img, -1, axis=1)
        self.img[:, -1] = spec
        self.occ = np.roll(self.occ, -1, axis=1)
        self.occ[0, -1] = float(decision)
        self.k += 1
        now = time.monotonic()
        if now - self.t_last >= self.interval and backlog <= self.max_backlog:
            self.t_last = now
            self.im.set_data(self.img)
            finite = self.img[np.isfinite(self.img)]
            if finite.size:
                self.im.set_clim(np.percentile(finite, 2), np.percentile(finite, 98))
            self.oc.set_data(self.occ)
            self.plt.pause(0.001)
