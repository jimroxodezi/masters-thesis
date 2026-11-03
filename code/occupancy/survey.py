"""
survey.py
---------
Band survey with rtl_power (design section 2).

* runs ``rtl_power`` over a frequency range with a FIXED gain (``-g``; the
  tool's automatic gain is never used) and without ``-T`` (bias tee off),
  saves the CSV with a JSON sidecar (UTC times, position, day/night label,
  notes, software version);
* parses rtl_power CSVs (also existing ones);
* plots mean power against frequency and a waterfall to PNG;
* lists plateaus (LTE carriers appear as flat plateaus 5-20 MHz wide) and idle
  stretches at least 1.92 MHz wide, nearest the carrier first, as candidate
  Class B windows (design section 3).

The plateau and idle-window lists are visual aids only; carriers are confirmed
with an LTE cell-search tool, and the Class B window is confirmed free with
survey plus cell search at capture time.

rtl_power CSV rows: date, time, Hz low, Hz high, Hz step, samples, dB, dB, ...
All hops of one integration interval share one timestamp (local time of the
host); bin i of a row lies at Hz low + i * Hz step.
"""

from __future__ import annotations

import csv
import shlex
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import provenance
from .capture import validate_gain
from .config import Config


@dataclass
class SurveyData:
    freqs_hz: np.ndarray        # (F,) bin frequencies, ascending
    times: list                 # (T,) timestamps as written by rtl_power (local time)
    power_db: np.ndarray        # (T, F) power per bin in dB (rtl_power's uncalibrated scale)

    @property
    def mean_db(self) -> np.ndarray:
        """Mean power over sweeps (averaged in linear power), dB."""
        return 10.0 * np.log10(np.mean(10.0 ** (self.power_db / 10.0), axis=0))


# ── rtl_power ─────────────────────────────────────────────────────────────────

def build_rtl_power_cmd(cfg: Config, f_start_hz: float, f_stop_hz: float, gain_db, out_csv: Path,
                        bin_hz=None, integration_s=None, duration=None, ppm=None, crop=None,
                        window=None, device_index: int = 0) -> list[str]:
    s = cfg.survey
    gain_db = validate_gain(gain_db)
    if f_stop_hz <= f_start_hz:
        raise ValueError("stop frequency must exceed start frequency")
    if f_stop_hz > cfg.capture.max_centre_freq_hz:
        raise ValueError(f"stop frequency {f_stop_hz / 1e6:.1f} MHz exceeds the tuner limit "
                         f"{cfg.capture.max_centre_freq_hz / 1e6:.0f} MHz")
    cmd = [s.rtl_power,
           "-f", f"{int(f_start_hz)}:{int(f_stop_hz)}:{int(bin_hz or s.bin_hz)}",
           "-g", f"{gain_db:g}",
           "-i", str(int(integration_s or s.integration_s)),
           "-e", str(duration or s.duration),
           "-p", str(int(cfg.capture.ppm if ppm is None else ppm)),
           "-d", str(int(device_index))]
    crop = s.crop_fraction if crop is None else crop
    if crop:
        cmd += ["-c", f"{float(crop):g}"]
    window = s.window if window is None else window
    if window:
        cmd += ["-w", window]
    # never "-T": the bias tee stays off (design section 1)
    cmd.append(str(out_csv))
    return cmd


def run_survey(cfg: Config, f_start_hz: float, f_stop_hz: float, gain_db, out_dir: Path, *,
               label: str = "", position_id: str = "", notes: str = "", carrier_hz=None,
               runner=subprocess.run, log=print, bias_tee_note: str = "off (-T not passed)",
               **kw) -> dict:
    """Run one rtl_power survey, then parse and plot it. Returns the output paths."""
    if shutil.which(cfg.survey.rtl_power) is None and runner is subprocess.run:
        raise FileNotFoundError(f"{cfg.survey.rtl_power} not found; install the RTL-SDR Blog "
                                "driver build (see README)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = provenance.utc_now()
    stem = (f"survey_{t0.strftime('%Y%m%dT%H%M%SZ')}_{int(f_start_hz / 1e6)}-{int(f_stop_hz / 1e6)}MHz"
            + (f"_{label}" if label else "") + (f"_{position_id}" if position_id else ""))
    csv_path = out_dir / f"{stem}.csv"
    cmd = build_rtl_power_cmd(cfg, f_start_hz, f_stop_hz, gain_db, csv_path, **kw)
    log("running: " + " ".join(shlex.quote(c) for c in cmd))
    side = {"kind": "survey", "synthetic": False, "command": cmd, "utc_start": provenance.utc_iso(t0),
            "f_start_hz": f_start_hz, "f_stop_hz": f_stop_hz, "gain_db": float(gain_db),
            "gain_mode": "manual (-g)", "bias_tee": bias_tee_note, "label": label,
            "position_id": position_id, "notes": notes,
            "timestamps_in_csv": "local time of the host (rtl_power)",
            "software_version": provenance.software_version()}
    try:
        result = runner(cmd, check=False)
        side["return_code"] = getattr(result, "returncode", None)
    except KeyboardInterrupt:          # Ctrl-C also stops rtl_power; keep what it wrote
        side["interrupted_by_operator"] = True
        side["return_code"] = None
        log("survey interrupted; analysing the sweeps written so far")
    side["utc_end"] = provenance.utc_iso()
    provenance.write_json(csv_path.with_suffix(".json"), side, durable=True)
    if side["return_code"] not in (0, None):
        raise RuntimeError(f"rtl_power exited with code {side['return_code']}")
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"rtl_power wrote no data to {csv_path}")
    return analyse_survey(cfg, csv_path, carrier_hz=carrier_hz, log=log)


def parse_rtl_power_csv(path: Path) -> SurveyData:
    sweeps: "OrderedDict[str, dict]" = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            row = [c.strip() for c in row]
            stamp = f"{row[0]} {row[1]}"
            low, step = float(row[2]), float(row[4])
            vals = [float(v) if v not in ("", "nan", "-nan") else np.nan for v in row[6:]]
            sweep = sweeps.setdefault(stamp, {})
            for i, v in enumerate(vals):
                sweep[int(round(low + i * step))] = v
    if not sweeps:
        raise ValueError(f"no rtl_power rows in {path}")
    # the frequency grid is the one of the most complete sweep; incomplete sweeps
    # (e.g. the last one, cut off by the exit timer) are dropped
    sizes = {k: len(v) for k, v in sweeps.items()}
    full = max(sizes.values())
    keep = [k for k, n in sizes.items() if n == full]
    freqs = np.array(sorted(sweeps[keep[0]]), dtype=np.float64)
    power = np.array([[sweeps[k].get(int(fq), np.nan) for fq in freqs] for k in keep])
    return SurveyData(freqs_hz=freqs, times=keep, power_db=power)


# ── analysis aids ─────────────────────────────────────────────────────────────

def find_plateaus(freqs_hz: np.ndarray, mean_db: np.ndarray, threshold_db: float,
                  min_width_hz: float, merge_gap_hz: float = 200e3) -> tuple[list[dict], float]:
    """Contiguous runs at least threshold_db above the floor (5th percentile)."""
    m = np.asarray(mean_db, dtype=np.float64)
    floor = float(np.nanpercentile(m, 5))
    above = np.nan_to_num(m, nan=-np.inf) > floor + threshold_db
    runs, start = [], None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        if (not a or i == len(above) - 1) and start is not None:
            stop = i if a else i - 1
            runs.append([start, stop])
            start = None
    merged = []
    for r in runs:
        if merged and freqs_hz[r[0]] - freqs_hz[merged[-1][1]] <= merge_gap_hz:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    step = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 0.0
    out = []
    for a, b in merged:
        lo, hi = freqs_hz[a] - step / 2, freqs_hz[b] + step / 2
        if hi - lo >= min_width_hz:
            out.append({"start_hz": lo, "stop_hz": hi, "width_hz": hi - lo, "centre_hz": (lo + hi) / 2,
                        "level_db": float(np.nanmedian(m[a:b + 1])),
                        "above_floor_db": float(np.nanmedian(m[a:b + 1]) - floor)})
    return out, floor


def suggest_b_windows(cfg: Config, freqs_hz: np.ndarray, plateaus: list[dict],
                      carrier_hz: float | None = None) -> list[dict]:
    """Idle stretches that can hold a 1.92 MHz window (design section 3, Class B).

    The window is kept ``b_window_margin_hz`` away from every plateau and from
    the edges of the sweep; within each stretch it is placed as close to the
    carrier as possible. Nearest first when the carrier is given.
    """
    half = cfg.capture.sample_rate_hz / 2
    margin = cfg.survey.b_window_margin_hz
    lo_edge, hi_edge = float(freqs_hz[0]), float(freqs_hz[-1])
    busy = sorted((p["start_hz"], p["stop_hz"]) for p in plateaus)
    gaps, cursor = [], lo_edge
    for a, b in busy:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if hi_edge > cursor:
        gaps.append((cursor, hi_edge))
    out = []
    for a, b in gaps:
        lo, hi = a + margin + half, b - margin - half       # allowed window centres
        if hi < lo:
            continue
        target = carrier_hz if carrier_hz is not None else (lo + hi) / 2
        centre = float(min(max(target, lo), hi))
        out.append({"gap_start_hz": a, "gap_stop_hz": b, "gap_width_hz": b - a,
                    "window_centre_hz": centre,
                    "distance_to_carrier_hz": None if carrier_hz is None else abs(centre - carrier_hz)})
    if carrier_hz is not None:
        out.sort(key=lambda d: d["distance_to_carrier_hz"])
    return out


def plot_survey(data: SurveyData, out_prefix: Path, title: str = "", plateaus=(), suggestions=(),
                floor_db: float | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_prefix = Path(out_prefix)
    f_mhz = data.freqs_hz / 1e6
    paths = []
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(f_mhz, data.mean_db, lw=0.8, color="#1f4e79", label="mean power")
    if data.power_db.shape[0] > 1:
        ax.plot(f_mhz, np.nanmax(data.power_db, axis=0), lw=0.5, color="#999999", label="max hold")
    for p in plateaus:
        ax.axvspan(p["start_hz"] / 1e6, p["stop_hz"] / 1e6, color="#e69f00", alpha=0.2)
    for s in list(suggestions)[:3]:
        c = s["window_centre_hz"] / 1e6
        ax.axvspan(c - 0.96, c + 0.96, color="#009e73", alpha=0.25)
    if floor_db is not None:
        ax.axhline(floor_db, color="k", lw=0.5, ls=":", label="floor (5th percentile)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB, rtl_power scale)")
    ax.set_title(title or "Mean power (orange: plateaus, green: candidate idle 1.92 MHz windows)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = out_prefix.with_name(out_prefix.name + "_mean.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(data.power_db, aspect="auto", origin="upper", interpolation="nearest",
                   extent=[f_mhz[0], f_mhz[-1], data.power_db.shape[0], 0], cmap="viridis")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Sweep number (time runs downwards)")
    ax.set_title("Waterfall")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    p2 = out_prefix.with_name(out_prefix.name + "_waterfall.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths


def analyse_survey(cfg: Config, csv_path: Path, carrier_hz=None, log=print) -> dict:
    data = parse_rtl_power_csv(csv_path)
    mean_db = data.mean_db
    plateaus, floor = find_plateaus(data.freqs_hz, mean_db, cfg.survey.plateau_threshold_db,
                                    cfg.survey.plateau_min_width_hz)
    sugg = suggest_b_windows(cfg, data.freqs_hz, plateaus, carrier_hz)
    pngs = plot_survey(data, Path(csv_path).with_suffix(""), Path(csv_path).name, plateaus, sugg, floor)
    log(f"{len(data.times)} complete sweeps, {len(data.freqs_hz)} bins, "
        f"{data.freqs_hz[0] / 1e6:.3f}-{data.freqs_hz[-1] / 1e6:.3f} MHz; floor {floor:.1f} dB")
    log("plateaus (visual aid; confirm carriers with cell search):")
    for p in plateaus:
        log(f"  {p['start_hz'] / 1e6:9.3f} - {p['stop_hz'] / 1e6:9.3f} MHz  width {p['width_hz'] / 1e6:6.2f} MHz  "
            f"centre {p['centre_hz'] / 1e6:9.3f} MHz  +{p['above_floor_db']:.1f} dB")
    log("candidate Class B windows (idle stretch >= 1.92 MHz; confirm with cell search):")
    for s in sugg[:5]:
        d = s["distance_to_carrier_hz"]
        log(f"  centre {s['window_centre_hz'] / 1e6:9.3f} MHz in idle stretch "
            f"{s['gap_start_hz'] / 1e6:.3f}-{s['gap_stop_hz'] / 1e6:.3f} MHz"
            + ("" if d is None else f", {d / 1e6:.2f} MHz from the carrier"))
    summary = {"csv": str(csv_path), "pngs": [str(p) for p in pngs], "n_sweeps": len(data.times),
               "floor_db": floor, "plateaus": plateaus, "b_window_candidates": sugg,
               "carrier_hz": carrier_hz}
    provenance.write_json(Path(csv_path).with_name(Path(csv_path).stem + "_analysis.json"), summary)
    return summary
