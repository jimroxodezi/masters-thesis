"""
gaintrials.py
-------------
Summary of gain-trial captures (design section 1, choice of g1 and g2).

Before the first session, short captures are made at several tuner gain steps
(session ID "trial", then moved out of data/raw/ so that they are never used
as study data). This module summarises a folder of such captures, one row per
capture, sorted by start time:

    utc_start, local time (WAT), class, centre frequency, gain, duration,
    clipping fraction (from the capture record), mean power (dB, from the
    capture record), I/Q standard deviation in ADC steps (LSB), and, for
    captures of at least 10 s, the mean power in 0.5 s blocks (minimum,
    median, maximum and swing), which shows how much the carrier's power
    changes with traffic during a capture.

Command line:  python -m occupancy trials DIR [--out CSV]
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from . import provenance

BLOCK_S = 0.5           # [impl] block length for the power swing
MIN_SWING_S = 10.0      # [impl] captures shorter than this get no swing figures


def summarise_capture(meta_path: Path, utc_offset_h: float = 1.0) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    m, radio = meta["measurements"], meta["radio"]
    raw = np.fromfile(Path(meta_path).with_suffix(".cu8"), dtype=np.uint8)
    i, q = raw[0::2].astype(np.float64), raw[1::2].astype(np.float64)
    fs = float(radio.get("sample_rate_hz") or 1.92e6)
    n = int(m.get("n_samples") or i.size)
    t0 = provenance.parse_utc(meta["utc_start"])
    row = {"utc_start": meta["utc_start"],
           "local_time": (t0 + _dt.timedelta(hours=utc_offset_h)).strftime("%Y-%m-%d %H:%M:%S"),
           "class": meta["class"], "centre_freq_mhz": float(radio["centre_freq_hz"]) / 1e6,
           "gain_db": float(radio["gain_db"]), "duration_s": n / fs,
           "clipping_percent": 100.0 * float(m["clipping_fraction"]),
           "clipping_flagged": bool(m["clipping_flagged"]), "mean_power_db": float(m["mean_power_db"]),
           "iq_std_lsb": float(np.sqrt((i.var() + q.var()) / 2.0)),
           "block_power_min_db": None, "block_power_median_db": None, "block_power_max_db": None,
           "block_power_swing_db": None, "notes": meta.get("context", {}).get("notes", "")}
    if n / fs >= MIN_SWING_S:
        y = (i - 127.5) / 127.5 + 1j * (q - 127.5) / 127.5
        blk = int(BLOCK_S * fs)
        p = np.array([10 * np.log10(np.mean(np.abs(y[k:k + blk]) ** 2)) for k in range(0, y.size - blk + 1, blk)])
        row.update(block_power_min_db=float(p.min()), block_power_median_db=float(np.median(p)),
                   block_power_max_db=float(p.max()), block_power_swing_db=float(p.max() - p.min()))
    return row


def summarise_folder(folder: Path, utc_offset_h: float = 1.0) -> list[dict]:
    rows = [summarise_capture(p, utc_offset_h) for p in sorted(Path(folder).glob("*.json"))
            if Path(p).with_suffix(".cu8").exists()]
    return sorted(rows, key=lambda r: r["utc_start"])


def write_csv(rows: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path
