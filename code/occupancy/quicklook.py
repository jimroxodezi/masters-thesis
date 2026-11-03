"""
quicklook.py
------------
Per-capture quick-look figure and the labelling check (design section 3).

The quick-look figure shows a capture's mean spectrum (first few hundred ms)
and one spectrogram sample, so the operator can check the class right after
recording, and so the second labelling pass on a later day can inspect every
capture. ``labelling_template`` writes a CSV with one row per capture for that
second pass; ``labelling_agreement`` reports the agreement once it is filled.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import Config
from .dataset import CaptureIndex, dc_remove
from .iqio import open_raw, u8_to_complex
from .preprocess import freq_axis_hz, stft_db


def capture_quicklook(raw_path: Path, meta: dict, cfg: Config, out_png: Path, n_frames: int = 32) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = cfg.preprocess.frame_len
    raw = open_raw(raw_path)
    n = int(min(n_frames, raw.size // (2 * N)))
    if n == 0:
        raise ValueError("capture shorter than one frame")
    y = dc_remove(u8_to_complex(np.asarray(raw[:2 * N * n])).reshape(n, N)).astype(np.complex64)
    S = stft_db(y, cfg)                                        # (n, K, K) dB
    mean_spec = 10 * np.log10(np.mean(10 ** (S.astype(np.float64) / 10), axis=(0, 2)))
    f_khz = freq_axis_hz(cfg) / 1e3
    fc = meta["radio"]["centre_freq_hz"] / 1e6
    m = meta["measurements"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(f_khz, mean_spec, lw=0.8)
    a1.set_xlabel(f"frequency offset from {fc:.4f} MHz (kHz)")
    a1.set_ylabel("mean power per bin (dB)")
    a1.grid(alpha=0.3)
    a1.set_title(f"mean spectrum, first {n * N / cfg.capture.sample_rate_hz * 1e3:.0f} ms")
    T_ms = cfg.frame_duration_s * 1e3
    im = a2.imshow(S[0], aspect="auto", extent=[0, T_ms, f_khz[-1], f_khz[0]], cmap="viridis")
    a2.set_xlabel("time (ms)")
    a2.set_ylabel("frequency offset (kHz)")
    a2.set_title("spectrogram of frame 0 (128 x 128)")
    fig.colorbar(im, ax=a2, label="dB")
    syn = "  [SYNTHETIC]" if meta.get("synthetic") else ""
    fig.suptitle(f"{meta['capture_id']}  class {meta['class']}  {meta['gain_role']} = "
                 f"{meta['radio']['gain_db']:g} dB  clipping {100 * m['clipping_fraction']:.3f} %  "
                 f"mean power {m['mean_power_db']:.1f} dB{syn}", fontsize=9)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def labelling_template(index: CaptureIndex, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capture_id", "session_id", "position_id", "gain_role", "centre_freq_hz",
                    "first_pass_class", "second_pass_class", "cell_search_rerun", "second_pass_notes"])
        for r in index.records:
            w.writerow([r.capture_id, r.session_id, r.position_id, r.gain_role, r.centre_freq_hz,
                        r.cls, "", "", ""])
    return out_csv


def labelling_agreement(csv_path: Path) -> dict:
    """Agreement between the first and second labelling pass (rows with a second pass)."""
    occupied = {"A": 0, "B": 0, "C_centre": 1, "C_offset": 1}
    n = same_cls = same_occ = 0
    disagreements = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            second = (row.get("second_pass_class") or "").strip()
            if not second:
                continue
            first = row["first_pass_class"].strip()
            if second not in occupied:
                raise ValueError(f"{row['capture_id']}: unknown class {second!r}")
            n += 1
            same_cls += first == second
            same_occ += occupied[first] == occupied[second]
            if first != second:
                disagreements.append({"capture_id": row["capture_id"], "first": first, "second": second})
    return {"n_checked": n, "class_agreement": same_cls / n if n else None,
            "occupancy_label_agreement": same_occ / n if n else None, "disagreements": disagreements}
