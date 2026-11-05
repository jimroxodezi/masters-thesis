#!/usr/bin/env python3
"""Figure A.2: gain trials of 5 November 2026 at P1 (choice of g1 and g2).

Reads results/gain_trials_2026-11-05.csv, written by
  uv run python -m occupancy trials data/trials/2026-11-05_gain_trials \
      --out ../results/gain_trials_2026-11-05.csv
and draws (a) the clipping of 2 s captures centred on the carrier (773 MHz)
against the gain, with the 0.1 % limit and the repeat at 37.2 dB taken during
a traffic peak, and (b) the mean power of 2 s idle captures (765.37 MHz)
against the gain. The chosen gains are marked. Measured data. Run from the
repository root:
  uv run --no-project --with matplotlib python figures/src/figA-gains.py figures/figA-gains.pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "gain_trials_2026-11-05.csv"
G1, G2 = 29.7, 19.7
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#2C2C2A", "#5F5E5A", "#E6E4DC"


def main(out):
    lm = "/usr/share/texmf/fonts/opentype/public/lm/"
    for f in ("lmroman10-regular.otf", "lmroman10-italic.otf", "lmroman10-bold.otf"):
        font_manager.fontManager.addfont(lm + f)
    plt.rcParams.update({
        "font.family": "LM Roman 10", "mathtext.fontset": "cm",
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "pdf.fonttype": 3, "savefig.dpi": 300,
    })
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ("gain_db", "duration_s", "clipping_percent", "mean_power_db"):
            r[k] = float(r[k])
    carrier = [r for r in rows if r["class"] == "C_centre"]
    sweep = [r for r in carrier if r["duration_s"] < 10][:-1]          # the descending series
    peak = [r for r in carrier if r["duration_s"] < 10][-1]            # the repeat at 37.2 dB
    long_ = [r for r in carrier if r["duration_s"] >= 10]
    idle = sorted([r for r in rows if r["class"] == "B"], key=lambda r: r["gain_db"])

    fig, (a, b) = plt.subplots(2, 1, figsize=(5.9, 5.6))
    xs = [r["gain_db"] for r in sweep]
    a.plot(xs, [r["clipping_percent"] for r in sweep], "o-", color=BLUE, ms=3.5, lw=1,
           label="Carrier, 2 s captures (07:57--07:58)")
    a.plot([peak["gain_db"]], [peak["clipping_percent"]], "s", color=ORANGE, ms=5,
           label=f"Repeat at {peak['gain_db']:g} dB, traffic peak (07:58)")
    for r in long_:
        a.plot([r["gain_db"]], [r["clipping_percent"]], "D", color=INK, ms=4, mfc="white",
               label=f"Carrier, {r['duration_s']:.0f} s capture at {r['gain_db']:g} dB")
    a.axhline(0.1, color=INK2, lw=0.8, ls="--")
    a.text(43.0, 0.13, "limit 0.1 %", fontsize=8, color=INK2)
    a.set_yscale("symlog", linthresh=1e-3, linscale=0.5)
    a.set_ylim(-1e-4, 6)
    a.set_ylabel("Clipped samples (%)")
    b.plot([r["gain_db"] for r in idle], [r["mean_power_db"] for r in idle], "o-", color=BLUE, ms=3.5, lw=1,
           label="Idle window, 2 s captures")
    b.set_ylabel("Mean power (dB)")
    b.set_xlabel("Tuner gain (dB)")
    for ax in (a, b):
        for g, name in ((G1, "$g_1$"), (G2, "$g_2$")):
            ax.axvline(g, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(18, 51)
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.40), frameon=False)
    b.legend(loc="upper left", frameon=False)
    ymax = b.get_ylim()[1]
    for g, name in ((G1, "$g_1$"), (G2, "$g_2$")):
        b.text(g + 0.3, ymax - 0.6, name, fontsize=9, color=INK)
    a.text(0.0, 1.02, "(a)", transform=a.transAxes, fontsize=9)
    b.text(0.0, 1.02, "(b)", transform=b.transAxes, fontsize=9)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figA-gains.pdf")
