#!/usr/bin/env python3
"""Figure A.1: band survey at P1, mean power of each candidate range.

Reads the rtl_power survey files and the CellSearch outputs that the field
work saved in code/data/survey/ (night and, once made, day runs), and draws
one panel per candidate range: the mean power over the sweeps for each run,
with the carriers confirmed by the cell search shaded (centre +- half the
occupied width, N_RB x 0.18 MHz) and plateaus where no LTE cell was found
marked. Measured data, not simulation. Run from code/ so that the study's
survey parser is used:
  uv run python ../figures/src/figA-survey.py ../figures/figA-survey.pdf
"""
import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from occupancy.survey import parse_rtl_power_csv  # noqa: E402

SURVEY = ROOT / "code" / "data" / "survey"
RANGES = (("758-803MHz", 758, 803), ("790-894MHz", 790, 894), ("925-960MHz", 925, 960))
RUNS = (("night", "#2a78d6", "Night (22:00)"), ("day", "#eb6834", "Day (07:00)"))
INK, INK2, SHADE = "#2C2C2A", "#5F5E5A", "#D3D1C7"


def cell_searches(label):
    """{search centre MHz: [(cell id, fc MHz, n_rb)] or [] if no cell} for one run."""
    out = {}
    for f in sorted(SURVEY.glob(f"cellsearch_{label}_*MHz.txt")):
        centre = float(re.search(r"_(\d+(?:\.\d+)?)MHz", f.name).group(1))
        cells = []
        for line in f.read_text(errors="replace").splitlines():
            m = re.match(r"\s*(FDD|TDD)\s+(\d+)\s+\d+\s+([\d.]+)M\s+\S+\s+\S+\s+\S\s+(\d+)\s", line)
            if m:
                cells.append((int(m.group(2)), float(m.group(3)), int(m.group(4))))
        out[centre] = cells
    return out


def main(out):
    lm = "/usr/share/texmf/fonts/opentype/public/lm/"
    for f in ("lmroman10-regular.otf", "lmroman10-italic.otf", "lmroman10-bold.otf"):
        font_manager.fontManager.addfont(lm + f)
    plt.rcParams.update({
        "font.family": "LM Roman 10", "mathtext.fontset": "cm",
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E4DC", "grid.linewidth": 0.5,
        "pdf.fonttype": 3, "savefig.dpi": 300,
    })
    searches = cell_searches("night")
    fig, axes = plt.subplots(3, 1, figsize=(5.9, 7.4))
    for ax, (tag, lo, hi) in zip(axes, RANGES):
        for label, colour, name in RUNS:
            files = sorted(glob.glob(str(SURVEY / f"survey_*_{tag}_{label}_P1.csv")))
            if not files:
                continue
            d = parse_rtl_power_csv(files[-1])
            ax.plot(d.freqs_hz / 1e6, d.mean_db, color=colour, lw=0.6, label=name)
        ymin, ymax = ax.get_ylim()
        for centre, cells in sorted(searches.items()):
            if not lo <= centre <= hi:
                continue
            if cells:
                fc, n_rb = cells[0][1], cells[0][2]
                half = n_rb * 0.18 / 2
                ax.axvspan(fc - half, fc + half, color=SHADE, alpha=0.6, lw=0, zorder=0)
                ids = ", ".join(str(c[0]) for c in cells)
                xl = (max(lo, fc - half) + min(hi, fc + half)) / 2    # centre of the visible part
                ax.text(xl, ymax, f"LTE, cell{'s' if len(cells) > 1 else ''} {ids}", ha="center",
                        va="bottom", fontsize=8, color=INK)
            else:
                ax.text(centre, ymax, "no LTE cell", ha="center", va="bottom", fontsize=8, color=INK2,
                        style="italic")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Mean power (dB)")
    axes[-1].set_xlabel("Frequency (MHz)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(h_pad=1.6)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}; cell searches: {searches}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figA-survey.pdf")
