#!/usr/bin/env python3
"""Figure 2.6: analytic detection curves of the energy detector.

Computes P_D against SNR at P_FA = 0.05 for the normalized energy statistic
T = (1/N) sum |y(n)|^2 (Chapter 2, eqs. 2.3-2.5), using the exact chi-square
distribution: N*T ~ (sigma^2 / 2) * chi2(2N) under H0 and
((sigma^2 + sigma_s^2) / 2) * chi2(2N) under H1 (Axell et al., 2010, p. 323;
Yucek & Arslan, 2009, p. 121).

Noise uncertainty follows Tandra & Sahai (2008): the noise power is only known
to lie in [sigma_n^2 / rho, rho * sigma_n^2], rho = 10^(delta/10). Worst case:
the threshold is set for P_FA at the highest noise power and P_D is evaluated
at the lowest. SNR is defined against the nominal noise power sigma_n^2 = 1.
The SNR wall is (rho^2 - 1) / rho (their eq. 2).

These are analytic curves, not measurements. Run from the repository root:
  uv run --no-project --with matplotlib --with numpy --with scipy \
      python figures/src/fig2-ed-pd-snr.py figures/fig2-ed-pd-snr.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.stats import chi2

PFA = 0.05
DELTA_DB = 1.0                      # noise uncertainty in dB
NS = (10**3, 10**5)                 # numbers of complex samples
SNR_DB = np.arange(-25.0, 0.0001, 0.05)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#5F5E5A"   # validated categorical pair (dataviz skill)
INK, INK2 = "#2C2C2A", "#5F5E5A"


def pd_curve(n, rho):
    """Worst-case P_D for n samples with uncertainty factor rho (rho = 1: known noise)."""
    sigma_hi, sigma_lo = rho, 1.0 / rho            # extreme noise powers (nominal = 1)
    lam = sigma_hi * chi2.ppf(1.0 - PFA, 2 * n) / (2 * n)
    snr = 10.0 ** (SNR_DB / 10.0)
    return chi2.sf(2 * n * lam / (sigma_lo + snr), 2 * n)


def main(out):
    lm = "/usr/share/texmf/fonts/opentype/public/lm/"
    for f in ("lmroman10-regular.otf", "lmroman10-italic.otf", "lmroman10-bold.otf"):
        font_manager.fontManager.addfont(lm + f)
    plt.rcParams.update({
        "font.family": "LM Roman 10", "mathtext.fontset": "cm",
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#D3D1C7", "grid.linewidth": 0.5,
        "pdf.fonttype": 3, "savefig.dpi": 300,
    })
    rho = 10.0 ** (DELTA_DB / 10.0)
    wall_db = 10.0 * np.log10((rho**2 - 1.0) / rho)

    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    for n in NS:
        ax.plot(SNR_DB, pd_curve(n, 1.0), color=BLUE, lw=1.6, ls="-",
                label="Noise power known" if n == NS[0] else None)
        ax.plot(SNR_DB, pd_curve(n, rho), color=ORANGE, lw=1.6, ls=(0, (5, 2.5)),
                label=f"{DELTA_DB:.0f} dB noise uncertainty, worst case" if n == NS[0] else None)
    ax.axvline(wall_db, color=GRAY, lw=1.0, ls=":", zorder=1)
    ax.text(wall_db - 0.3, 0.06, "SNR wall\n" + rf"${wall_db:.1f}$ dB", ha="right", va="bottom", color=INK2, fontsize=9)

    # direct labels for the number of samples: beside each known-noise curve at P_D = 0.5,
    # and one shared label for the two uncertain curves, which both pile up at the wall
    for n in NS:
        pd = pd_curve(n, 1.0)
        x = SNR_DB[np.argmin(np.abs(pd - 0.5))]
        exp = int(round(np.log10(n)))
        ax.text(x - 0.4, 0.5, rf"$N = 10^{exp}$", ha="right", va="center", color=INK, fontsize=9)
    ax.annotate(r"$N = 10^3$ and $10^5$", xy=(wall_db + 0.15, 0.62), xytext=(-8.6, 0.62),
                ha="left", va="center", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))

    ax.set_xlim(-25, 0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"Probability of detection, $P_{\mathrm{D}}$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    if out.endswith(".pdf") and len(sys.argv) > 2:
        fig.savefig(sys.argv[2], bbox_inches="tight", facecolor="white", dpi=150)
    print(f"SNR wall for {DELTA_DB} dB: {wall_db:.2f} dB")
    for n in NS:
        for r, name in ((1.0, "known"), (rho, "uncertain")):
            pd = pd_curve(n, r)
            print(f"N={n:>6} {name:9s}: P_D=0.5 at {SNR_DB[np.argmin(np.abs(pd - 0.5))]:.2f} dB, "
                  f"P_D=0.9 at {SNR_DB[np.argmin(np.abs(pd - 0.9))]:.2f} dB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig2-ed-pd-snr.pdf")
