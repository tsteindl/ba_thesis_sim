"""Render the combined-lever study: results/lever_cube.csv -> results/fig_levers.png (+ console summary).

Panels:
  A/B  2D heatmap of the reverse-engineering budget ratio over (phi_min, phi_max) at eps=1e-6,
       at the 50% and 90% convergence thresholds -> the ridge / optimum.
  C    advantage-vs-precision for three representative priors, 50% (solid) and 90% (dashed).
  D    breaking points: ratio vs phi_min (phi_max=0.1) and ratio vs phi_max (phi_min=1e-4), eps=1e-6.

    python analysis/render_levers.py
"""
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

CUBE = "results/lever_cube.csv"


def load(path=CUBE):
    """(phi_min, phi_max, eps, threshold_pct, algo) -> ratio_vs_brute (float or nan)."""
    d = {}
    for r in csv.DictReader(open(path)):
        v = r["ratio_vs_brute"]
        d[(float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]),
           int(r["threshold_pct"]), r["algo"])] = (float(v) if v not in ("", "None", None) else float("nan"))
    return d


def get(d, pmin, pmax, eps, thr, algo="reverse_eng"):
    for (a, b, c, t, al), v in d.items():
        if al == algo and t == thr and abs(math.log10(c) - math.log10(eps)) < 1e-6 \
           and abs(a - pmin) <= 1e-12 + 1e-6 * a and abs(b - pmax) <= 1e-9:
            return v
    return float("nan")


def oracle(pmin, pmax):
    R = pmax / pmin
    return math.log(R) / (1 - 1 / R)


def heatmap(ax, d, thr, pmins, pmaxs, eps=1e-6):
    M = np.array([[get(d, pm, px, eps, thr) for px in pmaxs] for pm in pmins])
    norm = TwoSlopeNorm(vmin=min(0.5, np.nanmin(M)), vcenter=1.0, vmax=max(1.5, np.nanmax(M)))
    ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(pmaxs)), [f"{x:g}" for x in pmaxs])
    ax.set_yticks(range(len(pmins)), [f"{x:g}" for x in pmins])
    for i in range(len(pmins)):
        for j in range(len(pmaxs)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black", fontweight="bold" if v == np.nanmax(M) else "normal")
    ax.set(xlabel=r"$\phi_{\max}$", ylabel=r"$\phi_{\min}$",
           title=f"RE budget ratio @ {thr}%  ($\\varepsilon=10^{{-6}}$)")


def main():
    if not os.path.exists(CUBE):
        sys.exit(f"{CUBE} not found -- run analysis/combined_lever_study.py first")
    d = load()
    pmins = [0.01, 1e-3, 1e-4, 3e-5, 1e-5]
    pmaxs = [0.05, 0.1, 0.15, 0.2, 0.3]

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False, "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    heatmap(axes[0, 0], d, 50, pmins, pmaxs)
    heatmap(axes[0, 1], d, 90, pmins, pmaxs)

    # Panel C: advantage vs precision for three representative priors.
    ax = axes[1, 0]
    priors = [(0.01, 0.1, "U(0.01, 0.1)  narrow", "#7f7f7f"),
              (1e-4, 0.1, "U(1e-4, 0.1)  wide", "#1f77b4"),
              (1e-4, 0.2, "U(1e-4, 0.2)  wide+raised", "#d62728")]
    epss = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for pmin, pmax, name, col in priors:
        for thr, ls, mk in [(50, "-", "o"), (90, "--", "s")]:
            y = [get(d, pmin, pmax, e, thr) for e in epss]
            ax.plot(epss, y, ls + mk, color=col, alpha=0.9,
                    label=f"{name} @{thr}%" if thr == 50 else None)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel=r"$\varepsilon$  (precision; smaller = harder)",
           ylabel="RE budget ratio vs brute", title="Advantage vs precision  (solid=50%, dashed=90%)")
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="upper left")

    # Panel D: breaking points.
    ax = axes[1, 1]
    pmin_axis = [0.01, 1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    pmax_axis = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    for thr, ls in [(50, "-"), (90, "--")]:
        y1 = [get(d, pm, 0.1, 1e-6, thr) for pm in pmin_axis]
        ax.plot(range(len(pmin_axis)), y1, ls + "o", color="#1f77b4",
                label=f"vs $\\phi_{{\\min}}$ ($\\phi_{{\\max}}$=0.1) @{thr}%")
        y2 = [get(d, 1e-4, px, 1e-6, thr) for px in pmax_axis]
        ax.plot(range(len(pmax_axis)), y2, ls + "^", color="#d62728",
                label=f"vs $\\phi_{{\\max}}$ ($\\phi_{{\\min}}$=1e-4) @{thr}%")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    n = max(len(pmin_axis), len(pmax_axis))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{pmin_axis[i]:g}\n/{pmax_axis[i]:g}" if i < len(pmax_axis) else f"{pmin_axis[i]:g}\n/-"
                        for i in range(n)], fontsize=7)
    ax.set(xlabel=r"blue: $\phi_{\min}$ down  |  red: $\phi_{\max}$ up", ylabel="RE budget ratio vs brute",
           title="Breaking points  ($\\varepsilon=10^{-6}$)")
    ax.legend(fontsize=7, loc="lower center")

    fig.suptitle("Combining the levers: precision (eps) x dynamic range (phi_min / phi_max)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig("results/fig_levers.png", dpi=140)
    print("wrote results/fig_levers.png")

    # Console summary for LEVERS.md.
    print("\n=== max realized RE ratio per threshold (over all scenarios) ===")
    for thr in (50, 80, 90, 95):
        best = max(((v, k) for k, v in d.items() if k[3] == thr and k[4] == "reverse_eng" and np.isfinite(v)),
                   default=(float("nan"), None))
        if best[1]:
            pm, px, e, _, _ = best[1]
            print(f"  @{thr}%: {best[0]:.3f}x  at U({pm:g},{px:g}), eps={e:.0e}  "
                  f"(oracle E[phi_max/phi]={oracle(pm, px):.2f}x)")
    print("\n=== phi_min lever at phi_max=0.1 (is it inert at 90%?) ===")
    for thr in (50, 90):
        vals = [(pm, get(d, pm, 0.1, 1e-6, thr)) for pm in pmin_axis]
        print(f"  @{thr}% (eps=1e-6): " + "  ".join(f"{pm:g}:{v:.3f}" for pm, v in vals if np.isfinite(v)))


if __name__ == "__main__":
    main()
