"""Two panels for results/BINARY_STORY.md: what the exploration does, and what reuse would buy."""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

REUSE = [  # setting, re, re_reuse, bs, bs_reuse, bs_reuse_acc  (scratchpad/reuse2.py)
    ("U(1e-4,1e-3) e-4", 91.10, 91.88, 89.12, 91.98, 91.98),
    ("U(1e-4,1e-3) e-6", 55.00, 55.27, 51.88, 64.71, 53.77),
    ("U(1e-4,1e-2) e-4", 47.70, 51.48, 48.24, 50.74, 50.74),
    ("U(1e-4,1e-2) e-6", 59.86, 60.15, 58.95, 93.25, 58.66),
    ("U(1e-4,1e-1) e-4", 59.07, 59.56, 58.89, 59.20, 59.20),
    ("U(1e-4,1e-1) e-6", 61.80, 61.77, 59.50, 75.64, 59.26),
    ("U(1e-3,1e-2) e-3", 91.12, 91.88, 86.63, 92.03, 92.03),
    ("U(1e-3,1e-2) e-4", 50.98, 53.15, 50.21, 52.81, 52.81),
    ("U(1e-3,1e-2) e-5", 54.87, 54.98, 52.19, 64.99, 53.95),
    ("U(1e-3,1e-2) e-6", 56.20, 56.33, 55.02, 73.39, 54.93),
    ("U(1e-3,1e-2) e-7", 56.96, 56.97, 55.77, 56.63, 56.12),
    ("U(1e-3,1e-2) e-8", 57.28, 57.26, 56.16, 56.06, 56.10),
    ("U(1e-3,1e-1) e-4", 59.03, 59.33, 58.48, 58.91, 58.91),
    ("U(1e-3,1e-1) e-6", 61.60, 61.49, 60.03, 79.64, 60.08),
    ("U(5e-3,5e-2) e-4", 53.99, 53.80, 50.52, 67.48, 52.81),
    ("U(5e-3,5e-2) e-6", 56.84, 57.10, 56.01, 57.61, 55.99),
    ("U(1e-2,1e-1) e-3", 52.42, 53.33, 49.91, 62.82, 54.22),
    ("U(1e-2,1e-1) e-4", 55.77, 55.83, 51.19, 75.23, 52.25),
    ("U(1e-2,1e-1) e-5", 57.17, 57.40, 55.77, 73.89, 55.67),
    ("U(1e-2,1e-1) e-6", 57.69, 57.81, 56.76, 57.23, 56.73),
    ("U(1e-2,1e-1) e-7", 57.68, 57.67, 56.78, 57.00, 57.03),
    ("U(1e-2,1e-1) e-8", 57.63, 57.63, 56.74, 56.88, 56.75),
    ("U(1e-5,1e-2) e-5", 58.65, 58.55, 56.20, 58.35, 58.35),
]


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.35]})

    rows = list(csv.DictReader(open("results/binary_diagnostics.csv")))
    pr = np.array([float(r["probes"]) for r in rows])
    ex = np.array([100 * float(r["explore_share"]) for r in rows])
    one = pr < 1.5
    ax.scatter(pr[one], ex[one], s=52, color=ORANGE, edgecolor="white", lw=0.9, zorder=3,
               label=f"no bisection at all ({one.sum()} of {len(pr)})")
    ax.scatter(pr[~one], ex[~one], s=52, color=BLUE, edgecolor="white", lw=0.9, zorder=3,
               label=f"bisection runs ({(~one).sum()} of {len(pr)})")
    ax.set_xlabel("probes taken by the exploration (mean over trials)")
    ax.set_ylabel("exploration share of budget (%)")
    ax.set_title("At its own tuned optimum, the bisection is\nall-or-nothing", fontsize=11, loc="left")
    ax.grid(axis="y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 17)

    lbl = [r[0] for r in REUSE]
    re_r = np.array([r[2] for r in REUSE])
    bs = np.array([r[3] for r in REUSE])
    bsR = np.array([r[4] for r in REUSE])
    bsA = np.array([r[5] for r in REUSE])
    order = np.argsort(bsR - re_r)
    y = np.arange(len(REUSE))
    ax2.hlines(y, re_r[order], bsR[order], color="#d8d7d2", lw=2.2, zorder=1)
    ax2.scatter(bs[order], y, s=34, color=ORANGE, zorder=3, label="binary search, as shipped")
    ax2.scatter(re_r[order], y, s=34, marker="|", lw=2, color=INK2, zorder=4,
                label="reverse engineering (+ its own pilot reused)")
    ax2.scatter(bsA[order], y, s=30, facecolor="white", edgecolor=AQUA, lw=1.6, zorder=3,
                label="+ reuse of ACCEPTED probes only")
    ax2.scatter(bsR[order], y, s=44, color=BLUE, zorder=5, label="+ reuse of ALL probes, unwrapped")
    ax2.set_yticks(y)
    ax2.set_yticklabels([lbl[i] for i in order], fontsize=7.5)
    ax2.set_xlabel("convergence rate (%)")
    ax2.set_title("Reusing the exploration shots is what makes the bisection pay",
                  fontsize=11, loc="left")
    ax2.grid(axis="x")
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig("results/fig_binary_story.png", bbox_inches="tight")
    print("wrote results/fig_binary_story.png")


if __name__ == "__main__":
    main()
