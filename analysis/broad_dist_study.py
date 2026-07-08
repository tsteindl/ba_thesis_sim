"""Where do linear / binary search beat reverse engineering?

For broad uniform priors extending toward pi/2, N_min is small (1-2), so reverse engineering's
single-shot depth inference overshoots and caps out, while the iterative linear/binary searches stay
robust. Sweeps the budget for a few wide phi_max, saves results/broad_dist.csv and results/fig_broad.png.
    python analysis/broad_dist_study.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_binary_search as BIN,
    find_phi_fixed_budget_binary_search_anneal_m as BINA,
    find_phi_fixed_budget_reverse_engineering as RE,
)

R_TUNE, R_TEST = 1500, 30000
PMIN, EPS = 0.01, 1e-3
COL = {"brute": "#7f7f7f", "linear": "#2ca02c", "binary": "#ff7f0e", "reverse_eng": "#1f77b4"}
MARK = {"brute": "o", "linear": "^", "binary": "D", "reverse_eng": "s"}
NAME = {"brute": "Brute force", "linear": "Linear search", "binary": "Binary search", "reverse_eng": "Reverse engineering"}


def best(fn, grid, pmax, b):
    _m, arg, _ = E.grid_search_max(fn, {**grid, "budget": [int(b)]}, R_TUNE, PMIN, pmax, EPS, 42)
    return 100 * E.success_rate(fn, arg, R_TEST, PMIN, pmax, EPS, 2024)


def best_binary(pmax, b):
    m = np.unique(np.geomspace(10, 3000, 14).astype(int))
    r1 = best(BIN, {"m_exploration": m, "safeguard": [0, 1, 2], "conf": [0.5, 0.65, 0.8, 0.9, 0.95]}, pmax, b)
    r2 = best(BINA, {"m_exploration": m, "safeguard": [0, 1, 2], "conf": [0.5, 0.8, 0.9],
                     "max_b_steps_sub": [1, 2, 3], "delta": [2, 5]}, pmax, b)
    return max(r1, r2)


def sweep(pmax, budgets):
    m = np.unique(np.geomspace(3, 3000, 20).astype(int))
    out = {"budget": list(budgets), "brute": [], "linear": [], "binary": [], "reverse_eng": []}
    for b in budgets:
        out["brute"].append(100 * E.success_rate(BF, {"budget": int(b)}, R_TEST, PMIN, pmax, EPS, 2024))
        out["linear"].append(best(LIN, {"m_exploration": m, "lookback_window": [1, 2, 3, 5],
                                         "safeguard": [0, 1, 2, 5], "inc": [1, 2, 5]}, pmax, b))
        out["binary"].append(best_binary(pmax, b))
        out["reverse_eng"].append(best(RE, {"m_exploration": m, "safeguard": [0.8, 0.85, 0.9, 0.95]}, pmax, b))
    return out


def main():
    cases = [(np.pi / 2, "pi/2"), (np.pi / 4, "pi/4")]
    budgets = np.geomspace(3e3, 2e6, 9)
    data = {tag: sweep(pmax, budgets) for pmax, tag in cases}

    os.makedirs("results", exist_ok=True)
    with open("results/broad_dist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_max", "budget", "brute", "linear", "binary", "reverse_eng"])
        for _pmax, tag in cases:
            d = data[tag]
            for i, b in enumerate(d["budget"]):
                w.writerow([tag, int(b)] + [round(d[a][i], 2) for a in ("brute", "linear", "binary", "reverse_eng")])

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (pmax, tag) in zip(axes, cases):
        d = data[tag]
        for a in ("brute", "linear", "binary", "reverse_eng"):
            ax.plot(d["budget"], d[a], MARK[a] + "-", color=COL[a], label=NAME[a],
                    alpha=0.95 if a in ("reverse_eng", "linear") else 0.7)
        ltag = tag.replace("pi", "\\pi")
        ax.set(xscale="log", ylim=(0, 103), title=f"$\\phi \\sim U(0.01,\\ {ltag})$,  $\\varepsilon=10^{{-3}}$")
        ax.set_xlabel("budget  $C = N\\cdot m$  (log scale)", fontsize=16)
        ax.set_ylabel("% of trials converged", fontsize=16)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Broad distributions: reverse engineering plateaus while linear search stays robust", fontsize=18)
    fig.tight_layout()
    fig.savefig("results/fig_broad.png", dpi=140)
    print("wrote results/broad_dist.csv and results/fig_broad.png")

    for _pmax, tag in cases:
        d = data[tag]
        cap = max(d["reverse_eng"])
        print(f"  U(0.01,{tag}): RE ceiling ~{cap:.0f}%; linear reaches {max(d['linear']):.0f}%")


if __name__ == "__main__":
    main()
