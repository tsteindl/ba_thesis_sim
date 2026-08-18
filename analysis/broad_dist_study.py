"""Where do linear / binary search beat reverse engineering?

For broad uniform priors extending toward pi/2, N_min is small (1-2), so the pilot estimate is
imprecise and reverse engineering's single-shot depth inference is at its most fragile. Sweeps the
budget for a few wide phi_max, saves results/broad_dist.csv and results/fig_broad.png.

Two corrections relative to the published Table 3.4:

  * the exploration grid used to stop at m' = 3,000. With N_min = 1 the pilot needs *far* more shots
    than that before the inferred depth is usable, so the published "reverse engineering plateaus"
    curve was grid-limited rather than algorithmic. The grid now scales with the regime, exactly as
    in analysis/extensive_sweep.py. The `reverse_eng_m3k` column reproduces the old cap so the
    correction stays attributable.
  * the statistical safeguard (qmetrology/safeguard.py) is reported next to the tuned constant.

    python analysis/broad_dist_study.py [--quick]
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
from qmetrology.oracle import heisenberg_rate
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_binary_search as BIN,
    find_phi_fixed_budget_binary_search_anneal_m as BINA,
    find_phi_fixed_budget_reverse_engineering as RE,
    find_phi_fixed_budget_binary_search_risk as BIN_R,
    find_phi_fixed_budget_reverse_engineering_risk as RE_R,
)

QUICK = "--quick" in sys.argv
R_TUNE, R_TEST = (400, 4000) if QUICK else (1500, 30000)
PMIN, EPS = 0.01, 1e-3
SERIES = ["brute", "linear", "binary", "binary_risk", "reverse_eng", "reverse_eng_risk",
          "reverse_eng_m3k", "oracle"]
# colour carries the algorithm identity; the statistical-safeguard variant reuses its parent's hue
# and is separated by dash + open marker, so no new categorical hue is introduced.
COL = {"brute": "#7f7f7f", "linear": "#2ca02c", "binary": "#ff7f0e", "binary_risk": "#ff7f0e",
       "reverse_eng": "#1f77b4", "reverse_eng_risk": "#1f77b4", "reverse_eng_m3k": "#9ecae1",
       "oracle": "#000000"}
MARK = {"brute": "o", "linear": "^", "binary": "D", "binary_risk": "D",
        "reverse_eng": "s", "reverse_eng_risk": "s", "reverse_eng_m3k": "s", "oracle": "*"}
STYLE = {"binary_risk": "--", "reverse_eng_risk": "--", "reverse_eng_m3k": ":",
         "oracle": ":"}
FILL = {"binary_risk": "none", "reverse_eng_risk": "none"}
NAME = {"brute": "Brute force", "linear": "Linear search", "binary": "Binary search",
        "binary_risk": "Binary search", "reverse_eng": "Reverse engineering (tuned C, wide grid)",
        "reverse_eng_risk": "Reverse engineering",
        "reverse_eng_m3k": "Reverse engineering (published, m'≤3000)",
        "oracle": "Heisenberg ceiling (Eq. 3.4 at $N_{opt}$)"}


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def m_grid(pmax, lo, n):
    """Exploration-size grid scaled to the regime (same rule as analysis/extensive_sweep.py).

    The ceiling is raised to 500k here: at phi_max = pi/2 the constant-C arm tunes to the very top of
    the grid, so a lower cap would understate the baseline it is being compared against.
    """
    hi = int(np.clip(0.6724 / (n_min(pmax) * EPS**2), 3000, 500_000))
    return np.unique(np.geomspace(lo, hi, n).astype(int))


def best(fn, grid, pmax, b):
    _m, arg, _ = E.grid_search_max(fn, {**grid, "budget": [int(b)]}, R_TUNE, PMIN, pmax, EPS, 42)
    return 100 * E.success_rate(fn, arg, R_TEST, PMIN, pmax, EPS, 2024)


def best_binary(pmax, b):
    m = m_grid(pmax, 10, 16)
    r1 = best(BIN, {"m_exploration": m, "safeguard": [0, 1, 2], "conf": [0.5, 0.65, 0.8, 0.9, 0.95]}, pmax, b)
    r2 = best(BINA, {"m_exploration": m, "safeguard": [0, 1, 2], "conf": [0.5, 0.8, 0.9],
                     "max_b_steps_sub": [1, 2, 3], "delta": [2, 5]}, pmax, b)
    return max(r1, r2)


def sweep(pmax, budgets):
    m = m_grid(pmax, 3, 22)
    m_old = np.unique(np.geomspace(3, 3000, 20).astype(int))   # the published cap
    out = {"budget": list(budgets), **{k: [] for k in SERIES}}
    for b in budgets:
        out["brute"].append(100 * E.success_rate(BF, {"budget": int(b)}, R_TEST, PMIN, pmax, EPS, 2024))
        out["linear"].append(best(LIN, {"m_exploration": m, "lookback_window": [1, 2, 3, 5],
                                        "safeguard": [0, 1, 2, 5], "inc": [1, 2, 5]}, pmax, b))
        out["binary"].append(best_binary(pmax, b))
        out["binary_risk"].append(best(BIN_R, {"m_exploration": m_grid(pmax, 10, 16),
                                               "conf": [0.5, 0.65, 0.8, 0.9], "eps_target": [EPS]}, pmax, b))
        out["reverse_eng"].append(best(RE, {"m_exploration": m, "safeguard": [0.8, 0.85, 0.9, 0.95]}, pmax, b))
        out["reverse_eng_risk"].append(best(RE_R, {"m_exploration": m, "eps_target": [EPS]}, pmax, b))
        out["reverse_eng_m3k"].append(best(RE, {"m_exploration": m_old,
                                                "safeguard": [0.8, 0.85, 0.9, 0.95]}, pmax, b))
        # the ceiling: analytic (Eq. 3.4 at N_opt), so no simulation and no Monte-Carlo noise.
        # The exact argmax oracle is deliberately NOT used: near the aliasing edge it converges off
        # the deterministic readout rather than from the data (qmetrology/oracle.py).
        out["oracle"].append(100 * heisenberg_rate(int(b), EPS, PMIN, pmax))
        print(f"    B={b:>12,.0f}  " + "  ".join(f"{k.split('_')[0][:4]}{out[k][-1]:5.1f}" for k in SERIES),
              flush=True)
    return out


def main():
    cases = [(np.pi / 2, "pi/2"), (np.pi / 4, "pi/4")]
    budgets = np.geomspace(3e3, 2e6, 5 if QUICK else 9)
    data = {}
    for pmax, tag in cases:
        print(f"=== U({PMIN},{tag}) ===", flush=True)
        data[tag] = sweep(pmax, budgets)

    os.makedirs("results", exist_ok=True)
    with open("results/broad_dist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_max", "budget"] + SERIES)
        for _pmax, tag in cases:
            d = data[tag]
            for i, b in enumerate(d["budget"]):
                w.writerow([tag, int(b)] + [round(d[a][i], 2) for a in SERIES])

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False, "font.size": 11})
    plotted = [a for a in SERIES if a != "binary"]      # binary_risk IS binary search now
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (pmax, tag) in zip(axes, cases):
        d = data[tag]
        for a in plotted:
            ax.plot(d["budget"], d[a], marker=MARK[a], ls=STYLE.get(a, "-"), color=COL[a],
                    label=NAME[a], markersize=5, markerfacecolor=FILL.get(a, COL[a]),
                    alpha=0.95 if a.startswith(("reverse", "linear")) else 0.75)
        ltag = tag.replace("pi", "\\pi")
        ax.set(xscale="log", ylim=(0, 103), title=f"$\\phi \\sim U(0.01,\\ {ltag})$,  $\\varepsilon=10^{{-3}}$")
        ax.set_xlabel("budget  $C = N\\cdot m$  (log scale)", fontsize=13)
        ax.set_ylabel("% of trials converged", fontsize=13)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Broad distributions: reverse engineering's plateau was an exploration-grid artefact",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig("results/fig_broad.png", dpi=140)
    print("wrote results/broad_dist.csv and results/fig_broad.png")

    for _pmax, tag in cases:
        d = data[tag]
        print(f"  U(0.01,{tag}): RE published-grid ceiling ~{max(d['reverse_eng_m3k']):.0f}%; "
              f"RE wide grid {max(d['reverse_eng']):.0f}%; RE statistical {max(d['reverse_eng_risk']):.0f}%; "
              f"linear {max(d['linear']):.0f}%; binary statistical {max(d['binary_risk']):.0f}%")


if __name__ == "__main__":
    main()
