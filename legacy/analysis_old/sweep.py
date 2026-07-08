"""Comprehensive map of WHERE adaptive beats brute force.

Sweeps prior (uniform vs log-uniform) x eps x phi_max x dynamic-range x budget, and
records the honest (de-biased) % converged for brute / linear / reverse-engineering.
Writes results/sweep_data.csv for downstream analysis.

    python analysis/sweep.py
"""
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_reverse_engineering as RE,
)

R_TUNE, R_TEST = 700, 20000
GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 3000, 12).astype(int)),
            "lookback_window": [1, 2, 5], "safeguard": [1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 8000, 14).astype(int)),
           "safeguard": [0.8, 0.85, 0.9, 0.95]}

PRIORS = ["uniform", "loguniform"]
EPS = [1e-3, 1e-4]
PHI_MAX = {"0.1": 0.1, "pi/2": np.pi / 2}
ORDERS = [0.5, 1.0, 2.0, 3.0]          # dynamic range = log10(phi_max/phi_min)
BUDGETS = [1e4, 1e5, 1e6, 1e7]         # spans the eps=1e-3 and 1e-4 regimes


def brute(budget, pmin, pmax, eps, dist):
    return 100 * E.success_rate(BF, {"budget": int(budget)}, R_TEST, pmin, pmax, eps, 2024, phi_dist=dist)


def best_adaptive(fn, grid, budget, pmin, pmax, eps, dist):
    g = {**grid, "budget": [int(budget)]}
    res = E.grid_full(fn, g, R_TUNE, pmin, pmax, eps, 42, phi_dist=dist)
    win = max(res, key=lambda r: r[0])[2]
    return 100 * E.success_rate(fn, win, R_TEST, pmin, pmax, eps, 2024, phi_dist=dist)


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    combos = list(itertools.product(PRIORS, EPS, PHI_MAX.items(), ORDERS, BUDGETS))
    print(f"{len(combos)} cells...")
    for i, (dist, eps, (pmxname, pmax), orders, budget) in enumerate(combos):
        pmin = pmax / (10 ** orders)
        b = brute(budget, pmin, pmax, eps, dist)
        lin = best_adaptive(LIN, GRID_LIN, budget, pmin, pmax, eps, dist)
        re = best_adaptive(RE, GRID_RE, budget, pmin, pmax, eps, dist)
        best = max(lin, re)
        best_algo = "linear" if lin >= re else "reverse_eng"
        rows.append(dict(prior=dist, eps=eps, phi_max=pmxname, orders=orders, phi_min=round(pmin, 6),
                         budget=int(budget), brute=round(b, 1), linear=round(lin, 1), re=round(re, 1),
                         best_adaptive=round(best, 1), best_margin=round(best - b, 1), best_algo=best_algo))
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(combos)}")
    with open("results/sweep_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # quick console highlights: best margin per (prior, regime)
    print("\nTop margins by prior:")
    for dist in PRIORS:
        top = sorted((r for r in rows if r["prior"] == dist), key=lambda r: -r["best_margin"])[:5]
        print(f"  [{dist}]")
        for r in top:
            print(f"    eps={r['eps']:.0e} phi_max={r['phi_max']:>4} orders={r['orders']} budget={r['budget']:>9,} "
                  f"| brute {r['brute']:5.1f}  best {r['best_adaptive']:5.1f} ({r['best_algo']}) margin +{r['best_margin']:.1f}")
    print("\nWrote results/sweep_data.csv")


if __name__ == "__main__":
    main()
