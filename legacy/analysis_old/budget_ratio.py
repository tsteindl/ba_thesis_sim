"""Budget-ratio (resource-advantage) metric — budget-robust, unlike a fixed-budget snapshot.

For a target convergence T, find the budget each method needs to reach T, and report
    resource advantage = budget_brute(T) / budget_adaptive(T).
A value of 4x means adaptive reaches the same convergence with 1/4 the budget. This is
the natural way to state a Heisenberg-vs-SQL-style advantage and does not depend on
picking one lucky budget.  python analysis/budget_ratio.py
"""
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

R_TUNE, R_TEST = 500, 15000
GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 3000, 10).astype(int)),
            "lookback_window": [1, 2], "safeguard": [1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 8000, 12).astype(int)), "safeguard": [0.8, 0.9]}


def rate_brute(b, pmin, pmax, eps, dist):
    return E.success_rate(BF, {"budget": int(b)}, R_TEST, pmin, pmax, eps, 2024, phi_dist=dist)


def rate_adaptive(fn, grid, b, pmin, pmax, eps, dist):
    g = {**grid, "budget": [int(b)]}
    res = E.grid_full(fn, g, R_TUNE, pmin, pmax, eps, 42, phi_dist=dist)
    win = max(res, key=lambda r: r[0])[2]
    return E.success_rate(fn, win, R_TEST, pmin, pmax, eps, 2024, phi_dist=dist)


def crossing(budgets, rates, target):
    r = np.asarray(rates)
    if target <= r[0]:
        return budgets[0]
    if target >= r[-1]:
        return float("inf")
    return float(np.exp(np.interp(target, r, np.log(budgets))))


def analyse(name, pmin, pmax, eps, dist, budgets):
    br = [rate_brute(b, pmin, pmax, eps, dist) for b in budgets]
    li = [rate_adaptive(LIN, GRID_LIN, b, pmin, pmax, eps, dist) for b in budgets]
    re = [rate_adaptive(RE, GRID_RE, b, pmin, pmax, eps, dist) for b in budgets]
    ad = np.maximum(li, re)
    print(f"\n### {name}")
    for T in (0.5, 0.9):
        bb, ba = crossing(budgets, br, T), crossing(budgets, ad, T)
        ratio = bb / ba if np.isfinite(ba) and ba > 0 else float("nan")
        print(f"  target {int(T*100)}% converged:  brute needs {bb:12,.0f} | adaptive needs {ba:12,.0f}"
              f"  ->  {ratio:5.1f}x less budget")


if __name__ == "__main__":
    B_LO = np.geomspace(3e3, 3e6, 12)
    B_HI = np.geomspace(3e5, 3e8, 12)
    analyse("NARROW  U(0.01,0.1), eps=1e-3", 0.01, 0.1, 1e-3, "uniform", B_LO)
    analyse("NARROW  U(0.01,0.1), eps=1e-4", 0.01, 0.1, 1e-4, "uniform", B_HI)
    analyse("BROAD   logU(1e-4,0.1) 3 orders, eps=1e-3", 1e-4, 0.1, 1e-3, "loguniform", B_LO)
    analyse("BROAD   logU(1e-4,0.1) 3 orders, eps=1e-4", 1e-4, 0.1, 1e-4, "loguniform", B_HI)
    analyse("BROAD   logU(1.6e-3,pi/2) 3 orders, eps=1e-3", np.pi / 2 / 1000, np.pi / 2, 1e-3, "loguniform", B_LO)
