"""Which convergence THRESHOLD and which PRECISION give the best honest story?

For uniform U(phi_min, phi_max), sweep budget, grid-tune each algorithm at every budget
(optimal params change per budget), build de-biased convergence curves for brute vs
best-adaptive, and report the BUDGET RATIO = budget_brute(p*)/budget_adaptive(p*) at
several target thresholds p*.  Shows where adaptive wins and where brute overtakes it.
    python analysis/threshold_precision.py
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

R_TUNE, R_TEST = 800, 25000
GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 5000, 14).astype(int)),
            "lookback_window": [1, 2, 5], "safeguard": [0, 1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 20000, 16).astype(int)),
           "safeguard": [0.8, 0.85, 0.9, 0.95]}
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


def brute_curve(budgets, pmin, pmax, eps):
    return np.array([E.success_rate(BF, {"budget": int(b)}, R_TEST, pmin, pmax, eps, 2024) for b in budgets])


def adaptive_curve(budgets, pmin, pmax, eps):
    out = []
    for b in budgets:
        best = 0.0
        for fn, g in [(LIN, GRID_LIN), (RE, GRID_RE)]:
            res = E.grid_full(fn, {**g, "budget": [int(b)]}, R_TUNE, pmin, pmax, eps, 42)
            win = max(res, key=lambda r: r[0])[2]
            best = max(best, E.success_rate(fn, win, R_TEST, pmin, pmax, eps, 2024))
        out.append(best)
    return np.array(out)


def crossing(budgets, rates, T):
    r = np.asarray(rates)
    if T <= r[0] or T >= r[-1]:
        return float("nan")
    return float(np.exp(np.interp(T, r, np.log(budgets))))


def run(name, pmin, pmax, eps, budgets):
    br, ad = brute_curve(budgets, pmin, pmax, eps), adaptive_curve(budgets, pmin, pmax, eps)
    print(f"\n### {name}  (eps={eps:.0e})")
    print(f"  {'target':>7} | {'brute budget':>14} | {'adaptive budget':>16} | ratio")
    for T in THRESHOLDS:
        bb, ba = crossing(budgets, br, T), crossing(budgets, ad, T)
        if np.isfinite(bb) and np.isfinite(ba):
            flag = "  <-- adaptive wins" if bb / ba > 1.05 else ("  (~tie)" if bb / ba > 0.95 else "  <-- BRUTE wins")
            print(f"  {int(T*100):>6}% | {bb:14,.0f} | {ba:16,.0f} | {bb/ba:4.2f}x{flag}")
        else:
            print(f"  {int(T*100):>6}% | (outside swept budget range)")


if __name__ == "__main__":
    for pmin, pmax, tag in [(0.01, 0.1, "U(0.01,0.1)"), (0.001, 0.1, "U(0.001,0.1)")]:
        run(f"{tag}", pmin, pmax, 1e-3, np.geomspace(3e3, 5e6, 13))
        run(f"{tag}", pmin, pmax, 1e-4, np.geomspace(3e5, 5e8, 13))
    run("U(0.01,0.1)", 0.01, 0.1, 1e-5, np.geomspace(3e7, 5e10, 13))
