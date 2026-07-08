"""Extensive, precision-scaled VARIABLE-budget grid search for ALL algorithms (unchanged
algorithms) — the honest best mean budget each can reach for >=90% convergence, vs brute.

Grids shift with precision (m_exploitation range ~ 1/eps^2), per the request. De-biased:
tune on seed 42, then among the cheapest tune-passing configs validate on seed 2024 and report
the cheapest that still reaches 90%.
    python analysis/variable_extensive.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_linear_search as LIN,
    find_phi_binary_search as BIN,
    find_phi_reverse_engineering as RE,
)

R_TUNE, R_TEST = 4000, 30000
TARGET = 0.90


def grids(eps):
    """Precision-scaled: exploitation budget ~ 1/eps^2, wide range."""
    b_hi = int(0.7 / eps ** 2)                    # ~ brute budget scale for 90%
    mfin = np.unique(np.geomspace(200, 40 * b_hi, 18).astype(int))   # m_exploitation, WIDE + shifted
    mexp = np.unique(np.geomspace(10, 30000, 9).astype(int))
    return {
        "linear": (LIN, {"m_exploration": mexp, "m_exploitation": mfin,
                         "lookback_window": [1, 2, 5], "safeguard": [0, 1, 2], "inc": [1, 2]}),
        "binary": (BIN, {"m_exploration": mexp, "m_exploitation": mfin,
                         "safeguard": [0, 1, 2], "conf": [0.5, 0.8, 0.9, 0.95]}),
        "reverse_eng": (RE, {"m_exploration": mexp, "m_exploitation": mfin,
                             "safeguard": [0.8, 0.85, 0.9, 0.95]}),
    }


def brute_budget_90(pmin, pmax, eps):
    """Sweep brute budget to the 90% crossing (its natural 'budget for 90%')."""
    budgets = np.geomspace(0.02 / eps ** 2, 5 / eps ** 2, 22)
    prev = None
    for b in budgets:
        r = E.success_rate(BF, {"budget": int(b)}, R_TEST, pmin, pmax, eps, 2024)
        if r >= TARGET and prev is not None:
            (b0, r0) = prev
            f = (TARGET - r0) / (r - r0)
            return float(np.exp(np.log(b0) + f * (np.log(b) - np.log(b0))))
        prev = (b, r)
    return float("nan")


def best_variable(fn, grid, pmin, pmax, eps):
    """Cheapest mean-budget config reaching 90% (tune seed 42 -> validate seed 2024)."""
    res = E.grid_full(fn, grid, R_TUNE, pmin, pmax, eps, 42)          # (rate, mean_budget, params)
    cand = sorted((r for r in res if r[0] >= TARGET and np.isfinite(r[1])), key=lambda r: r[1])
    for _rt, _bt, params in cand[:12]:
        rt, bt = E.rate_and_budget(fn, params, R_TEST, pmin, pmax, eps, 2024)
        if rt >= TARGET:
            return bt, rt, params
    return float("nan"), 0.0, None


def run(pmin, pmax, eps):
    bb = brute_budget_90(pmin, pmax, eps)
    print(f"\n=== U({pmin:g},{pmax:g}), eps={eps:.0e}   brute@90% = {bb:,.0f} ===")
    for name, (fn, g) in grids(eps).items():
        b, r, _ = best_variable(fn, g, pmin, pmax, eps)
        if np.isfinite(b):
            tag = "WINS" if b < bb / 1.03 else ("~ties" if b < bb * 1.03 else "LOSES")
            print(f"   {name:12s} {b:14,.0f}  (rate {r:.3f})   ratio vs brute {bb/b:4.2f}x  <- {tag}")
        else:
            print(f"   {name:12s} (no config reached 90%)")


if __name__ == "__main__":
    for pmin, pmax, eps in [(0.01, 0.1, 1e-3), (0.01, 0.1, 1e-4)]:
        run(pmin, pmax, eps)
