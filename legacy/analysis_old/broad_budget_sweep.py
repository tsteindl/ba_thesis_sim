"""Does the adaptive advantage for BROAD distributions grow with budget?

Hypothesis: at budget=10,000 the two-phase methods can't afford to explore AND
exploit when phi spans a wide range, so they barely beat brute force. Give them
more budget and the gap should widen. Honest (de-biased) % converged.
    python analysis/broad_budget_sweep.py
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

PHI_MIN, EPS = 0.01, 1e-3
R_TUNE, R_TEST = 1000, 30000
GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 1000, 12).astype(int)),
            "lookback_window": [1, 2, 5], "safeguard": [1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 3000, 12).astype(int)),
           "safeguard": [0.8, 0.85, 0.9, 0.95]}


def best(fn, grid, budget, phi_max):
    """Grid-tune at this budget (seed 42), de-bias the winner (seed 2024). % converged."""
    g = {**grid, "budget": [budget]}
    res = E.grid_full(fn, g, R_TUNE, PHI_MIN, phi_max, EPS, 42)
    win = max(res, key=lambda r: r[0])[2]
    return 100 * E.success_rate(fn, win, R_TEST, PHI_MIN, phi_max, EPS, 2024)


def brute(budget, phi_max):
    return 100 * E.success_rate(BF, {"budget": budget}, R_TEST, PHI_MIN, phi_max, EPS, 2024)


if __name__ == "__main__":
    for phi_max, name in [(np.pi / 2, "pi/2"), (np.pi / 8, "pi/8")]:
        print(f"\n=== broad phi ~ U(0.01, {name}) — honest % converged, and margin over brute ===")
        print(f"{'budget':>8} | {'brute':>6} | {'linear':>14} | {'reverse_eng':>16}")
        for budget in [10_000, 30_000, 100_000, 300_000, 1_000_000]:
            b = brute(budget, phi_max)
            lin = best(LIN, GRID_LIN, budget, phi_max)
            re = best(RE, GRID_RE, budget, phi_max)
            print(f"{budget:>8,} | {b:6.1f} | {lin:6.1f} ({lin-b:+5.1f}) | {re:6.1f} ({re-b:+5.1f})")
