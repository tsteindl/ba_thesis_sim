"""Fix a SINGLE phi (one 'experiment') and ask: what budget does each algorithm need to
converge that phi >=90% of the time? Tune each per-phi (fair, since it's one experiment).
This tests the user's claim that per-phi, variable RE beats brute at some budget.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.sim import simulate_errors
from qmetrology.algorithms import find_phi_reverse_engineering as RE

pmin, pmax, eps, R = 0.001, 0.01, 1e-4, 4000
N_min = int(np.pi // (2 * pmax))  # 157


def rate_at(fn_kind, phi, budget=None, m_fin=None, m_exp=500, sg=0.9):
    seeds = np.random.default_rng(2024).integers(0, 2**63, R)
    s = 0; bud = 0.0
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        if fn_kind == "brute":
            m = max(1, int(budget / N_min))
            ph = simulate_errors(rng, phi, m, N_min); b = m * N_min
        else:
            ph, b = RE(rng, phi, pmax, pmin, m_exploration=m_exp, m_exploitation=int(m_fin), safeguard=sg)
        bud += b
        if abs(ph - phi) < eps: s += 1
    return s / R, bud / R


def brute_budget_90(phi):
    for b in np.geomspace(20_000, 3_000_000, 40):
        if rate_at("brute", phi, budget=b)[0] >= 0.90:
            return b
    return float("nan")


def re_budget_90(phi):
    best = float("inf")
    for mf in np.geomspace(1_000, 3_000_000, 40):
        r, b = rate_at("re", phi, m_fin=mf)
        if r >= 0.90:
            best = min(best, b)
    return best


print(f"single-phi 'budget for 90%'  (U(0.001,0.01), eps=1e-4, N_min={N_min})\n")
print(f"  {'phi':>7} | {'brute budget':>13} | {'RE budget':>13} | RE vs brute")
for phi in [0.009, 0.005, 0.003, 0.0015]:
    bb, br = brute_budget_90(phi), re_budget_90(phi)
    print(f"  {phi:7.4f} | {bb:13,.0f} | {br:13,.0f} | x{bb/br:.2f}")
