"""Confidence-Calibrated Reverse Engineering (CCRE, "Algorithm 8").

A same-family improvement on reverse engineering (qmetrology.algorithms
.find_phi_fixed_budget_reverse_engineering): still single-batch arccos inversion,
phi_hat = arccos(sqrt(p_hat))/N, staying strictly inside the principal branch
(N*phi < pi/2) at every measurement — no phase-unwrapping, unlike ladder/. That means CCRE
cannot leave the SQL cost class (cost-to-eps ~ 1/eps^2, same as brute/linear/binary/RE); its
target is RE's ~1.72x plateau, not the ladder's unbounded growth (see ccre/results/CCRE.md for
the honest ceiling derivation, E[phi_max/phi] ~ 2.56x under U(0.01,0.1)).

RE has two weaknesses this fixes:
  1. a single exploration shot with no correction — CCRE allows a small FIXED number of cheap
     confirmation rounds that each tighten phi_hat before committing the bulk of the budget;
  2. an ad hoc fixed multiplicative safeguard (0.9) that ignores how precise the exploration
     estimate actually was — CCRE replaces it with a confidence-bound safe depth,
     N_safe = floor(pi / (2*(phi_hat + z*sigma))),  z = norm.ppf(conf),
     sigma = 1/(2*N*sqrt(m)) (the same CRB formula binary search and ladder already use — see
     qmetrology/algorithms.py's find_phi_fixed_budget_binary_search and
     ladder/algorithms.py's _next_depth n_principal branch). This gives a PROVABLE per-round
     overshoot bound P(overshoot) ~ 1-conf, unlike RE's ungoverned constant or binary search's
     aliasing-broken bisection past the first fold.

Same uniform signature as qmetrology.algorithms:
    find_phi_*(rng, phi, phi_max, phi_min, **params) -> (phi_hat, budget_used)
"""
import numpy as np
from scipy.stats import norm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.sim import simulate_errors


def _safe_next_N(phi_hat, N_cur, m_round, z, N_max):
    """Deepest N such that phi_hat + z*sigma stays under the pi/2 branch fold, where
    sigma = 1/(2*N_cur*sqrt(m_round)) is the CRB sd of the estimate just measured at N_cur."""
    sigma = 1.0 / (2 * N_cur * np.sqrt(m_round))
    return min(max(int(np.pi // (2 * (phi_hat + z * sigma))), 1), N_max)


def find_phi_fixed_budget_ccre(rng, phi, phi_max, phi_min, budget,
                               m_round=200, conf=0.95, n_rounds=1, trace=None):
    """Fixed-budget CCRE.

    Round 1 (exploration) measures m_round shots at N_min; up to n_rounds-1 further
    confirmation rounds each measure m_round shots at the current confidence-safe depth,
    tightening phi_hat before the final round spends whatever budget remains at the last
    safe depth. n_rounds=1 degenerates to RE's shape with safeguard -> conf.
    `trace`, if a list, collects (N, m, phi_hat) per round (diagnostics only).
    """
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)
    z = norm.ppf(conf)
    if m_round * N_min > budget:
        return np.inf, budget

    N = N_min
    phi_hat = 0.0
    budget_used = 0
    while phi_hat == 0:  # resample an all-hit draw (same guard as RE)
        phi_hat = simulate_errors(rng, phi, m_round, N)
        budget_used += m_round * N
    if trace is not None:
        trace.append((N, m_round, phi_hat))

    for _ in range(n_rounds - 1):  # confirmation rounds 2..n_rounds
        remaining = budget - budget_used
        N_next = _safe_next_N(phi_hat, N, m_round, z, N_max)
        if N_next <= N or m_round * N_next > remaining:
            break
        phi_hat = simulate_errors(rng, phi, m_round, N_next)
        budget_used += m_round * N_next
        N = N_next
        if trace is not None:
            trace.append((N, m_round, phi_hat))

    remaining = budget - budget_used
    if remaining <= 0:
        return phi_hat, budget_used
    N_final = _safe_next_N(phi_hat, N, m_round, z, N_max)
    m = int(remaining / N_final)
    if m < 1:
        return phi_hat, budget_used
    phi_hat = simulate_errors(rng, phi, m, N_final)
    budget_used += m * N_final
    if trace is not None:
        trace.append((N_final, m, phi_hat))
    return phi_hat, budget_used
