"""Reverse Engineering with statistical safeguard (Algorithm 6)."""
import numpy as np

from safeguard import statistical_safeguard
from sim import simulate


def estimate_phi_reverse_engineering(rng, phi, phi_min, phi_max, budget,
                                     exploration_shots, epsilon):
    N_min = max(int(np.floor(np.pi / (2 * phi_max))), 1)
    N_max = max(int(np.floor(np.pi / (2 * phi_min))), 1)
    used_budget = 0
    pilot_estimate = 0.0

    while pilot_estimate == 0:
        pilot_cost = exploration_shots * N_min
        if used_budget + pilot_cost > budget:
            return np.nan
        pilot_estimate = simulate(rng, phi, N_min, exploration_shots)
        used_budget += pilot_cost

    remaining_budget = budget - used_budget
    if remaining_budget <= 0:
        return pilot_estimate
    pilot_variance = 1 / (4 * exploration_shots * N_min ** 2)
    N_star = statistical_safeguard(
        remaining_budget, N_min, N_max, epsilon,
        pilot_estimate, pilot_variance, phi_min, phi_max)
    m = remaining_budget // N_star
    return pilot_estimate if m < 1 else simulate(rng, phi, N_star, m)

