"""Binary Search with deepest accepted pilot and statistical safeguard (Algorithm 5)."""
import numpy as np
from scipy.special import ndtri

from safeguard import statistical_safeguard
from sim import simulate


def _lower_threshold(reference, N, exploration_shots, alpha):
    sd = 1 / (2 * N * np.sqrt(exploration_shots))
    return reference + ndtri(alpha) * sd


def estimate_phi_binary_search(rng, phi, phi_min, phi_max, budget,
                               exploration_shots, alpha, epsilon):
    N_min = max(int(np.floor(np.pi / (2 * phi_max))), 1)
    N_max = max(int(np.floor(np.pi / (2 * phi_min))), 1)
    if exploration_shots * N_min > budget:
        return np.nan

    N = N_min
    accepted_estimate = simulate(rng, phi, N, exploration_shots)
    accepted_N = N
    used_budget = exploration_shots * N
    threshold = _lower_threshold(accepted_estimate, N, exploration_shots, alpha)

    lower, upper = N_min, N_max
    N += (upper - N) // 2
    while True:
        if used_budget + exploration_shots * N > budget:
            break
        estimate = simulate(rng, phi, N, exploration_shots)
        used_budget += exploration_shots * N
        old_N = N

        if estimate < threshold:
            upper = old_N
            N -= (N - lower) // 2
        else:
            accepted_estimate, accepted_N = estimate, old_N
            lower = old_N
            N += (upper - N) // 2
            threshold = _lower_threshold(
                accepted_estimate, N, exploration_shots, alpha)
        if N == old_N:
            break

    remaining_budget = budget - used_budget
    if remaining_budget <= 0:
        return accepted_estimate
    pilot_variance = 1 / (4 * exploration_shots * accepted_N ** 2)
    N_star = statistical_safeguard(
        remaining_budget, N_min, N_max, epsilon,
        accepted_estimate, pilot_variance, phi_min, phi_max)
    N_star = max(N_star, N_min)
    m = remaining_budget // N_star
    return accepted_estimate if m < 1 else simulate(rng, phi, N_star, m)
