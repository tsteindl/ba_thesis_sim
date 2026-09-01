"""Linear Search from Algorithm 4."""
import numpy as np

from sim import simulate


def estimate_phi_linear_search(rng, phi, phi_min, phi_max, budget,
                               exploration_shots, lookback, safeguard,
                               increment=1):
    N_min = max(int(np.floor(np.pi / (2 * phi_max))), 1)
    N_max = max(int(np.floor(np.pi / (2 * phi_min))), 1)
    N = N_min
    estimates = []
    previous_mean = 0.0
    decreasing_steps = 0
    used_budget = 0
    N_guess = N

    while used_budget + exploration_shots * N <= budget:
        estimate = simulate(rng, phi, N, exploration_shots)
        estimates.append(estimate)
        used_budget += exploration_shots * N
        N_guess = N

        running_mean = np.mean(estimates)
        if len(estimates) >= 2 and running_mean < previous_mean:
            decreasing_steps += 1
        else:
            decreasing_steps = 0
        previous_mean = running_mean

        if decreasing_steps >= lookback:
            N_guess = N - lookback * increment
            break
        if N >= N_max or used_budget >= budget:
            N_guess = N
            break
        N = min(N + increment, N_max)

    if not estimates:
        return np.nan
    remaining_budget = budget - used_budget
    N_star = max(N_guess - safeguard, 1)
    m = remaining_budget // N_star
    return estimates[-1] if m < 1 else simulate(rng, phi, N_star, m)
