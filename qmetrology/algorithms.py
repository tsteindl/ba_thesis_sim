"""The phase-search algorithms (Chapter 3).

Uniform signature: find_phi_*(rng, phi, phi_max, phi_min, **params) -> (phi_hat, budget_used).
The fixed-budget variants cap the total cost N*m at `budget`; the budget needed for a target
convergence is found by sweeping the budget (analysis/extensive_sweep.py).

The variable-budget ("run until confident") formulations live in legacy/variable_algorithms.py.
Only reverse-engineering-lite (Algorithm 7) is kept here in variable form, as Table 3.1 reports it.
"""
import numpy as np
from scipy.stats import norm

from .sim import simulate_errors


# Separable baseline (N = 1)
def find_phi_fixed_budget_separable(rng, phi, phi_max, phi_min, budget):
    """Separable protocol: a single N=1 circuit measured `budget` times."""
    N = 1
    m = int(budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, m * N


# Brute force (Algorithm 3): fix N = N_min, spend all budget on m
def find_phi_fixed_budget_brute_force(rng, phi, phi_max, phi_min, budget):
    N_min = max(np.pi // (2 * phi_max), 1)
    m = int(budget / N_min)
    phi_hat = simulate_errors(rng, phi, m, N_min)
    return phi_hat, m * N_min


# Linear search (Algorithm 4): scan N upward, detect overshoot via lookback
def find_phi_fixed_budget_linear_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return np.inf, budget

    phi_hat_list = []
    counter = 0
    budget_used = 0
    r_mean = 0
    done = False
    while not done:
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)

        r_mean_new = np.mean(phi_hat_list)
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean:
            counter += 1
        else:
            counter = 0
        r_mean = r_mean_new

        if counter >= lookback_window:
            N -= lookback_window * inc
            done = True
        elif N >= N_max or budget_used >= budget:
            done = True
        else:
            N += inc

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        return phi_hat_list[idx], budget_used

    N = max(1, N - safeguard)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


# Binary search (Algorithm 5): divide-and-conquer on N with a confidence bound
def find_phi_fixed_budget_binary_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        safeguard=1, conf=0.95):
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return np.inf, budget

    phi_hat = simulate_errors(rng, phi, m_exploration, N)
    budget_used = m_exploration * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))

    lb, ub = N_min, N_max
    N += (ub - N) // 2

    done = False
    while not done:
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += m_exploration * N
        temp_N = N
        if phi_hat < phi_1:
            N -= (N - lb) // 2
            ub = temp_N
        else:
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        if temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used

    N = max(N - safeguard, 1)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_anneal_m(rng, phi, phi_max, phi_min, m_exploration, budget, safeguard=1, conf=0.95, max_b_steps_sub=3, delta=5):
    """Binary-search variant that anneals m_exploration up as N grows (m *= (N/N_old)^2 on each step right)."""
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = max(np.pi//(2*phi_min), 1)
    N = N_min
    max_b_steps = np.ceil(np.log2(N_max - N_min)) - max_b_steps_sub
    if m_exploration * N > budget:
        return np.inf, budget
    phi_hat = simulate_errors(rng, phi, m_exploration, N)
    budget_used = m_exploration * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
    lb = N_min
    ub = N_max
    N += (ub - N) // 2
    done = False
    b_steps = 0
    while not done:
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += m_exploration * N
        step_left = (N - lb) // 2
        step_right = (ub - N) // 2
        if ub - lb <= delta:
            break
        N_old = N
        if phi_hat < phi_1:
            if step_left == 0:
                done = True
                break
            ub = N
            N -= step_left
        else:
            if step_right == 0:
                done = True
                break
            lb = N
            N += step_right
            m_exploration = max(int(m_exploration * (N / N_old)**2), 20)
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        b_steps += 1
        if b_steps >= max_b_steps or budget_used >= budget:
            done = True
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used
    N = max(N - safeguard, 1)
    m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


# Reverse engineering (Algorithm 6) and its "lite" variant (Algorithm 7)
def find_phi_fixed_budget_reverse_engineering(rng, phi, phi_max, phi_min, m_exploration,
                                              budget, safeguard=0.9):
    N_min = max(np.pi // (2 * phi_max), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    N = int(np.pi // (2 * phi_hat) * safeguard)
    remaining_budget = budget - budget_used
    if remaining_budget <= 0 or N <= 1:
        return phi_hat, budget_used
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += N * m
    return phi_hat, budget_used


def find_phi_reverse_engineering_lite(rng, phi, phi_max, phi_min, m_exploration=100,
                                      m_exploitation=10):
    """Algorithm 7: infer N from one estimate, then exploit with a small m (no safeguard).
    Variable-budget; reported only in Table 3.1."""
    N_min = max(np.pi // (2 * phi_max), 1)
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    N = int(np.pi // (2 * phi_hat))
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += N * m_exploitation
    return phi_hat, budget_used


# Registry used by config/tables.
FIXED_BUDGET = {
    "separable": find_phi_fixed_budget_separable,
    "brute": find_phi_fixed_budget_brute_force,
    "linear": find_phi_fixed_budget_linear_search,
    "binary": find_phi_fixed_budget_binary_search,
    "reverse_eng": find_phi_fixed_budget_reverse_engineering,
}
