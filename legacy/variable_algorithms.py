"""Archived variable-budget algorithm formulations (not used by the thesis results).

The thesis uses the fixed-budget versions in qmetrology/algorithms.py. Kept for provenance.
"""
import numpy as np
from scipy.stats import norm

from qmetrology.sim import simulate_errors


def find_phi_brute_force(rng, phi, phi_max, phi_min, m):
    N_min = max(np.pi // (2 * phi_max), 1)
    phi_hat = simulate_errors(rng, phi, m, N_min)
    return phi_hat, m * N_min


def find_phi_linear_search(rng, phi, phi_max, phi_min, m_exploration=10,
                           m_exploitation=1_000, lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = np.pi // (2 * phi_min)
    N = max(1, N_min)

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
            N -= lookback_window
            done = True
        elif N >= N_max:
            done = True
        else:
            N += inc

    N = max(1, N - safeguard)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    return phi_hat, budget_used


def find_phi_fixed_budget_linear_search_windowed(rng, phi, phi_max, phi_min, m_exploration,
                                                 budget, window=10, lookback_window=5,
                                                 safeguard=1, inc=1):
    """Linear search whose overshoot test uses a trailing-`window` mean of estimates (unused)."""
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

        r_mean_new = np.mean(phi_hat_list[-window:])
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean:
            counter += 1
        else:
            counter = 0
        r_mean = r_mean_new

        if counter >= lookback_window:
            N -= lookback_window
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


def find_phi_binary_search(rng, phi, phi_max, phi_min, m_exploration=400,
                           m_exploitation=1_000, safeguard=1, conf=0.95):
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = np.pi // (2 * phi_min)
    N = max(1, N_min)

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
        if temp_N == N or N < N_min or N > N_max:
            done = True

    N = max(1, N - safeguard)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    return phi_hat, budget_used


def find_phi_binary_search_anneal_m(rng, phi, phi_max, phi_min, m_exploration=400, m_exploitation=1_000, safeguard=1, conf=0.95, max_b_steps_sub=3, delta=5, annealing_factor=4):
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = np.pi//(2*phi_min)
    N = N_min
    max_b_steps = np.ceil(np.log2(N_max - N_min)) - max_b_steps_sub
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
        if b_steps >= max_b_steps:
            done = True
    N -= safeguard
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    return phi_hat, budget_used


def find_phi_binary_search_delayed(rng, phi, phi_max, phi_min, m_exploration, m_exploitation, m_reference=100, lookback_window=1, safeguard=1, conf=0.95):
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)
    phi_hat_list = []
    N_history = []
    budget_used = 0
    lb = N_min
    ub = N_max
    phi_ref = simulate_errors(rng, phi, m_reference, N_min)
    budget_used += N_min * m_reference
    phi_hat_list.append(phi_ref)
    N_history.append(N_min)
    phi_1 = norm.ppf(1 - conf, phi_ref, np.sqrt(1 / (4 * m_reference * N_min**2)))
    N = N_min + (ub - N_min) // 2
    done = False
    while not done:
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_history.append(N)
        temp_N = N
        recent = phi_hat_list[-lookback_window:]
        is_overshot = (len(phi_hat_list) >= lookback_window + 1 and all(r < phi_1 for r in recent))
        if is_overshot:
            ub = N_history[-lookback_window]
            N = lb + (ub - lb) // 2
        else:
            lb = N
            N += (ub - N) // 2
        if ub < lb or temp_N == N or N < N_min or N > N_max:
            done = True
    N = max(lb - safeguard, 1)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_delayed(rng, phi, phi_max, phi_min, m_exploration, budget, m_reference=100, lookback_window=1, safeguard=1, conf=0.95):
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)
    phi_hat_list = []
    N_history = []
    budget_used = 0
    lb = N_min
    ub = N_max
    if m_reference * N_min > budget:
        return np.inf, budget
    phi_ref = simulate_errors(rng, phi, m_reference, N_min)
    budget_used += N_min * m_reference
    phi_hat_list.append(phi_ref)
    N_history.append(N_min)
    phi_1 = norm.ppf(1 - conf, phi_ref, np.sqrt(1 / (4 * m_reference * N_min**2)))
    N = N_min + (ub - N_min) // 2
    done = False
    while not done:
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_history.append(N)
        temp_N = N
        recent = phi_hat_list[-lookback_window:]
        is_overshot = (len(phi_hat_list) >= lookback_window + 1 and all(r < phi_1 for r in recent))
        if is_overshot:
            ub = N_history[-lookback_window]
            N = lb + (ub - lb) // 2
        else:
            lb = N
            N += (ub - N) // 2
        if ub < lb or temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat_list[-1], budget_used
    N = max(lb - safeguard, 1)
    m = int(remaining_budget / N)
    if m <= 0:
        return phi_hat_list[-1], budget_used
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


def find_phi_reverse_engineering(rng, phi, phi_max, phi_min, m_exploration=50,
                                 m_exploitation=400, safeguard=0.9):
    N_min = max(np.pi // (2 * phi_max), 1)
    phi_hat = 0
    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used = m_exploration * N_min
    N = int(np.pi // (2 * phi_hat) * safeguard)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += N * m_exploitation
    return phi_hat, budget_used
