import numpy as np
from sim import simulate_errors

def find_phi_linear_search(rng, phi, phi_max, phi_min, m_exploration=10, m_exploitation=1_000, lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = np.pi//(2*phi_min)
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
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean: #TODO check if still works
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
    
    # remaining_budget = budget-budget_used
    # if remaining_budget <= 0:
    #     return budget_used, results[-lookback_window]

    N = max(1, N - safeguard) # to avoid overshooting
    
    # m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    
    return phi_hat, budget_used


def find_phi_linear_search_windowed(rng, phi, phi_max, phi_min, m_exploration=10, m_exploitation=1_000, window=10, lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = np.pi//(2*phi_min)
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
        
        r_mean_new = np.mean(phi_hat_list[-window:])
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean: #TODO check if still works
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
    
    # remaining_budget = budget-budget_used
    # if remaining_budget <= 0:
    #     return budget_used, results[-lookback_window]

    N = max(1, N - safeguard) # to avoid overshooting
    
    # m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += m_exploitation * N
    
    return phi_hat, budget_used

def find_phi_fixed_budget_linear_search(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = max(np.pi//(2*phi_min), 1)
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
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean: #TODO check if still works
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

    N = max(1, N - safeguard) # to avoid overshooting

    m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N

    return phi_hat, budget_used


def find_phi_fixed_budget_linear_search_windowed(rng, phi, phi_max, phi_min, m_exploration, budget, window=10, lookback_window=5, safeguard=1, inc=1):
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = max(np.pi//(2*phi_min), 1)
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

    N = max(1, N - safeguard) # to avoid overshooting

    m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N

    return phi_hat, budget_used