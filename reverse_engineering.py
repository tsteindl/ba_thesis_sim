import numpy as np
from sim import simulate_errors

def find_phi_reverse_engineering(rng, phi, phi_max, phi_min, m_exploration=50, m_exploitation=400, safeguard=0.9):
    N_min = np.pi//(2*phi_max)
    
    phi_hat = 0

    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used = m_exploration * N_min
    
    N = int(np.pi//(2*phi_hat)*safeguard)
    
    # remaining_budget = budget - budget_used
    # if remaining_budget <= 0:
        # return result, budget_used
    
    # shots = int(remaining_budget/n)
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    
    budget_used += N * m_exploitation
    
    return phi_hat, budget_used