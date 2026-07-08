import numpy as np
from sim import simulate_errors


def find_phi_brute_force(rng, phi, phi_max, phi_min, m):
    N_min = max(np.pi//(2*phi_max), 1)

    phi_hat = simulate_errors(rng, phi, m, N_min)
    budget_used = m * N_min
    
    return phi_hat, budget_used


def find_phi_fixed_budget_brute_force(rng, phi, phi_max, phi_min, budget):
    N_min = max(np.pi//(2*phi_max), 1)

    m = int(budget/N_min)
    phi_hat = simulate_errors(rng, phi, m, N_min)
    budget_used = m * N_min
    
    return phi_hat, budget_used