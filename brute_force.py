import numpy as np
from sim import simulate_errors


def find_phi_brute_force(rng, phi, phi_max, phi_min, m):
    N_min = np.pi//(2*phi_max)

    phi_hat = simulate_errors(rng, phi, m, N_min)
    budget_used = m * N_min
    
    return phi_hat, budget_used