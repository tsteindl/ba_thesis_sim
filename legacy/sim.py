import numpy as np

def simulate(rng, phi, m, n, n_simulations=1):
    shots_mat = rng.random([n_simulations, m]) < np.cos(n*phi)**2
    return shots_mat[0] if n_simulations == 1 else shots_mat

def simulate_errors(rng, phi, m, N, n_simulations=1):
    N = max(N, 1)
    
    if n_simulations == 1:
        shots_vec = simulate(rng, phi, m, N, n_simulations)
        hits = np.sum(shots_vec)
        prob_zero = hits/m
        phi_hat = 1/N * np.arccos(np.sqrt(prob_zero))

    else:
        shots_mat = simulate(rng, phi, m, N, n_simulations)
        hits_vec = np.sum(shots_mat, axis=1)
        prob_zero_vec = hits_vec/m
        phi_estimates_vec = 1/N * np.arccos(np.sqrt(prob_zero_vec))

        phi_hat = np.mean(phi_estimates_vec) 

    return phi_hat