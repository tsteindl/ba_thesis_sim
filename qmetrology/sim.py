"""Circuit simulation and the phase estimator.

The entangled circuit reads out "0" with probability p0 = cos^2(N*phi). Drawing m shots and
inverting gives phi_hat = arccos(sqrt(hits/m)) / N.

Two equivalent samplers for the hit count: 'bernoulli' draws m shots and sums them (O(m));
'binomial' draws hits ~ Binomial(m, p0) directly (O(1) in m). Default is 'binomial'.
"""
import numpy as np

DEFAULT_METHOD = "binomial"  # "binomial" | "bernoulli"


def set_method(method):
    """Set the global default sampler ('binomial' or 'bernoulli')."""
    global DEFAULT_METHOD
    assert method in ("binomial", "bernoulli")
    DEFAULT_METHOD = method


def simulate(rng, phi, m, n, n_simulations=1):
    shots_mat = rng.random([n_simulations, m]) < np.cos(n * phi) ** 2
    return shots_mat[0] if n_simulations == 1 else shots_mat


def simulate_errors(rng, phi, m, N, n_simulations=1, method=None):
    N = max(N, 1)
    method = method or DEFAULT_METHOD

    if method == "binomial":
        if m <= 0:  # no budget left
            return np.nan
        p = np.cos(N * phi) ** 2
        if n_simulations == 1:
            hits = rng.binomial(m, p)
            return 1 / N * np.arccos(np.sqrt(hits / m))
        hits = rng.binomial(m, p, size=n_simulations)
        return np.mean(1 / N * np.arccos(np.sqrt(hits / m)))

    if n_simulations == 1:
        shots_vec = simulate(rng, phi, m, N, n_simulations)
        hits = np.sum(shots_vec)
        prob_zero = hits / m
        phi_hat = 1 / N * np.arccos(np.sqrt(prob_zero))
    else:
        shots_mat = simulate(rng, phi, m, N, n_simulations)
        hits_vec = np.sum(shots_mat, axis=1)
        prob_zero_vec = hits_vec / m
        phi_estimates_vec = 1 / N * np.arccos(np.sqrt(prob_zero_vec))
        phi_hat = np.mean(phi_estimates_vec)

    return phi_hat
