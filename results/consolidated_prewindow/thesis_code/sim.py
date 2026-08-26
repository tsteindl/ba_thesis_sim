"""Maximally entangled phase-estimation measurement used by every listing."""
import numpy as np


def simulate(rng, phi, N, m):
    """Draw m outcomes and return arccos(sqrt(p_hat_0))/N."""
    if m < 1:
        return np.nan
    p_zero = np.cos(N * phi) ** 2
    zero_count = rng.binomial(m, p_zero)
    return np.arccos(np.sqrt(zero_count / m)) / N

