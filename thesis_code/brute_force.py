"""Brute-force baseline from Algorithm 3."""
import numpy as np

from sim import simulate


def estimate_phi_brute_force(rng, phi, phi_max, budget):
    N_min = max(int(np.floor(np.pi / (2 * phi_max))), 1)
    m = budget // N_min
    return simulate(rng, phi, N_min, m)

