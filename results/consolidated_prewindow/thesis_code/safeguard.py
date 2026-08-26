"""Statistical safeguard shared by Binary Search and Reverse Engineering."""
import numpy as np
from scipy.special import ndtr


def statistical_safeguard(remaining_budget, N_min, N_max, epsilon,
                          pilot_estimate, pilot_variance, phi_min, phi_max):
    """Return the measured implementation's risk-optimal value of N.

    N_min is retained in the thesis-facing signature. Binary Search applies it as a final floor;
    Reverse Engineering uses the complete range from one to N_max, as in the measured code.
    """
    if remaining_budget <= 0 or pilot_variance <= 0:
        return N_min
    pilot_sd = np.sqrt(pilot_variance)
    lower_mass = ndtr((phi_min - pilot_estimate) / pilot_sd)
    upper_mass = ndtr((phi_max - pilot_estimate) / pilot_sd)
    normalizer = upper_mass - lower_mass
    if normalizer <= 0:
        return N_min

    def score(candidates):
        candidates = np.asarray(candidates, dtype=float)
        alias_limits = np.pi / (2 * candidates)
        p_valid = ((ndtr((alias_limits - pilot_estimate) / pilot_sd) - lower_mass)
                   / normalizer)
        p_valid = np.clip(p_valid, 0, 1)
        p_accurate = 2 * ndtr(2 * epsilon * np.sqrt(candidates * remaining_budget)) - 1
        return p_valid * p_accurate

    lower = 1
    phi_lower = np.clip(pilot_estimate - 5 * pilot_sd, phi_min, phi_max)
    upper = min(int(np.pi / (2 * phi_lower)) + 1, N_max)
    upper = max(upper, 1)
    for _ in range(3):
        if upper - lower <= 1:
            break
        candidates = np.unique(np.geomspace(lower, upper, 128).astype(int))
        best = int(np.argmax(score(candidates)))
        lower = int(candidates[max(best - 1, 0)])
        upper = int(candidates[min(best + 1, len(candidates) - 1)])
    candidates = np.arange(lower, upper + 1, dtype=int)
    return int(candidates[np.argmax(score(candidates))])
