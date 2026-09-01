"""Statistical safeguard shared by Binary Search and Reverse Engineering."""
import numpy as np
from scipy.special import ndtr


def statistical_safeguard(remaining_budget, N_min, N_max, epsilon,
                          pilot_estimate, pilot_variance, phi_min, phi_max):
    """Return the risk-optimal exploitation depth: the exact argmax over N_min, ..., N_max.

    Every integer of the closed interval is scored in one vectorised pass, so the returned depth is
    the exact maximiser of the score over the admissible set -- no coarse-to-fine search, no cap and
    no cutoff.  The accuracy factor uses the whole number of shots the depth can be paid for,
    m = floor(B / N).  Both algorithms use the same range: N_min = floor(pi / 2 phi_max) is safe for
    the entire prior support, so no admissible phase justifies going shallower.
    """
    if remaining_budget <= 0 or pilot_variance <= 0:
        return N_min
    pilot_sd = np.sqrt(pilot_variance)
    lower_mass = ndtr((phi_min - pilot_estimate) / pilot_sd)
    upper_mass = ndtr((phi_max - pilot_estimate) / pilot_sd)
    normalizer = upper_mass - lower_mass
    if normalizer <= 0:
        return N_min

    candidates = np.arange(N_min, max(N_max, N_min) + 1, dtype=np.int64)
    alias_limits = np.pi / (2 * candidates.astype(float))
    p_valid = ((ndtr((alias_limits - pilot_estimate) / pilot_sd) - lower_mass) / normalizer)
    p_valid = np.clip(p_valid, 0, 1)
    shots = int(remaining_budget) // candidates
    p_accurate = np.where(
        shots >= 1,
        2 * ndtr(2 * epsilon * candidates.astype(float) * np.sqrt(shots)) - 1,
        0.0,
    )
    return int(candidates[int(np.argmax(p_valid * p_accurate))])
