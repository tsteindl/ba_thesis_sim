"""Exact posterior for phi given the whole exploration history — the informative use of a bisection.

Eq. (3.8) needs P(phi < pi/2N | data). The normal-approximation route (qmetrology/safeguard.py) gets
it from ONE probe, via the estimator phi_hat = arccos(sqrt(hits/m))/N and its asymptotic law. That
route has to throw away every probe the overshoot test flags, because arccos cannot be inverted past
pi/2 -- and those are precisely the most informative probes.

The likelihood does not have that problem. For a probe of m shots at depth N,

    hits ~ Binomial(m, cos^2(N phi))

is a valid statement about phi for EVERY phi, aliased or not. So the exact posterior under the uniform
prior of A3 is just the normalised product over all K probes,

    p(phi | data)  ~  prod_i  cos^2(N_i phi)^{hits_i} (1 - cos^2(N_i phi))^{m - hits_i},
                      phi in [phi_min, phi_max],

which is one-dimensional and can be evaluated on a grid. Nothing is discarded and no asymptotics are
invoked. A probe that reads hits ~ 0 at depth N contributes a sharp ridge at N phi ~ pi/2, i.e.
phi ~ pi/(2N) -- exactly the "N_guess is close to N_opt" information a bisection produces, entered as
evidence instead of as a hard classification.

The multimodality that makes arccos ambiguous (cos^2 is periodic, so a deep probe alone cannot tell
phi from phi + pi/N) is resolved by the shallow probes: the opening probe at N_min is unimodal over
the prior by construction (A2), so the product has a single dominant mode.
"""
import numpy as np
from scipy.special import ndtr

_EPS = 1e-300


def phi_grid(phi_min, phi_max, n=4096):
    """Grid the prior support. n=4096 resolves pi/(2N) spacing up to N ~ 1e3 comfortably."""
    return np.linspace(float(phi_min), float(phi_max), int(n))


def log_likelihood(phi, probes, m):
    """sum_i log Binomial(hits_i; m, cos^2(N_i phi)), up to a phi-independent constant."""
    out = np.zeros_like(phi)
    for N, hits in probes:
        p0 = np.cos(float(N) * phi) ** 2
        out += hits * np.log(p0 + _EPS) + (m - hits) * np.log1p(-p0 + _EPS)
    return out


def posterior(probes, m, phi_min, phi_max, n_grid=4096):
    """(phi grid, normalised posterior weights) under a uniform prior on [phi_min, phi_max]."""
    phi = phi_grid(phi_min, phi_max, n_grid)
    lp = log_likelihood(phi, probes, m)
    lp -= lp.max()
    w = np.exp(lp)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return phi, np.full_like(phi, 1.0 / len(phi))
    return phi, w / s


def depth_from_posterior(probes, m, phi_min, phi_max, budget, eps, N_max=None, n_grid=4096):
    """N* = argmax_N P(phi < pi/2N | data) * [2 Phi(2 eps sqrt(N B)) - 1], exact posterior.

    Same objective as Eq. (3.8); only the first factor is computed differently -- from the whole
    history by its exact likelihood rather than from one probe by its asymptotic law.
    """
    phi, w = posterior(probes, m, phi_min, phi_max, n_grid)
    cdf = np.cumsum(w)                      # P(phi <= phi_k | data)
    hi = int(N_max) if N_max else max(int(np.pi / (2.0 * phi[0])), 1)
    hi = max(min(hi, 1 << 22), 1)
    N = np.arange(1, hi + 1, dtype=np.int64)
    # P(phi < pi/(2N)) by locating the threshold in the grid
    thr = np.pi / (2.0 * N)
    idx = np.searchsorted(phi, thr, side="left") - 1
    p_safe = np.where(idx < 0, 0.0, cdf[np.clip(idx, 0, len(cdf) - 1)])
    p_safe = np.where(thr >= phi[-1], 1.0, p_safe)
    p_conv = 2.0 * ndtr(2.0 * eps * np.sqrt(N.astype(float) * float(budget))) - 1.0
    return int(N[int(np.argmax(p_safe * p_conv))])


def hits_from_estimate(phi_hat, N, m):
    """Recover the sufficient statistic from a reported estimate (phi_hat = arccos(sqrt(k/m))/N)."""
    if not np.isfinite(phi_hat):
        return 0
    return int(round(m * np.cos(float(N) * float(phi_hat)) ** 2))
