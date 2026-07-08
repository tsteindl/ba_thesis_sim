"""Phase-unwrapping ladder (iterative phase estimation a la Kitaev / Higgins et al.).

Why a new protocol: every algorithm in qmetrology/algorithms.py inverts a single batch
with  phi_hat = arccos(sqrt(p_hat)) / N,  which is unambiguous only while N*phi < pi/2.
That caps the usable depth at N <= pi/(2 phi), so brute AND all adaptive protocols scale
as  delta_phi = 1/(2 sqrt(N C))  with N bounded  =>  cost-to-eps ~ 1/eps^2  for everyone,
and the adaptive advantage saturates at the cap ratio (~1.7x for U(0.01, 0.1)).

The cap is informational, not physical.  cos^2(N phi) determines N*phi only up to the
branch set {+-arccos(sqrt(p_hat)) + k pi}; a *previous, coarser* estimate whose confidence
interval is narrower than the branch spacing selects the correct branch.  So the ladder
grows N geometrically stage by stage, unwrapping each measurement with the last estimate,
and N can pass pi/(2 phi) indefinitely:  delta_phi ~ sqrt(m_stage)/C  — Heisenberg scaling,
cost-to-eps ~ 1/eps.

Model note: the thesis circuit has no controllable measurement phase, so instead of the
textbook phase feedback we *steer the depth*: choose N so that N*phi_hat sits on an odd
multiple of pi/4, where the false branches are maximally far (pi/2) and the endpoint
degeneracies (p ~ 0, 1) are avoided.

References: Kitaev, quant-ph/9511026 (1995); Higgins et al., Nature 450, 393 (2007);
Berry et al., PRA 80, 052114 (2009).

Same uniform signature as qmetrology.algorithms:
    find_phi_*(rng, phi, phi_max, phi_min, **params) -> (phi_hat, budget_used)
"""
import numpy as np

HALF_PI = np.pi / 2


def _sample_p0(rng, phi, m, N):
    """Hit fraction of m shots at depth N. Matches qmetrology.sim's 'binomial' sampler:
    the hit count of m Bernoulli(cos^2(N phi)) shots IS Binomial(m, cos^2(N phi))."""
    return rng.binomial(int(m), np.cos(N * phi) ** 2) / m


def _measure_update(rng, phi, m, N, phi_prev, phi_min, phi_max):
    """One stage: measure at depth N, unwrap with the previous estimate, return
    (phi_hat, crb_sd).  Candidates for N*phi are {+-arccos(sqrt(p_hat)) + k pi};
    keep the one closest to N*phi_prev.  The estimate is clamped to the prior
    support [phi_min, phi_max] (always valid: phi is drawn from it)."""
    p_hat = _sample_p0(rng, phi, m, N)
    theta0 = np.arccos(np.sqrt(p_hat))  # folded value, in [0, pi/2]
    t = N * phi_prev
    cand_plus = theta0 + np.round((t - theta0) / np.pi) * np.pi
    cand_minus = -theta0 + np.round((t + theta0) / np.pi) * np.pi
    theta = cand_plus if abs(cand_plus - t) <= abs(cand_minus - t) else cand_minus
    phi_hat = min(max(theta / N, phi_min), phi_max)
    return phi_hat, 1.0 / (2.0 * N * np.sqrt(m))  # CRB sd, phi-independent


def _next_depth(phi_hat, h, m_next, N_cur, phi_min, phi_max, z, n_cap, safety):
    """Deepest safe next N (>= N_cur; == N_cur means no growth possible).

    The safety bound combines two regimes (take the better):
      * principal:  N * (phi_hat + h) <= pi/2  — the whole interval stays on the first
        branch, inversion unique (this is the classic cap; big jump while h >~ phi_hat);
      * unwrapped:  branch selection succeeds if the prior offset N*h plus the folded
        measurement noise (z-sigma: z/(2 sqrt(m)) in theta, doubled for the fold) stays
        inside a `safety` fraction of the pi/4 half-spacing of a steered target.
    Below the bound, ALWAYS steer: snap N so N*phi_hat lands on the largest odd multiple
    of pi/4 that fits (max distance pi/2 to the false branches, no p~0,1 endpoint
    degeneracy).  Landing unsteered next to a fold (N*phi mod pi near 0 or pi/2) is the
    dominant failure mode: there the mirror branch is arbitrarily close.  Only when no
    steered point exists between N_cur and the bound (early stages with h >~ phi_hat)
    fall back to the principal-branch depth, where inversion needs no branch choice.
    """
    hi = min(phi_hat + h, phi_max)
    # fold buffer: keep the interval under pi/2 AND the centre at least pi/8 away from
    # the fold (otherwise the mirror branch is a coin flip); only binds once h < phi_hat/3
    n_principal = int(min(HALF_PI / hi, 3 * np.pi / 8 / phi_hat))
    margin = safety * (np.pi / 4) - z / np.sqrt(m_next)
    n_unwrap = int(margin / h) if margin > 0 else 0
    n_bound = max(n_principal, n_unwrap)
    if n_cap:
        n_bound = min(n_bound, n_cap)
    if n_bound <= N_cur:
        return N_cur
    # steered target: largest odd multiple of pi/4 whose depth fits under the bound
    j = int((n_bound * phi_hat / (np.pi / 4) - 1) // 2)
    while j >= 0:
        n_steer = int(round((2 * j + 1) * (np.pi / 4) / phi_hat))
        if n_steer <= n_bound:
            break
        j -= 1  # rounding pushed it past the bound
    if j >= 0 and N_cur < n_steer <= n_bound:
        return n_steer
    # no steered point reachable: principal-branch depth is safe without steering
    return max(min(n_principal, n_bound), N_cur)


def find_phi_fixed_budget_ladder(rng, phi, phi_max, phi_min, budget, m_stage=100,
                                 z=2.0, final_frac=0.5, safety=0.75, n_cap=None,
                                 trace=None):
    """Fixed-budget phase-unwrapping ladder.

    Rungs of m_stage shots at geometrically deepening N (each unwrapped by the previous
    estimate); when the next rung would eat into the `final_frac` reserve of the
    remaining budget (or depth cannot grow), the entire remainder is spent in one final
    measurement at the deepest safe N.  n_cap=None: depth unbounded (Heisenberg
    scaling); n_cap=int: hardware-capped variant.  `trace`, if a list, collects
    (N, m, phi_hat, crb_sd) per stage (diagnostics only).
    """
    N_min = int(max(np.pi // (2 * phi_max), 1))
    if m_stage * N_min > budget:
        return np.inf, budget

    # stage 0: principal branch by construction (N_min * phi <= pi/2)
    N = N_min
    phi_hat, sd = _measure_update(rng, phi, m_stage, N, 0.0, phi_min, phi_max)
    budget_used = N * m_stage
    if trace is not None:
        trace.append((N, m_stage, phi_hat, sd))

    while True:
        remaining = budget - budget_used
        N_next = _next_depth(phi_hat, z * sd, m_stage, N, phi_min, phi_max, z, n_cap, safety)
        if N_next == N or N_next * m_stage > (1 - final_frac) * remaining:
            # final stage: dump the remainder at the deepest affordable depth
            m_fin = int(remaining // N_next)
            if m_fin < 1:
                N_next, m_fin = N, int(remaining // N)
            if m_fin >= 1:
                phi_hat, sd = _measure_update(rng, phi, m_fin, N_next, phi_hat, phi_min, phi_max)
                budget_used += N_next * m_fin
                if trace is not None:
                    trace.append((N_next, m_fin, phi_hat, sd))
            return phi_hat, budget_used
        phi_hat, sd = _measure_update(rng, phi, m_stage, N_next, phi_hat, phi_min, phi_max)
        budget_used += N_next * m_stage
        N = N_next
        if trace is not None:
            trace.append((N, m_stage, phi_hat, sd))


def find_phi_fixed_budget_ladder_capped(rng, phi, phi_max, phi_min, **params):
    """Hardware-capped ladder: N <= pi/(2 phi_min) — the same maximal depth the existing
    protocols already query (linear/binary scan up to it), so this is the apples-to-apples
    'same hardware' comparison; only the *inference* differs."""
    n_cap = int(max(np.pi // (2 * phi_min), 1))
    return find_phi_fixed_budget_ladder(rng, phi, phi_max, phi_min, n_cap=n_cap, **params)


LADDERS = {
    "ladder_capped": find_phi_fixed_budget_ladder_capped,
    "ladder_unbounded": find_phi_fixed_budget_ladder,
}
