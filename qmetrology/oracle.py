"""The omniscient benchmark: the depth N a protocol would pick if it already knew phi.

Every adaptive algorithm in Chapter 3 spends part of its budget learning phi only so that it can
choose the exploitation depth N. This module removes that problem entirely: given the true phi, it
returns the depth that maximises the convergence probability at budget C. The resulting convergence
rate is an *upper bound* for the whole family — no exploration schedule, safeguard, or stopping rule
can beat an algorithm that already knows the answer — which turns the reported budget ratios from
"adaptive beats brute force" into "adaptive closes x% of the gap to what is achievable at all".

Three ceilings are provided, and the differences between them are themselves results:

  `alias_depth`     N = floor(pi / (2 phi)) — the largest depth that does not alias (N phi <= pi/2).
                    This is what Eq. (3.6) reverse-engineers from the pilot, with C_safe = 1.
                    Perfect knowledge of phi, no backoff.

  `optimal_depth`   N* = argmax_N P(|phi_hat - phi| < eps), evaluated exactly from the binomial law.
                    The tightest valid bound — but see the degeneracy warning below: it is *valid*
                    everywhere and *informative* only where 2 phi^2 / pi > eps.

  `heisenberg_prob` the same perfect depth choice, but with the accuracy required to come from the
                    sampling distribution (Eq. 3.4) instead of from the aliasing constant. Not a
                    strict bound on the exact estimator, but the ceiling that means something:
                    non-degenerate everywhere, analytic, and directly comparable to Eq. (3.7).

**Report `heisenberg_prob` as the headline ceiling.** The exact oracle is the right object
mathematically and the wrong one rhetorically: in scenarios where the whole prior satisfies
phi < sqrt(pi eps / 2) it answers from the choice of N alone and reports absurd advantages
(~400x at phi ~ U(0.001,0.01), eps = 1e-4). `degeneracy_share` quantifies how much of a given
prior is affected, and should be quoted wherever the exact oracle is.

`alias_depth` is *not* optimal, and that is the point: at N = floor(pi/(2 phi)) the readout
probability p0 = cos^2(N phi) sits against the boundary p0 ~ 0, where the arccos estimator is
boundary-clamped rather than asymptotically normal (the Var = 1/(4 N^2 m) law of Eq. (3.4) assumes
p0 (1-p0) >> 1/m). The residual pi/2 - N phi < phi leaves a systematic error of order phi^2, which is
harmless at eps = 10^-3 and fatal at eps = 10^-6. The optimal depth therefore backs off from the
aliasing bound even with *perfect* information — the safeguard is not only a hedge against pilot
noise, and a protocol that inferred phi exactly would still need one.

Exact convergence probability. With K ~ Binomial(m, cos^2(N phi)) and phi_hat = arccos(sqrt(K/m))/N,

    |phi_hat - phi| < eps  <=>  A < arccos(sqrt(K/m)) < B,   A = max(N(phi-eps), 0)
                                                             B = min(N(phi+eps), pi/2)
                           <=>  m cos^2 B < K < m cos^2 A     (arccos is decreasing)

so P = F(ceil(m cos^2 A) - 1) - F(floor(m cos^2 B)) with F the Binomial(m, cos^2(N phi)) CDF. No
simulation and no normal approximation enter, so the bound holds at every budget, not just
asymptotically.
"""
import numpy as np
from scipy.special import ndtr   # standard normal CDF, vectorised
from scipy.stats import binom

_EXHAUSTIVE_MAX = 4096   # scan every admissible depth when there are at most this many
_N_SEARCH = 256          # candidates per refinement round, beyond that
_N_ROUNDS = 3            # geometric refinement rounds before the final integer scan
_TOP_WINDOW = 2048       # depths below the aliasing bound always scanned exactly


def alias_depth(phi, budget):
    """Largest non-aliasing depth, N = floor(pi/(2 phi)), capped so that m = budget/N >= 1."""
    n = max(int(np.pi // (2.0 * phi)), 1)
    return max(min(n, int(budget)), 1)


def convergence_prob(N, phi, budget, eps):
    """Exact P(|phi_hat - phi| < eps) at depth N with m = floor(budget/N) shots.

    Vectorised over N. Returns 0 where the depth is infeasible (m = 0) or the convergence window is
    empty (N(phi-eps) >= pi/2, i.e. even the shallowest admissible reading aliases).
    """
    N = np.atleast_1d(np.asarray(N, dtype=np.int64))
    m = np.asarray(budget, dtype=np.int64) // N

    lo_ang = N * (phi - eps)                           # A, unclipped
    hi_ang = N * (phi + eps)                           # B, unclipped
    # arccos(.) always lands in [0, pi/2]: a bound outside that range is simply not binding, and
    # must NOT be clipped onto pi/2 — that would drop the K = 0 reading, which is exactly the one
    # taken when the depth sits at the aliasing edge.
    feasible = (m >= 1) & (lo_ang < np.pi / 2)
    if not feasible.any():
        return np.zeros(N.shape)

    m_safe = np.where(feasible, m, 1)
    p0 = np.cos(N * phi) ** 2
    # K <= k_hi  <=>  arccos(sqrt(K/m)) > A   (no constraint when A <= 0)
    k_hi = np.where(lo_ang > 0, np.ceil(m_safe * np.cos(np.minimum(lo_ang, np.pi / 2)) ** 2) - 1, m_safe)
    p = binom.cdf(k_hi, m_safe, p0)
    # K >  k_lo  <=>  arccos(sqrt(K/m)) < B   (no constraint when B >= pi/2, the common case at
    # large N — skipping the second CDF there is worth it, it dominates the cost of the whole search)
    binding = hi_ang < np.pi / 2
    if binding.any():
        k_lo = np.where(binding, np.floor(m_safe * np.cos(np.minimum(hi_ang, np.pi / 2)) ** 2), -1.0)
        p = p - binom.cdf(k_lo, m_safe, p0)
    return np.where(feasible, np.clip(p, 0.0, 1.0), 0.0)


def expected_hits(N, phi, budget):
    """m * p0 — the expected number of 0-outcomes at depth N with m = floor(budget/N) shots.

    The single quantity that decides whether the measurement carries information. When it drops
    below ~1 the readout is deterministic (K = 0 every time) and phi_hat collapses to the constant
    pi/(2N), which is not an estimate at all; above ~30 the estimator is in its regular regime and
    the exact binomial probability agrees with the asymptotic law of Eq. (3.4) to about 1%.
    """
    N = np.atleast_1d(np.asarray(N, dtype=np.int64))
    return (np.asarray(budget, dtype=np.int64) // N) * np.cos(N * np.asarray(phi, float)) ** 2


def _argmax_over(cand, phi, budget, eps, min_hits=0.0):
    p = convergence_prob(cand, phi, budget, eps)
    if min_hits > 0:
        # discard depths where the readout is deterministic: there phi_hat = pi/(2N) regardless of
        # the data, so a high "convergence probability" only reflects that N was chosen from phi
        p = np.where(expected_hits(cand, phi, budget) >= min_hits, p, -1.0)
    return int(cand[int(np.argmax(p))])


def optimal_depth(phi, budget, eps, N_max=None, min_hits=0.0):
    """N* = argmax_N P(|phi_hat - phi| < eps) — the depth an omniscient protocol would choose.

    The admissible range is 1 <= N <= N_hi, with N_hi the largest depth whose convergence window is
    non-empty. Whenever that range holds at most `_EXHAUSTIVE_MAX` depths — which covers every prior
    used in the thesis, since N_hi ~ pi/(2 phi_min) — **every** depth is evaluated, so the returned
    depth is the exact maximiser and the oracle really is an upper bound.

    The objective is not perfectly unimodal (m = floor(budget/N) and the discreteness of the binomial
    put small kinks in it), so a purely geometric search can settle on a local maximum: measured
    against an exhaustive scan it missed the true argmax in ~2% of draws, by up to 4 percentage
    points of convergence probability. Refinement is therefore used only for the very wide priors
    (phi_min <~ 1e-4) where an exhaustive scan is too costly, and even then it is combined with an
    exact scan of the `_TOP_WINDOW` depths below the aliasing bound, where the maximum sits unless
    the budget is too small to resolve phi at all.
    """
    budget = int(budget)
    if budget < 1 or eps <= 0 or not np.isfinite(phi) or phi <= 0:
        return 1
    hi = budget
    if phi > eps:                                # beyond this every reading aliases -> P = 0
        hi = min(hi, int(np.pi / (2.0 * (phi - eps))) + 1)
    if N_max is not None:
        hi = min(hi, int(N_max))
    hi = max(int(hi), 1)

    if hi <= _EXHAUSTIVE_MAX:                    # exact
        return max(_argmax_over(np.arange(1, hi + 1, dtype=np.int64), phi, budget, eps, min_hits), 1)

    lo = 1
    best = [_argmax_over(np.arange(max(hi - _TOP_WINDOW, 1), hi + 1, dtype=np.int64),
                         phi, budget, eps, min_hits)]
    for _ in range(_N_ROUNDS):
        if hi - lo <= 1:
            break
        cand = np.unique(np.geomspace(lo, hi, _N_SEARCH).astype(np.int64))
        i = cand.tolist().index(_argmax_over(cand, phi, budget, eps, min_hits))
        best.append(int(cand[i]))
        lo, hi = int(cand[max(i - 1, 0)]), int(cand[min(i + 1, len(cand) - 1)])
    best.append(_argmax_over(np.arange(lo, hi + 1, dtype=np.int64), phi, budget, eps, min_hits))
    return max(_argmax_over(np.array(best, dtype=np.int64), phi, budget, eps, min_hits), 1)


def best_convergence_prob(phi, budget, eps, N_max=None):
    """P(converge) at the optimal depth — the per-phi ceiling, without simulating anything."""
    return float(convergence_prob(optimal_depth(phi, budget, eps, N_max), phi, budget, eps)[0])


# --------------------------------------------------------------------------------------------
# The exact oracle above degenerates, and the reason is worth stating precisely.
#
# At the aliasing edge the readout is deterministic: with p0 = cos^2(N phi) ~ 0 every shot returns
# K = 0, so phi_hat = pi/(2N) *regardless of the data*. An oracle that knows phi can therefore choose
# N = floor(pi/(2 phi)) and have the answer handed to it by that constant, to absolute accuracy
#
#     |pi/(2 floor(pi/2phi)) - phi|  ~  2 phi^2 / pi,
#
# using a handful of shots and no averaging at all. Whenever phi < sqrt(pi eps / 2) that error is
# already below the tolerance, and the "budget needed" collapses to almost nothing — at
# phi ~ U(0.001,0.01), eps = 1e-4 the whole prior is in that regime and the oracle reports a
# meaningless ~400x advantage.
#
# This is a property of the model, not a bug: the depth is a continuous-valued choice, so choosing it
# with exact knowledge of phi smuggles phi into the estimate. It makes the exact oracle a valid but
# vacuous bound there. The useful ceiling is the one where the accuracy has to come from the
# *statistics* rather than from the choice of N, i.e. the asymptotic law of Eq. (3.4) evaluated at the
# best admissible depth — which is exactly Eq. (3.7) of the thesis with N = N_opt.
def heisenberg_prob(phi, budget, eps):
    """P(converge) at depth N = floor(pi/(2 phi)) under Eq. (3.4): 2 Phi(2 eps sqrt(N C)) - 1.

    The non-degenerate ceiling. It grants perfect knowledge of the optimal depth but still requires
    the estimate to be resolved by the sampling distribution, so it cannot exploit the deterministic
    aliasing readout. Being analytic it carries no Monte-Carlo noise.
    """
    N = np.maximum(np.floor(np.pi / (2.0 * np.asarray(phi, float))), 1.0)
    N = np.minimum(N, float(budget))
    return np.clip(2.0 * ndtr(2.0 * eps * np.sqrt(N * float(budget))) - 1.0, 0.0, 1.0)


def heisenberg_rate(budget, eps, phi_min, phi_max, n_grid=20001):
    """Convergence rate of the non-degenerate ceiling under phi ~ U(phi_min, phi_max).

    Averaged by quadrature over the prior rather than by sampling, so the curve is exact — there is
    no simulation and therefore no error bar to attach to it.
    """
    phi = np.linspace(phi_min, phi_max, n_grid)
    return float(np.trapezoid(heisenberg_prob(phi, budget, eps), phi) / (phi_max - phi_min))


def degeneracy_share(eps, phi_min, phi_max):
    """Share of the prior where the exact oracle can answer from the aliasing constant alone.

    That is where 2 phi^2 / pi < eps. Reported next to the exact oracle so a reader can see at a
    glance whether its number means anything in a given scenario.
    """
    phi_star = np.sqrt(np.pi * eps / 2.0)
    lo, hi = float(phi_min), float(phi_max)
    return float(np.clip((min(phi_star, hi) - lo) / (hi - lo), 0.0, 1.0))
