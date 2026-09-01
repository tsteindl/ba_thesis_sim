"""Oracle calculations for the fixed-budget phase-estimation comparison.

The thesis results use one simple reference: the oracle is handed
N_opt = floor(pi/(2 phi)), takes m = floor(B/N_opt) whole shots, and its convergence probability is
computed from the asymptotic normal law that saturates the QCRB. `oracle_rate` averages that law
exactly over the discrete N_opt distribution of a uniform prior, and `oracle_budget_for_rate` finds
the smallest integer budget reaching a requested convergence probability. Neither function draws
artificial estimator errors.

`convergence_prob` is the exact-binomial probability at one depth, used by the linear-search
detector study; `ceiling_rate` and `heisenberg_rate` are the looser analytic ceilings the broad-prior
study plots.
"""
import numpy as np
from scipy.special import ndtr, ndtri   # standard normal CDF and inverse CDF, vectorised
from scipy.stats import binom


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


def _depth_factor(budget, n_max):
    """g(n) = max over 1 <= N <= n of  N * sqrt(floor(budget / N)).

    This is the quantity the ceiling's accuracy depends on: with m = floor(budget/N) integer shots at
    depth N, Eq. (3.4) gives sd = 1/(2 N sqrt(m)), so P(converge) = 2 Phi(2 eps N sqrt(m)) - 1 and
    only the product N sqrt(m) matters.

    WHY THE MAXIMUM IS NEEDED. Ignoring the floor, N sqrt(m) = sqrt(N * budget) is increasing in N,
    so the deepest admissible depth N_opt is trivially best. With integer shots that stops being
    true: N_opt can waste a large fraction of the budget to the floor when m is small. At
    budget = 896, phi = 0.01 the aliasing bound is N_opt = 157, which affords m = 5 and spends only
    785; N = 149 affords m = 6, spends 894, and is strictly better (N sqrt(m) = 365 vs 351). A
    ceiling evaluated at N_opt alone would therefore not be a ceiling -- an implementable protocol
    could beat it. Taking the maximum over admissible depths restores the bound.

    Computed as a running maximum over one scan, so the whole table costs O(n_max).
    """
    N = np.arange(1, int(n_max) + 1, dtype=float)
    m = np.floor(float(budget) / N)
    return np.maximum.accumulate(np.where(m >= 1.0, N * np.sqrt(np.maximum(m, 0.0)), 0.0))


def ceiling_prob(phi, budget, eps, n_max=None):
    """P(converge) for the tightest admissible depth, with INTEGER shots.

    The reported ceiling. Same spirit as `heisenberg_prob` -- perfect knowledge of the depth, but the
    accuracy must come from the sampling distribution rather than from the deterministic aliasing
    readout -- and additionally honest about the budget actually spendable: a protocol takes
    m = floor(budget/N) whole shots, so N*m <= budget, and `heisenberg_prob`'s 1/(4 N budget)
    silently assumes fractional shots.

    The difference is at most ~0.3 pp, and only at the lowest budgets where m is a handful of shots;
    `heisenberg_prob` is a valid but slightly loose (optimistic) bound everywhere.
    """
    phi = np.asarray(phi, dtype=float)
    nm = int(n_max if n_max is not None else np.nanmax(
        np.maximum(np.floor(np.pi / (2.0 * phi)), 1.0)))
    nm = max(min(nm, int(budget)), 1)
    g = _depth_factor(budget, nm)
    n_opt = np.clip(np.maximum(np.floor(np.pi / (2.0 * phi)), 1.0).astype(np.int64), 1, nm)
    return np.clip(2.0 * ndtr(2.0 * eps * g[n_opt - 1]) - 1.0, 0.0, 1.0)




def oracle_shots_for_rate(target, eps, N):
    """Smallest whole-shot count attaining ``target`` under the asymptotic QCRB law.

    With a locally unbiased, asymptotically efficient estimator,

        phi_hat - phi ~ Normal(0, 1 / (4 m N^2)),

    so P(|phi_hat - phi| < eps) = 2 Phi(2 eps N sqrt(m)) - 1.  Inverting this
    expression gives the required shot count directly; no random errors are drawn.
    """
    target, eps, N = float(target), float(eps), int(N)
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    if eps <= 0.0 or N < 1:
        raise ValueError("eps and N must be positive")
    z = float(ndtri((1.0 + target) / 2.0))
    return max(int(np.ceil((z / (2.0 * eps * N)) ** 2)), 1)


def _uniform_nopt_distribution(phi_min, phi_max):
    """Possible N_opt values and their exact probabilities under a uniform phase prior.

    floor(pi/(2 phi)) = n on (pi/(2(n+1)), pi/(2n)].  Intersecting those intervals
    with the prior support turns the prior average into a short finite sum instead of a
    Monte-Carlo estimate or a dense quadrature grid.
    """
    phi_min, phi_max = float(phi_min), float(phi_max)
    if not 0.0 < phi_min < phi_max:
        raise ValueError("require 0 < phi_min < phi_max")
    n_lo = max(int(np.floor(np.pi / (2.0 * phi_max))), 1)
    n_hi = max(int(np.floor(np.pi / (2.0 * phi_min))), 1)
    n = np.arange(n_lo, n_hi + 1, dtype=np.int64)
    lo = np.maximum(phi_min, np.pi / (2.0 * (n + 1.0)))
    hi = np.minimum(phi_max, np.pi / (2.0 * n))
    weights = np.maximum(hi - lo, 0.0) / (phi_max - phi_min)
    live = weights > 0.0
    n, weights = n[live], weights[live]
    # The intervals partition the support; normalising only removes floating-point roundoff.
    weights = weights / weights.sum()
    return n, weights


def oracle_rate(budget, eps, phi_min, phi_max):
    """Prior-averaged convergence rate of the QCRB oracle, without simulation.

    The oracle is handed N_opt(phi).  At hard budget B it takes the maximum affordable
    number of whole shots, m = floor(B/N_opt), and its conditional convergence probability
    follows from the asymptotic normal law that saturates the QCRB.  For a uniform prior the
    average is an exact finite sum over the possible integer values of N_opt.

    "Exact" here means exact under that asymptotic model; it is not an exact finite-shot
    binomial calculation.
    """
    budget, eps = int(budget), float(eps)
    if budget < 1 or eps <= 0.0:
        raise ValueError("budget and eps must be positive")
    n_opt, weights = _uniform_nopt_distribution(phi_min, phi_max)
    m = budget // n_opt
    prob = 2.0 * ndtr(2.0 * eps * n_opt * np.sqrt(m)) - 1.0
    return float(np.dot(weights, np.clip(prob, 0.0, 1.0)))


def oracle_abs_error_quantile(budget, quantile, phi_min, phi_max):
    """Quantile of ``|phi_hat-phi|`` for the analytic QCRB Oracle.

    Conditional on N_opt, the absolute error is half-normal with
    sigma = 1/(2 N_opt sqrt(m)).  The uniform phase prior therefore gives an exact finite mixture
    over the possible N_opt values.  Its scalar quantile is found deterministically; no estimator
    errors or phase values are sampled.
    """
    budget, quantile = int(budget), float(quantile)
    if budget < 1 or not 0.0 < quantile < 1.0:
        raise ValueError("budget must be positive and quantile must lie between zero and one")
    n_opt, weights = _uniform_nopt_distribution(phi_min, phi_max)
    m = budget // n_opt
    live = m > 0
    if float(weights[live].sum()) < quantile:
        return float("inf")
    sigma = 1.0 / (2.0 * n_opt[live] * np.sqrt(m[live]))
    weights = weights[live]

    def cdf(x):
        return float(np.dot(weights, 2.0 * ndtr(x / sigma) - 1.0))

    lo, hi = 0.0, float(np.max(sigma))
    while cdf(hi) < quantile:
        hi *= 2.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if cdf(mid) >= quantile:
            hi = mid
        else:
            lo = mid
    return hi


def oracle_error_variance(budget, phi_min, phi_max):
    """Prior-averaged QCRB variance at known N_opt and whole-shot count floor(B/N_opt)."""
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    n_opt, weights = _uniform_nopt_distribution(phi_min, phi_max)
    m = budget // n_opt
    if np.any(m == 0):
        return float("inf")
    return float(np.dot(weights, 1.0 / (4.0 * n_opt ** 2 * m)))


def oracle_budget_for_rate(target, eps, phi_min, phi_max):
    """Smallest integer hard budget whose prior-averaged oracle rate reaches ``target``.

    Integer shots make the rate a monotone step function, so there is no single closed-form
    inverse after averaging over the different N_opt values.  A doubling search followed by
    integer bisection finds the exact first step and normally needs only a few dozen evaluations
    of the finite sum in `oracle_rate`.
    """
    target = float(target)
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    lo, hi = 1, 1
    while oracle_rate(hi, eps, phi_min, phi_max) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if oracle_rate(mid, eps, phi_min, phi_max) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo




def ceiling_rate(budget, eps, phi_min, phi_max, n_grid=20001):
    """Ensemble convergence rate of the ceiling under phi ~ U(phi_min, phi_max).

    Averaged over the prior by QUADRATURE, not by sampling. The averaging is required whatever
    formula is used for P(converge | phi), because that probability depends on phi through
    N_opt = floor(pi/(2 phi)); the only alternative is Monte-Carlo integration, which computes the
    same integral with added sampling error. Being deterministic, this carries no error bar --
    the trapezoid discretisation is ~1e-6 relative at the default grid, a thousand times below the
    Monte-Carlo error of any simulated row.
    """
    n_max = max(int(np.pi // (2.0 * phi_min)), 1)
    phi = np.linspace(phi_min, phi_max, n_grid)
    return float(np.trapezoid(ceiling_prob(phi, budget, eps, n_max=n_max), phi)
                 / (phi_max - phi_min))


