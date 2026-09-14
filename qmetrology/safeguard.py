"""Statistically derived exploitation depth — replaces the grid-tuned safeguard constants.

CORE ASSUMPTIONS
----------------
The rule below is only as good as these. Full derivation and the numbers behind each verdict:
the thesis (Section 3.2.2).

  A1  The estimator is asymptotically normal with Var(phi_hat) = 1/(4 N^2 m), *independent of phi*.
      EXACT to leading order -- the p0(1-p0) factors cancel, the Fisher information is a constant
      4N^2. Everything else here depends on this constancy. Degrades when m*p0 = m*cos^2(N phi) is
      small (the hits=0 atom censors the estimator); trustworthy while m*p0 >~ 10.

  A2  The pilot likelihood is unimodal over the prior support, i.e. N0*phi <= pi/2 for all admissible
      phi, so arccos inverts without an aliasing ambiguity.
      EXACT BY CONSTRUCTION for reverse engineering, whose pilot is taken at N0 = N_min =
      floor(pi/(2 phi_max)) -- that is precisely why N_min is defined that way. WEAKER for binary
      search, whose pilot is the deepest non-overshooting probe: there it is the bisection's own
      overshoot test, not construction, that rules out the aliased branch.

  A3  phi has a uniform prior on [phi_min, phi_max].
      LITERALLY TRUE, not a modelling convenience: experiments.py draws phi that way in every trial.
      This is what turns "phi ~ N(phi_hat0, sigma^2)" from a Bayesian belief into the true
      conditional distribution of phi given the pilot, over the ensemble the thesis reports on.
      (phi is of course fixed *within* a run -- see A3' below.)

  A3' The quantity being optimised is the ensemble-average convergence rate, not a guarantee for one
      fixed phi. That average IS the reported metric, so the objective is an unbiased estimate of it.
      A per-phi worst-case guarantee would need a minimax rule instead, and would be more
      conservative.

  A4  The posterior's truncation at phi_min / phi_max is dropped.
      CONSERVATIVE at the upper end (forgetting phi <= phi_max only over-states overshoot risk); the
      lower end is enforced exactly instead, by capping N at N_max = floor(pi/(2 phi_min)).

  A5  Overshooting implies the trial never converges.
      SLIGHTLY CONSERVATIVE: a marginal overshoot still converges while phi - pi/(2N) < eps, a band
      of width ~ pi*eps/(2 phi^2) in N. Negligible at tight eps; up to ~11% of N at phi=0.01, eps=1e-3.

  A6  In the convergence factor N is treated as fixed, though it is chosen from the pilot.
      APPROXIMATION: the exploitation shots are fresh, so the estimate is not biased, but its
      marginal law is a mixture over N rather than the single Gaussian assumed.

  A7  The chosen operating point stays inside A1's validity region.
      NOT ENFORCED, and checked empirically instead: it holds at moderate/high budget (<2% of trials
      outside), but ~24% of trials at budget 1e4 land where m*p0 < 10. The rule still wins there
      (+5.0 pp), so the failure is benign -- those trials would mostly have failed at any depth --
      but the low-budget result is empirically, not theoretically, justified.

Both the reverse-engineering safety factor `C_safe` (Eq. 3.6) and the binary-search decrement `s`
exist for one reason: the depth inferred from a pilot estimate may overshoot N_opt = floor(pi/2phi),
which invalidates the estimator. Their values were previously chosen by parameter grid search.

They can instead be derived from the asymptotic sampling distribution that the binary-search
overshoot criterion already relies on (Eq. 3.4),

    phi_hat ~ N(phi, 1/(4 N^2 m))     =>     sd(phi_hat) = 1 / (2 N sqrt(m)),

which is exact for this estimator: with p0 = cos^2(N phi), error propagation on
phi_hat = arccos(sqrt(hits/m))/N gives Var = [p0(1-p0)/m] / (dp0/dphi)^2 = 1/(4 N^2 m),
independent of phi (it saturates the Fisher information).

Reading Eq. (3.4) as a likelihood for the unknown phi given the pilot (phi_hat0, N0, m'), a candidate
exploitation depth N carries two quantifiable risks:

    P(no overshoot)  = P(phi < pi/(2N))      = Phi( (pi/(2N) - phi_hat0) / sigma )
    P(converge | N)  = P(|phi_hat - phi| < eps) with the *same* law at (N, m = floor(B/N)),
                       i.e.  2 Phi( 2 eps N sqrt(floor(B/N)) ) - 1.

The exploitation depth is the one that maximises their product,

    N* = argmax_{N_min <= N <= N_max}  P(phi < pi/(2N) | pilot) * ( 2 Phi(2 eps N sqrt(floor(B/N))) - 1 ),

which needs no tuned constant at all. Both factors move in opposite directions: a deeper circuit is
more precise (second factor grows roughly as sqrt(N)) but more likely to alias (first factor falls).

TWO THINGS ARE EXACT HERE, AND ONE IS NOT.

  * The MAXIMISATION is exact. Every integer of {N_min, ..., N_max} is scored by one vectorised
    evaluation and the argmax is taken directly. There is no coarse-to-fine bracketing, no local
    interval, no 5-sigma cutoff and no hard cap, so the returned depth cannot depend on the shape of
    a search schedule or on an unproven unimodality of the score.

  * The SHOT COUNT is exact. The accuracy factor uses m = floor(B/N), the whole number of shots the
    depth can actually be paid for, rather than the smooth substitution m ~ B/N (which gives
    2 Phi(2 eps sqrt(N B)) - 1). The two agree to O(N/B); they differ visibly exactly where the
    depth is a material fraction of the budget, which is where the rule is deciding.

  * "Exact" does NOT modify the statistical model. A1-A7 above are unchanged, and the score is still
    a Gaussian approximation to a discrete problem. What is now exact is the numerical maximisation
    of that approximate score over the admissible set.

THE ADMISSIBLE SET IS [N_min, N_max], AND BOTH ENDS ARE DELIBERATE.

    N_min = floor(pi / (2 phi_max))   the deepest circuit that stays on the first identifiable
                                      branch for the WHOLE prior support (N_min * phi <= pi/2 for
                                      every admissible phi, with equality possible only at an
                                      endpoint). It is therefore safe by construction, and it is the
                                      depth the opening pilot is already taken at. Running the
                                      exploitation shallower than the guaranteed-safe baseline is
                                      not a trade-off the safeguard should be able to make, so
                                      N < N_min is excluded rather than merely disfavoured.
    N_max = floor(pi / (2 phi_min))   beyond it the truncated posterior puts zero mass on "no
                                      overshoot", so the score is identically zero.

Because m is now a floor, the score is not monotone below N_min and no proof is offered that some
smaller integer could never carry a marginally larger numerical score; the claim is that N* is the
exact maximiser over the CONSTRAINED range. On the 177 active scenario/budget points of the thesis
matrix a direct integer audit found N_min's accuracy factor to be at least as large as that of every
N < N_min anyway, so nothing is excluded that would have won.

The implied multiplicative safeguard C_eff = N* / floor(pi/(2 phi_hat0)) is therefore *adaptive*:
it tightens when the pilot is relatively imprecise (small phi, small N0 sqrt(m')) and relaxes toward
1 as the budget grows. A constant C cannot track either axis, which is what the grid search was
compensating for.
"""
from functools import lru_cache

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised and faster than scipy.stats.norm.cdf


def pilot_sd(N, m):
    """Asymptotic sd of a pilot estimate taken at depth N with m shots — sqrt of Eq. (3.4)."""
    return 1.0 / (2.0 * N * np.sqrt(m))


def _p_safe(N, phi_hat, sigma, support=None):
    """P(phi < pi/(2N) | pilot). `support` = (phi_min, phi_max) uses the exact TRUNCATED posterior;
    None uses the untruncated normal (assumption A4)."""
    t = np.pi / (2.0 * np.asarray(N, dtype=float))
    if support is None:
        return ndtr((t - phi_hat) / sigma)
    lo, hi = support
    Fa, Fb = ndtr((lo - phi_hat) / sigma), ndtr((hi - phi_hat) / sigma)
    den = Fb - Fa
    if not np.isfinite(den) or den < 1e-12:   # posterior has collapsed onto the nearer boundary
        return (t > min(max(phi_hat, lo), hi)).astype(float)
    return np.clip((ndtr((t - phi_hat) / sigma) - Fa) / den, 0.0, 1.0)


def _score(N, phi_hat, sigma, budget, eps, support=None):
    """P(no overshoot) * P(|phi_hat - phi| < eps) for candidate depths N (array or scalar).

    The accuracy factor is evaluated at the WHOLE number of exploitation shots the depth can be paid
    for, m = floor(B/N) -- what the algorithm actually spends -- not at the smooth m = B/N. With
    N^2 m in Eq. (3.4) that gives 2 Phi(2 eps N sqrt(m)) - 1, and 0 where the depth is unaffordable
    (m = 0), so an N the trial could not buy a single shot at can never win the argmax.
    """
    Ni = np.asarray(N, dtype=np.int64)
    m = int(budget) // np.maximum(Ni, 1)
    p_acc = np.where(m >= 1, 2.0 * ndtr(2.0 * eps * Ni.astype(float) * np.sqrt(m)) - 1.0, 0.0)
    return _p_safe(Ni.astype(float), phi_hat, sigma, support) * p_acc


@lru_cache(maxsize=64)
def _candidates(N_min, N_max):
    """The closed integer interval [N_min, N_max], cached and read-only.

    Pure allocation avoidance: the tuning sweep makes O(10^8) safeguard calls and the active
    scenarios use only three distinct (N_min, N_max) pairs, so the same array is rebuilt millions of
    times otherwise. It is frozen because callers share it; `_score` only reads.
    """
    c = np.arange(N_min, N_max + 1, dtype=np.int64)
    c.flags.writeable = False
    return c


def risk_optimal_depth(phi_hat, sigma, budget, eps, *, N_min, N_max, support=None):
    """Exploitation depth maximising P(no overshoot) x P(converge), by exhaustive enumeration.

    Every integer of the closed interval [N_min, N_max] is scored in one vectorised pass and the
    argmax is returned, so the result is the exact maximiser of the (approximate, Gaussian) score
    over the admissible set -- see the module docstring for what "exact" does and does not claim.
    `np.argmax` breaks an exact tie toward the SHALLOWEST depth, which is the conservative side.

    phi_hat, sigma : pilot estimate and its asymptotic sd (use `pilot_sd`).
    budget         : budget remaining for the exploitation shot (m = floor(budget / N)).
    eps            : target tolerance |phi_hat - phi| < eps.
    N_min, N_max   : REQUIRED, keyword-only. The admissible depth range, normally
                     floor(pi/(2 phi_max)) .. floor(pi/(2 phi_min)) from the prior support (binary
                     search may lower N_max to the depth the bisection settled on). They are
                     keyword-only and mandatory so that no call site can silently fall back to a
                     lower bound of 1 and pick a depth shallower than the guaranteed-safe baseline.
    support        : optional (phi_min, phi_max). Given, the overshoot factor uses the exact
                     truncated-normal posterior instead of the untruncated one. Off by default:
                     analysis/truncation_check.py measures the difference as <= 0.14 pp of
                     convergence (SE 0.25 pp) at every operating point the thesis reports, because
                     the prior spans pi*sqrt(m')*(1 - phi_min/phi_max) >= 14 sigma there.

    Returns N_min on invalid input (non-finite pilot, non-positive sigma/budget/eps) -- the
    guaranteed-safe baseline, never 1. If the remaining budget cannot buy a single shot even at
    N_min the score is zero everywhere and N_min is returned as well; the CALLER is responsible for
    taking its no-exploitation path rather than spending a shot it cannot afford.
    """
    lo = max(int(N_min), 1)
    hi = max(int(N_max), lo)
    if not np.isfinite(phi_hat) or sigma <= 0 or budget <= 0 or eps <= 0:
        return lo
    cand = _candidates(lo, hi)
    return int(cand[int(np.argmax(_score(cand, phi_hat, sigma, budget, eps, support)))])


def effective_C(phi_hat, sigma, budget, eps, *, N_min, N_max):
    """Diagnostic: the multiplicative safeguard N*/floor(pi/(2 phi_hat)) the rule implies."""
    naive = max(int(np.pi // (2 * phi_hat)), 1)
    return risk_optimal_depth(phi_hat, sigma, budget, eps, N_min=N_min, N_max=N_max) / naive
