"""Statistically derived exploitation depth — replaces the grid-tuned safeguard constants.

CORE ASSUMPTIONS
----------------
The rule below is only as good as these. Full derivation and the numbers behind each verdict:
results/SAFEGUARD_DERIVATION.md.

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
    P(converge | N)  = P(|phi_hat - phi| < eps) with the *same* law at (N, m = B/N),
                       and since N^2 m = N B this is  2 Phi( 2 eps sqrt(N B) ) - 1.

The exploitation depth is the one that maximises their product,

    N* = argmax_{1 <= N <= N_max}  Phi( (pi/(2N) - phi_hat0)/sigma ) * ( 2 Phi( 2 eps sqrt(N B) ) - 1 ),

which needs no tuned constant at all. Both factors move in opposite directions: a deeper circuit is
more precise (second factor increases as sqrt(N)) but more likely to alias (first factor falls).

The implied multiplicative safeguard C_eff = N* / floor(pi/(2 phi_hat0)) is therefore *adaptive*:
it tightens when the pilot is relatively imprecise (small phi, small N0 sqrt(m')) and relaxes toward
1 as the budget grows. A constant C cannot track either axis, which is what the grid search was
compensating for.
"""
import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised and faster than scipy.stats.norm.cdf

_N_SEARCH = 128   # candidates per refinement round
_N_ROUNDS = 3     # geometric refinement rounds before the final exact integer scan
_HARD_CAP = 1 << 45  # only reached when the pilot carries no usable information at all


def pilot_sd(N, m):
    """Asymptotic sd of a pilot estimate taken at depth N with m shots — sqrt of Eq. (3.4)."""
    return 1.0 / (2.0 * N * np.sqrt(m))


def _p_safe(N, phi_hat, sigma, support=None):
    """P(phi < pi/(2N) | pilot). `support` = (phi_min, phi_max) uses the exact TRUNCATED posterior;
    None uses the untruncated normal (assumption A4). See results/SAFEGUARD_DERIVATION.md §3.7."""
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
    """P(no overshoot) * P(|phi_hat - phi| < eps) for candidate depths N (array or scalar)."""
    Nf = np.asarray(N, dtype=float)
    p_conv = 2.0 * ndtr(2.0 * eps * np.sqrt(Nf * float(budget))) - 1.0
    return _p_safe(Nf, phi_hat, sigma, support) * p_conv


def risk_optimal_depth(phi_hat, sigma, budget, eps, N_max=None, support=None):
    """Exploitation depth maximising P(no overshoot) x P(converge).

    phi_hat, sigma : pilot estimate and its asymptotic sd (use `pilot_sd`).
    budget         : budget remaining for the exploitation shot (m = budget / N).
    eps            : target tolerance |phi_hat - phi| < eps.
    N_max          : hard cap, normally floor(pi/(2 phi_min)) from the prior support (and, for
                     binary search, the depth the bisection settled on — the safeguard only reduces N).
    support        : optional (phi_min, phi_max). Given, the overshoot factor uses the exact
                     truncated-normal posterior instead of the untruncated one. Off by default:
                     analysis/truncation_check.py measures the difference as <= 0.14 pp of
                     convergence (SE 0.25 pp) at every operating point the thesis reports, because
                     the prior spans pi*sqrt(m')*(1 - phi_min/phi_max) >= 14 sigma there.
    """
    if not np.isfinite(phi_hat) or sigma <= 0 or budget <= 0 or eps <= 0:
        return 1

    # Above pi/(2N) < phi_hat - 5 sigma the overshoot term is numerically dead; the prior support
    # (N_max) bounds the search when the pilot is too noisy to bound it on its own.
    phi_lo = phi_hat - 5.0 * sigma
    if support is not None:
        # the truncated posterior has no mass outside [phi_min, phi_max], so the 5-sigma bound is
        # only meaningful once clamped to the support -- without this a pilot that lands outside
        # the prior range would cut the search short at a depth the posterior does not rule out.
        phi_lo = min(max(phi_lo, support[0]), support[1])
    hi = _HARD_CAP if phi_lo <= 0 else int(np.pi / (2.0 * phi_lo)) + 1
    if N_max is not None:
        hi = min(hi, int(N_max))
    hi, lo = max(int(hi), 1), 1

    for _ in range(_N_ROUNDS):
        if hi - lo <= 1:
            break
        cand = np.unique(np.geomspace(lo, hi, _N_SEARCH).astype(np.int64))
        i = int(np.argmax(_score(cand, phi_hat, sigma, budget, eps, support)))
        lo, hi = int(cand[max(i - 1, 0)]), int(cand[min(i + 1, len(cand) - 1)])

    cand = np.arange(lo, hi + 1, dtype=np.int64)
    return max(int(cand[int(np.argmax(_score(cand, phi_hat, sigma, budget, eps, support)))]), 1)


def effective_C(phi_hat, sigma, budget, eps, N_max=None):
    """Diagnostic: the multiplicative safeguard N*/floor(pi/(2 phi_hat)) the rule implies."""
    naive = max(int(np.pi // (2 * phi_hat)), 1)
    return risk_optimal_depth(phi_hat, sigma, budget, eps, N_max) / naive
