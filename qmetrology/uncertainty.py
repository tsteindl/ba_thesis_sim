"""Error bars for the reported numbers.

Every headline figure in the thesis is one of two kinds, and each carries a different uncertainty:

  * a **convergence rate** — a binomial proportion over R independent trials. Its Monte-Carlo error is
    exact and needs nothing but the rate and R: a Wilson score interval. At R = 40,000 the half-width
    is at most 0.49 percentage points (worst case p = 0.5).

  * a **budget to reach p\*** (and the ratio of two of them) — not measured but *derived*, by
    log-interpolating the convergence-vs-budget curve to the point where it crosses p\*. Two things
    make it uncertain: the Monte-Carlo error of the rates the curve is built from, and the finite
    spacing of the budget grid the curve is interpolated over. Both are quantified here from the data
    already in results/performance_curves.csv — no re-simulation.

`crossing_ci` propagates the first by parametric bootstrap: resample every curve point from
Binomial(R, p_hat)/R, re-interpolate the crossing, and take percentiles. `grid_sensitivity` bounds
the second by re-deriving the crossing from the two half-density subgrids; log-linear interpolation
error is O(h^2) in the grid log-spacing, so the full-grid error is ~1/4 of the deviation reported
there.

One caveat is worth stating in the thesis rather than hiding: all algorithms are evaluated on the
same seed list, so a given trial index draws the *same* phi for every algorithm and every budget.
Ratios are therefore paired, and a bootstrap that resamples the two curves independently — as this
one does — **overstates** the ratio's uncertainty. The intervals are conservative, not optimistic.

What is *not* covered: the sampling variance of the grid search itself (which configuration wins is
random, tuned at R_TUNE = 2,000). De-biasing removes its bias — the winner is re-validated on an
independent seed — but not its variance. Quantifying that needs repeated tuning seeds; see
analysis/tuning_stability.py, which does it for the headline cells only.
"""
import numpy as np

Z95 = 1.959963984540054


def wilson(p, n, z=Z95):
    """Wilson score interval for a binomial proportion (returns fractions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def crossing(budgets, rates, T):
    """Log-interpolated budget at which the convergence curve first reaches rate T (nan if never).

    Single definition, shared by the sweep (analysis/run.py) and the error bars here, so
    the interval is always built around exactly the number that is reported.
    """
    b, r = np.asarray(budgets, float), np.asarray(rates, float)
    if len(b) == 0 or T <= r[0] or T > r[-1]:
        return float("nan")
    for i in range(1, len(r)):
        if r[i] >= T:
            if r[i] == r[i - 1]:
                return float(b[i])
            f = (T - r[i - 1]) / (r[i] - r[i - 1])
            return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1]))))
    return float("nan")


def _resample(rates, R, rng, n_boot):
    """n_boot parametric-bootstrap replicates of a convergence curve."""
    p = np.clip(np.asarray(rates, float), 0.0, 1.0)
    return rng.binomial(R, p, size=(n_boot, len(p))) / R


def crossing_ci(budgets, rates, T, R, n_boot=2000, seed=0, ci=95):
    """(point estimate, lo, hi) for the budget at which `rates` reaches T.

    Monte-Carlo uncertainty only: each curve point is resampled from Binomial(R, p_hat)/R and the
    crossing re-derived. Replicates whose curve never reaches T are dropped and reported via the
    returned coverage, which stays at 1.0 unless the crossing sits at the very edge of the swept
    budget range.
    """
    point = crossing(budgets, rates, T)
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan"), 0.0
    draws = _resample(rates, R, np.random.default_rng(seed), n_boot)
    xs = np.array([crossing(budgets, d, T) for d in draws])
    ok = np.isfinite(xs)
    if ok.sum() < 0.5 * n_boot:
        return point, float("nan"), float("nan"), float(ok.mean())
    a = (100 - ci) / 2
    lo, hi = np.percentile(xs[ok], [a, 100 - a])
    return point, float(lo), float(hi), float(ok.mean())


def ratio_ci(budgets, rates_ref, rates_alg, T, R, n_boot=2000, seed=0, ci=95, budgets_alg=None):
    """(point, lo, hi) for the budget ratio B_ref(T) / B_alg(T).

    `budgets_alg` defaults to `budgets`; pass it when the two curves were evaluated on different
    abscissae (the omniscient ceilings get extra low-budget points, since their crossing can fall
    below the grid the sweep chose around brute force).

    The two curves are resampled independently, which ignores the seed pairing between algorithms;
    the result is an unpaired Monte-Carlo interval with no general conservative-coverage guarantee.
    """
    b_a = budgets if budgets_alg is None else budgets_alg
    b_ref, b_alg = crossing(budgets, rates_ref, T), crossing(b_a, rates_alg, T)
    if not (np.isfinite(b_ref) and np.isfinite(b_alg)):
        return float("nan"), float("nan"), float("nan"), 0.0
    rng = np.random.default_rng(seed)
    d_ref = _resample(rates_ref, R, rng, n_boot)
    d_alg = _resample(rates_alg, R, rng, n_boot)
    xs = np.array([crossing(budgets, x, T) / crossing(b_a, y, T)
                   for x, y in zip(d_ref, d_alg)])
    ok = np.isfinite(xs)
    if ok.sum() < 0.5 * n_boot:
        return b_ref / b_alg, float("nan"), float("nan"), float(ok.mean())
    a = (100 - ci) / 2
    lo, hi = np.percentile(xs[ok], [a, 100 - a])
    return b_ref / b_alg, float(lo), float(hi), float(ok.mean())


def grid_sensitivity(budgets, rates, T):
    """Largest relative deviation of the crossing when re-derived from the half-density subgrids.

    A direct, assumption-free probe of how much the budget grid's finite spacing (not the Monte-Carlo
    noise) moves the reported number. Log-linear interpolation error scales as the square of the grid
    log-spacing, so the full grid carries roughly a quarter of the deviation returned here.
    """
    full = crossing(budgets, rates, T)
    if not np.isfinite(full) or len(budgets) < 6:
        return float("nan")
    devs = []
    for k in (0, 1):
        sub = crossing(np.asarray(budgets)[k::2], np.asarray(rates)[k::2], T)
        if np.isfinite(sub):
            devs.append(abs(sub / full - 1.0))
    return max(devs) if devs else float("nan")
