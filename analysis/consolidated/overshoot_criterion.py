"""Why binary search's overshoot criterion is theoretically justified (Section 3.2.2).

The rule declares an overshoot when a probe at N reads below a threshold built from the deepest
accepted estimate:

    declare overshoot   <=>   phi_hat_N  <  phi_1,     phi_1 = phi_hat_acc + z_alpha /(2 N sqrt(m')).

THE CENTRAL POINT: this needs no Gaussian assumption at all, because it is *exactly* a one-sided
binomial test on the raw count. Since phi_hat = arccos(sqrt(K/m'))/N is strictly DECREASING in K,

    { phi_hat_N < phi_1 }   ==   { K > m' cos^2(N phi_1) }          (exact, every outcome)

so the rule fires precisely when the number of "0" outcomes exceeds a critical count
k* = m' cos^2(N phi_1). Its false-alarm probability is therefore an exact binomial tail under
K ~ Bin(m', cos^2(N phi)) -- Equation (2.41) of the thesis, which is exact, not an approximation.

What the normal law is used for is ONLY placing k*: z_alpha converts a nominal level into a
threshold. Misplacing k* changes the test's SIZE, not its VALIDITY -- the rule remains a legitimate
monotone test on K whichever critical value is used. So the criterion does not rest on Gaussianity;
Gaussianity is a computational shortcut for a quantity that has an exact binomial expression, and
this module measures how good that shortcut is.

A second, purely deterministic fact bounds the misses. arccos returns a value in [0, pi/2], so
every estimate obeys N phi_hat <= pi/2, while overshooting means N phi > pi/2. Hence at ANY
overshooting N and for EVERY outcome,

    phi_hat  <=  pi/(2N)  <  phi,

i.e. an overshooting probe always reads below the truth. Against a known phi the rule could never
miss; it misses only because phi_hat_acc is itself noisy, and at alpha = 0.5 that is the only
source of misses.

    python analysis/consolidated/overshoot_criterion.py

Writes overshoot_size.csv (the justification), overshoot_power.csv, overshoot_operating.csv and
overshoot_bracket_walk.csv.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import binom, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_io import path

# A round, illustrative shot count -- deliberately NOT one of the sweep's tuned values, which
# Chapter 3 has not introduced yet.
SHOTS = (50, 200, 800)
MAIN_SHOTS = 200

# The reference qualities binary search actually produces, from bracket_walk() below: the opening
# comparison comes from N_min while the probe sits near the middle of the range (rho ~ 5.7); by the
# time the bracket has closed onto N_opt the reference is a few percent shallower (rho ~ 1.05).
REFERENCES = ((None, "perfect"), (1.05, "late"), (5.73, "first"))
ALPHAS = (0.5, 0.05)
X_GRID = np.round(np.linspace(0.80, 1.40, 241), 6)


# --------------------------------------------------------------------------- exact binomial view
def critical_count(m, N, phi_1):
    """k* such that {phi_hat_N < phi_1} == {K > k*}. Exact, from monotonicity of arccos."""
    return m * np.cos(N * phi_1) ** 2


def verify_equivalence(trials=200_000, seed=3):
    """Check {phi_hat < phi_1} == {K > k*} on random (m, N, phi_1, K). Returns the mismatch count."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        m = int(rng.integers(2, 600))
        N = int(rng.integers(2, 200))
        phi_1 = rng.uniform(0.0, np.pi / (2 * N))
        k = int(rng.integers(0, m + 1))
        if (np.arccos(np.sqrt(k / m)) / N < phi_1) != (k > critical_count(m, N, phi_1)):
            bad += 1
    return bad


def exact_size(m, theta, alpha, N=80):
    """Exact false-alarm probability of the implemented test at a safe depth.

    The reference is set to the truth, which isolates the normal law's placement of the critical
    value from the separate error of plugging in a noisy phi_hat_acc. Returns None where the
    normal threshold falls outside the estimator's range [0, pi/(2N)], where the test is vacuous.
    """
    phi = theta / N
    phi_1 = norm.ppf(alpha, phi, 1.0 / (2 * N * np.sqrt(m)))
    if not (0.0 <= N * phi_1 <= np.pi / 2):
        return None
    return float(binom.sf(np.floor(critical_count(m, N, phi_1)), m, np.cos(theta) ** 2))


def size_table(shots=(50, 114, 200, 400, 800), confs=(0.5, 0.66, 0.95),
               thetas=np.linspace(0.3, 1.5, 121)):
    """Exact size of the test against its nominal level, over the admissible depths."""
    rows = []
    for conf in confs:
        alpha = 1.0 - conf                       # the implementation uses norm.ppf(1 - conf, ...)
        for m in shots:
            sizes = [s for s in (exact_size(m, float(t), alpha) for t in thetas) if s is not None]
            if not sizes:
                continue
            a = np.asarray(sizes)
            rows.append(dict(conf=conf, alpha_nominal=round(alpha, 4), m_exploration=m,
                             size_min=round(float(a.min()), 4),
                             size_median=round(float(np.median(a)), 4),
                             size_max=round(float(a.max()), 4),
                             max_abs_deviation=round(float(np.max(np.abs(a - alpha))), 4)))
    return rows


def probe_law(m, x):
    """Exact law of (phi_hat - phi)/sigma_N at depth x = N/N_opt, sorted increasing.

    sigma_N = 1/(2 N sqrt(m')), so the standardised law depends only on (m', x).
    """
    theta = x * np.pi / 2
    k = np.arange(m + 1)
    w = binom.pmf(k, m, np.cos(theta) ** 2)
    z = 2.0 * np.sqrt(m) * (np.arccos(np.sqrt(k / m)) - theta)
    order = np.argsort(z)
    return z[order], w[order]


def power(m, x, rho, alpha):
    """Exact P(declare overshoot) for a probe at depth x against a reference rho times shallower."""
    z_probe, w_probe = probe_law(m, x)
    if rho is None:                                   # phi_hat_acc = phi exactly
        return float(w_probe[z_probe < norm.ppf(alpha)].sum())
    z_ref, w_ref = probe_law(m, x / rho)
    cutoffs = rho * z_ref + norm.ppf(alpha)
    cum = np.cumsum(w_probe)
    idx = np.searchsorted(z_probe, cutoffs, side="left")
    return float(w_ref @ np.where(idx > 0, cum[np.clip(idx - 1, 0, None)], 0.0))


def curve(m, rho, alpha, grid=X_GRID):
    return np.array([power(m, float(x), rho, alpha) for x in grid])


def detect_from(grid, pw, target=0.9):
    """Smallest depth x > 1 at which the rule fires with probability `target`."""
    past = grid > 1.0
    g, p = grid[past], pw[past]
    hit = np.nonzero(p >= target)[0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(g[0])
    x0, x1, y0, y1 = g[i - 1], g[i], p[i - 1], p[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(g[i])


def missed_error(x):
    """Relative error left by an undetected overshoot at depth x = N/N_opt.

    phi_hat/phi = 2 N_opt/N - 1 = 2/x - 1, so the relative error is 2(1 - 1/x).
    """
    return 2.0 * (1.0 - 1.0 / x)


def bracket_walk(phi, n_min=15, n_max=157, steps=6):
    """The probes Algorithm 6 takes, and the reference quality rho = N/N_acc at each.

    Shows that the bracket narrows in step with the difficulty of the decision: the probes landing
    near N_opt are the late ones, by which time the reference is only a few percent shallower.
    """
    n_opt = int(np.pi // (2 * phi))
    lo, hi, n_acc = n_min, n_max, n_min
    n = n_min + (n_max - n_min) // 2
    out = []
    for step in range(steps):
        safe = n <= n_opt
        out.append(dict(phi=phi, N_opt=n_opt, step=step + 1, N=n, N_acc=n_acc,
                        rho=round(n / n_acc, 3), x=round(n * phi / (np.pi / 2), 3),
                        truth="safe" if safe else "overshoot"))
        if safe:
            n_acc, lo = n, n
            n = n + (hi - n) // 2
        else:
            hi = n
            n = n - (n - lo) // 2
        if hi - lo <= 1:
            break
    return out


def _write(name, rows):
    with open(path(name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(path(""), exist_ok=True)

    bad = verify_equivalence()
    print("EXACTNESS CHECK")
    print(f"  {{phi_hat < phi_1}} == {{K > k*}} : {bad} mismatches in 200,000 random cases")
    assert bad == 0, "the binomial restatement of the test does not hold"
    print("  -> the overshoot rule is exactly a one-sided binomial test on the count K.\n")

    sizes = size_table()
    _write("overshoot_size.csv", sizes)
    print("EXACT SIZE of the implemented test (reference set to the truth)")
    print("  {:>5} {:>8} {:>6} {:>22} {:>9}".format(
        "conf", "nominal", "shots", "exact size range", "max dev"))
    for r in sizes:
        print(f"  {r['conf']:>5.2f} {r['alpha_nominal']:>8.2f} {r['m_exploration']:>6} "
              f"{r['size_min']:>10.3f} - {r['size_max']:<9.3f} {r['max_abs_deviation']:>9.3f}")
    print()
    rows, oprows = [], []
    for m in SHOTS:
        for alpha in ALPHAS:
            for rho, tag in REFERENCES:
                pw = curve(m, rho, alpha)
                for x, p in zip(X_GRID, pw):
                    rows.append(dict(m_exploration=m, alpha=alpha, reference=tag,
                                     rho="" if rho is None else rho,
                                     x=x, power=round(float(p), 6)))
                x90 = detect_from(X_GRID, pw)
                err = missed_error(x90) if np.isfinite(x90) else float("nan")
                oprows.append(dict(
                    m_exploration=m, alpha=alpha, reference=tag,
                    rho="" if rho is None else rho,
                    false_alarm_at_0p9=round(float(pw[np.argmin(abs(X_GRID - 0.90))]), 4),
                    detect_from=round(x90, 4),
                    missed_error=round(err, 4),
                ))
    _write("overshoot_power.csv", rows)
    _write("overshoot_operating.csv", oprows)
    _write("overshoot_bracket_walk.csv",
           [r for phi in (0.02, 0.05) for r in bracket_walk(phi)])

    print("Correctness: at any overshooting N, phi_hat <= pi/(2N) < phi, for every outcome.")
    print("So with a perfect reference the rule cannot miss; misses come only from reference noise.")
    print(f"\noperating characteristic at m' = {MAIN_SHOTS}")
    print(f"  {'alpha':>6} {'reference':>10} {'false alarms':>13} {'detects from':>13} "
          f"{'if missed, error':>17}")
    for o in oprows:
        if o["m_exploration"] != MAIN_SHOTS:
            continue
        print(f"  {o['alpha']:>6.2f} {o['reference']:>10} "
              f"{100*o['false_alarm_at_0p9']:>12.0f}% {o['detect_from']:>13.3f} "
              f"{100*o['missed_error']:>16.1f}%")

    print("\nhow the bracket narrows (Algorithm 6, N in {15,...,157})")
    for phi in (0.02, 0.05):
        print(f"  phi={phi}:  " + "  ".join(
            f"N/N_opt={w['x']:.3f} (rho={w['rho']:.2f})" for w in bracket_walk(phi)))
    print("  -> the probes landing near N_opt are the late ones, where rho is already ~1")
    print(f"\nwrote overshoot_power.csv, overshoot_operating.csv, overshoot_bracket_walk.csv "
          f"to {path('')}")


if __name__ == "__main__":
    main()
