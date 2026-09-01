"""How much does the TRUNCATED posterior matter? (the retired assumption A4)

The posterior for phi given the pilot is a normal truncated to the prior support [phi_min, phi_max].
qmetrology/safeguard.py now uses that exact form; this script measures what the untruncated shortcut
would have cost, in three steps:

  1. the analytic condition -- the prior spans W = pi sqrt(m') (1 - phi_min/phi_max) standard
     deviations of the pilot, so truncation can only distort the tails when W is small;
  2. the worst-case shift in the chosen depth N*, scanned over phi_hat0, budget and m';
  3. the end-to-end convergence rate both ways, de-biased exactly like every other number in the
     thesis (tune on seed 42, validate on seed 2024).

    python analysis/truncation_check.py [--quick]

Prints a report; the conclusions are written up in results/SAFEGUARD_DERIVATION.md (SS 3.6-3.7).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

QUICK = "--quick" in sys.argv
R_TUNE, R_TEST = (500, 5000) if QUICK else (2000, 40_000)

# operating points that carry the reported results
SCEN = [(0.01, 0.1, 1e-3, 10_000), (0.01, 0.1, 1e-3, 45_000), (0.001, 0.01, 1e-4, 400_000),
        (0.01, 0.1, 1e-4, 3_000_000), (0.01, np.pi / 2, 1e-3, 887_240)]
if QUICK:
    SCEN = SCEN[:2]


def _re(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, truncated):
    """Reverse engineering, with the truncation of the overshoot factor switched on or off.

    Local copy rather than the library function, so both arms stay comparable no matter which
    form qmetrology/algorithms.py currently defaults to.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat, used = 0, 0
    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        used += m_exploration * N_min
    rem = budget - used
    if rem <= 0:
        return phi_hat, used
    N = risk_optimal_depth(phi_hat, pilot_sd(N_min, m_exploration), rem, eps_target,
                           N_min=max(int(N_min), 1), N_max=max(int(N_max), int(N_min), 1),
                           support=(phi_min, phi_max) if truncated else None)
    m = int(rem / N)
    return simulate_errors(rng, phi, m, N), used + N * m


def RE_TRUNC(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target):
    return _re(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, True)


def RE_PLAIN(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target):
    return _re(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, False)


def debias(fn, grid, pmin, pmax, eps, B):
    _m, arg, _ = E.grid_search_max(fn, {**grid, "budget": [int(B)]}, R_TUNE, pmin, pmax, eps, 42)
    return 100 * E.success_rate(fn, arg, R_TEST, pmin, pmax, eps, 2024)


def worst_shift(pmin, pmax, eps, B):
    """Largest |dN*|/N* over the pilot values and exploration sizes the tuner can reach."""
    N0 = max(int(np.pi // (2 * pmax)), 1)
    N_max = max(int(np.pi / (2 * pmin)), 1)
    worst = 0.0
    for mp in [20, 70, 200, 1000, 16074]:
        sd = pilot_sd(N0, mp)
        for q in np.linspace(0, 4, 17):          # sweep phi_hat0 through the boundary layer
            ph = pmin + q * sd
            if ph > pmax:
                continue
            a = risk_optimal_depth(ph, sd, B, eps, N_min=N0, N_max=N_max)
            b = risk_optimal_depth(ph, sd, B, eps, N_min=N0, N_max=N_max, support=(pmin, pmax))
            worst = max(worst, abs(b - a) / a)
    return worst


def main():
    print("1. When can truncation matter?   W = pi sqrt(m') (1 - phi_min/phi_max)\n")
    print(f"   {'prior':>22} {'m_pilot':>8} {'W (sigma)':>10} {'% near a boundary':>18}")
    for pmin, pmax in [(0.01, 0.1), (0.001, 0.01), (0.001, 0.1), (0.01, np.pi / 2), (0.0001, 0.4)]:
        for mp in (70, 1000):
            N0 = max(int(np.pi // (2 * pmax)), 1)
            W = (pmax - pmin) / pilot_sd(N0, mp)
            tag = "U(%g, %.4g)" % (pmin, pmax)
            print(f"   {tag:>22} {mp:>8} {W:>10.1f} {min(100 * 4 / W, 100):>17.1f}%")
    print("\n   Every prior in the thesis has phi_min/phi_max <= 0.1 and every selected m' >= 70,")
    print("   so W >= 14. For binary search the pilot sits deeper: sigma smaller, W larger.\n")

    print("2. Worst-case shift in the chosen depth\n")
    print(f"   {'operating point':>44} {'worst |dN*|/N*':>15} {'-> in sigma_final':>18}")
    for pmin, pmax, eps, B in SCEN:
        w = worst_shift(pmin, pmax, eps, B)
        tag = "U(%g, %.4g), eps=%.0e, B=%s" % (pmin, pmax, eps, format(B, ","))
        print(f"   {tag:>44} {100 * w:>14.1f}% {100 * (1 / np.sqrt(1 - w) - 1):>17.1f}%")

    print(f"\n3. End-to-end convergence, de-biased (tune 42 / validate 2024, R={R_TEST:,})")
    print(f"   standard error ~{100 * np.sqrt(0.25 / R_TEST):.2f} pp\n")
    deltas = []
    for pmin, pmax, eps, B in SCEN:
        hi = int(np.clip(0.6724 / (max(1, int(np.pi / (2 * pmax))) * eps ** 2), 200, 300_000))
        grid = {"m_exploration": np.unique(np.geomspace(20, hi, 20).astype(int)), "eps_target": [eps]}
        a = debias(RE_PLAIN, grid, pmin, pmax, eps, B)
        b = debias(RE_TRUNC, grid, pmin, pmax, eps, B)
        deltas.append(b - a)
        print(f"   U({pmin:g},{pmax:.3g}) eps={eps:.0e} B={B:>9,}: untruncated {a:6.2f}%  "
              f"truncated {b:6.2f}%  delta {b - a:+.2f} pp", flush=True)
    d = np.array(deltas)
    se = 100 * np.sqrt(0.25 / R_TEST)          # per-arm SE; the difference of two arms is ~sqrt(2) x
    tol = 2 * np.sqrt(2) * se                  # ~2 sigma on the difference
    verdict = ("inside" if np.abs(d).max() <= tol else "OUTSIDE")
    print(f"\n   => largest deviation {np.abs(d).max():+.2f} pp, mean {d.mean():+.2f} pp — "
          f"{verdict} the {tol:.2f} pp two-sigma band on a difference of two arms.")
    if verdict == "OUTSIDE":
        print("      NOTE: at low R this is dominated by tuning noise (the grid search picks a "
              "different m' per arm). Re-run without --quick before drawing any conclusion.")
    print("   The truncated form is the default; pass support=None for the untruncated one.")


if __name__ == "__main__":
    main()
