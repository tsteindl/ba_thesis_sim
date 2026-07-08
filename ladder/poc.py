"""Proof-of-concept checks for the phase-unwrapping ladder.

1. Deterministic per-stage traces at several phi: depth growth, unwrap correctness,
   error shrinkage.
2. Quick R=2,000 statistics (test-style seeding, same scheme as qmetrology.experiments):
   convergence at eps=1e-4 vs brute / RE at matched budgets, plus the branch-miss rate
   (error > 10 eps => a wrong-branch catastrophe rather than ordinary noise).

    python ladder/poc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.algorithms import (  # noqa: E402
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_reverse_engineering as RE,
)
from ladder.algorithms import (  # noqa: E402
    find_phi_fixed_budget_ladder as LAD,
    find_phi_fixed_budget_ladder_capped as LADC,
)

PMIN, PMAX, EPS = 0.01, 0.1, 1e-4


def trace_run(phi, budget, seed=7, **kw):
    rng = np.random.default_rng(seed)
    tr = []
    phi_hat, used = LAD(rng, phi, PMAX, PMIN, budget, trace=tr, **kw)
    print(f"\nphi={phi}  budget={budget:,}  used={used:,}  final err={abs(phi_hat - phi):.2e}")
    prev_err = None
    for i, (N, m, ph, sd) in enumerate(tr):
        err = abs(ph - phi)
        theta = N * phi
        print(f"  stage {i}: N={N:>9,}  m={m:>9,}  N*phi={theta:8.2f} rad  "
              f"phi_hat={ph:.8f}  err={err:.2e}  (crb sd={sd:.1e})")
        prev_err = err
    return abs(phi_hat - phi)


def stats(fn, budget, R=2000, seed=2024, **params):
    """Same seeding scheme as qmetrology.experiments (seed -> per-trial seeds -> phi)."""
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    errs = np.empty(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            phi = rng.uniform(PMIN, PMAX)
            ph, _ = fn(rng, phi, PMAX, PMIN, budget=budget, **params)
            errs[i] = abs(ph - phi)
    conv = float(np.mean(errs < EPS))
    miss = float(np.mean(errs > 10 * EPS))  # branch-miss proxy
    return conv, miss, float(np.nanmedian(errs))


def main():
    print("=== 1. deterministic traces (unbounded ladder, defaults) ===")
    for phi in (0.011, 0.05, 0.099):
        for seed in (7, 8):
            trace_run(phi, 1_000_000, seed=seed)

    print("\n=== 2. quick R=2000 comparison, eps=1e-4, U(0.01,0.1) ===")
    print(f"{'budget':>12}  {'brute':>14}  {'RE(canon-ish)':>14}  {'ladder-cap':>14}  {'ladder-unb':>14}")
    for budget in (100_000, 200_000, 400_000, 1_000_000, 4_500_000):
        row = [f"{budget:>12,}"]
        for fn, params in [(BF, {}), (RE, {"m_exploration": 1000, "safeguard": 0.9}),
                           (LADC, {}), (LAD, {})]:
            conv, miss, med = stats(fn, budget, **params)
            row.append(f"{100 * conv:5.1f}% ({100 * miss:4.1f}%)")
        print("  ".join(row) + "   [conv% (miss>10eps%)]")


if __name__ == "__main__":
    main()
