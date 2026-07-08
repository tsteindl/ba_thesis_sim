"""Proof-of-concept checks for CCRE.

1. Deterministic per-round traces at several phi (high conf=0.999, so the branch invariant
   holds with overwhelming probability and traces read cleanly), and error shrinks round-over-round.
2. Overshoot-rate check: the confidence bound is PROBABILISTIC by design (a one-sided z=conf
   bound), so at conf=0.95 we EXPECT the principal-branch invariant to fail on the final round
   about 1-conf=5% of the time (that's the point -- a provable, tunable bound vs RE's ungoverned
   fixed safeguard). Verify the empirical rate over many seeds tracks 1-conf.
3. Quick R=2,000 comparison vs reverse engineering at matched budgets, eps=1e-4, U(0.01,0.1).

    python ccre/poc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.algorithms import find_phi_fixed_budget_reverse_engineering as RE  # noqa: E402
from ccre.algorithms import find_phi_fixed_budget_ccre as CCRE  # noqa: E402

PMIN, PMAX, EPS = 0.01, 0.1, 1e-4


def trace_run(phi, budget, seed, **kw):
    rng = np.random.default_rng(seed)
    tr = []
    phi_hat, used = CCRE(rng, phi, PMAX, PMIN, budget, trace=tr, **kw)
    print(f"\nphi={phi}  budget={budget:,}  used={used:,}  final err={abs(phi_hat - phi):.2e}")
    ok = True
    for i, (N, m, ph) in enumerate(tr):
        invariant = N * phi < np.pi / 2
        ok &= invariant
        print(f"  round {i}: N={N:>6,}  m={m:>8,}  N*phi={N*phi:6.3f} (<pi/2: {invariant})  "
              f"phi_hat={ph:.8f}  err={abs(ph - phi):.2e}")
    assert ok, f"principal-branch invariant violated for phi={phi} seed={seed} (unexpected at high conf)"
    return abs(phi_hat - phi)


def overshoot_rate(conf, n_rounds=2, R=5000, seed=999, budget=1_000_000, m_round=200):
    """Empirical P(final N*phi > pi/2) vs the predicted one-sided (1-conf)."""
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    over = 0
    for s in seeds:
        rng = np.random.default_rng(int(s))
        phi = rng.uniform(PMIN, PMAX)
        tr = []
        CCRE(rng, phi, PMAX, PMIN, budget, m_round=m_round, conf=conf, n_rounds=n_rounds, trace=tr)
        if tr and tr[-1][0] * phi >= np.pi / 2:
            over += 1
    return over / R


def stats(fn, budget, R=2000, seed=2024, **params):
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    errs = np.empty(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            phi = rng.uniform(PMIN, PMAX)
            ph, _ = fn(rng, phi, PMAX, PMIN, budget=budget, **params)
            errs[i] = abs(ph - phi)
    conv = float(np.mean(errs < EPS))
    miss = float(np.mean(errs > 10 * EPS))
    return conv, miss


def main():
    print("=== 1. deterministic traces (n_rounds=2, conf=0.999 -- high margin for a clean demo) ===")
    for phi in (0.011, 0.05, 0.099):
        for seed in (7, 8):
            trace_run(phi, 1_000_000, seed, m_round=200, conf=0.999, n_rounds=2)
    print("\nAll principal-branch invariants held (as expected at conf=0.999).")

    print("\n=== 1b. overshoot-rate check: empirical vs predicted (1-conf), R=5000 ===")
    for conf in (0.80, 0.90, 0.95, 0.99):
        rate = overshoot_rate(conf)
        print(f"  conf={conf:.2f}  predicted overshoot~{100 * (1 - conf):.1f}%  "
              f"empirical={100 * rate:.2f}%")

    print("\n=== 2. quick R=2000 comparison, eps=1e-4, U(0.01,0.1) ===")
    print(f"{'budget':>12}  {'RE(canon)':>16}  {'CCRE n=1':>16}  {'CCRE n=2':>16}  {'CCRE n=3':>16}")
    for budget in (100_000, 200_000, 400_000, 1_000_000, 4_500_000):
        row = [f"{budget:>12,}"]
        for fn, params in [(RE, {"m_exploration": 1000, "safeguard": 0.9}),
                           (CCRE, {"m_round": 200, "conf": 0.95, "n_rounds": 1}),
                           (CCRE, {"m_round": 200, "conf": 0.95, "n_rounds": 2}),
                           (CCRE, {"m_round": 200, "conf": 0.95, "n_rounds": 3})]:
            conv, miss = stats(fn, budget, **params)
            row.append(f"{100 * conv:5.1f}% ({100 * miss:4.2f}%)")
        print("  ".join(row) + "   [conv% (miss>10eps%)]")


if __name__ == "__main__":
    main()
