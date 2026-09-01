"""Timing preflight for the exact statistical safeguard.

    python analysis/safeguard_timing.py

Exhaustive enumeration replaced a coarse-to-fine search, so one safeguard call now costs a vectorised
pass over the whole [N_min, N_max] interval instead of ~4 x 128 candidates. That is cheap per call
and irrelevant inside one trial; the reason to measure it is the MULTIPLIER. In `--max` mode one
scenario/budget point makes on the order of 8.7e5 safeguard calls for `binary_deep` and 2.8e5 for
`reverse_eng_risk`, and the eight active scenarios hold 177 such points -- roughly 2e8 calls before
duplicate-grid reductions. Microseconds per call turn into hours of aggregate CPU there.

The active thesis matrix uses exactly three intervals:

    narrow_e3 .. narrow_e8    15 .. 157       143 candidates
    small_e4                 157 .. 1570    1,414 candidates
    wide_e4                   15 .. 1570    1,556 candidates

There is no active scenario with N_max > 1570; tightening epsilon lengthens the budget grid but does
not widen the candidate interval.

This is a DIAGNOSTIC gate, not an acceptance threshold. If 15..1570 comes out far above ~0.5 ms per
call, profile the candidate allocation and `_p_safe` before launching the sweep -- do not respond by
reintroducing an approximate search.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qmetrology.safeguard import pilot_sd, risk_optimal_depth

# (label, phi_min, phi_max, eps, budget) -- one representative operating point per interval
CASES = [
    ("narrow  15..157",   0.01,  0.1,  1e-3, 10_000,    10_000),
    ("small  157..1570",  0.001, 0.01, 1e-4, 1_000_000,  5_000),
    ("wide    15..1570",  0.001, 0.1,  1e-4, 200_000,    5_000),
]
GATE_MS = 0.5          # diagnostic trigger for the widest active interval


def bounds(phi_min, phi_max):
    lo = max(int(np.pi // (2 * phi_max)), 1)
    return lo, max(int(np.pi // (2 * phi_min)), lo)


def measure(phi_min, phi_max, eps, budget, n_calls, seed=20260901):
    """Time `n_calls` calls on pilots drawn the way the algorithms actually produce them.

    The pilot values VARY across calls, so the argmax lands in different places and neither branch
    prediction nor a single cached score can flatter the measurement. `_candidates` legitimately
    caches the interval array -- that is the production behaviour and is what should be timed.
    """
    N_min, N_max = bounds(phi_min, phi_max)
    rng = np.random.default_rng(seed)
    phis = rng.uniform(phi_min, phi_max, n_calls)
    m_pilot = rng.integers(20, 2000, n_calls)
    sigmas = pilot_sd(N_min, m_pilot)
    support = (phi_min, phi_max)

    # warm the interval cache and the scipy dispatch, so the timed loop measures steady state
    for i in range(50):
        risk_optimal_depth(float(phis[i]), float(sigmas[i]), budget, eps,
                           N_min=N_min, N_max=N_max, support=support)

    depths = np.empty(n_calls, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n_calls):
        depths[i] = risk_optimal_depth(float(phis[i]), float(sigmas[i]), budget, eps,
                                       N_min=N_min, N_max=N_max, support=support)
    dt = time.perf_counter() - t0
    assert np.all((depths >= N_min) & (depths <= N_max))
    return dict(N_min=N_min, N_max=N_max, n_cand=N_max - N_min + 1, n_calls=n_calls,
                total_s=dt, us_per_call=1e6 * dt / n_calls,
                distinct_depths=int(len(np.unique(depths))))


def main():
    print(f"safeguard timing preflight   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}", flush=True)
    print(f"{'interval':<18} {'cand':>6} {'calls':>8} {'total s':>9} {'us/call':>9} "
          f"{'distinct N*':>12}")
    rows = []
    for (label, pmin, pmax, eps, budget, n) in CASES:
        r = measure(pmin, pmax, eps, budget, n)
        rows.append((label, r))
        print(f"{label:<18} {r['n_cand']:>6,} {r['n_calls']:>8,} {r['total_s']:>9.3f} "
              f"{r['us_per_call']:>9.1f} {r['distinct_depths']:>12,}", flush=True)

    wide = dict(rows)["wide    15..1570"]
    ms = wide["us_per_call"] / 1000.0
    print()
    print(f"widest active interval (15..1570): {ms:.4f} ms/call   "
          f"[diagnostic trigger {GATE_MS} ms/call]")
    # the aggregate the multiplier actually implies, at the widest interval's rate
    calls = 1_150_000 * 177
    print(f"upper bound on the sweep's safeguard cost at that rate: "
          f"{calls * wide['us_per_call'] / 1e6 / 3600:.2f} CPU-hours "
          f"({calls:,} calls, pre-dedup, single core)")
    if ms > GATE_MS:
        print("\nABOVE THE TRIGGER. Profile the candidate allocation and _p_safe before launching "
              "the sweep. Do NOT reintroduce an approximate search.")
        return 1
    print("\nbelow the trigger -- enumeration is not the bottleneck for the active matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
