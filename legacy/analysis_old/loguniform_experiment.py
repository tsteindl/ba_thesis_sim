"""Is a LOG-uniform prior (a true 'wide dynamic range') where adaptive beats brute?

Brute force fixes N=N_min, so its error ~1/(2 N_min sqrt(m)) is independent of phi:
its convergence rate depends only on (phi_max, budget), NOT on the shape of the prior.
Adaptive uses N ~ pi/(2 phi), so it converges far more easily for SMALL phi. Hence the
adaptive advantage lives in the small-phi tail — which a linear-uniform prior barely
samples, but a log-uniform prior (orders of magnitude) samples heavily.

We compare honest (de-biased) % converged for brute / linear / reverse-eng under
uniform vs log-uniform priors, over increasing dynamic range and budget.
    python analysis/loguniform_experiment.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E  # for _pool context / wilson
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_reverse_engineering as RE,
)

EPS = 1e-3
R_TUNE, R_TEST = 800, 20000
GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 2000, 12).astype(int)),
            "lookback_window": [1, 2, 5], "safeguard": [1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 5000, 12).astype(int)),
           "safeguard": [0.8, 0.85, 0.9, 0.95]}


def _chunk(task):
    fn, phi_min, phi_max, params, seeds, logu = task
    lo, hi = np.log(phi_min), np.log(phi_max)
    succ = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = float(np.exp(rng.uniform(lo, hi))) if logu else float(rng.uniform(phi_min, phi_max))
            ph, _ = fn(rng, phi, phi_max, phi_min, **params)
            succ += abs(ph - phi) < EPS
    return succ


def rate(fn, params, R, phi_min, phi_max, seed, logu, n_jobs=20):
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 3) if len(c)]
    tasks = [(fn, phi_min, phi_max, params, c, logu) for c in chunks]
    with E._pool(n_jobs) as ex:
        return 100 * sum(ex.map(_chunk, tasks)) / R


def best(fn, grid, budget, phi_min, phi_max, logu):
    """Grid-tune (seed 42) then de-bias the winner (seed 2024)."""
    names = list(grid) + ["budget"]
    combos = [dict(zip(names, v)) for v in product(*grid.values(), [budget])]
    tuned = [(rate(fn, c, R_TUNE, phi_min, phi_max, 42, logu), c) for c in combos]
    win = max(tuned, key=lambda t: t[0])[1]
    return rate(fn, win, R_TEST, phi_min, phi_max, 2024, logu)


if __name__ == "__main__":
    phi_max = np.pi / 2  # N_min = 1: brute is the pure low-N baseline
    for budget in [10_000, 100_000]:
        for logu in [False, True]:
            tag = "log-uniform" if logu else "linear-uniform"
            print(f"\n=== {tag} prior on [phi_min, pi/2],  budget={budget:,},  eps=1e-3 ===")
            print(f"{'phi_min':>8} {'orders':>7} | {'brute':>6} | {'linear':>15} | {'reverse_eng':>15}")
            for phi_min in [0.1, 0.03, 0.01, 0.003, 0.001]:
                orders = np.log10(phi_max / phi_min)
                b = rate(BF, {"budget": budget}, R_TEST, phi_min, phi_max, 2024, logu)
                lin = best(LIN, GRID_LIN, budget, phi_min, phi_max, logu)
                re = best(RE, GRID_RE, budget, phi_min, phi_max, logu)
                print(f"{phi_min:>8} {orders:>6.1f}x | {b:6.1f} | {lin:6.1f} ({lin-b:+5.1f}) | {re:6.1f} ({re-b:+5.1f})")
