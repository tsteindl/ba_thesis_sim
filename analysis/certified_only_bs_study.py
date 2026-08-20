"""Focused test of certified-only sequential bisection.

Unlike ``conservative_update_study.py``, this variant does not spend resources on a midpoint whose
outcome cannot be admitted to the final pilot.  Before each prospective bisection probe it requires

    P(phi < pi/(2N_mid) | previously certified data) >= tau.

If the condition fails, exploration stops.  If it passes, the outcome is measured, used for the
bisection branch, and inverse-variance pooled into the pilot supplied to the statistical safeguard.
Thus every bisection step has a direct role in the final choice of exploitation depth, and the
method automatically falls back to the opening-pilot RE rule when no safe refinement is available.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import _p_safe, pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors


TAUS = (0.99, 0.999)
ARMS = ("re", "stop_099", "stop_0999")
R_TUNE = 500
R_TEST = 10_000
SEED_TUNE = 8_103_917
SEED_TEST = 6_277_019
OUT = "results/certified_only_bs.csv"
POINTS = (
    ("headline low", 0.01, 0.1, 1e-3, 10_000),
    ("tight medium", 0.01, 0.1, 1e-4, 878_661),
    ("small phase", 0.001, 0.01, 1e-4, 464_457),
    ("wide prior", 0.0001, 0.1, 1e-4, 1_026_274),
)


def explore_certified(rng, phi, pmax, pmin, m, budget, conf, tau):
    n_min = max(int(np.pi // (2 * pmax)), 1)
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    if m * n_min > budget:
        return None
    pilot = float(simulate_errors(rng, phi, m, n_min))
    used = m * n_min
    sigma = pilot_sd(n_min, m)
    precision = 1.0 / sigma**2
    n_guess = n_sup if not np.isfinite(pilot) or pilot <= 0 else min(
        max(int(np.pi // (2 * pilot)), n_min), n_sup)
    lb, ub = n_min, n_guess
    probes = 1
    contaminated = False

    while ub - lb > 1:
        n = lb + (ub - lb) // 2
        if n <= lb or n >= ub or used + m * n > budget:
            break
        # The safety decision is predictable: it is made before observing this probe.
        q_safe = float(_p_safe(n, pilot, sigma, support=(pmin, pmax)))
        if q_safe < tau:
            break
        threshold = norm.ppf(1 - conf, loc=pilot, scale=pilot_sd(n, m))
        new_pilot = float(simulate_errors(rng, phi, m, n))
        used += m * n
        probes += 1
        contaminated = contaminated or n > n_opt
        if new_pilot < threshold:
            ub = n
        else:
            lb = n
        new_precision = 1.0 / pilot_sd(n, m)**2
        pilot = (pilot * precision + new_pilot * new_precision) / (precision + new_precision)
        precision += new_precision
        sigma = 1.0 / np.sqrt(precision)
    return pilot, sigma, used, probes, float(contaminated)


def _finish(seed, phi, pmin, pmax, eps, budget, pilot, sigma, used):
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    remaining = budget - used
    if remaining <= 0:
        return (0.0, used / budget, 0.0, 0.0, 50.0)
    depth = risk_optimal_depth(pilot, sigma, remaining, eps, N_max=n_sup,
                               support=(pmin, pmax))
    shots = int(remaining / depth)
    rng = np.random.default_rng([int(seed) % (2**32), int(depth) % (2**32),
                                 int(shots) % (2**32), 499_979])
    estimate = simulate_errors(rng, phi, shots, depth)
    return (float(abs(estimate - phi) < eps), used / budget, depth / n_opt,
            float(depth > n_opt), min(abs(pilot - phi) / sigma, 50.0))


def _one(seed, pmin, pmax, eps, budget, m, conf):
    seed = int(seed)
    phi = float(np.random.default_rng(seed).uniform(pmin, pmax))
    n0 = max(int(np.pi // (2 * pmax)), 1)
    stream = [seed % (2**32), 104_729]
    ph0 = float(simulate_errors(np.random.default_rng(stream), phi, m, n0))
    spec = {"re": (ph0, pilot_sd(n0, m), m * n0, 1, 0.0)}
    for tau in TAUS:
        spec[f"stop_{str(tau).replace('.', '')}"] = explore_certified(
            np.random.default_rng(stream), phi, pmax, pmin, m, budget, conf, tau)
    result = {}
    for arm, (pilot, sigma, used, probes, contaminated) in spec.items():
        finish = _finish(seed, phi, pmin, pmax, eps, budget, pilot, sigma, used)
        # success, exploration, probes, updates, P(update), contaminated, depth/Nopt,
        # overshoot, standardized pilot error
        result[arm] = (finish[0], finish[1], probes, probes - 1, float(probes > 1),
                       contaminated, finish[2], finish[3], finish[4])
    return result


def _task(args):
    pmin, pmax, eps, budget, cfg, seeds = args
    sums = {arm: np.zeros(9) for arm in ARMS}
    for seed in seeds:
        trial = _one(seed, pmin, pmax, eps, budget, cfg[0], cfg[1])
        for arm in ARMS:
            sums[arm] += trial[arm]
    return sums, len(seeds)


def evaluate(configs, repetitions, point, seed):
    _label, pmin, pmax, eps, budget = point
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=repetitions)
    chunks = [x for x in np.array_split(seeds, E.N_JOBS * 2) if len(x)]
    tasks = [(pmin, pmax, eps, budget, cfg, chunk) for cfg in configs for chunk in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
        output = list(executor.map(_task, tasks, chunksize=1))
    result = {arm: [] for arm in ARMS}
    k = len(chunks)
    for i in range(len(configs)):
        parts = output[i * k:(i + 1) * k]
        n = sum(x[1] for x in parts)
        for arm in ARMS:
            result[arm].append(sum((x[0][arm] for x in parts), start=np.zeros(9)) / n)
    return result


def grid(pmax, eps, budget):
    n_min = max(int(np.pi // (2 * pmax)), 1)
    m_hi = min(int(np.clip(0.6724 / (n_min * eps**2), 200, 100_000)),
               budget // n_min)
    ms = np.unique(np.geomspace(3, m_hi, 8).astype(int))
    return [(int(m), conf) for m, conf in product(ms, (0.5, 0.8, 0.95))]


def main():
    fields = ("point", "arm", "rate", "mcse_pp", "exploration_share", "probes",
              "certified_updates", "p_updated", "contaminated", "depth_over_nopt",
              "overshoot", "pilot_abs_z", "m", "conf")
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for i, point in enumerate(POINTS, 1):
        configs = grid(point[2], point[3], point[4])
        tuning = evaluate(configs, R_TUNE, point, SEED_TUNE)
        winners = {arm: configs[int(np.argmax([x[0] for x in tuning[arm]]))]
                   for arm in ARMS}
        unique = list(dict.fromkeys(winners.values()))
        tests = {cfg: evaluate([cfg], R_TEST, point, SEED_TEST) for cfg in unique}
        print(f"[{i}/{len(POINTS)}] {point[0]}", flush=True)
        for arm in ARMS:
            cfg = winners[arm]
            stats = tests[cfg][arm][0]
            rate = stats[0]
            row = {"point": point[0], "arm": arm, "rate": 100 * rate,
                   "mcse_pp": 100 * np.sqrt(rate * (1 - rate) / R_TEST),
                   "exploration_share": stats[1], "probes": stats[2],
                   "certified_updates": stats[3], "p_updated": 100 * stats[4],
                   "contaminated": 100 * stats[5], "depth_over_nopt": stats[6],
                   "overshoot": 100 * stats[7], "pilot_abs_z": stats[8],
                   "m": cfg[0], "conf": cfg[1]}
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  {arm:10s} {100*rate:6.2f}% probes={stats[2]:4.2f} "
                  f"updates={stats[3]:4.2f} ({100*stats[4]:5.1f}%) "
                  f"contam={100*stats[5]:5.3f}% expl={100*stats[1]:4.1f}% "
                  f"|z|={stats[8]:4.2f} m={cfg[0]} c={cfg[1]:.2f}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
