"""Can conservative, sequential pilot updates give binary search a defensible role?

The statistical safeguard is allowed to use a later probe only when that probe was certified as
unaliased *before its outcome was observed*.  This is the key distinction from using the "deepest
accepted" estimate: acceptance selects on the same noisy value and invalidates the ordinary Gaussian
pilot likelihood.

All updating arms start from the safe opening probe, set the initial binary-search upper bound to
N_guess=floor(pi/(2 phi_hat_0)), and run the existing lower-threshold bisection.  Before measuring a
midpoint N, the current truncated-normal pilot posterior supplies

    q_safe(N) = P(phi < pi/(2N) | data already observed).

The new measurement may update the safeguard pilot only if q_safe >= tau.  The decision is therefore
independent of that measurement's outcome.  Two update rules are tested:

    last_tau   replace the pilot by the latest pre-certified estimate;
    pool_tau   inverse-variance-pool every pre-certified estimate.

Pooling is the statistically coherent version: for safe probes the likelihoods are independent and
their precisions are 4*m*N_i^2.  ``contaminated`` records the empirical share of runs in which at
least one supposedly safe update actually had N > N_opt.

Baselines:
    re          opening pilot only, no bisection;
    current     full-prior bisection, but opening pilot feeds the safeguard;
    guess_deep  N_guess-restricted bisection, deepest accepted estimate feeds the safeguard.

This is a focused method-selection experiment, not a production algorithm.  It writes
results/conservative_update.csv and does not modify the thesis sweep.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_story_study import explore as explore_current
from focused_bs_resolution import explore_to_guess


TAUS = (0.95, 0.99, 0.999)
GATED = [f"{mode}_{str(tau).replace('.', '')}" for mode in ("last", "pool") for tau in TAUS]
ARMS = ["re", "current", "guess_deep"] + GATED

R_TUNE = 350
R_TEST = 6_000
SEED_TUNE = 2_305_843
SEED_TEST = 4_808_817
OUT = "results/conservative_update.csv"

POINTS = [
    ("headline low", 0.01, 0.1, 1e-3, 10_000),
    ("tight medium", 0.01, 0.1, 1e-4, 878_661),
    ("small phase", 0.001, 0.01, 1e-4, 464_457),
    ("wide prior", 0.0001, 0.1, 1e-4, 1_026_274),
]

# success, exploration share, probes, certified updates, P(any update), contaminated,
# final N/Nopt, final overshoot, standardized pilot error
NSTATS = 9


def explore_gated(rng, phi, pmax, pmin, m, budget, conf, tau, mode):
    """N_guess-restricted bisection with pre-outcome certification of pilot updates."""
    n_min = max(int(np.pi // (2 * pmax)), 1)
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    if m * n_min > budget:
        return None

    ph = simulate_errors(rng, phi, m, n_min)
    used = m * n_min
    sigma = pilot_sd(n_min, m)
    mu = float(ph)
    precision = 1.0 / sigma**2
    n_guess = n_sup if not np.isfinite(ph) or ph <= 0 else min(
        max(int(np.pi // (2 * ph)), n_min), n_sup)
    lb, ub = n_min, n_guess
    probes = 1
    certified = 1
    contaminated = False

    while ub - lb > 1:
        n = lb + (ub - lb) // 2
        if n <= lb or n >= ub or used + m * n > budget:
            break

        # This gate is evaluated before seeing the new outcome.  It may therefore select a depth,
        # but it cannot select a favorable realization of phi_hat at that depth.
        q_safe = float(_p_safe(n, mu, sigma, support=(pmin, pmax)))
        threshold = norm.ppf(1 - conf, loc=mu, scale=pilot_sd(n, m))
        new_ph = float(simulate_errors(rng, phi, m, n))
        used += m * n
        probes += 1

        if new_ph < threshold:
            ub = n
        else:
            lb = n

        if q_safe >= tau:
            certified += 1
            contaminated = contaminated or n > n_opt
            new_sigma = pilot_sd(n, m)
            if mode == "pool":
                new_precision = 1.0 / new_sigma**2
                mu = (mu * precision + new_ph * new_precision) / (precision + new_precision)
                precision += new_precision
                sigma = 1.0 / np.sqrt(precision)
            else:
                mu, sigma = new_ph, new_sigma

    return mu, sigma, used, probes, certified, float(contaminated)


def _one(seed, pmin, pmax, eps, budget, m, conf):
    seed = int(seed)
    phi = float(np.random.default_rng(seed).uniform(pmin, pmax))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    stream = [seed % (2**32), 104_729]

    oc = explore_current(np.random.default_rng(stream), phi, pmax, pmin, m, budget, conf)
    og = explore_to_guess(np.random.default_rng(stream), phi, pmax, pmin, m, budget, conf)
    if oc is None or og is None:
        return None

    _ph, _n, used_c, _deep_c, _nd_c, ph0, n0, _lc, _uc, hist_c = oc
    used_g, ph0_g, n0_g, deep_g, ndeep_g, _last_g, _nlast_g, _lg, _ug, probes_g = og
    assert np.isclose(ph0, ph0_g) and n0 == n0_g

    # arm -> pilot, sigma, exploration cost, probes, certified count, contaminated
    spec = {
        "re": (ph0, pilot_sd(n0, m), m * n0, 1, 1, 0.0),
        "current": (ph0, pilot_sd(n0, m), used_c, len(hist_c), 1, 0.0),
        # Diagnostic only: unlike the gated arms this estimate was selected by its own outcome.
        "guess_deep": (deep_g, pilot_sd(ndeep_g, m), used_g, probes_g,
                       1 + int(ndeep_g != n0), float(ndeep_g > n_opt)),
    }
    for mode in ("last", "pool"):
        for tau in TAUS:
            name = f"{mode}_{str(tau).replace('.', '')}"
            out = explore_gated(np.random.default_rng(stream), phi, pmax, pmin, m, budget,
                                conf, tau, mode)
            if out is None:
                return None
            spec[name] = out

    result = {}
    for arm in ARMS:
        pilot, sigma, used, probes, certified, contaminated = spec[arm]
        rem = budget - used
        if rem <= 0 or not np.isfinite(pilot) or not np.isfinite(sigma) or sigma <= 0:
            result[arm] = (0.0, min(used / budget, 1.0), probes, certified - 1,
                           float(certified > 1), contaminated, 0.0, 0.0, 50.0)
            continue
        depth = risk_optimal_depth(pilot, sigma, rem, eps, N_max=n_sup,
                                   support=(pmin, pmax))
        shots = int(rem / depth)
        r2 = np.random.default_rng([seed % (2**32), int(depth) % (2**32),
                                    int(shots) % (2**32), 499_979])
        estimate = simulate_errors(r2, phi, shots, depth)
        result[arm] = (
            float(abs(estimate - phi) < eps), used / budget, probes, certified - 1,
            float(certified > 1), contaminated, depth / n_opt, float(depth > n_opt),
            min(abs(pilot - phi) / sigma, 50.0),
        )
    return result


def _task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    sums = {arm: [0.0] * NSTATS for arm in ARMS}
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for seed in seeds:
            trial = _one(seed, pmin, pmax, eps, budget, cfg["m"], cfg["conf"])
            n += 1
            if trial is None:
                continue
            for arm in ARMS:
                for j, value in enumerate(trial[arm]):
                    sums[arm][j] += value
    return sums, n


def evaluate(configs, repetitions, pmin, pmax, eps, budget, seed):
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=repetitions)
    chunks = [chunk for chunk in np.array_split(seeds, E.N_JOBS * 2) if len(chunk)]
    tasks = [(pmin, pmax, eps, budget, cfg, chunk) for cfg in configs for chunk in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
        output = list(executor.map(_task, tasks, chunksize=1))
    per = {arm: [] for arm in ARMS}
    k = len(chunks)
    for i in range(len(configs)):
        parts = output[i * k:(i + 1) * k]
        total = sum(part[1] for part in parts)
        for arm in ARMS:
            per[arm].append(tuple(sum(part[0][arm][j] for part in parts) / total
                                  for j in range(NSTATS)))
    return per


def grid(pmax, eps):
    n_min = max(int(np.pi // (2 * pmax)), 1)
    m_hi = int(np.clip(0.6724 / (n_min * eps**2), 200, 100_000))
    ms = np.unique(np.geomspace(3, m_hi, 8).astype(int))
    return [{"m": int(m), "conf": conf} for m, conf in product(ms, (0.5, 0.8, 0.95))]


def main():
    os.makedirs("results", exist_ok=True)
    fields = ["point", "phi_min", "phi_max", "eps", "budget", "arm", "rate", "mcse_pp",
              "exploration_share", "probes", "certified_updates", "p_updated", "contaminated",
              "depth_over_nopt", "overshoot", "pilot_abs_z", "m", "conf"]
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"conservative update study: {len(POINTS)} points, R_tune={R_TUNE}, R_test={R_TEST}",
          flush=True)
    for i, (label, pmin, pmax, eps, budget) in enumerate(POINTS, 1):
        configs = grid(pmax, eps)
        tuning = evaluate(configs, R_TUNE, pmin, pmax, eps, budget, SEED_TUNE)
        winners = {arm: configs[int(np.argmax([x[0] for x in tuning[arm]]))] for arm in ARMS}
        unique = []
        for cfg in winners.values():
            if cfg not in unique:
                unique.append(cfg)
        tested = {tuple(cfg.values()): evaluate([cfg], R_TEST, pmin, pmax, eps, budget,
                                                SEED_TEST) for cfg in unique}
        print(f"\n[{i}/{len(POINTS)}] {label}: U({pmin:g},{pmax:g}), eps={eps:g}, B={budget:,}",
              flush=True)
        for arm in ARMS:
            cfg = winners[arm]
            stats = tested[tuple(cfg.values())][arm][0]
            rate = stats[0]
            row = {
                "point": label, "phi_min": pmin, "phi_max": pmax, "eps": eps,
                "budget": budget, "arm": arm, "rate": round(100 * rate, 4),
                "mcse_pp": round(100 * np.sqrt(rate * (1 - rate) / R_TEST), 4),
                "exploration_share": round(stats[1], 6), "probes": round(stats[2], 4),
                "certified_updates": round(stats[3], 4), "p_updated": round(100 * stats[4], 4),
                "contaminated": round(100 * stats[5], 4), "depth_over_nopt": round(stats[6], 5),
                "overshoot": round(100 * stats[7], 4), "pilot_abs_z": round(stats[8], 4),
                "m": cfg["m"], "conf": cfg["conf"],
            }
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  {arm:11s} {100*rate:6.2f}%  probes={stats[2]:5.2f}  "
                  f"updates={stats[3]:4.2f} ({100*stats[4]:5.1f}%)  "
                  f"contam={100*stats[5]:5.2f}%  expl={100*stats[1]:5.1f}%  "
                  f"|z|={stats[8]:4.2f}  m={cfg['m']:6d} c={cfg['conf']:.2f}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
