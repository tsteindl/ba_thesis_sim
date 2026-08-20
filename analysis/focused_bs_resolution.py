"""Focused check of low-disruption ways to make bisection inform the safeguard.

This is deliberately smaller than the full thesis sweep.  It answers one design question at six
representative operating points, with every arm tuned on one seed and evaluated on an independent
seed:

``re``
    The statistical-safeguard reverse-engineering rule.  It buys only the opening probe.
``current``
    The current binary exploration, but the safeguard uses the opening probe and the prior cap.
    Hence the bisection cannot affect the final depth.
``phi0_U`` / ``phi0_L``
    Let the current bisection cap the safeguard at its rejected/accepted bound.
``deep_L``
    The closest implementation of "latest accepted phi-hat, N_min ... N_guess=L".
``guess_deep`` / ``guess_deep_L``
    First infer N_guess=floor(pi/(2 phi_hat_0)), then bisect only [N_min,N_guess], and feed the
    deepest accepted estimate to the safeguard (with the prior/L cap respectively).
``guess_last_L``
    As above, but literally feed the most recent probe, even if it was rejected.
``guess_fresh_L``
    Re-measure the selected L with a fresh, independent batch before invoking the safeguard.  This
    removes the acceptance-selection bias, at the cost of one confirmation probe.

The new ``guess_*`` family is not proposed as a thesis algorithm.  It is included because it is the
most plausible small repair suggested by the current discussion.  If it cannot beat ``re`` or the
decorative ``current`` arm here, there is no case for expanding it into the full sweep.

Run from the repository root:

    python analysis/focused_bs_resolution.py

The script never touches the published sweep files.  It writes results/focused_bs_resolution.csv.
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
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_story_study import explore as explore_current


R_TUNE = 500
R_TEST = 6_000
SEED_TUNE = 710_231
SEED_TEST = 910_229
OUT = "results/focused_bs_resolution.csv"

ARMS = [
    "re",
    "current",
    "phi0_U",
    "phi0_L",
    "deep_L",
    "guess_deep",
    "guess_deep_L",
    "guess_last_L",
    "guess_fresh_L",
]

# Low/high budget for the headline prior, plus narrow- and wide-prior stress tests.
POINTS = [
    ("headline low", 0.01, 0.1, 1e-3, 10_000),
    ("headline high", 0.01, 0.1, 1e-3, 48_613),
    ("tight medium", 0.01, 0.1, 1e-4, 878_661),
    ("tight high", 0.01, 0.1, 1e-4, 4_861_325),
    ("small phase", 0.001, 0.01, 1e-4, 464_457),
    ("wide prior", 0.0001, 0.1, 1e-4, 1_026_274),
]

# success, exploration fraction, probes, depth/Nopt, final overshoot, differs from RE
NSTATS = 6


def explore_to_guess(rng, phi, phi_max, phi_min, m, budget, conf):
    """Algorithm 5 restricted to [N_min, floor(pi/(2 phi_hat_0))]."""
    n_min = max(int(np.pi // (2 * phi_max)), 1)
    n_sup = max(int(np.pi // (2 * phi_min)), 1)
    if m * n_min > budget:
        return None

    n = n_min
    ph = simulate_errors(rng, phi, m, n)
    used = m * n
    ph0, n0 = ph, n
    phi_cut = norm.ppf(1 - conf, ph, pilot_sd(n, m))

    if not np.isfinite(ph) or ph <= 0:
        n_guess = n_sup
    else:
        n_guess = min(max(int(np.pi // (2 * ph)), n_min), n_sup)
    lb, ub = n_min, n_guess
    deep_ph, deep_n = ph, n
    last_ph, last_n = ph, n
    probes = 1

    while ub - lb > 1:
        n = lb + (ub - lb) // 2
        if n <= lb or n >= ub or used + m * n > budget:
            break
        ph = simulate_errors(rng, phi, m, n)
        used += m * n
        probes += 1
        last_ph, last_n = ph, n
        if ph < phi_cut:
            ub = n
        else:
            lb = n
            deep_ph, deep_n = ph, n
            # The current code uses the newly accepted estimate as the next reference.
            phi_cut = norm.ppf(1 - conf, ph, pilot_sd(n, m))

    return used, ph0, n0, deep_ph, deep_n, last_ph, last_n, lb, ub, probes


def _one(seed, pmin, pmax, eps, budget, m, conf):
    seed = int(seed)
    rng_phi = np.random.default_rng(seed)
    phi = float(rng_phi.uniform(pmin, pmax))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)

    # Same pilot stream for the two explorations.  Their paths may diverge after the opening draw.
    oc = explore_current(np.random.default_rng([seed % (2**32), 101]), phi, pmax, pmin,
                         m, budget, conf)
    og = explore_to_guess(np.random.default_rng([seed % (2**32), 101]), phi, pmax, pmin,
                          m, budget, conf)
    if oc is None or og is None:
        return None

    _ph, _n, used_c, deep_c, n_deep_c, ph0, n0, lc, uc, probes_c_raw = oc
    probes_c = len(probes_c_raw)
    used_g, ph0_g, n0_g, deep_g, n_deep_g, last_g, _n_last_g, lg, _ug, probes_g = og
    assert np.isclose(ph0, ph0_g) and n0 == n0_g

    # arm -> (pilot estimate, pilot depth, cap, exploration spend, probe count)
    spec = {
        "re": (ph0, n0, n_sup, m * n0, 1),
        "current": (ph0, n0, n_sup, used_c, probes_c),
        "phi0_U": (ph0, n0, min(n_sup, max(int(uc), 1)), used_c, probes_c),
        "phi0_L": (ph0, n0, min(n_sup, max(int(lc), 1)), used_c, probes_c),
        "deep_L": (deep_c, n_deep_c, min(n_sup, max(int(lc), 1)), used_c, probes_c),
        "guess_deep": (deep_g, n_deep_g, n_sup, used_g, probes_g),
        "guess_deep_L": (deep_g, n_deep_g, min(n_sup, max(int(lg), 1)), used_g, probes_g),
        "guess_last_L": (last_g, max(int(_n_last_g), 1), min(n_sup, max(int(lg), 1)),
                         used_g, probes_g),
    }

    # A fresh measurement at selected L is independent of the event that selected L.  Its likelihood
    # is therefore an ordinary Eq. (3.4) likelihood conditional on L, unlike ``deep_g``.
    used_fresh = used_g + m * max(int(lg), 1)
    if used_fresh <= budget:
        fresh = simulate_errors(np.random.default_rng([seed % (2**32), 307]), phi, m,
                                max(int(lg), 1))
        spec["guess_fresh_L"] = (fresh, max(int(lg), 1), min(n_sup, max(int(lg), 1)),
                                 used_fresh, probes_g + 1)
    else:
        spec["guess_fresh_L"] = (np.nan, 1, 1, used_g, probes_g)

    chosen = {}
    raw = {}
    for arm in ARMS:
        pilot, npilot, cap, used, probes = spec[arm]
        rem = budget - used
        if rem <= 0 or not np.isfinite(pilot):
            chosen[arm] = 0
            raw[arm] = (0.0, min(used / budget, 1.0), probes, 0.0, 0.0)
            continue
        depth = risk_optimal_depth(pilot, pilot_sd(npilot, m), rem, eps,
                                   N_max=max(int(cap), 1), support=(pmin, pmax))
        shots = int(rem / depth)
        # Same (phi, N, m) gives the same exploitation draw in every arm.
        r2 = np.random.default_rng([seed % (2**32), int(depth) % (2**32),
                                    int(shots) % (2**32), 401])
        est = simulate_errors(r2, phi, shots, depth)
        chosen[arm] = depth
        raw[arm] = (float(abs(est - phi) < eps), used / budget, probes,
                    depth / n_opt, float(depth > n_opt))

    return {arm: raw[arm] + (float(chosen[arm] != chosen["re"]),) for arm in ARMS}


def _task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    sums = {a: [0.0] * NSTATS for a in ARMS}
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for seed in seeds:
            out = _one(seed, pmin, pmax, eps, budget, cfg["m"], cfg["conf"])
            n += 1
            if out is None:
                continue
            for arm in ARMS:
                for j, value in enumerate(out[arm]):
                    sums[arm][j] += value
    return sums, n


def evaluate(configs, repetitions, pmin, pmax, eps, budget, seed):
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=repetitions)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, chunk) for cfg in configs for chunk in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
        out = list(executor.map(_task, tasks, chunksize=1))
    per = {a: [] for a in ARMS}
    k = len(chunks)
    for i in range(len(configs)):
        parts = out[i * k:(i + 1) * k]
        total = sum(part[1] for part in parts)
        for arm in ARMS:
            per[arm].append(tuple(sum(part[0][arm][j] for part in parts) / total
                                  for j in range(NSTATS)))
    return per


def grid(pmax, eps):
    nmin = max(int(np.pi // (2 * pmax)), 1)
    m_hi = int(np.clip(0.6724 / (nmin * eps**2), 200, 100_000))
    ms = np.unique(np.geomspace(3, m_hi, 10).astype(int))
    return [{"m": int(m), "conf": conf} for m, conf in product(ms, [0.5, 0.8, 0.95])]


def main():
    os.makedirs("results", exist_ok=True)
    fields = ["point", "phi_min", "phi_max", "eps", "budget", "arm", "rate", "mcse_pp",
              "exploration_share", "probes", "depth_over_nopt", "overshoot", "differs_from_re",
              "m", "conf"]
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"focused BS resolution: {len(POINTS)} points, R_tune={R_TUNE}, R_test={R_TEST}", flush=True)
    for i, (label, pmin, pmax, eps, budget) in enumerate(POINTS, 1):
        configs = grid(pmax, eps)
        tuned = evaluate(configs, R_TUNE, pmin, pmax, eps, budget, SEED_TUNE)
        winners = {arm: configs[int(np.argmax([x[0] for x in tuned[arm]]))] for arm in ARMS}
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
                "exploration_share": round(stats[1], 5), "probes": round(stats[2], 4),
                "depth_over_nopt": round(stats[3], 5), "overshoot": round(100 * stats[4], 4),
                "differs_from_re": round(100 * stats[5], 4), "m": cfg["m"],
                "conf": cfg["conf"],
            }
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  {arm:15s} {100*rate:6.2f}%  probes={stats[2]:5.2f}  "
                  f"expl={100*stats[1]:5.1f}%  "
                  f"N/Nopt={stats[3]:.3f}  over={100*stats[4]:5.2f}%  "
                  f"m={cfg['m']:6d} c={cfg['conf']:.2f}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
