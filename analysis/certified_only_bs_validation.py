"""High-replication, no-retuning replay of certified-only BS versus RE."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certified_only_bs_study import POINTS, explore_certified


SOURCE = "results/certified_only_bs.csv"
OUT = "results/certified_only_bs_validation.csv"
SEED = 1_907_413
TAU = 0.999


def _finish(seed, phi, pmin, pmax, eps, budget, pilot, sigma, used):
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    remaining = budget - used
    depth = risk_optimal_depth(pilot, sigma, remaining, eps, N_max=n_sup,
                               support=(pmin, pmax))
    shots = int(remaining / depth)
    rng = np.random.default_rng([int(seed) % (2**32), int(depth) % (2**32),
                                 int(shots) % (2**32), 499_979])
    estimate = simulate_errors(rng, phi, shots, depth)
    return (float(abs(estimate - phi) < eps), depth / n_opt, float(depth > n_opt),
            min(abs(pilot - phi) / sigma, 50.0))


def _one(seed, point, re_cfg, bs_cfg):
    _label, pmin, pmax, eps, budget = point
    seed = int(seed)
    phi = float(np.random.default_rng(seed).uniform(pmin, pmax))
    n0 = max(int(np.pi // (2 * pmax)), 1)
    stream = [seed % (2**32), 104_729]
    m_re = re_cfg[0]
    ph0 = float(simulate_errors(np.random.default_rng(stream), phi, m_re, n0))
    re = _finish(seed, phi, pmin, pmax, eps, budget, ph0, pilot_sd(n0, m_re), m_re*n0)

    m_bs, conf_bs = bs_cfg
    pilot, sigma, used, probes, contaminated = explore_certified(
        np.random.default_rng(stream), phi, pmax, pmin, m_bs, budget, conf_bs, TAU)
    bs = _finish(seed, phi, pmin, pmax, eps, budget, pilot, sigma, used)
    delta = bs[0] - re[0]
    return (re[0], bs[0], delta, delta * delta, used / budget, probes, probes - 1,
            float(probes > 1), contaminated, bs[1], bs[2], bs[3], re[1], re[2], re[3])


def _task(args):
    point, re_cfg, bs_cfg, seeds = args
    total = np.zeros(15)
    for seed in seeds:
        total += _one(seed, point, re_cfg, bs_cfg)
    return total, len(seeds)


def _configs():
    with open(SOURCE, newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for point in POINTS:
        label = point[0]
        by_arm = {r["arm"]: r for r in rows if r["point"] == label}
        result[label] = (point,
                         (int(by_arm["re"]["m"]), float(by_arm["re"]["conf"])),
                         (int(by_arm["stop_0999"]["m"]),
                          float(by_arm["stop_0999"]["conf"])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50_000)
    args = parser.parse_args()
    rows = []
    for i, (label, (point, re_cfg, bs_cfg)) in enumerate(_configs().items(), 1):
        seeds = np.random.default_rng(SEED).integers(0, 2**63, size=args.reps)
        chunks = [x for x in np.array_split(seeds, E.N_JOBS * 4) if len(x)]
        tasks = [(point, re_cfg, bs_cfg, x) for x in chunks]
        with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
            parts = list(executor.map(_task, tasks, chunksize=1))
        sums = sum((x[0] for x in parts), start=np.zeros(15))
        n = sum(x[1] for x in parts)
        means = sums / n
        paired_se = np.sqrt(max(means[3] - means[2]**2, 0.0) / (n - 1))
        row = {
            "point": label, "repetitions": n, "re_rate": 100 * means[0],
            "bs_rate": 100 * means[1], "paired_difference_pp": 100 * means[2],
            "paired_mcse_pp": 100 * paired_se,
            "paired_ci_low_pp": 100 * (means[2] - 1.96 * paired_se),
            "paired_ci_high_pp": 100 * (means[2] + 1.96 * paired_se),
            "exploration_share": means[4], "probes": means[5],
            "certified_updates": means[6], "p_updated": 100 * means[7],
            "contaminated": 100 * means[8], "bs_depth_over_nopt": means[9],
            "bs_overshoot": 100 * means[10], "bs_pilot_abs_z": means[11],
            "re_depth_over_nopt": means[12], "re_overshoot": 100 * means[13],
            "re_pilot_abs_z": means[14], "re_m": re_cfg[0], "re_conf": re_cfg[1],
            "bs_m": bs_cfg[0], "bs_conf": bs_cfg[1],
        }
        rows.append(row)
        print(f"[{i}/{len(POINTS)}] {label}: RE={row['re_rate']:.3f}% "
              f"BS={row['bs_rate']:.3f}% delta={row['paired_difference_pp']:+.3f} "
              f"+/- {1.96*row['paired_mcse_pp']:.3f} pp, updates={row['certified_updates']:.2f}, "
              f"contam={row['contaminated']:.3f}%", flush=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
