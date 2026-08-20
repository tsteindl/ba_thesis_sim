"""High-replication validation of the promising 99% pooled-update rule.

Configurations are read from the independent tune/test study in
``results/conservative_update.csv``.  This replay compares each arm at its own previously selected
configuration and uses common phase draws, reporting a paired Monte Carlo standard error for the
success-rate difference.  No parameters are selected on the validation replications.
"""

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
from conservative_update_study import explore_gated


SOURCE = "results/conservative_update.csv"
POINTS = ("tight medium", "wide prior")
SEED = 7_401_991


def _finish(seed, phi, pmin, pmax, eps, budget, pilot, sigma, used):
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(int(np.pi // (2 * phi)), 1)
    remaining = budget - used
    if remaining <= 0:
        return 0.0, 0.0, 0.0, 0.0
    depth = risk_optimal_depth(pilot, sigma, remaining, eps, N_max=n_sup,
                               support=(pmin, pmax))
    shots = int(remaining / depth)
    rng = np.random.default_rng([int(seed) % (2**32), int(depth) % (2**32),
                                 int(shots) % (2**32), 499_979])
    estimate = simulate_errors(rng, phi, shots, depth)
    return (float(abs(estimate - phi) < eps), depth / n_opt, float(depth > n_opt),
            min(abs(pilot - phi) / sigma, 50.0))


def _one(seed, point, re_cfg, pool_cfg, tau):
    pmin, pmax, eps, budget = point
    seed = int(seed)
    phi = float(np.random.default_rng(seed).uniform(pmin, pmax))
    n0 = max(int(np.pi // (2 * pmax)), 1)
    stream = [seed % (2**32), 104_729]

    m_re = re_cfg[0]
    ph0 = float(simulate_errors(np.random.default_rng(stream), phi, m_re, n0))
    sig0 = pilot_sd(n0, m_re)
    re_success, re_ratio, re_over, re_z = _finish(
        seed, phi, pmin, pmax, eps, budget, ph0, sig0, m_re * n0)

    m_pool, conf_pool = pool_cfg
    gated = explore_gated(np.random.default_rng(stream), phi, pmax, pmin, m_pool, budget,
                          conf_pool, tau, "pool")
    if gated is None:
        return None
    pilot, sigma, used, probes, certified, contaminated = gated
    pool_success, pool_ratio, pool_over, pool_z = _finish(
        seed, phi, pmin, pmax, eps, budget, pilot, sigma, used)
    delta = pool_success - re_success
    return (
        re_success, pool_success, delta, delta * delta,
        used / budget, probes, certified - 1, float(certified > 1), contaminated,
        pool_ratio, pool_over, pool_z, re_ratio, re_over, re_z,
    )


def _task(args):
    point, re_cfg, pool_cfg, tau, seeds = args
    total = np.zeros(15)
    n = 0
    for seed in seeds:
        row = _one(seed, point, re_cfg, pool_cfg, tau)
        if row is not None:
            total += row
            n += 1
    return total, n


def _load(tau):
    rows = list(csv.DictReader(open(SOURCE, newline="")))
    arm_name = f"pool_{str(tau).replace('.', '')}"
    selected = {}
    for label in POINTS:
        by_arm = {r["arm"]: r for r in rows if r["point"] == label}
        re = by_arm["re"]
        pool = by_arm[arm_name]
        point = (float(re["phi_min"]), float(re["phi_max"]), float(re["eps"]),
                 int(re["budget"]))
        selected[label] = (point, (int(re["m"]), float(re["conf"])),
                           (int(pool["m"]), float(pool["conf"])))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50_000)
    parser.add_argument("--tau", type=float, choices=(0.99, 0.999), default=0.99)
    args = parser.parse_args()
    configs = _load(args.tau)
    tau_tag = str(args.tau).replace(".", "")
    arm_label = f"pool_{tau_tag}"
    out_path = ("results/conservative_update_validation.csv" if args.tau == 0.99 else
                f"results/conservative_update_validation_{tau_tag}.csv")
    os.makedirs("results", exist_ok=True)
    output = []
    for i, (label, (point, re_cfg, pool_cfg)) in enumerate(configs.items(), 1):
        seeds = np.random.default_rng(SEED).integers(0, 2**63, size=args.reps)
        chunks = [x for x in np.array_split(seeds, E.N_JOBS * 4) if len(x)]
        tasks = [(point, re_cfg, pool_cfg, args.tau, x) for x in chunks]
        with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
            parts = list(executor.map(_task, tasks, chunksize=1))
        sums = sum((x[0] for x in parts), start=np.zeros(15))
        n = sum(x[1] for x in parts)
        means = sums / n
        diff_se = np.sqrt(max(means[3] - means[2] ** 2, 0.0) / (n - 1))
        re_se = np.sqrt(means[0] * (1 - means[0]) / n)
        pool_se = np.sqrt(means[1] * (1 - means[1]) / n)
        row = {
            "point": label, "repetitions": n,
            "re_rate": 100 * means[0], "re_mcse_pp": 100 * re_se,
            f"{arm_label}_rate": 100 * means[1], f"{arm_label}_mcse_pp": 100 * pool_se,
            "paired_difference_pp": 100 * means[2], "paired_mcse_pp": 100 * diff_se,
            "paired_ci_low_pp": 100 * (means[2] - 1.96 * diff_se),
            "paired_ci_high_pp": 100 * (means[2] + 1.96 * diff_se),
            "exploration_share": means[4], "probes": means[5],
            "certified_updates": means[6], "p_updated": 100 * means[7],
            "contaminated": 100 * means[8], "pool_depth_over_nopt": means[9],
            "pool_overshoot": 100 * means[10], "pool_pilot_abs_z": means[11],
            "re_depth_over_nopt": means[12], "re_overshoot": 100 * means[13],
            "re_pilot_abs_z": means[14], "re_m": re_cfg[0], "re_conf": re_cfg[1],
            "pool_m": pool_cfg[0], "pool_conf": pool_cfg[1],
        }
        output.append(row)
        print(f"[{i}/{len(configs)}] {label}: RE={row['re_rate']:.3f}%  "
              f"{arm_label}={row[f'{arm_label}_rate']:.3f}%  "
              f"delta={row['paired_difference_pp']:+.3f} +/- "
              f"{1.96*row['paired_mcse_pp']:.3f} pp (95% MC interval), "
              f"updates={row['certified_updates']:.2f}, "
              f"contam={row['contaminated']:.3f}%", flush=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
