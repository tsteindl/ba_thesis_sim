"""What does Algorithm 5's exploration actually DO? The numbers a discussion section must report.

Reporting a bisection without reporting how often it bisects is the blunder this script exists to
prevent. For every scenario, at three budgets spanning the live range, it tunes binary search over the
same grid the sweep uses (with the m floor lowered to 3 -- see the note below) and then reports, at the
winning configuration:

    probes         mean number of circuits the exploration ran (1.00 = no bisection happened at all)
    p_ge2, p_ge4   share of trials taking at least 2 / at least 4 probes
    steps_useful   mean number of probes that actually moved the bracket
    explore_share  mean fraction of the budget spent before the exploitation shot
    N_over_Nopt    mean exploitation depth as a fraction of the true aliasing limit
    p_exact        share of trials landing exactly at integer N_opt
    p_within_1     share of trials landing within 1% of N_opt
    p_at_opt       share of trials landing within 5% of N_opt (legacy column name)
    p_within_10    share of trials landing within 10% of N_opt
    overshoot      share of trials with N > N_opt (the estimator aliases)
    L_over_Nopt    mean bisection lower bound as a fraction of N_opt (>1 means L itself aliased)

THE m FLOOR. analysis/extensive_sweep.py tunes reverse engineering and binary search over
m_exploration >= 20 but linear search over >= 3. In scenarios where N_min is large relative to the
budget the pilot then costs most of the budget by construction: at U(1e-4,1e-3), eps=1e-4, B=36,435 the
m=20 pilot costs 31,400 of 36,435 (86%), and lowering the floor to 3 moves reverse engineering from
28.3% to 76.0%. Two of 23 scenarios are affected this way (the other is U(1e-3,1e-2), eps=1e-3, +46.8
pp); elsewhere the median effect is -0.03 pp. This script uses the corrected floor.

    python analysis/binary_diagnostics.py [--quick]
"""
import csv
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_story_study import explore          # verified == _binary_search_explore

QUICK = "--quick" in sys.argv
R_TUNE = 300 if QUICK else 1000
R_TEST = 3000 if QUICK else 15_000
OUT = "results/binary_diagnostics.csv"
FIELDS = ["rate", "probes", "p_ge2", "p_ge4", "steps_useful", "explore_share",
          "N_over_Nopt", "p_exact", "p_within_1", "p_at_opt", "p_within_10",
          "overshoot", "L_over_Nopt"]


def _one(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = explore(rng, phi, pmax, pmin, m, budget, conf)
    if out is None:
        return None
    _p, _N, used, _pa, _Na, phi_0, N_0, L, _U, probes = out
    rem = budget - used
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    if rem <= 0:
        return (0.0, len(probes), float(len(probes) >= 2), float(len(probes) >= 4),
                0.0, used / budget, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                max(int(L), 1) / n_opt)
    N = risk_optimal_depth(phi_0, pilot_sd(N_0, m), rem, eps, N_max=n_sup, support=(pmin, pmax))
    mm = int(rem / N)
    est = simulate_errors(np.random.default_rng([int(seed) % (2 ** 32), 15485863]), phi, mm, N)
    # a probe is "useful" if it moved the bracket, i.e. every probe after the opening one
    rel = abs(N - n_opt) / n_opt
    return (float(abs(est - phi) < eps), len(probes), float(len(probes) >= 2),
            float(len(probes) >= 4), float(len(probes) - 1), used / budget,
            N / n_opt, float(N == n_opt), float(rel <= 0.01), float(rel <= 0.05),
            float(rel <= 0.10), float(N > n_opt), max(int(L), 1) / n_opt)


def _task(t):
    pmin, pmax, eps, budget, cfg, seeds = t
    acc = [0.0] * len(FIELDS)
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                continue
            for i, v in enumerate(r):
                acc[i] += v
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed):
    seeds = np.random.default_rng(seed).integers(0, 2 ** 63, size=R)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    k = len(chunks)
    res = []
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[1] for p in part)
        res.append([sum(p[0][j] for p in part) / tot for j in range(len(FIELDS))])
    return res


def grid_for(pmax, eps):
    n_min = max(1, int(np.floor(np.pi / (2 * pmax))))
    m_hi = int(np.clip(0.6724 / (n_min * eps ** 2), 200, 300_000))
    m_b = np.unique(np.geomspace(3, m_hi, 14).astype(int))
    return [{"m_exploration": int(m), "conf": c} for m, c in product(m_b, [0.5, 0.8, 0.95])]


def main():
    by = collections.defaultdict(list)
    meta = {}
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] != "binary_risk":
                continue
            if 0.05 < float(r["rate"]) < 0.95:
                by[r["setting"]].append(int(r["budget"]))
                meta[r["setting"]] = (float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]))
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["setting", "budget", "m", "conf"] + FIELDS)
    print(f"{'setting':28s} {'budget':>15s} {'m*':>7s} {'rate':>6s} {'probes':>7s} {'>=2':>6s} "
          f"{'>=4':>6s} {'expl%':>6s} {'N/Nopt':>7s} {'exact':>6s} {'within5':>7s} "
          f"{'over%':>6s} {'L/Nopt':>7s}",
          flush=True)
    for s in sorted(by):
        bs = sorted(by[s])
        if not bs:
            continue
        pick = [bs[0], bs[len(bs) // 2], bs[-1]] if len(bs) >= 3 else bs
        pmin, pmax, eps = meta[s]
        for B in pick:
            cfgs = grid_for(pmax, eps)
            tune = evaluate(cfgs, R_TUNE, pmin, pmax, eps, B, 42)
            cfg = cfgs[int(np.argmax([t[0] for t in tune]))]
            d = evaluate([cfg], R_TEST, pmin, pmax, eps, B, 2024)[0]
            with open(OUT, "a", newline="") as f:
                csv.writer(f).writerow([s, B, cfg["m_exploration"], cfg["conf"]]
                                       + [round(v, 4) for v in d])
            print(f"{s[:28]:28s} {B:15,d} {cfg['m_exploration']:7d} {100*d[0]:6.2f} {d[1]:7.2f} "
                  f"{100*d[2]:5.1f}% {100*d[3]:5.1f}% {100*d[5]:5.1f}% {d[6]:7.3f} "
                  f"{100*d[7]:5.1f}% {100*d[9]:6.1f}% {100*d[11]:5.1f}% {d[12]:7.3f}",
                  flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
