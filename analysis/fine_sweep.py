"""Definitive re-tune with grid bounds that cannot bind, and two-stage refinement.

WHY. In results/story_winners.csv the winning m_exploration sits ON a grid endpoint at a large share of
operating points -- linear 37.5% at the minimum, reverse engineering 32.1%, binary search 23.5% (and
1.3-3.8% at the maximum). A boundary solution means the optimum is outside the search box, so those
numbers understate the algorithms by an unknown amount. analysis/extensive_sweep.py uses
geomspace(20, m_hi) for reverse engineering and binary search, geomspace(3, m_hi) for linear, with
m_hi = clip(brute90, 200, 300000) -- and that clip is active in 20 of 23 scenarios.

BOUNDS THAT CANNOT BIND.
    lower   m' = 1, the physical minimum (one shot).
    upper   m' = budget // N_min, the largest pilot that can be afforded at all: a probe at the opening
            depth costs m' * N_min, so beyond this the algorithm cannot take even its first probe and
            the configuration scores zero by construction.
Both endpoints are reported, so a boundary hit is now informative rather than an artifact.

TWO-STAGE SEARCH. A single grid fine enough to span 1 .. budget/N_min would be enormous, so the search
is coarse-then-fine: a log-spaced sweep over the full feasible range, then a refinement by a factor of
four either side of the stage-1 winner, at higher R. Both stages tune on seed 42; the winner is
validated on seed 2024.

    python fine_sweep.py [--quick] [--budgets K] [--algos a,b,c]
"""
import collections
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

ROOT = "/home/tsteindl/Programming/ba_thesis_sim"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors
from qmetrology.algorithms import (find_phi_fixed_budget_reverse_engineering_risk as RE_R,
                                   find_phi_fixed_budget_linear_search as LIN)
from binary_story_study import explore as bs_explore

QUICK = "--quick" in sys.argv
K_BUD = int(sys.argv[sys.argv.index("--budgets") + 1]) if "--budgets" in sys.argv else 8
ALGOS = (sys.argv[sys.argv.index("--algos") + 1].split(",") if "--algos" in sys.argv
         else ["binary_deep", "re_fixed", "linear_fixed"])
R1 = 200 if QUICK else 600
R2 = 400 if QUICK else 1500
R_TEST = 3000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
CONFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
OUT = os.path.join(ROOT, "results/fine_sweep.csv")


def m_bounds(pmax, budget):
    n_min = max(int(np.pi // (2 * pmax)), 1)
    return 1, max(2, int(budget // n_min))


def _bin_trial(seed, pmin, pmax, eps, B, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    o = bs_explore(rng, phi, pmax, pmin, m, B, conf)
    if o is None:
        return 0.0
    _p, _N, used, phi_acc, N_acc, _p0, _N0, L, U, probes = o
    rem = B - used
    if rem <= 0 or not np.isfinite(phi_acc):
        return 0.0
    n_min = max(int(np.pi // (2 * pmax)), 1)
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m), rem, eps, N_min=min(n_min, n_sup),
                           N_max=n_sup, support=(pmin, pmax))
    mm = int(rem / N)
    if mm < 1:
        return 0.0
    est = simulate_errors(np.random.default_rng([int(seed) % (2 ** 32), 15485863]), phi, mm, N)
    return float(abs(est - phi) < eps)


def _bin_task(t):
    pmin, pmax, eps, B, cfg, seeds = t
    s = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in seeds:
            s += _bin_trial(k, pmin, pmax, eps, B, cfg["m"], cfg["conf"])
    return s, len(seeds)


def eval_binary(cfgs, R, pmin, pmax, eps, B, seed):
    seeds = np.random.default_rng(seed).integers(0, 2 ** 63, size=R)
    ch = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    tasks = [(pmin, pmax, eps, B, c, k) for c in cfgs for k in ch]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_bin_task, tasks, chunksize=1))
    n = len(ch)
    return [sum(o[0] for o in out[i * n:(i + 1) * n]) / sum(o[1] for o in out[i * n:(i + 1) * n])
            for i in range(len(cfgs))]


def _score(kind, ms, disc, R, seed, pmin, pmax, eps, B):
    if kind == "binary_deep":
        cfgs = [{"m": int(m), "conf": c} for m, c in product(ms, disc)]
        return cfgs, eval_binary(cfgs, R, pmin, pmax, eps, B, seed)
    if kind == "re_fixed":
        g = {"m_exploration": [int(m) for m in ms], "eps_target": [eps], "budget": [B]}
        res = E.grid_full(RE_R, g, R, pmin, pmax, eps, seed)
        return [{"m": c["m_exploration"]} for _r, _b, c in res], [x[0] for x in res]
    g = {"m_exploration": [int(m) for m in ms], "budget": [B],
         "lookback_window": [1, 2, 5], "safeguard": [0, 1, 2], "inc": [1, 2, 5]}
    res = E.grid_full(LIN, g, R, pmin, pmax, eps, seed)
    return ([{"m": c["m_exploration"], **{k: v for k, v in c.items()
                                          if k not in ("m_exploration", "budget")}}
             for _r, _b, c in res], [x[0] for x in res])


def two_stage(kind, pmin, pmax, eps, B):
    lo, hi = m_bounds(pmax, B)
    n1 = 24 if kind != "linear_fixed" else 16
    ms1 = np.unique(np.geomspace(lo, hi, n1).astype(np.int64))
    disc = CONFS if kind == "binary_deep" else [None]
    c1, r1 = _score(kind, ms1, disc, R1, SEED_TUNE, pmin, pmax, eps, B)
    w1 = c1[int(np.argmax(r1))]
    at_lo, at_hi = w1["m"] <= ms1.min(), w1["m"] >= ms1.max()
    ms2 = np.unique(np.clip(np.geomspace(max(lo, w1["m"] / 4), min(hi, w1["m"] * 4), 13),
                            lo, hi).astype(np.int64))
    if kind == "binary_deep":
        i = CONFS.index(w1["conf"])
        disc2 = CONFS[max(0, i - 1):i + 2]
    else:
        disc2 = [None]
    c2, r2 = _score(kind, ms2, disc2, R2, SEED_TUNE, pmin, pmax, eps, B)
    win = c2[int(np.argmax(r2))]
    if kind == "binary_deep":
        rate = eval_binary([win], R_TEST, pmin, pmax, eps, B, SEED_TEST)[0]
    elif kind == "re_fixed":
        rate = E.success_rate(RE_R, {"m_exploration": win["m"], "eps_target": eps, "budget": B},
                              R_TEST, pmin, pmax, eps, SEED_TEST)
    else:
        p = {k: v for k, v in win.items() if k != "m"}
        rate = E.success_rate(LIN, {"m_exploration": win["m"], "budget": B, **p},
                              R_TEST, pmin, pmax, eps, SEED_TEST)
    return rate, win, lo, hi, at_lo, at_hi


def main():
    # only the informative band: budgets where the reference adaptive curve is neither floored nor
    # saturated. Outside it every configuration ties, the argmax is arbitrary, and a boundary flag
    # would be meaningless.
    cur = collections.defaultdict(list)
    meta = {}
    with open(os.path.join(ROOT, "results/story_curves.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "reverse_eng_risk" and 0.03 < float(r["rate"]) < 0.97:
                cur[r["setting"]].append(int(r["budget"]))
                meta[r["setting"]] = (float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]))
    settings = sorted(cur)
    if QUICK:
        settings = settings[:1]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["setting", "phi_min", "phi_max", "eps", "budget", "algo", "rate",
                                "m", "params", "m_lo", "m_hi", "at_min", "at_max"])
    for si, s in enumerate(settings, 1):
        pmin, pmax, eps = meta[s]
        buds = sorted(cur[s])
        idx = np.unique(np.linspace(0, len(buds) - 1, min(K_BUD, len(buds))).astype(int))
        buds = [buds[i] for i in idx]
        print(f"\n=== [{si}/{len(settings)}] {s} ===", flush=True)
        for B in buds:
            line = f"   B={B:>16,}"
            for a in ALGOS:
                rate, win, lo, hi, alo, ahi = two_stage(a, pmin, pmax, eps, B)
                with open(OUT, "a", newline="") as f:
                    csv.writer(f).writerow([s, pmin, pmax, eps, B, a, round(rate, 6), win["m"],
                                            str({k: v for k, v in win.items() if k != "m"}),
                                            lo, hi, int(alo), int(ahi)])
                line += (f"  {a.split('_')[0][:3]} {100*rate:6.2f} (m={win['m']:,}"
                         f"{'!LO' if alo else ''}{'!HI' if ahi else ''})")
            print(line, flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
