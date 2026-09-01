"""How should N_guess (the upper end of the Eq. 3.8 search) be chosen in binary search?

Run AFTER the budget guard was added to the exploration loops and AFTER the pilot was switched to the
opening probe phi_0, so it measures the algorithm as it now stands. Three choices:

  support   N_guess = N_max = floor(pi/(2 phi_min)) -- the prior support. The bisection then only
            serves to buy a pilot; it does not restrict the depth.
  L         N_guess = L, the bisection's lower bound (deepest probe not flagged as an overshoot).
  min       N_guess = min(L, N_max).

All three share the identical exploration on every trial (common random numbers), so the comparison is
paired: the only thing that differs is the cap applied to the same Eq. (3.8) search. Each variant is
grid-tuned over the exact grid analysis/extensive_sweep.py uses under --max (26 exploration sizes x 5
confidences) on seed 42, then validated on seed 2024 at R = 40,000.

The mean budget consumed is recorded for every arm: a comparison is only meaningful at equal cost.

    python analysis/binary_depth_sweep.py [--quick] [--resume] [--stride N]
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import _binary_search_explore
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

QUICK = "--quick" in sys.argv
RESUME = "--resume" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 2
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
CAPS = ["support", "L", "min"]
OUT = "results/binary_depth_sweep.csv"


def _one_trial(seed, pmin, pmax, eps, budget, m_exploration, conf):
    """One trial, evaluated under all three caps off a single shared exploration.

    Returns {cap: (converged, spent, N, overshot)} or None if the trial was refused.
    """
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = _binary_search_explore(rng, phi, pmax, pmin, m_exploration, budget, conf)
    if out is None:
        return None
    _ph, _Nb, used, _pa, _Na, phi_0, N_0, L, _U, _hist = out
    rem = budget - used
    n_opt = max(1, int(np.pi // (2 * phi)))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    caps = {"support": n_sup, "L": max(int(L), 1), "min": min(max(int(L), 1), n_sup)}
    res = {}
    for k, (name, cap) in enumerate(caps.items()):
        if rem <= 0 or not np.isfinite(phi_0):
            res[name] = (0.0, used, 0, 0.0)
            continue
        N = risk_optimal_depth(phi_0, pilot_sd(N_0, m_exploration), rem, eps,
                               N_min=min(max(int(np.pi // (2 * pmax)), 1), cap),
                               N_max=cap, support=(pmin, pmax))
        m = int(rem / N)
        # an exploitation stream per variant: the exploration is shared (paired), the final shots
        # are independent so one variant's draw cannot leak into another's
        r2 = np.random.default_rng([int(seed) % (2**32), k, 7919])
        ph = simulate_errors(r2, phi, m, N)
        res[name] = (float(abs(ph - phi) < eps), used + m * N, N, float(N > n_opt))
    return res


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    acc = {c: [0.0, 0.0, 0.0] for c in CAPS}   # successes, spent, overshoots
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one_trial(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                for c in CAPS:
                    acc[c][1] += budget      # a refused trial still costs the budget it was given
                continue
            for c in CAPS:
                conv, spent, _N, over = r[c]
                acc[c][0] += conv
                acc[c][1] += spent
                acc[c][2] += over
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed, n_jobs=None):
    """{cap: [(rate, spent_ratio, overshoot_rate) per cfg]} — all caps from one pass over the trials."""
    n_jobs = n_jobs or E.N_JOBS
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=E._CTX) as ex:
        out = list(ex.map(_eval_task, tasks, chunksize=2))
    per = {c: [] for c in CAPS}
    k = len(chunks)
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[1] for p in part)
        for c in CAPS:
            s = sum(p[0][c][0] for p in part)
            b = sum(p[0][c][1] for p in part)
            o = sum(p[0][c][2] for p in part)
            per[c].append((s / tot, b / tot / budget, o / tot))
    return per


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def grid_for(pmax, eps):
    m_hi = int(np.clip(brute90(pmax, eps), 200, 300_000))
    m_b = np.unique(np.geomspace(20, m_hi, 26).astype(int))
    return [{"m_exploration": int(m), "conf": c}
            for m, c in product(m_b, [0.5, 0.65, 0.8, 0.9, 0.95])]


def load_points():
    """Budgets where binary search was live in the previous sweep — the interesting range."""
    pts = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] != "binary_risk":
                continue
            if not (0.01 < float(r["rate"]) < 0.99):
                continue
            pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                            phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                            budget=int(r["budget"])))
    return pts


HEADER = (["setting", "phi_min", "phi_max", "eps", "budget"]
          + [f"{c}_{k}" for c in CAPS for k in ("rate", "spent", "overshoot", "m", "conf")])


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[::STRIDE]
    if QUICK:
        pts = pts[::40]
    done = set()
    if RESUME and os.path.exists(OUT):
        with open(OUT, newline="") as f:
            done = {(r["setting"], int(r["budget"])) for r in csv.DictReader(f)}
        pts = [p for p in pts if (p["setting"], p["budget"]) not in done]
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} operating points ({len(done)} cached), stride {STRIDE}", flush=True)

    last = None
    for i, p in enumerate(pts, 1):
        if p["setting"] != last:
            print(f"\n=== {p['setting']} ===", flush=True)
            last = p["setting"]
        cfgs = grid_for(p["phi_max"], p["eps"])
        tune = evaluate(cfgs, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], p["budget"], SEED_TUNE)
        row = [p["setting"], p["phi_min"], p["phi_max"], p["eps"], p["budget"]]
        summary = []
        for c in CAPS:
            j = int(np.argmax([t[0] for t in tune[c]]))
            best = cfgs[j]
            test = evaluate([best], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                            SEED_TEST)[c][0]
            row += [round(100 * test[0], 3), round(test[1], 4), round(100 * test[2], 3),
                    best["m_exploration"], best["conf"]]
            summary.append(f"{c} {100*test[0]:6.2f}")
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] B={p['budget']:>15,}  " + "  ".join(summary), flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
