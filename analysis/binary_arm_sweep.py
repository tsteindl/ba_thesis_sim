"""Confirm the binary-search pilot choice under the CURRENT (budget-guarded) code.

analysis/binary_pilot_sweep.py answered "opening probe or deepest non-flagged probe?" for the
algorithm as it stood *before* the exploration loops were budget-guarded. The guard changed binary
search materially (+10.7 pp at the Table 3.1 point, because a large exploration size stopped being
self-defeating), so the pilot comparison has to be redone against the current code.

Arms are (pilot, N_guess) pairs, all evaluated off a single shared exploration per trial (common
random numbers), so the comparison is paired and costs one pass rather than one per arm:

    first/support   opening probe at N_min, search capped by the prior support   <- proposed default
    deep/support    deepest probe not flagged as an overshoot, same cap          <- previous default
    first/L         opening probe, search capped at the bisection's lower bound

Each arm is grid-tuned over the exact grid analysis/extensive_sweep.py uses under --max, on seed 42,
and validated on seed 2024 at R = 40,000. Mean spend is recorded per arm.

    python analysis/binary_arm_sweep.py [--quick] [--resume] [--stride N]
"""
import csv
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
ARMS = [("first", "support"), ("deep", "support"), ("first", "L")]
NAMES = [f"{p}_{c}" for p, c in ARMS]
OUT = "results/binary_arm_sweep.csv"


def _one_trial(seed, pmin, pmax, eps, budget, m_exploration, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = _binary_search_explore(rng, phi, pmax, pmin, m_exploration, budget, conf)
    if out is None:
        return None
    _ph, _Nb, used, phi_acc, N_acc, phi_0, N_0, L, _U, _hist = out
    rem = budget - used
    n_opt = max(1, int(np.pi // (2 * phi)))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_lo = min(max(int(np.pi // (2 * pmax)), 1), n_sup)
    res = {}
    for k, (pilot, cap) in enumerate(ARMS):
        ph_p, N_p = (phi_0, N_0) if pilot == "first" else (phi_acc, N_acc)
        cap_v = n_sup if cap == "support" else max(int(L), 1)
        if rem <= 0 or not np.isfinite(ph_p):
            res[NAMES[k]] = (0.0, used, 0.0)
            continue
        N = risk_optimal_depth(ph_p, pilot_sd(N_p, m_exploration), rem, eps,
                               N_min=min(n_lo, cap_v), N_max=cap_v, support=(pmin, pmax))
        m = int(rem / N)
        r2 = np.random.default_rng([int(seed) % (2**32), k, 104729])
        est = simulate_errors(r2, phi, m, N)
        res[NAMES[k]] = (float(abs(est - phi) < eps), used + m * N, float(N > n_opt))
    return res


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    acc = {n: [0.0, 0.0, 0.0] for n in NAMES}
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one_trial(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                for k in NAMES:
                    acc[k][1] += budget
                continue
            for k in NAMES:
                c, b, o = r[k]
                acc[k][0] += c
                acc[k][1] += b
                acc[k][2] += o
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed, n_jobs=None):
    n_jobs = n_jobs or E.N_JOBS
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=E._CTX) as ex:
        out = list(ex.map(_eval_task, tasks, chunksize=2))
    per = {n: [] for n in NAMES}
    k = len(chunks)
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[1] for p in part)
        for nm in NAMES:
            per[nm].append((sum(p[0][nm][0] for p in part) / tot,
                            sum(p[0][nm][1] for p in part) / tot / budget,
                            sum(p[0][nm][2] for p in part) / tot))
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
    pts = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "binary_risk" and 0.01 < float(r["rate"]) < 0.99:
                pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                                phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                                budget=int(r["budget"])))
    return pts


HEADER = (["setting", "phi_min", "phi_max", "eps", "budget"]
          + [f"{n}_{k}" for n in NAMES for k in ("rate", "spent", "overshoot", "m", "conf")])


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
        for nm in NAMES:
            j = int(np.argmax([t[0] for t in tune[nm]]))
            best = cfgs[j]
            t = evaluate([best], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                         SEED_TEST)[nm][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3),
                    best["m_exploration"], best["conf"]]
            summary.append(f"{nm} {100*t[0]:6.2f}")
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] B={p['budget']:>15,}  " + "  ".join(summary), flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
