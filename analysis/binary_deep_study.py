"""Binary search as a sequentially sharpening pilot: a large-grid study of the story we want to tell.

THE STORY UNDER TEST. Algorithm 5's bisection walks to deeper and deeper non-aliased depths, so the
deepest ACCEPTED probe is a pilot at depth N_acc >> N_min, and by Eq. (3.4) its spread
sigma = 1/(2 N_acc sqrt(m')) is smaller than reverse engineering's by exactly N_acc/N_min. Feeding that
pilot to Eq. (3.8) is what makes binary search a different computation from reverse engineering rather
than the same one with a smaller exploitation budget.

Two things decide whether the story survives contact with the data:
  (i)  does the sharper pilot actually buy convergence, once BOTH variants are tuned properly?
  (ii) can the acceptance test be made strict enough that the pilot is essentially never aliased,
       without the bisection becoming so conservative that the pilot stops being deep?

THE ACCEPTANCE TEST. Algorithm 5 flags a probe when phi_hat < phi_1 with
phi_1 = norm.ppf(1 - conf, phi_hat_prev, sigma_new): a ONE-SIDED test at alpha = 1 - conf that treats
the previous estimate as exact and charges all the noise to the new probe. Three variants are run,
each with its own exploration (independent shot streams, common phi per trial):

    1s   one-sided at alpha = 1 - conf, sigma = sigma_new                      [shipped]
    2s   two-sided at alpha = 1 - conf: reject below phi_prev - z_{1-a/2} s
         AND above phi_prev + z_{1-a/2} s. Note the lower threshold alone is just `1s` at a higher
         conf, which the conf grid already covers -- so what `2s` isolates is the UPPER rejection
         region, i.e. whether a probe reading too HIGH should also be treated as suspect.
    1c   one-sided, but with the honest combined spread sigma = sqrt(sigma_prev^2 + sigma_new^2)

GRID. The statistical safeguard removed the `safeguard`/`s` parameter, so only (m_exploration, conf)
remain and the grid can be much finer than the sweep's: 20 exploration sizes x 9 confidence levels =
180 configurations, tuned per operating point AND per arm on seed 42, tested on seed 2024.

Also reported, for each arm, the config that maximises the rate SUBJECT TO the exploitation depth
overshooting N_opt in at most MAX_OVER of trials -- the "keep the overshoot probability very low"
version of the story, at its measured cost.

    python analysis/binary_deep_study.py [--quick] [--stride N]
"""
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
from qmetrology.algorithms import find_phi_fixed_budget_reverse_engineering_risk as RE_R

QUICK = "--quick" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 20
R_TUNE = 400 if QUICK else 1000
R_TEST = 4000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
TESTS = ["1s", "2s", "1c"]
PILOTS = ["first", "deep"]
ARMS = [f"{p}_{t}" for t in TESTS for p in PILOTS]
MAX_OVER = 0.01          # "very low overshoot" = at most 1% of trials alias
CONFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
N_M = 20
OUT = "results/binary_deep.csv"
NSTAT = 8


def explore(rng, phi, phi_max, phi_min, m, budget, conf, test):
    """Algorithm 5's bisection under one of the three acceptance tests.

    Identical to _binary_search_explore in every other respect: same opening probe at N_min, same
    arithmetic step, same guard that tests the budget BEFORE spending it, same termination.
    """
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)
    N = N_min
    if m * N > budget:
        return None
    ph = simulate_errors(rng, phi, m, N)
    used = m * N
    prev, N_prev = ph, N
    phi_acc, N_acc, phi_0, N_0 = ph, N, ph, N
    n_probes, n_acc = 1, 1
    lb, ub = N_min, N_max
    N += (ub - N) // 2
    a = 1.0 - conf
    while True:
        if used + m * N > budget:
            break
        ph = simulate_errors(rng, phi, m, N)
        used += m * N
        n_probes += 1
        t = N
        s_new = pilot_sd(N, m)
        if test == "1s":
            lo, hi = prev + norm.ppf(a) * s_new, np.inf
        elif test == "2s":
            z = norm.ppf(1.0 - a / 2.0)
            lo, hi = prev - z * s_new, prev + z * s_new
        else:                                   # "1c": honest combined spread
            s = np.sqrt(pilot_sd(N_prev, m) ** 2 + s_new ** 2)
            lo, hi = prev + norm.ppf(a) * s, np.inf
        ok = (ph >= lo) and (ph <= hi) and np.isfinite(ph)
        if not ok:
            N -= (N - lb) // 2
            ub = t
        else:
            n_acc += 1
            phi_acc, N_acc = ph, t
            N += (ub - N) // 2
            lb = t
            prev, N_prev = ph, t
        if t == N or N < N_min or N > N_max or used >= budget:
            break
    return used, phi_acc, N_acc, phi_0, N_0, n_probes, n_acc


def _one_trial(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    n_min = max(int(np.pi // (2 * pmax)), 1)
    res = {}
    for ti, test in enumerate(TESTS):
        o = explore(np.random.default_rng([int(seed) % (2 ** 32), ti + 1]), phi, pmax, pmin, m,
                    budget, conf, test)
        if o is None:
            return None
        used, phi_acc, N_acc, phi_0, N_0, n_probes, n_acc = o
        rem = budget - used
        for pil in PILOTS:
            a = f"{pil}_{test}"
            ph, N0 = (phi_0, N_0) if pil == "first" else (phi_acc, N_acc)
            if rem <= 0 or not np.isfinite(ph):
                res[a] = (0.0, used / budget, 0.0, 0.0, n_probes, n_acc, N_acc / n_min,
                          float(N_acc > n_opt))
                continue
            N = risk_optimal_depth(ph, pilot_sd(N0, m), rem, eps, N_min=min(n_min, n_sup),
                                   N_max=n_sup, support=(pmin, pmax))
            mm = int(rem / N)
            est = simulate_errors(np.random.default_rng([int(seed) % (2 ** 32), 15485863]),
                                  phi, mm, N)
            res[a] = (float(abs(est - phi) < eps), (used + mm * N) / budget, float(N > n_opt),
                      N / n_opt, n_probes, n_acc, N_acc / n_min, float(N_acc > n_opt))
    return res


def _task(t):
    pmin, pmax, eps, budget, cfg, seeds = t
    acc = {a: [0.0] * NSTAT for a in ARMS}
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one_trial(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                for a in ARMS:
                    acc[a][1] += 1.0
                continue
            for a in ARMS:
                for i, v in enumerate(r[a]):
                    acc[a][i] += v
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed):
    seeds = np.random.default_rng(seed).integers(0, 2 ** 63, size=R)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    k = len(chunks)
    per = {a: [] for a in ARMS}
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[1] for p in part)
        for a in ARMS:
            per[a].append(tuple(sum(p[0][a][j] for p in part) / tot for j in range(NSTAT)))
    return per


def grid_for(pmax, eps):
    n_min = max(1, int(np.floor(np.pi / (2 * pmax))))
    m_hi = int(np.clip(0.6724 / (n_min * eps ** 2), 200, 300_000))
    ms = np.unique(np.geomspace(3, m_hi, N_M).astype(int))
    return [{"m_exploration": int(m), "conf": c} for m, c in product(ms, CONFS)]


def load_points():
    pts = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "binary_risk" and 0.05 < float(r["rate"]) < 0.95:
                pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                                phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                                budget=int(r["budget"])))
    return pts


KEYS = ("rate", "spent", "overshoot", "depth_ratio", "probes", "accepted", "sd_gain", "pilot_alias")
HEADER = (["setting", "phi_min", "phi_max", "eps", "budget", "re_rate", "n_cfg"]
          + [f"{a}_{k}" for a in ARMS for k in KEYS + ("m", "conf")]
          + [f"{a}_safe_{k}" for a in ARMS for k in ("rate", "overshoot", "m", "conf")])


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[::STRIDE]
    if QUICK:
        pts = pts[::12]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} points | grid {N_M} m x {len(CONFS)} conf = {N_M*len(CONFS)} cfgs "
          f"| R_tune={R_TUNE} R_test={R_TEST}", flush=True)
    for i, p in enumerate(pts, 1):
        cfgs = grid_for(p["phi_max"], p["eps"])
        tune = evaluate(cfgs, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], p["budget"], SEED_TUNE)
        ms = sorted({c["m_exploration"] for c in cfgs})
        rr = E.grid_full(RE_R, {"m_exploration": ms, "eps_target": [p["eps"]],
                                "budget": [p["budget"]]}, R_TUNE, p["phi_min"], p["phi_max"],
                         p["eps"], SEED_TUNE)
        _r, _b, rcfg = max(rr, key=lambda x: x[0])
        re_rate = E.success_rate(RE_R, rcfg, R_TEST, p["phi_min"], p["phi_max"], p["eps"], SEED_TEST)
        win = {a: cfgs[int(np.argmax([t[0] for t in tune[a]]))] for a in ARMS}
        for a in ARMS:                       # same arm, constrained to a very low overshoot rate
            ok = [j for j, t in enumerate(tune[a]) if t[2] <= MAX_OVER]
            if not ok:
                ok = [int(np.argmin([t[2] for t in tune[a]]))]
            win[a + "_safe"] = cfgs[max(ok, key=lambda j: tune[a][j][0])]
        uniq = []
        for c in win.values():
            if c not in uniq:
                uniq.append(c)
        tested = {id(c): evaluate([c], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                                  SEED_TEST) for c in uniq}
        key = {(c["m_exploration"], c["conf"]): id(c) for c in uniq}
        row = [p["setting"], p["phi_min"], p["phi_max"], p["eps"], p["budget"],
               round(100 * re_rate, 3), len(cfgs)]
        got = {}
        for a in ARMS:
            c = win[a]
            t = tested[key[(c["m_exploration"], c["conf"])]][a][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3), round(t[3], 4),
                    round(t[4], 2), round(t[5], 2), round(t[6], 3), round(100 * t[7], 3),
                    c["m_exploration"], c["conf"]]
            got[a] = 100 * t[0]
        for a in ARMS:
            c = win[a + "_safe"]
            t = tested[key[(c["m_exploration"], c["conf"])]][a][0]
            row += [round(100 * t[0], 3), round(100 * t[2], 3), c["m_exploration"], c["conf"]]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] {p['setting'][:22]:22s} B={p['budget']:>13,} "
              f"RE {100*re_rate:6.2f} | "
              + " ".join(f"{a} {got[a]:6.2f}" for a in ARMS)
              + f" | best={max(got, key=got.get)}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
