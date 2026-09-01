"""Can binary search be made a genuine contender again?

With the pilot taken from the opening probe and the Eq. (3.8) search capped only by the prior support,
the bisection contributes nothing to the depth decision -- binary search becomes reverse engineering
plus wasted exploration. This script tests every way of putting the bisection's information back to
work, all off the SAME exploration per trial (common random numbers), so the arms are paired:

  first     pilot = opening probe, cap = prior support            (the degenerate baseline)
  deep      pilot = deepest non-flagged probe, cap = prior support
  post      EXACT posterior over every probe (qmetrology/posterior.py), cap = prior support.
            Uses the flagged probes too: a probe reading ~0 hits at depth N says phi ~ pi/(2N), which
            is the sharpest statement the exploration produces and the one the overshoot test throws
            away. Measured sd is 6-11x tighter than the opening probe's.
  L-s       N* = max(L - s, 1) for s in 0,1,2,3,5 -- the plain additive safeguard on the bisection's
            lower bound, with no distributional argument at all.
  postL     exact posterior, but capped at L (does the bisection's bound still help once the
            posterior already knows what the bisection knows?)

Also reports reverse engineering at the same budget and m', because "beats the baseline" is not the
bar -- binary search has to beat the algorithm it currently collapses into.

    python analysis/binary_rescue_study.py [--quick] [--stride N]
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
from qmetrology.posterior import depth_from_posterior
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors
from qmetrology.algorithms import find_phi_fixed_budget_reverse_engineering_risk as RE_R

QUICK = "--quick" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 4
R_TUNE = 400 if QUICK else 1500
R_TEST = 4000 if QUICK else 30_000
SEED_TUNE, SEED_TEST = 42, 2024
S_VALUES = [0, 1, 2, 3, 5]
ARMS = ["first", "deep", "post", "postL"] + [f"L{s}" for s in S_VALUES]
OUT = "results/binary_rescue.csv"


def explore(rng, phi, phi_max, phi_min, m, budget, conf):
    """Algorithm 5's bisection, recording (N, hits) for every probe.

    Same arithmetic and same RNG consumption as qmetrology.algorithms._binary_search_explore; hits is
    recorded because it is the sufficient statistic the exact posterior needs (phi_hat throws away the
    branch information once N phi > pi/2).
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)
    if m * N > budget:
        return None
    p0 = np.cos(N * phi) ** 2
    hits = int(rng.binomial(m, p0))
    phi_hat = float(np.arccos(np.sqrt(hits / m)) / N)
    used = m * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N**2)))
    probes = [(N, hits)]
    acc = [(N, phi_hat)]
    phi_0, N_0 = phi_hat, N
    lb, ub = N_min, N_max
    N += (ub - N) // 2
    done = False
    while not done:
        if used + m * N > budget:
            break
        hits = int(rng.binomial(m, np.cos(N * phi) ** 2))
        phi_hat = float(np.arccos(np.sqrt(hits / m)) / N)
        used += m * N
        probes.append((N, hits))
        temp_N = N
        if phi_hat < phi_1:
            N -= (N - lb) // 2
            ub = temp_N
        else:
            acc.append((temp_N, phi_hat))
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N**2)))
        if temp_N == N or N < N_min or N > N_max or used >= budget:
            done = True
    return probes, acc, used, phi_0, N_0, lb


def _one_trial(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = explore(rng, phi, pmax, pmin, m, budget, conf)
    if out is None:
        return None
    probes, acc, used, phi_0, N_0, L = out
    rem = budget - used
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    depths = {}
    if rem > 0:
        n_min = min(max(int(np.pi // (2 * pmax)), 1), n_sup)
        depths["first"] = risk_optimal_depth(phi_0, pilot_sd(N_0, m), rem, eps,
                                             N_min=n_min, N_max=n_sup, support=(pmin, pmax))
        pa, Na = acc[-1]
        depths["deep"] = risk_optimal_depth(pa, pilot_sd(Na, m), rem, eps,
                                            N_min=n_min, N_max=n_sup, support=(pmin, pmax))
        depths["post"] = depth_from_posterior(probes, m, pmin, pmax, rem, eps, N_max=n_sup)
        depths["postL"] = depth_from_posterior(probes, m, pmin, pmax, rem, eps,
                                               N_max=max(int(L), 1))
        for s in S_VALUES:
            depths[f"L{s}"] = max(int(L) - s, 1)
    res = {}
    for k, a in enumerate(ARMS):
        N = depths.get(a)
        if N is None or rem <= 0:
            res[a] = (0.0, used, 0.0)
            continue
        mm = int(rem / N)
        r2 = np.random.default_rng([int(seed) % (2**32), k, 15485863])
        est = simulate_errors(r2, phi, mm, N)
        res[a] = (float(abs(est - phi) < eps), used + mm * N, float(N > n_opt))
    return res


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one_trial(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                for a in ARMS:
                    acc[a][1] += budget
                continue
            for a in ARMS:
                c, b, o = r[a]
                acc[a][0] += c
                acc[a][1] += b
                acc[a][2] += o
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed, n_jobs=None):
    n_jobs = n_jobs or E.N_JOBS
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=E._CTX) as ex:
        out = list(ex.map(_eval_task, tasks, chunksize=1))
    per = {a: [] for a in ARMS}
    k = len(chunks)
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[1] for p in part)
        for a in ARMS:
            per[a].append((sum(p[0][a][0] for p in part) / tot,
                           sum(p[0][a][1] for p in part) / tot / budget,
                           sum(p[0][a][2] for p in part) / tot))
    return per


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def grid_for(pmax, eps):
    m_hi = int(np.clip(brute90(pmax, eps), 200, 300_000))
    m_b = np.unique(np.geomspace(20, m_hi, 14).astype(int))
    return [{"m_exploration": int(m), "conf": c}
            for m, c in product(m_b, [0.5, 0.8, 0.95])]


def load_points():
    pts = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "binary_risk" and 0.02 < float(r["rate"]) < 0.98:
                pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                                phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                                budget=int(r["budget"])))
    return pts


HEADER = (["setting", "phi_min", "phi_max", "eps", "budget", "re_rate"]
          + [f"{a}_{k}" for a in ARMS for k in ("rate", "spent", "overshoot", "m", "conf")])


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[::STRIDE]
    if QUICK:
        pts = pts[::30]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} operating points, stride {STRIDE}", flush=True)
    last = None
    for i, p in enumerate(pts, 1):
        if p["setting"] != last:
            print(f"\n=== {p['setting']} ===", flush=True)
            last = p["setting"]
        cfgs = grid_for(p["phi_max"], p["eps"])
        tune = evaluate(cfgs, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], p["budget"], SEED_TUNE)
        # reverse engineering, tuned over the same exploration sizes -- the bar to beat
        ms = sorted({c["m_exploration"] for c in cfgs})
        rr = E.grid_full(RE_R, {"m_exploration": ms, "eps_target": [p["eps"]],
                                "budget": [p["budget"]]}, R_TUNE, p["phi_min"], p["phi_max"],
                         p["eps"], SEED_TUNE)
        _r, _b, rcfg = max(rr, key=lambda x: x[0])
        re_rate = E.success_rate(RE_R, rcfg, R_TEST, p["phi_min"], p["phi_max"], p["eps"], SEED_TEST)
        row = [p["setting"], p["phi_min"], p["phi_max"], p["eps"], p["budget"],
               round(100 * re_rate, 3)]
        # each arm's own winning config, then ONE test pass per distinct config: a pass already
        # evaluates every arm, so evaluating per-arm would repeat the same work up to len(ARMS) times
        win = {a: cfgs[int(np.argmax([t[0] for t in tune[a]]))] for a in ARMS}
        uniq = []
        for c in win.values():
            if c not in uniq:
                uniq.append(c)
        tested = {id(c): evaluate([c], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                                  SEED_TEST) for c in uniq}
        key = {(c["m_exploration"], c["conf"]): id(c) for c in uniq}
        best_of = {}
        for a in ARMS:
            cfg = win[a]
            t = tested[key[(cfg["m_exploration"], cfg["conf"])]][a][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3),
                    cfg["m_exploration"], cfg["conf"]]
            best_of[a] = 100 * t[0]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        top = max(best_of, key=best_of.get)
        print(f"  [{i}/{len(pts)}] B={p['budget']:>14,}  RE {100*re_rate:6.2f} | "
              f"first {best_of['first']:6.2f}  deep {best_of['deep']:6.2f}  "
              f"post {best_of['post']:6.2f}  postL {best_of['postL']:6.2f}  "
              f"bestL {max(best_of[f'L{s}'] for s in S_VALUES):6.2f} | best={top}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
