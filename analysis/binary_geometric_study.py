"""Giving binary search a job back, WITHOUT the exact posterior.

Diagnosis (analysis/binary_story_study.py). Algorithm 5's bisection steps to the ARITHMETIC midpoint
of [N_min, N_max]. At a wide prior that first jump is enormous -- for U(1e-4, 1e-1) it is N ~ 7861
against N_min = 15 -- so the probe costs m' * N_mid, which exceeds the whole budget, the budget guard
fires, and the exploration stops after the opening probe. The bisection is then literally reverse
engineering's pilot: same probe, same depth, at a higher price. Measured at the tuned configuration,
the exploration takes exactly 1.00 probes at 3 of 4 sampled operating points while spending 1-2 % of
the budget. It is not the safeguard that collapses binary search into reverse engineering; it is a
bisection that cannot afford its own first step.

The fix does not touch the overshoot criterion (Eq. 3.4-3.5) or the safeguard (Eq. 3.8): step to the
GEOMETRIC mean of the bracket instead of the arithmetic one. N spans orders of magnitude -- N_max/N_min
= phi_max/phi_min, up to 10^4 here -- so halving log N is the natural bisection for it, and it reaches
the same bracket precision in the same log2 number of steps while its first probe costs
m' * sqrt(N_min N_max) instead of m' * (N_min + N_max)/2.

Arms, all using the EXISTING normal-law safeguard (qmetrology/safeguard.py), none using the posterior:

    a_first      arithmetic step, pilot = opening probe, cap = prior support   [shipped, = RE]
    g_first      geometric  step, pilot = opening probe, cap = prior support
    g_deep       geometric  step, pilot = deepest ACCEPTED probe
    g_capU       geometric, pilot = opening,        cap = U (shallowest rejected)
    g_deep_capU  geometric, pilot = deepest accepted, cap = U
    g_capL       geometric, pilot = opening,        cap = L (deepest accepted)
    g_deep_capL  geometric, pilot = deepest accepted, cap = L
    g_bracket    geometric, pilot = opening, prior truncated to (pi/2U, pi/2L)

Both explorations are run on every trial from the same drawn phi (independent shot streams), and the
exploitation draw depends only on (trial, N, m), so two arms choosing the same depth get the same shot
record. `differs` reports how often an arm's depth differs from a_first's -- i.e. whether the bisection
changed the decision at all, which is the question the story turns on.

    python analysis/binary_geometric_study.py [--quick] [--stride N]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_story_study import explore as explore_arith   # verified == _binary_search_explore

QUICK = "--quick" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 10
R_TUNE = 400 if QUICK else 1000
R_TEST = 4000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
ARMS = ["a_first", "g_first", "g_deep", "g_capU", "g_deep_capU", "g_capL", "g_deep_capL", "g_bracket"]
MIN_PROBES = 3          # what "the bisection actually ran" means
# Every arm is reported twice: tuned freely, and tuned subject to the exploration taking at least
# MIN_PROBES probes on average. The gap between the two is the price of keeping a real bisection.
OUT_ARMS = ARMS + [a + "_f" for a in ARMS]
OUT = "results/binary_geometric.csv"


def explore_geo(rng, phi, phi_max, phi_min, m, budget, conf):
    """The bisection of Algorithm 5 with a geometric step. Everything else is unchanged: the same
    overshoot test phi_hat < phi_1 with phi_1 from Eq. (3.4), the same before-spending budget guard,
    the same termination when the bracket stops moving."""
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)
    N = N_min
    if m * N > budget:
        return None
    ph = simulate_errors(rng, phi, m, N)
    used = m * N
    phi_1 = norm.ppf(1 - conf, ph, np.sqrt(1 / (4 * m * N ** 2)))
    phi_acc, N_acc, phi_0, N_0 = ph, N, ph, N
    probes = [(N, ph, True)]
    lb, ub = N_min, N_max
    N = max(int(round(np.sqrt(lb * ub))), lb + 1) if ub > lb else lb
    while True:
        if used + m * N > budget:
            break
        ph = simulate_errors(rng, phi, m, N)
        used += m * N
        t = N
        if ph < phi_1:
            probes.append((t, ph, False))
            ub = t
        else:
            probes.append((t, ph, True))
            phi_acc, N_acc = ph, t
            lb = t
            phi_1 = norm.ppf(1 - conf, ph, np.sqrt(1 / (4 * m * t ** 2)))
        N = int(round(np.sqrt(lb * ub)))
        if N <= lb or N >= ub or t == N or used >= budget:
            break
    return ph, N, used, phi_acc, N_acc, phi_0, N_0, lb, ub, probes


def _depth(spec, rem, eps):
    ph, sd, cap, sup = spec
    return risk_optimal_depth(ph, sd, rem, eps, N_max=cap, support=sup)


def _one_trial(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    full = (pmin, pmax)
    specs, useds = {}, {}

    oa = explore_arith(np.random.default_rng([int(seed) % (2 ** 32), 1]), phi, pmax, pmin, m,
                       budget, conf)
    og = explore_geo(np.random.default_rng([int(seed) % (2 ** 32), 2]), phi, pmax, pmin, m,
                     budget, conf)
    if oa is None or og is None:
        return None
    for tag, o in (("a", oa), ("g", og)):
        _p, _N, used, phi_acc, N_acc, phi_0, N_0, L, U, probes = o
        L, U = max(int(L), 1), max(int(U), 1)
        sd0, sdA = pilot_sd(N_0, m), pilot_sd(N_acc, m)
        lo = min(max(np.pi / (2 * U), pmin), pmax)
        hi = min(max(np.pi / (2 * L), pmin), pmax)
        brLU = (lo, hi) if lo < hi else full
        useds[tag] = used
        specs[f"{tag}_first"] = (phi_0, sd0, n_sup, full)
        specs[f"{tag}_deep"] = (phi_acc, sdA, n_sup, full)
        specs[f"{tag}_capU"] = (phi_0, sd0, min(n_sup, U), full)
        specs[f"{tag}_deep_capU"] = (phi_acc, sdA, min(n_sup, U), full)
        specs[f"{tag}_capL"] = (phi_0, sd0, min(n_sup, L), full)
        specs[f"{tag}_deep_capL"] = (phi_acc, sdA, min(n_sup, L), full)
        specs[f"{tag}_bracket"] = (phi_0, sd0, n_sup, brLU)
        specs[f"{tag}_nprobe"] = len(probes)

    res, chosen = {}, {}
    for a in ARMS:
        used = useds[a[0]]
        rem = budget - used
        if rem <= 0:
            chosen[a] = 0
            res[a] = (0.0, used / budget, 0.0, 0.0, 0.0, specs[f"{a[0]}_nprobe"])
            continue
        N = _depth(specs[a], rem, eps)
        mm = int(rem / N)
        r2 = np.random.default_rng([int(seed) % (2 ** 32), 15485863])
        est = simulate_errors(r2, phi, mm, N)
        chosen[a] = N
        res[a] = (float(abs(est - phi) < eps), (used + mm * N) / budget, float(N > n_opt),
                  N / n_opt, 0.0, specs[f"{a[0]}_nprobe"])
    for a in ARMS:
        res[a] = res[a][:4] + (float(chosen[a] != chosen["a_first"]),) + res[a][5:]
    return res


NSTAT = 6


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
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


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed, n_jobs=None):
    n_jobs = n_jobs or E.N_JOBS
    seeds = np.random.default_rng(seed).integers(0, 2 ** 63, size=R)
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
            per[a].append(tuple(sum(p[0][a][j] for p in part) / tot for j in range(NSTAT)))
    return per


def grid_for(pmax, eps):
    n_min = max(1, int(np.floor(np.pi / (2 * pmax))))
    m_hi = int(np.clip(0.6724 / (n_min * eps ** 2), 200, 300_000))
    m_b = np.unique(np.geomspace(5, m_hi, 13).astype(int))
    return [{"m_exploration": int(m), "conf": c} for m, c in product(m_b, [0.5, 0.8, 0.95])]


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
          + [f"{a}_{k}" for a in OUT_ARMS
             for k in ("rate", "spent", "overshoot", "depth_ratio", "differs", "probes", "m", "conf")])


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[::STRIDE]
    if QUICK:
        pts = pts[::20]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} points, stride {STRIDE}, R_tune={R_TUNE} R_test={R_TEST}", flush=True)
    last = None
    for i, p in enumerate(pts, 1):
        if p["setting"] != last:
            print(f"\n=== {p['setting']} ===", flush=True)
            last = p["setting"]
        cfgs = grid_for(p["phi_max"], p["eps"])
        tune = evaluate(cfgs, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], p["budget"], SEED_TUNE)
        ms = sorted({c["m_exploration"] for c in cfgs})
        rr = E.grid_full(RE_R, {"m_exploration": ms, "eps_target": [p["eps"]],
                                "budget": [p["budget"]]}, R_TUNE, p["phi_min"], p["phi_max"],
                         p["eps"], SEED_TUNE)
        _r, _b, rcfg = max(rr, key=lambda x: x[0])
        re_rate = E.success_rate(RE_R, rcfg, R_TEST, p["phi_min"], p["phi_max"], p["eps"], SEED_TEST)
        row = [p["setting"], p["phi_min"], p["phi_max"], p["eps"], p["budget"], round(100 * re_rate, 3)]
        win = {a: cfgs[int(np.argmax([t[0] for t in tune[a]]))] for a in ARMS}
        for a in ARMS:                      # same arm, tuned under the >= MIN_PROBES constraint
            ok = [i for i, t in enumerate(tune[a]) if t[5] >= MIN_PROBES]
            if not ok:                      # no configuration bisects at all here
                ok = [int(np.argmax([t[5] for t in tune[a]]))]
            win[a + "_f"] = cfgs[max(ok, key=lambda i: tune[a][i][0])]
        uniq = []
        for c in win.values():
            if c not in uniq:
                uniq.append(c)
        tested = {id(c): evaluate([c], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                                  SEED_TEST) for c in uniq}
        key = {(c["m_exploration"], c["conf"]): id(c) for c in uniq}
        got = {}
        for a in OUT_ARMS:
            cfg = win[a]
            t = tested[key[(cfg["m_exploration"], cfg["conf"])]][a.removesuffix("_f")][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3), round(t[3], 4),
                    round(100 * t[4], 3), round(t[5], 2), cfg["m_exploration"], cfg["conf"]]
            got[a] = 100 * t[0]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] B={p['budget']:>13,} RE {100*re_rate:6.2f} | "
              + " ".join(f"{a.replace('_','')} {got[a]:6.2f}" for a in ARMS) + " || forced "
              + " ".join(f"{a.replace('_','')} {got[a + '_f']:6.2f}" for a in ("g_first", "g_deep",
                                                                              "g_deep_capU", "g_capL"))
              + f" | best={max(got, key=got.get)}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
