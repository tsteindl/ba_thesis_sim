"""Can binary search keep the Eq. (3.8) safeguard AND still be more than reverse engineering?

With the pilot taken from the opening probe at N_min and the depth search capped only by the prior
support, Algorithm 5's bisection contributes nothing to the depth decision: it buys the same pilot
reverse engineering buys, at a higher price. This script measures every way of giving the bisection a
job that does NOT require the exact-posterior machinery -- i.e. every variant that Eq. (3.4) and
Eq. (3.8) already justify as written.

All arms share ONE exploration per trial (common random numbers), so the comparison is paired: the only
thing that differs is what is read off that exploration.

    first        pilot = opening probe (N_min),      cap = prior support     [the degenerate shipped one]
    deep         pilot = deepest ACCEPTED probe,     cap = prior support
    capU         pilot = opening probe,              cap = U  (shallowest REJECTED probe)
    capL         pilot = opening probe,              cap = L  (deepest accepted probe)
    deep_capU    pilot = deepest accepted,           cap = U
    deep_capL    pilot = deepest accepted,           cap = L
    bracketU     pilot = opening probe, prior TRUNCATED from below to phi > pi/(2U), cap = support
    bracket      pilot = opening probe, prior truncated to (pi/2U, pi/2L),           cap = support

The bisection brackets N_opt from both sides: L is the deepest probe it accepted (evidence phi < pi/2L)
and U the shallowest it rejected (evidence phi > pi/2U). capU/bracketU use only the SAFE side -- a
type-I error there costs precision, never aliasing -- while capL/bracket use the side whose errors
force an overshoot.

The exploration is a verbatim copy of qmetrology.algorithms._binary_search_explore (asserted identical
in verify(), which compares every returned quantity over 2,000 paired trials); it exists only to also
return U and the accept flags, which the shipped function does not.

    python analysis/binary_story_study.py [--quick] [--stride N] [--verify]
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
from qmetrology.algorithms import (_binary_search_explore,
                                   find_phi_fixed_budget_reverse_engineering_risk as RE_R)

QUICK = "--quick" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 8
R_TUNE = 400 if QUICK else 1200
R_TEST = 4000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
ARMS = ["first", "deep", "capU", "capL", "deep_capU", "deep_capL", "bracketU", "bracket"]
OUT = "results/binary_story.csv"


def explore(rng, phi, phi_max, phi_min, m, budget, conf):
    """_binary_search_explore, additionally returning ub and the accept flag of every probe."""
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)
    if m * N > budget:
        return None
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used = m * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N**2)))
    phi_acc, N_acc = phi_hat, N
    phi_0, N_0 = phi_hat, N
    probes = [(N, phi_hat, True)]
    lb, ub = N_min, N_max
    N += (ub - N) // 2
    done = False
    while not done:
        if budget_used + m * N > budget:
            break
        phi_hat = simulate_errors(rng, phi, m, N)
        budget_used += m * N
        temp_N = N
        if phi_hat < phi_1:
            probes.append((temp_N, phi_hat, False))
            N -= (N - lb) // 2
            ub = temp_N
        else:
            probes.append((temp_N, phi_hat, True))
            phi_acc, N_acc = phi_hat, temp_N
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N**2)))
        if temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True
    return phi_hat, N, budget_used, phi_acc, N_acc, phi_0, N_0, lb, ub, probes


def verify(R=2000):
    """explore() must reproduce _binary_search_explore exactly, value for value."""
    bad = 0
    for s in range(R):
        pmin, pmax = [(1e-4, 1e-1), (1e-3, 1e-2), (1e-2, 1e-1)][s % 3]
        m, conf = [20, 60, 385][s % 3], [0.5, 0.8, 0.95][s % 3]
        B = [10_000, 900_000, 50_000_000][s % 3]
        r1 = np.random.default_rng(s)
        phi = float(r1.uniform(pmin, pmax))
        a = _binary_search_explore(r1, phi, pmax, pmin, m, B, conf)
        r2 = np.random.default_rng(s)
        phi2 = float(r2.uniform(pmin, pmax))
        b = explore(r2, phi2, pmax, pmin, m, B, conf)
        if (a is None) != (b is None):
            bad += 1
            continue
        if a is None:
            continue
        # (phi_hat, N, used, phi_acc, N_acc, phi_0, N_0, lb) must match; b has ub, probes appended
        if not all(np.allclose(x, y, equal_nan=True) for x, y in zip(a[:8], b[:8])):
            bad += 1
    print(f"verify: {R - bad}/{R} trials identical to _binary_search_explore")
    return bad == 0


def _one_trial(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = explore(rng, phi, pmax, pmin, m, budget, conf)
    if out is None:
        return None
    _ph, _Nb, used, phi_acc, N_acc, phi_0, N_0, L, U, _probes = out
    rem = budget - used
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    L, U = max(int(L), 1), max(int(U), 1)
    sd0, sdA = pilot_sd(N_0, m), pilot_sd(N_acc, m)
    full = (pmin, pmax)
    # the bisection's evidence, expressed as a phi interval: rejected at U => phi > pi/2U,
    # accepted at L => phi < pi/2L. Clipped to the prior and guarded against an empty intersection.
    lo = min(max(np.pi / (2 * U), pmin), pmax)
    hi = min(max(np.pi / (2 * L), pmin), pmax)
    brU = (lo, pmax) if lo < pmax else full
    brLU = (lo, hi) if lo < hi else full

    spec = {
        "first":     (phi_0,   sd0, n_sup,          full),
        "deep":      (phi_acc, sdA, n_sup,          full),
        "capU":      (phi_0,   sd0, min(n_sup, U),  full),
        "capL":      (phi_0,   sd0, min(n_sup, L),  full),
        "deep_capU": (phi_acc, sdA, min(n_sup, U),  full),
        "deep_capL": (phi_acc, sdA, min(n_sup, L),  full),
        "bracketU":  (phi_0,   sd0, n_sup,          brU),
        "bracket":   (phi_0,   sd0, n_sup,          brLU),
    }
    n_lo = min(max(int(np.pi // (2 * pmax)), 1), n_sup)
    res, chosen = {}, {"first": 0}
    for k, a in enumerate(ARMS):
        if rem <= 0:
            chosen[a] = 0
            res[a] = (0.0, used, 0.0, 0.0, 0.0)
            continue
        ph, sd, cap, sup = spec[a]
        N = risk_optimal_depth(ph, sd, rem, eps, N_min=min(n_lo, cap), N_max=cap, support=sup)
        mm = int(rem / N)
        # common random numbers: the exploitation draw depends on the trial and on (N, m), not on
        # which arm asked for it, so two arms that choose the same depth get literally the same shot
        # record and an arm-to-arm difference reflects the depth decision alone.
        r2 = np.random.default_rng([int(seed) % (2**32), 15485863])
        est = simulate_errors(r2, phi, mm, N)
        chosen[a] = N
        res[a] = (float(abs(est - phi) < eps), used + mm * N, float(N > n_opt), N / n_opt,
                  float(N != chosen["first"]))
    return res


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    acc = {a: [0.0] * 5 for a in ARMS}
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
                for i, v in enumerate(r[a]):
                    acc[a][i] += v
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
            per[a].append(tuple(sum(p[0][a][j] for p in part) / tot for j in range(5)))
    return per


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def grid_for(pmax, eps):
    m_hi = int(np.clip(0.6724 / (n_min(pmax) * eps ** 2), 200, 300_000))
    m_b = np.unique(np.geomspace(20, m_hi, 12).astype(int))
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
          + [f"{a}_{k}" for a in ARMS
             for k in ("rate", "spent", "overshoot", "depth_ratio", "differs", "m", "conf")])


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[::STRIDE]
    if QUICK:
        pts = pts[::20]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} operating points, stride {STRIDE}, R_tune={R_TUNE} R_test={R_TEST}", flush=True)
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
        uniq = []
        for c in win.values():
            if c not in uniq:
                uniq.append(c)
        tested = {id(c): evaluate([c], R_TEST, p["phi_min"], p["phi_max"], p["eps"], p["budget"],
                                  SEED_TEST) for c in uniq}
        key = {(c["m_exploration"], c["conf"]): id(c) for c in uniq}
        got = {}
        for a in ARMS:
            cfg = win[a]
            t = tested[key[(cfg["m_exploration"], cfg["conf"])]][a][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3), round(t[3], 4),
                    round(100 * t[4], 3), cfg["m_exploration"], cfg["conf"]]
            got[a] = 100 * t[0]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] B={p['budget']:>14,}  RE {100*re_rate:6.2f} | "
              + "  ".join(f"{a} {got[a]:6.2f}" for a in ARMS)
              + f" | best={max(got, key=got.get)}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        verify(400)
        main()
