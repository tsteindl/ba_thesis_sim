"""Giving the bisection a job that Eq. (3.4) can already justify: unwrap the rejected probes.

THE OBSERVATION
---------------
The estimator returns phi_hat = arccos(sqrt(k/m))/N, which by the range of arccos is the distance from
N phi to the nearest multiple of pi, divided by N:

    phi_hat  ->  d(N phi)/N,     d(theta) = min_j |theta - j pi|.

So a probe does not lose the magnitude of phi when it overshoots -- it FOLDS it. The set of phases
consistent with a reading phi_hat at depth N is exactly

    A_N(phi_hat) = { j pi / N  +/-  phi_hat  :  j = 0, 1, 2, ... },

an arithmetic ladder of spacing pi/N. The overshoot test discards such a probe because arccos cannot
tell which element is the truth. But a *shallower* probe already localises phi, and if it localises it
to better than half the alias spacing, the correct element is simply the one nearest to it. Picking it
costs one subtraction, and the resulting estimate has the SAME sampling spread as an accepted probe at
that depth, sigma = 1/(2 N sqrt(m)) -- Eq. (3.4), unchanged.

That is classical phase unwrapping, and it is what makes a ladder of depths worth walking. It needs no
posterior, no grid and no new distributional assumption: it is Eq. (3.4) plus the arithmetic of Eq. (7).

THE IDENTIFIABILITY GATE
------------------------
Unwrapping probe i is only safe while the running estimate (mu, sigma) resolves the alias spacing. The
nearest-element rule errs iff |mu - phi| > pi/(2 N_i), which under Eq. (3.4) has probability
2[1 - Phi(pi/(2 N_i sigma))]. Requiring that to stay below the same alpha = 1 - conf the overshoot test
already uses gives the gate

    pi / (2 N_i sigma)  >=  z_{(1+conf)/2} ,

i.e. no new tuned constant -- the bisection's own confidence level does double duty.

ARMS (all with the existing Eq. 3.8 safeguard, none with the exact posterior)

    first     pilot = opening probe at N_min                        [shipped; identical to RE]
    deep      pilot = deepest ACCEPTED probe
    reflect   pilot = deepest probe overall, reflected to pi/N - phi_hat when it was rejected
    unwrap    sequential unwrap + inverse-variance pooling of every gated probe
    unwrapd   as unwrap, but the pilot is the single deepest gated probe (no pooling)
    bracket   pilot = opening probe, prior truncated to (pi/2U, pi/2L) by the bisection's bracket

Diagnostics recorded for every arm: overshoot rate, N*/N_opt, how often the depth differs from
`first`, the pilot's spread relative to `first`'s, the pilot's error in units of its own claimed sd
(calibration -- a branch error shows up here as a blow-up), and the branch-error rate itself.

    python analysis/binary_unwrap_study.py [--quick] [--stride N] [--geo]
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
from binary_story_study import explore as explore_arith      # verified == _binary_search_explore
from binary_geometric_study import explore_geo

QUICK = "--quick" in sys.argv
GEO = "--geo" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 10
R_TUNE = 400 if QUICK else 1000
R_TEST = 4000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
ARMS = ["first", "deep", "reflect", "unwrap", "unwrapd", "bracket"]
OUT = "results/binary_unwrap%s.csv" % ("_geo" if GEO else "")
EXPLORE = explore_geo if GEO else explore_arith


def alias_candidates(phi_hat, N, pmin, pmax):
    """Every phase consistent with the reading: {j pi/N +/- phi_hat} inside the prior support."""
    if not np.isfinite(phi_hat):
        return np.empty(0)
    j = np.arange(0, int(N * pmax / np.pi) + 2)
    c = np.concatenate([j * np.pi / N + phi_hat, j * np.pi / N - phi_hat])
    return c[(c >= pmin) & (c <= pmax)]


def unwrap_chain(probes, m, pmin, pmax, conf):
    """Walk the probes in order, unwrapping each one the running estimate can still resolve.

    Returns (mu_pool, sd_pool, mu_deep, sd_deep, n_used, N_deep_used) where the *_pool pair is the
    inverse-variance combination of every gated probe and the *_deep pair is the single deepest one.
    """
    z = norm.ppf(0.5 * (1.0 + conf))
    (N0, ph0, _a0) = probes[0]
    mu, prec = float(ph0), 1.0 / pilot_sd(N0, m) ** 2
    mu_d, N_d = float(ph0), int(N0)
    used = 1
    for N, ph, _acc in probes[1:]:
        sd_run = 1.0 / np.sqrt(prec)
        if z * sd_run > np.pi / (2.0 * N):        # alias spacing no longer resolvable -> skip
            continue
        cand = alias_candidates(ph, N, pmin, pmax)
        if cand.size == 0:
            continue
        c = float(cand[np.argmin(np.abs(cand - mu))])
        w = 1.0 / pilot_sd(N, m) ** 2
        mu = (mu * prec + c * w) / (prec + w)
        prec += w
        used += 1
        if N > N_d:
            mu_d, N_d = c, int(N)
    return mu, 1.0 / np.sqrt(prec), mu_d, pilot_sd(N_d, m), used, N_d


def _one_trial(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    out = EXPLORE(rng, phi, pmax, pmin, m, budget, conf)
    if out is None:
        return None
    _p, _N, used, phi_acc, N_acc, phi_0, N_0, L, U, probes = out
    rem = budget - used
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_opt = max(1, int(np.pi // (2 * phi)))
    L, U = max(int(L), 1), max(int(U), 1)
    full = (pmin, pmax)

    # deepest probe overall, reflected onto the first aliased branch when the test rejected it
    Nr, phr, accr = max(probes, key=lambda p: p[0])
    ph_ref = float(phr) if accr else float(np.pi / Nr - phr)
    mu_p, sd_p, mu_d, sd_d, n_used, N_used = unwrap_chain(probes, m, pmin, pmax, conf)

    lo = min(max(np.pi / (2 * U), pmin), pmax)
    hi = min(max(np.pi / (2 * L), pmin), pmax)
    brLU = (lo, hi) if lo < hi else full
    spec = {
        "first":   (phi_0,   pilot_sd(N_0, m),   n_sup, full),
        "deep":    (phi_acc, pilot_sd(N_acc, m), n_sup, full),
        "reflect": (ph_ref,  pilot_sd(Nr, m),    n_sup, full),
        "unwrap":  (mu_p,    sd_p,               n_sup, full),
        "unwrapd": (mu_d,    sd_d,               n_sup, full),
        "bracket": (phi_0,   pilot_sd(N_0, m),   n_sup, brLU),
    }
    res, chosen = {}, {}
    for a in ARMS:
        ph, sd, cap, sup = spec[a]
        if rem <= 0 or not np.isfinite(ph):
            chosen[a] = 0
            res[a] = (0.0, used / budget, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        N = risk_optimal_depth(ph, sd, rem, eps, N_max=cap, support=sup)
        mm = int(rem / N)
        r2 = np.random.default_rng([int(seed) % (2 ** 32), 15485863])
        est = simulate_errors(r2, phi, mm, N)
        chosen[a] = N
        res[a] = (float(abs(est - phi) < eps), (used + mm * N) / budget, float(N > n_opt),
                  N / n_opt, 0.0,
                  sd / pilot_sd(N_0, m),                 # pilot spread relative to `first`
                  min(abs(ph - phi) / sd, 50.0))         # pilot error in its own claimed sd
    for a in ARMS:
        res[a] = res[a][:4] + (float(chosen[a] != chosen["first"]),) + res[a][5:]
    # exploration diagnostics, identical for every arm
    steps = len(probes)
    diag = (steps, float(sum(1 for _, _, a_ in probes if a_)), used / budget, float(steps >= 2),
            float(steps >= 4), float(n_used), float(N_used) / n_opt,
            float(abs(mu_p - phi) > 3 * sd_p))           # unwrap branch failure proxy
    return res, diag


NSTAT, NDIAG = 7, 8


def _eval_task(task):
    pmin, pmax, eps, budget, cfg, seeds = task
    acc = {a: [0.0] * NSTAT for a in ARMS}
    dg = [0.0] * NDIAG
    n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one_trial(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"])
            n += 1
            if r is None:
                for a in ARMS:
                    acc[a][1] += 1.0
                continue
            res, diag = r
            for a in ARMS:
                for i, v in enumerate(res[a]):
                    acc[a][i] += v
            for i, v in enumerate(diag):
                dg[i] += v
    return acc, dg, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed, n_jobs=None):
    n_jobs = n_jobs or E.N_JOBS
    seeds = np.random.default_rng(seed).integers(0, 2 ** 63, size=R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, cfg, ch) for cfg in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=E._CTX) as ex:
        out = list(ex.map(_eval_task, tasks, chunksize=1))
    per, diags = {a: [] for a in ARMS}, []
    k = len(chunks)
    for i in range(len(cfgs)):
        part = out[i * k:(i + 1) * k]
        tot = sum(p[2] for p in part)
        for a in ARMS:
            per[a].append(tuple(sum(p[0][a][j] for p in part) / tot for j in range(NSTAT)))
        diags.append(tuple(sum(p[1][j] for p in part) / tot for j in range(NDIAG)))
    return per, diags


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


def load_top_gap(n):
    """The n operating points where the omniscient ceiling (oracle_hl) is furthest above reverse
    engineering -- i.e. the only places where a better use of the exploration could pay for itself."""
    import collections
    by = collections.defaultdict(dict)
    meta = {}
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["setting"], int(r["budget"]))
            by[k][r["algo"]] = float(r["rate"]) * 100
            meta[k] = (float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]))
    live = [(k, v) for k, v in by.items() if 2 < v["reverse_eng_risk"] < 98]
    live.sort(key=lambda kv: -(kv[1]["oracle_hl"] - kv[1]["reverse_eng_risk"]))
    out = []
    for (st, b), v in live[:n]:
        pmin, pmax, eps = meta[(st, b)]
        out.append(dict(setting=st, phi_min=pmin, phi_max=pmax, eps=eps, budget=b,
                        re=v["reverse_eng_risk"], oracle=v["oracle_hl"], brute=v["brute"]))
    return out


DIAGNAMES = ["probes", "accepted", "explore_share", "p_ge2", "p_ge4", "unwrapped",
             "Ndeep_over_Nopt", "unwrap_fail"]
HEADER = (["setting", "phi_min", "phi_max", "eps", "budget", "re_rate"]
          + [f"{a}_{k}" for a in ARMS
             for k in ("rate", "spent", "overshoot", "depth_ratio", "differs", "sd_ratio",
                       "pilot_z", "m", "conf")]
          + [f"win_{d}" for d in DIAGNAMES])


def main():
    os.makedirs("results", exist_ok=True)
    if "--top" in sys.argv:
        pts = load_top_gap(int(sys.argv[sys.argv.index("--top") + 1]))
    else:
        pts = load_points()[::STRIDE]
        if QUICK:
            pts = pts[::20]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} points, stride {STRIDE}, explore={'geometric' if GEO else 'arithmetic'}",
          flush=True)
    last = None
    for i, p in enumerate(pts, 1):
        if p["setting"] != last:
            print(f"\n=== {p['setting']} ===", flush=True)
            last = p["setting"]
        cfgs = grid_for(p["phi_max"], p["eps"])
        tune, _ = evaluate(cfgs, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], p["budget"], SEED_TUNE)
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
            t = tested[key[(cfg["m_exploration"], cfg["conf"])]][0][a][0]
            row += [round(100 * t[0], 3), round(t[1], 4), round(100 * t[2], 3), round(t[3], 4),
                    round(100 * t[4], 3), round(t[5], 4), round(t[6], 3),
                    cfg["m_exploration"], cfg["conf"]]
            got[a] = 100 * t[0]
        best = max(got, key=got.get)
        dcfg = win[best]
        d = tested[key[(dcfg["m_exploration"], dcfg["conf"])]][1][0]
        row += [round(v, 4) for v in d]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        extra = (f" [brute {p['brute']:.2f} oracle {p['oracle']:.2f} gap "
                 f"{p['oracle'] - p['re']:+.2f}]") if "oracle" in p else ""
        print(f"  [{i}/{len(pts)}] B={p['budget']:>13,} RE {100*re_rate:6.2f}{extra} | "
              + " ".join(f"{a} {got[a]:6.2f}" for a in ARMS)
              + f" | best={best} probes={d[0]:.1f} unwrapped={d[5]:.1f} expl={100*d[2]:.0f}%",
              flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
