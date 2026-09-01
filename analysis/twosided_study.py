"""Can the overshoot criterion be repaired instead of replaced?

The exact posterior (qmetrology/posterior.py) works, but it replaces the estimator-plus-normal-law
machinery of Chapter 3 wholesale. This asks whether a much smaller change to the EXISTING criterion
recovers most of the benefit -- specifically, whether it stops binary search's pilot from being
poisoned, so that the exploration still does real work and the algorithm does not collapse into
reverse engineering.

THE DIAGNOSIS. Eq. (3.5) flags a probe when it reads LOW:

    flag  if  phi_hat < phi_1 = quantile_{1-alpha}( N(phi_prev, sigma_prev^2) )

That is one-sided, and aliasing is not. Past the aliasing point the estimator returns pi/N - phi:
a reflection, which for N well beyond N_opt lands essentially anywhere in [0, pi/(2N)] -- ABOVE the
previous estimate as often as below. A bisection's second probe sits at N ~ N_max/2, i.e. far beyond
N_opt, so it is precisely the probe most likely to alias, and a one-sided test accepts it whenever the
reflection happens to read high. That single spurious acceptance sets L and the "deepest accepted"
pilot for the whole run.

THE FIX. Test both tails: a probe is consistent with the running estimate only if it is neither too
low nor too high.

    flag  if  phi_hat < phi_1  OR  phi_hat > phi_2,
    phi_1, phi_2 = the alpha/2 and 1-alpha/2 quantiles of N(phi_prev, sigma_prev^2)

One line, the same distributional argument, no new theory. Note it cannot help AT N_opt: there the two
branches coincide (pi/N - phi = phi when N = pi/2phi), so no test can separate them -- but there the
estimate is correct anyway. The damage is done far from N_opt, which is exactly where a two-sided test
bites.

Arms (all sharing one exploration per trial, so paired):
    deep_1s   deepest accepted probe as pilot, one-sided test   <- Chapter 3 as written
    deep_2s   deepest accepted probe as pilot, TWO-sided test
    L_2s      N = L with the two-sided test (is the bracket reliable now?)
    first     opening probe as pilot (degenerates to RE)
    post      exact posterior over all probes                    <- the alternative being weighed

    python analysis/twosided_study.py [--quick] [--stride N]
"""
import csv, os, sys
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
STRIDE = int(sys.argv[sys.argv.index("--stride")+1]) if "--stride" in sys.argv else 4
R_TUNE = 400 if QUICK else 1500
R_TEST = 4000 if QUICK else 30_000
SEED_TUNE, SEED_TEST = 42, 2024
ARMS = ["deep_1s", "deep_2s", "L_2s", "first", "post"]
OUT = "results/twosided.csv"


def explore(rng, phi, phi_max, phi_min, m, budget, conf, two_sided):
    """Algorithm 5's bisection; `two_sided` adds the upper-tail check to Eq. (3.5)."""
    N_min = max(np.pi // (2*phi_max), 1); N_max = max(np.pi // (2*phi_min), 1)
    N = max(1, N_min)
    if m*N > budget: return None
    k = int(rng.binomial(m, np.cos(N*phi)**2))
    ph = float(np.arccos(np.sqrt(k/m))/N); used = m*N
    sd = np.sqrt(1/(4*m*N**2))
    lo = norm.ppf((1-conf)/2 if two_sided else 1-conf, ph, sd)
    hi = norm.ppf(1-(1-conf)/2, ph, sd) if two_sided else np.inf
    probes = [(N, k)]; acc = [(N, ph)]
    phi_0, N_0 = ph, N
    lb, ub = N_min, N_max
    N += (ub-N)//2
    done = False
    while not done:
        if used + m*N > budget: break
        k = int(rng.binomial(m, np.cos(N*phi)**2))
        ph = float(np.arccos(np.sqrt(k/m))/N); used += m*N
        probes.append((N, k)); tN = N
        if ph < lo or ph > hi:
            N -= (N-lb)//2; ub = tN
        else:
            acc.append((tN, ph)); N += (ub-N)//2; lb = tN
            sd = np.sqrt(1/(4*m*N**2))
            lo = norm.ppf((1-conf)/2 if two_sided else 1-conf, ph, sd)
            hi = norm.ppf(1-(1-conf)/2, ph, sd) if two_sided else np.inf
        if tN == N or N < N_min or N > N_max or used >= budget: done = True
    return probes, acc, used, phi_0, N_0, lb


def _one(seed, pmin, pmax, eps, budget, m, conf):
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(pmin, pmax))
    nsup = max(int(np.pi//(2*pmin)), 1); nopt = max(1, int(np.pi//(2*phi)))
    nlo = min(max(int(np.pi//(2*pmax)), 1), nsup)
    out = {}
    for two in (False, True):
        r = explore(np.random.default_rng(int(seed)), phi, pmax, pmin, m, budget, conf, two)
        out[two] = r
    if out[False] is None or out[True] is None: return None
    res = {}
    depths = {}
    p1, a1, u1, f1, n1, L1 = out[False]
    p2, a2, u2, f2, n2, L2 = out[True]
    if budget - u1 > 0:
        pa, Na = a1[-1]
        depths["deep_1s"] = (risk_optimal_depth(pa, pilot_sd(Na, m), budget-u1, eps,
                                                N_min=nlo, N_max=nsup,
                                                support=(pmin, pmax)), u1)
        depths["first"] = (risk_optimal_depth(f1, pilot_sd(n1, m), budget-u1, eps,
                                              N_min=nlo, N_max=nsup,
                                              support=(pmin, pmax)), u1)
        depths["post"] = (depth_from_posterior(p1, m, pmin, pmax, budget-u1, eps, N_max=nsup), u1)
    if budget - u2 > 0:
        pa, Na = a2[-1]
        depths["deep_2s"] = (risk_optimal_depth(pa, pilot_sd(Na, m), budget-u2, eps,
                                                N_min=nlo, N_max=nsup,
                                                support=(pmin, pmax)), u2)
        depths["L_2s"] = (max(int(L2), 1), u2)
    for j, a in enumerate(ARMS):
        if a not in depths:
            res[a] = (0.0, budget, 0.0); continue
        N, used = depths[a]
        mm = int((budget-used)/N)
        r2 = np.random.default_rng([int(seed) % (2**32), j, 32452843])
        est = simulate_errors(r2, phi, mm, N)
        res[a] = (float(abs(est-phi) < eps), used+mm*N, float(N > nopt))
    return res


def _task(t):
    pmin, pmax, eps, budget, cfg, seeds = t
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}; n = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            r = _one(s, pmin, pmax, eps, budget, cfg["m_exploration"], cfg["conf"]); n += 1
            if r is None:
                for a in ARMS: acc[a][1] += budget
                continue
            for a in ARMS:
                c, b, o = r[a]; acc[a][0] += c; acc[a][1] += b; acc[a][2] += o
    return acc, n


def evaluate(cfgs, R, pmin, pmax, eps, budget, seed):
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=R)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS*2) if len(c)]
    tasks = [(pmin, pmax, eps, budget, c, ch) for c in cfgs for ch in chunks]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    per = {a: [] for a in ARMS}; k = len(chunks)
    for i in range(len(cfgs)):
        part = out[i*k:(i+1)*k]; tot = sum(p[1] for p in part)
        for a in ARMS:
            per[a].append((sum(p[0][a][0] for p in part)/tot,
                           sum(p[0][a][1] for p in part)/tot/budget,
                           sum(p[0][a][2] for p in part)/tot))
    return per


nmin = lambda pm: max(1, int(np.floor(np.pi/(2*pm))))
b90 = lambda pm, e: 0.6724/(nmin(pm)*e**2)


def grid_for(pmax, eps):
    m_hi = int(np.clip(b90(pmax, eps), 200, 300_000))
    return [{"m_exploration": int(m), "conf": c}
            for m, c in product(np.unique(np.geomspace(20, m_hi, 12).astype(int)),
                                [0.5, 0.8, 0.95])]


def load_points():
    pts = []
    for r in csv.DictReader(open("results/story_curves.csv")):
        if r["algo"] == "binary_risk" and 0.02 < float(r["rate"]) < 0.98:
            pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                            phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                            budget=int(r["budget"])))
    return pts


HEADER = (["setting","phi_min","phi_max","eps","budget","re_rate"]
          + [f"{a}_{k}" for a in ARMS for k in ("rate","spent","overshoot","m","conf")])

def main():
    pts = load_points()[::STRIDE]
    if QUICK: pts = pts[::30]
    csv.writer(open(OUT,"w",newline="")).writerow(HEADER)
    print(f"{len(pts)} points, stride {STRIDE}", flush=True)
    last=None
    for i,p in enumerate(pts,1):
        if p["setting"]!=last: print(f"\n=== {p['setting']} ===",flush=True); last=p["setting"]
        cfgs=grid_for(p["phi_max"],p["eps"])
        tune=evaluate(cfgs,R_TUNE,p["phi_min"],p["phi_max"],p["eps"],p["budget"],SEED_TUNE)
        ms=sorted({c["m_exploration"] for c in cfgs})
        rr=E.grid_full(RE_R,{"m_exploration":ms,"eps_target":[p["eps"]],"budget":[p["budget"]]},
                       R_TUNE,p["phi_min"],p["phi_max"],p["eps"],SEED_TUNE)
        _a,_b,rc=max(rr,key=lambda x:x[0])
        re_rate=E.success_rate(RE_R,rc,R_TEST,p["phi_min"],p["phi_max"],p["eps"],SEED_TEST)
        row=[p["setting"],p["phi_min"],p["phi_max"],p["eps"],p["budget"],round(100*re_rate,3)]
        win={a:cfgs[int(np.argmax([t[0] for t in tune[a]]))] for a in ARMS}
        uniq=[]
        for c in win.values():
            if c not in uniq: uniq.append(c)
        tested={(c["m_exploration"],c["conf"]):evaluate([c],R_TEST,p["phi_min"],p["phi_max"],
                                                        p["eps"],p["budget"],SEED_TEST) for c in uniq}
        got={}
        for a in ARMS:
            c=win[a]; t=tested[(c["m_exploration"],c["conf"])][a][0]
            row+=[round(100*t[0],3),round(t[1],4),round(100*t[2],3),c["m_exploration"],c["conf"]]
            got[a]=100*t[0]
        csv.writer(open(OUT,"a",newline="")).writerow(row)
        print(f"  [{i}/{len(pts)}] B={p['budget']:>13,} RE {100*re_rate:6.2f} | "
              + "  ".join(f"{a} {got[a]:6.2f}" for a in ARMS), flush=True)
    print(f"\nwrote {OUT}")

if __name__=="__main__": main()
