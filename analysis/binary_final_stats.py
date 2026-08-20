"""Every reported statistic for binary search, measured at the fine_sweep winning configuration.

One source of truth: results/fine_sweep.csv supplies (m', conf) at each of the 308 operating points;
this script re-runs the algorithm there and records the exploration and depth diagnostics, then joins
the brute-force baseline and the omniscient ceiling from results/story_curves.csv.

    python analysis/binary_final_stats.py
"""
import collections
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors
from binary_story_study import explore

R = 20_000
FIELDS = ["probes", "accepted", "explore_share", "pilot_gain", "pilot_alias",
          "depth_ratio", "abs_dist", "at_opt", "overshoot", "L_over_Nopt"]


def _task(t):
    pmin, pmax, eps, B, m, conf, seeds = t
    acc = [0.0] * len(FIELDS)
    n = 0
    n_min = max(int(np.pi // (2 * pmax)), 1)
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = float(rng.uniform(pmin, pmax))
            o = explore(rng, phi, pmax, pmin, m, B, conf)
            if o is None:
                continue
            _p, _N, used, phi_acc, N_acc, _p0, _N0, L, U, probes = o
            rem = B - used
            if rem <= 0 or not np.isfinite(phi_acc):
                continue
            n_opt = max(1, int(np.pi // (2 * phi)))
            N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m), rem, eps, N_max=n_sup,
                                   support=(pmin, pmax))
            N = max(N, min(n_min, n_sup))
            if int(rem / N) < 1:
                continue
            n += 1
            for i, v in enumerate((len(probes), sum(1 for _, _, a in probes if a), used / B,
                                   N_acc / n_min, float(N_acc > n_opt), N / n_opt,
                                   abs(N - n_opt) / n_opt, float(abs(N - n_opt) <= 0.05 * n_opt),
                                   float(N > n_opt), max(int(L), 1) / n_opt)):
                acc[i] += v
    return acc, n


def measure(pmin, pmax, eps, B, m, conf):
    seeds = np.random.default_rng(2024).integers(0, 2 ** 63, size=R)
    ch = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, [(pmin, pmax, eps, B, m, conf, c) for c in ch], chunksize=1))
    tot = sum(o[1] for o in out)
    return [sum(o[0][i] for o in out) / tot for i in range(len(FIELDS))]


def main():
    ref = collections.defaultdict(dict)
    for x in csv.DictReader(open("results/story_curves.csv")):
        ref[(x["setting"], int(x["budget"]))][x["algo"]] = float(x["rate"]) * 100
    new = collections.defaultdict(dict)
    for x in csv.DictReader(open("results/fine_sweep.csv")):
        new[(x["setting"], int(x["budget"]))][x["algo"]] = float(x["rate"]) * 100
    rows = [r for r in csv.DictReader(open("results/fine_sweep.csv")) if r["algo"] == "binary_deep"]
    out = []
    for r in rows:
        k = (r["setting"], int(r["budget"]))
        conf = float(eval(r["params"])["conf"])
        d = measure(float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]),
                    int(r["budget"]), int(r["m"]), conf)
        v = ref.get(k, {})
        rec = dict(setting=r["setting"], budget=int(r["budget"]), m=int(r["m"]), conf=conf,
                   rate=new[k]["binary_deep"], re=new[k]["re_fixed"], lin=new[k]["linear_fixed"],
                   brute=v.get("brute", float("nan")), oracle=v.get("oracle_hl", float("nan")))
        rec.update(dict(zip(FIELDS, d)))
        rec["vs_brute"] = rec["rate"] - rec["brute"]
        rec["share_hl"] = 100 * rec["rate"] / rec["oracle"] if rec["oracle"] else float("nan")
        out.append(rec)
    with open("results/binary_final_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    a = {k: np.array([o[k] for o in out], dtype=float) for k in out[0] if k != "setting"}
    fin = lambda v: v[np.isfinite(v)]
    bis = a["probes"] >= 1.5
    p = lambda k: (a[k].mean(), np.median(a[k]))
    print(f"binary search, deepest-accepted pilot — {len(out)} operating points, R={R:,}\n")
    print(f"  convergence rate                {a['rate'].mean():6.2f}%  (RE {a['re'].mean():.2f}%, "
          f"linear {a['lin'].mean():.2f}%, brute {fin(a['brute']).mean():.2f}%)")
    print(f"  improvement over brute force    {fin(a['vs_brute']).mean():+6.2f} pp  median "
          f"{np.median(fin(a['vs_brute'])):+.2f}")
    print(f"  share of the ceiling reached    {fin(a['share_hl']).mean():6.1f}%  median "
          f"{np.median(fin(a['share_hl'])):.1f}%  min {fin(a['share_hl']).min():.1f}%")
    print(f"  overshoot (N* > N_opt)          {100*a['overshoot'].mean():6.2f}%  median "
          f"{100*np.median(a['overshoot']):.2f}%  max {100*a['overshoot'].max():.2f}%")
    print(f"  depth N*/N_opt                  {a['depth_ratio'].mean():6.3f}  median "
          f"{np.median(a['depth_ratio']):.3f}")
    print(f"  mean |N*-N_opt| / N_opt         {100*a['abs_dist'].mean():6.1f}%  "
          f"(within 5% of N_opt in {100*a['at_opt'].mean():.1f}% of trials)")
    print(f"  exploration budget share        {100*a['explore_share'].mean():6.1f}%  median "
          f"{100*np.median(a['explore_share']):.1f}%  max {100*a['explore_share'].max():.1f}%")
    print(f"  probes per trial                {a['probes'].mean():6.2f}  (accepted "
          f"{a['accepted'].mean():.2f})")
    print(f"  pilot depth N_acc/N_min         {a['pilot_gain'].mean():6.2f}x  max "
          f"{a['pilot_gain'].max():.2f}x")
    print(f"  pilot itself aliased            {100*a['pilot_alias'].mean():6.2f}%")
    print(f"  bisection lower bound L/N_opt   {a['L_over_Nopt'].mean():6.3f}")
    print(f"\n  --- the {bis.sum()} points that genuinely bisect / the {(~bis).sum()} that do not ---")
    for lab, sel in (("bisecting", bis), ("single probe", ~bis)):
        print(f"  {lab:14s} rate {a['rate'][sel].mean():6.2f}%  vs RE "
              f"{(a['rate']-a['re'])[sel].mean():+5.2f} pp  probes {a['probes'][sel].mean():5.2f}  "
              f"pilot {a['pilot_gain'][sel].mean():.2f}x  explore "
              f"{100*a['explore_share'][sel].mean():.1f}%  overshoot "
              f"{100*a['overshoot'][sel].mean():.2f}%  N*/N_opt {a['depth_ratio'][sel].mean():.3f}")
    print("\nwrote results/binary_final_stats.csv")


if __name__ == "__main__":
    main()
