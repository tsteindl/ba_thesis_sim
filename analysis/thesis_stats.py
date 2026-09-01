"""The reporting table for binary search with the deepest-accepted pilot.

Joins results/binary_deep.csv (rate, overshoot, depth ratio, probes, pilot sharpening, pilot aliasing)
with results/story_curves.csv (brute-force baseline and the omniscient ceiling oracle_hl at the same
operating point) and results/binary_diagnostics.csv (exploration budget share), and emits one row per
operating point plus a per-scenario summary.

    python analysis/thesis_stats.py
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

ARM = "deep_1s"
R_MEASURE = 20_000


def _task(t):
    """Exploration diagnostics measured AT the deep-pilot tuned configuration (not joined in)."""
    pmin, pmax, eps, B, m, conf, seeds = t
    acc = [0.0] * 6
    n = 0
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    n_min = min(max(int(np.pi // (2 * pmax)), 1), n_sup)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = float(rng.uniform(pmin, pmax))
            o = explore(rng, phi, pmax, pmin, m, B, conf)
            if o is None:
                continue
            _p, _N, used, phi_acc, N_acc, _p0, _N0, L, U, probes = o
            rem = B - used
            n_opt = max(1, int(np.pi // (2 * phi)))
            if rem <= 0 or not np.isfinite(phi_acc):
                continue
            N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m), rem, eps,
                                   N_min=min(n_min, n_sup), N_max=n_sup, support=(pmin, pmax))
            n += 1
            acc[0] += used / B                       # exploration budget share
            acc[1] += len(probes)                    # probes
            acc[2] += N / n_opt                      # depth ratio
            acc[3] += abs(N - n_opt) / n_opt         # mean absolute distance from N_opt
            acc[4] += float(N > n_opt)               # overshoot
            acc[5] += float(N_acc > n_opt)           # pilot aliased
    return acc, n


def measure(pmin, pmax, eps, B, m, conf):
    seeds = np.random.default_rng(2024).integers(0, 2 ** 63, size=R_MEASURE)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, [(pmin, pmax, eps, B, m, conf, c) for c in chunks], chunksize=1))
    tot = sum(o[1] for o in out)
    return [sum(o[0][i] for o in out) / tot for i in range(6)]


def main():
    ref, meta = collections.defaultdict(dict), {}
    for x in csv.DictReader(open("results/story_curves.csv")):
        ref[(x["setting"], int(x["budget"]))][x["algo"]] = float(x["rate"]) * 100
    rows = list(csv.DictReader(open("results/binary_deep.csv")))
    out = []
    for r in rows:
        k = (r["setting"], int(r["budget"]))
        v = ref.get(k, {})
        g = lambda f: float(r[f"{ARM}_{f}"])
        mm = measure(float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]),
                     int(r["budget"]), int(r[f"{ARM}_m"]), float(r[f"{ARM}_conf"]))
        d = dict(setting=r["setting"], budget=int(r["budget"]),
                 rate=g("rate"), re=float(r["re_rate"]),
                 brute=v.get("brute", float("nan")), oracle=v.get("oracle_hl", float("nan")),
                 overshoot=g("overshoot"), depth_ratio=g("depth_ratio"),
                 dist_from_nopt=100 * mm[3],
                 probes=g("probes"), accepted=g("accepted"), sd_gain=g("sd_gain"),
                 pilot_alias=g("pilot_alias"),
                 explore_share=100 * mm[0],
                 m=int(r[f"{ARM}_m"]), conf=float(r[f"{ARM}_conf"]))
        d["vs_brute"] = d["rate"] - d["brute"]
        d["share_of_hl"] = 100 * d["rate"] / d["oracle"] if d["oracle"] else float("nan")
        out.append(d)

    with open("results/thesis_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    hdr = (f"{'scenario':26s} {'budget':>14s} {'rate':>6s} {'brute':>6s} {'+vs':>6s} "
           f"{'oracle':>6s} {'%HL':>6s} {'over%':>6s} {'N*/Nopt':>8s} {'probes':>7s} "
           f"{'expl%':>6s} {'pilot x':>8s} {'palias%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for d in out:
        print(f"{d['setting'][:26]:26s} {d['budget']:14,d} {d['rate']:6.2f} {d['brute']:6.2f} "
              f"{d['vs_brute']:+6.2f} {d['oracle']:6.2f} {d['share_of_hl']:5.1f}% "
              f"{d['overshoot']:5.2f}% {d['depth_ratio']:8.3f} {d['probes']:7.2f} "
              f"{d['explore_share']:5.1f}% {d['sd_gain']:7.2f}x {d['pilot_alias']:7.2f}%")

    a = {k: np.array([d[k] for d in out], dtype=float) for k in out[0] if k not in ("setting",)}
    fin = lambda v: v[np.isfinite(v)]
    print("\n=== pooled ===")
    print(f"  convergence rate                 mean {a['rate'].mean():6.2f}%   "
          f"(reverse engineering {a['re'].mean():.2f}%, brute force {fin(a['brute']).mean():.2f}%)")
    print(f"  improvement over brute force     mean {fin(a['vs_brute']).mean():+6.2f} pp   "
          f"median {np.median(fin(a['vs_brute'])):+.2f} pp")
    print(f"  share of the omniscient ceiling  mean {fin(a['share_of_hl']).mean():6.1f}%   "
          f"median {np.median(fin(a['share_of_hl'])):.1f}%   min {fin(a['share_of_hl']).min():.1f}%")
    print(f"  overshoot rate (N* > N_opt)      mean {a['overshoot'].mean():6.2f}%   "
          f"median {np.median(a['overshoot']):.2f}%   max {a['overshoot'].max():.2f}%")
    print(f"  exploitation depth N*/N_opt      mean {a['depth_ratio'].mean():6.3f}   "
          f"median {np.median(a['depth_ratio']):.3f}   mean |N*-N_opt|/N_opt "
          f"{a['dist_from_nopt'].mean():.1f}%")
    print(f"  probes taken                     mean {a['probes'].mean():6.2f}   "
          f"accepted {a['accepted'].mean():.2f}   1 probe at {100*(a['probes']<1.5).mean():.0f}% of points")
    print(f"  exploration budget share         mean {fin(a['explore_share']).mean():6.1f}%   "
          f"median {np.median(fin(a['explore_share'])):.1f}%")
    print(f"  pilot sharper than RE's          mean {a['sd_gain'].mean():6.2f}x  "
          f"max {a['sd_gain'].max():.2f}x")
    print(f"  pilot itself aliased             mean {a['pilot_alias'].mean():6.2f}%")
    print("\nwrote results/thesis_stats.csv")


if __name__ == "__main__":
    main()
