"""Does the exact posterior help ALL three adaptive algorithms, or only binary search?

The exact-posterior depth rule (qmetrology/posterior.py) was introduced to rescue binary search, whose
bisection otherwise contributes nothing to the depth decision. But linear search also visits many
depths, and reverse engineering's single pilot sits at N_min where the binomial is hard against its
boundary and the Gaussian of Eq. (3.4) fits badly. So both might gain too -- and if they gain as much,
binary search's advantage disappears.

Per algorithm the two variants share the identical exploration on every trial (common random
numbers), so normal-vs-posterior is paired within an algorithm. Across algorithms the comparison is
unpaired but uses the same scenarios, budgets, seeds and R.

    brute                          the baseline
    linear_s      / linear_post    tuned decrement s      vs exact posterior
    re_normal     / re_post        Eq. (3.8) normal law   vs exact posterior
    binary_first  / binary_post    Eq. (3.8) from phi_0   vs exact posterior

    python analysis/posterior_all_study.py [--quick] [--stride N]
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_io import live_points, path
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_linear_search_post as LIN_P,
    find_phi_fixed_budget_reverse_engineering_risk as RE_N,
    find_phi_fixed_budget_reverse_engineering_post as RE_P,
    find_phi_fixed_budget_binary_search_risk as BIN_N,
    find_phi_fixed_budget_binary_search_post as BIN_P,
)

QUICK = "--quick" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 4
R_TUNE = 400 if QUICK else 1500
R_TEST = 4000 if QUICK else 30_000
SEED_TUNE, SEED_TEST = 42, 2024
OUT = path("posterior_all.csv")
ALGOS = ["linear_s", "linear_post", "re_normal", "re_post", "binary_first", "binary_post"]


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def grids(pmax, eps, budget):
    m_hi = int(np.clip(brute90(pmax, eps), 200, 300_000))
    m_lin = np.unique(np.geomspace(3, m_hi, 12).astype(int))
    m_b = np.unique(np.geomspace(20, m_hi, 12).astype(int))
    b = [int(budget)]
    return {
        "linear_s": (LIN, {"m_exploration": m_lin, "lookback_window": [1, 2, 5],
                           "safeguard": [0, 1, 2, 5], "inc": [1, 2, 5], "budget": b}),
        "linear_post": (LIN_P, {"m_exploration": m_lin, "lookback_window": [1, 2, 5],
                                "inc": [1, 2, 5], "eps_target": [eps], "budget": b}),
        "re_normal": (RE_N, {"m_exploration": m_b, "eps_target": [eps], "budget": b}),
        "re_post": (RE_P, {"m_exploration": m_b, "eps_target": [eps], "budget": b}),
        "binary_first": (BIN_N, {"m_exploration": m_b, "conf": [0.5, 0.8, 0.95],
                                 "eps_target": [eps], "budget": b}),
        "binary_post": (BIN_P, {"m_exploration": m_b, "conf": [0.5, 0.8, 0.95],
                                "eps_target": [eps], "budget": b}),
    }





HEADER = (["setting", "phi_min", "phi_max", "eps", "budget", "brute"]
          + [f"{a}_{k}" for a in ALGOS for k in ("rate", "spent")])


def main():
    pts = live_points("reverse_eng_risk")[::STRIDE]
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
        args = (p["phi_min"], p["phi_max"], p["eps"])
        br = E.success_rate(BF, {"budget": p["budget"]}, R_TEST, *args, SEED_TEST)
        row = [p["setting"], *args, p["budget"], round(100 * br, 3)]
        got = {}
        for a in ALGOS:
            fn, g = grids(p["phi_max"], p["eps"], p["budget"])[a]
            res = E.grid_full(fn, g, R_TUNE, *args, SEED_TUNE)
            _r, _b, cfg = max(res, key=lambda x: x[0])
            rate, spent = E.rate_and_budget(fn, cfg, R_TEST, *args, SEED_TEST)
            row += [round(100 * rate, 3), round(spent / p["budget"], 4)]
            got[a] = 100 * rate
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        best = max(got, key=got.get)
        print(f"  [{i}/{len(pts)}] B={p['budget']:>14,} brute {100*br:6.2f} | "
              + "  ".join(f"{a.split('_')[0][:3]}{'P' if a.endswith('post') else 'N'} {got[a]:6.2f}"
                          for a in ALGOS) + f" | best={best}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
