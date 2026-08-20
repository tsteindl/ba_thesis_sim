"""Re-run the two algorithms whose numbers change, and rebuild the results-chapter tables.

WHAT CHANGES AND WHAT DOES NOT
  binary_deep   Algorithm 5 with the deepest-ACCEPTED probe as the pilot (results/ALGORITHM_BINARY.md)
                and the safeguard searching N in [N_min, N_max].
  re_fixed      Algorithm 6 unchanged, but tuned over m' >= 3 instead of the sweep's m' >= 20.
  unchanged     brute, separable, linear (its grid already starts at 3) and oracle_hl carry over from
                results/story_curves.csv -- brute/separable/oracle need no tuning, and linear was never
                subject to the m' floor.

THE m' FLOOR. analysis/extensive_sweep.py builds the exploration grid as geomspace(20, m_hi) for
reverse engineering and binary search but geomspace(3, m_hi) for linear search. Where N_min is large
relative to the budget a 20-shot pilot is unaffordable by construction: at U(1e-4,1e-3), eps=1e-4,
B=36,435 it costs 20*1570 = 31,400 of 36,435 (86%). Two of 23 scenarios are affected (+47.7 and
+46.8 pp); elsewhere the median effect is -0.03 pp.

    python analysis/results_tables.py [--quick] [--budgets K]
"""
import collections
import csv
import os
import sys
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import experiments as E
from qmetrology.algorithms import find_phi_fixed_budget_reverse_engineering_risk as RE_R
from binary_deep_study import ARMS, evaluate, grid_for  # noqa: F401  (grid_for rebuilt below)

QUICK = "--quick" in sys.argv
K_BUD = int(sys.argv[sys.argv.index("--budgets") + 1]) if "--budgets" in sys.argv else 16
R_TUNE = 300 if QUICK else 800
R_TEST = 3000 if QUICK else 20_000
SEED_TUNE, SEED_TEST = 42, 2024
CONFS = [0.5, 0.7, 0.8, 0.9, 0.95]
N_M = 16
OUT = "results/new_curves.csv"


def m_grid(pmax, eps):
    n_min = max(1, int(np.floor(np.pi / (2 * pmax))))
    m_hi = int(np.clip(0.6724 / (n_min * eps ** 2), 200, 300_000))
    return np.unique(np.geomspace(3, m_hi, N_M).astype(int))


def cfgs_for(pmax, eps):
    return [{"m_exploration": int(m), "conf": c} for m, c in product(m_grid(pmax, eps), CONFS)]


def load_curves():
    cur = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            cur[r["setting"]][r["algo"]].append((int(r["budget"]), float(r["rate"])))
            meta[r["setting"]] = (float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]))
    return cur, meta


def main():
    cur, meta = load_curves()
    settings = sorted(cur)
    if QUICK:
        settings = settings[:2]
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["setting", "phi_min", "phi_max", "eps", "budget", "algo", "rate",
                                "m", "conf"])
    for si, s in enumerate(settings, 1):
        pmin, pmax, eps = meta[s]
        buds = sorted(b for b, _ in cur[s]["brute"])
        idx = np.unique(np.linspace(0, len(buds) - 1, min(K_BUD, len(buds))).astype(int))
        buds = [buds[i] for i in idx]
        print(f"\n=== [{si}/{len(settings)}] {s} — {len(buds)} budgets ===", flush=True)
        for B in buds:
            cf = cfgs_for(pmax, eps)
            tune = evaluate(cf, R_TUNE, pmin, pmax, eps, B, SEED_TUNE)
            arm = "deep_1s"
            wc = cf[int(np.argmax([t[0] for t in tune[arm]]))]
            bt = evaluate([wc], R_TEST, pmin, pmax, eps, B, SEED_TEST)[arm][0][0]
            ms = sorted({c["m_exploration"] for c in cf})
            rr = E.grid_full(RE_R, {"m_exploration": ms, "eps_target": [eps], "budget": [B]},
                             R_TUNE, pmin, pmax, eps, SEED_TUNE)
            _r, _b, rcfg = max(rr, key=lambda x: x[0])
            rt = E.success_rate(RE_R, rcfg, R_TEST, pmin, pmax, eps, SEED_TEST)
            with open(OUT, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([s, pmin, pmax, eps, B, "binary_deep", round(bt, 6),
                            wc["m_exploration"], wc["conf"]])
                w.writerow([s, pmin, pmax, eps, B, "re_fixed", round(rt, 6),
                            rcfg["m_exploration"], ""])
            print(f"   B={B:>15,}  binary_deep {100*bt:6.2f}  re_fixed {100*rt:6.2f}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
