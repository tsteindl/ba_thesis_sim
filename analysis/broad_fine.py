"""The broad-prior scenarios (Table 3.4), re-tuned on the boundary-free grid.

analysis/fine_sweep.py covers the 23 uniform-prior scenarios of story_curves.csv (eps 1e-3 .. 1e-8).
It does NOT cover the broad priors of analysis/broad_dist_study.py, where phi_max reaches pi/4 and
pi/2 so that N_min = 1-2 and the pilot is at its least informative. That study has its own grid
asymmetry: binary search is tuned over m_grid(pmax, 10, 16) while linear search and reverse
engineering get m_grid(pmax, 3, 22) -- a higher floor and fewer points for binary.

Same treatment as fine_sweep: m' in [1, budget // N_min], two-stage refinement, boundary hits reported.

    python analysis/broad_fine.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fine_sweep as fs
from qmetrology import experiments as E
from qmetrology.algorithms import find_phi_fixed_budget_brute_force as BF
from qmetrology.oracle import heisenberg_rate

PMIN, EPS = 0.01, 1e-3
SCEN = [("pi/4", np.pi / 4), ("pi/2", np.pi / 2)]
OUT = "results/broad_fine.csv"


def main():
    budgets = sorted({int(r["budget"]) for r in csv.DictReader(open("results/broad_dist.csv"))})
    old = {}
    for r in csv.DictReader(open("results/broad_dist.csv")):
        old[(r["phi_max"], int(r["budget"]))] = r
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["phi_max", "budget", "algo", "rate", "m", "params",
                                "m_lo", "m_hi", "at_min", "at_max"])
    for tag, pmax in SCEN:
        print(f"\n=== U({PMIN}, {tag}) — N_min = {max(1, int(np.pi // (2 * pmax)))} ===", flush=True)
        for B in budgets:
            br = 100 * E.success_rate(BF, {"budget": int(B)}, 20_000, PMIN, pmax, EPS, 2024)
            oc = heisenberg_rate(int(B), EPS, PMIN, pmax)
            line = f"   B={B:>10,}  brute {br:6.2f}  ceiling {oc:6.2f}"
            for a in ("binary_deep", "re_fixed", "linear_fixed"):
                rate, win, lo, hi, alo, ahi = fs.two_stage(a, PMIN, pmax, EPS, int(B))
                with open(OUT, "a", newline="") as f:
                    csv.writer(f).writerow([tag, B, a, round(rate, 6), win["m"],
                                            str({k: v for k, v in win.items() if k != "m"}),
                                            lo, hi, int(alo), int(ahi)])
                line += (f"  {a.split('_')[0][:3]} {100*rate:6.2f}(m={win['m']:,}"
                         f"{'!LO' if alo else ''}{'!HI' if ahi else ''})")
            o = old.get((tag, B))
            if o:
                line += (f"  | old: bin_risk {float(o['binary_risk']):.2f} "
                         f"re_risk {float(o['reverse_eng_risk']):.2f} lin {float(o['linear']):.2f}")
            print(line, flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
