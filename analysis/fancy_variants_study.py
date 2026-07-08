"""Sweep the binary-anneal and larger-inc variants, and check for an RE-Lite win over brute.

Part 1: at the Table-3.1 cell, tune each adaptive algorithm over an extensive grid and compare to
brute. Part 2: the best RE-Lite margin over brute across several settings and budgets.
    python analysis/fancy_variants_study.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_binary_search as BIN,
    find_phi_fixed_budget_binary_search_anneal_m as BINA,
    find_phi_fixed_budget_reverse_engineering as RE,
    find_phi_reverse_engineering_lite as RL,
)

R_TUNE, R_TEST = 3000, 40000


def best(fn, grid, pmin, pmax, eps):
    _m, arg, _ = E.grid_search_max(fn, grid, R_TUNE, pmin, pmax, eps, 42)
    return 100 * E.success_rate(fn, arg, R_TEST, pmin, pmax, eps, 2024), arg


def part1():
    pmin, pmax, eps, B = 0.01, 0.1, 1e-3, 10000
    m = np.unique(np.geomspace(3, 3000, 24).astype(int))
    mb = np.unique(np.geomspace(10, 3000, 16).astype(int))
    print("=== Part 1: Table 3.1 cell (U(0.01,0.1), eps=1e-3, budget 10,000), de-biased ===")
    brute = 100 * E.success_rate(BF, {"budget": B}, R_TEST, pmin, pmax, eps, 2024)
    print(f"  brute force (baseline)         : {brute:5.1f}%")
    grids = {
        "linear (inc fixed)": (LIN, {"m_exploration": m, "lookback_window": [1, 2, 3, 5],
                                     "safeguard": [0, 1, 2, 5], "inc": [1, 2, 3, 5, 8], "budget": [B]}),
        "binary (plain)": (BIN, {"m_exploration": mb, "safeguard": [0, 1, 2],
                                 "conf": [0.5, 0.65, 0.8, 0.9, 0.95], "budget": [B]}),
        "binary anneal_m (extensive)": (BINA, {"m_exploration": mb, "safeguard": [0, 1, 2],
                                               "conf": [0.5, 0.8, 0.9], "max_b_steps_sub": [1, 2, 3, 4],
                                               "delta": [2, 5, 10], "budget": [B]}),
        "reverse engineering": (RE, {"m_exploration": m, "safeguard": [0.8, 0.85, 0.9, 0.95], "budget": [B]}),
    }
    for name, (fn, g) in grids.items():
        rate, arg = best(fn, g, pmin, pmax, eps)
        n = int(np.prod([len(v) for v in g.values()]))
        flag = "  <- beats brute" if rate > brute + 1 else ""
        print(f"  {name:31s}: {rate:5.1f}%   ({n:,} configs){flag}")
        print(f"      best: { {k:v for k,v in arg.items() if k!='budget'} }")


def part2():
    print("\n=== Part 2: RE-Lite win-hunt (best margin over brute at matched avg budget) ===")
    M_EXP = [8, 15, 25, 40, 60, 100, 150, 250, 400]
    M_FIN = [3, 5, 8, 12, 20, 35, 60, 120, 250]
    settings = [(0.01, 0.1, 1e-3), (0.001, 0.01, 1e-4), (0.005, 0.05, 1e-4), (0.001, 0.1, 1e-4), (0.0001, 0.01, 1e-4)]
    overall = None
    for pmin, pmax, eps in settings:
        cfg = []
        for me in M_EXP:
            for mf in M_FIN:
                r, b = E.rate_and_budget(RL, {"m_exploration": me, "m_exploitation": mf}, R_TUNE, pmin, pmax, eps, 42)
                cfg.append((b, r, me, mf))
        # scan a range of budgets; best RE-lite within budget vs brute at that budget
        c = max(b for b, *_ in cfg)
        best_margin = None
        for B in np.geomspace(500, c, 14):
            cand = sorted((x for x in cfg if x[0] <= B), key=lambda x: -x[1])[:4]
            for _bt, _rt, me, mf in cand:
                rv, bv = E.rate_and_budget(RL, {"m_exploration": me, "m_exploitation": mf}, R_TEST, pmin, pmax, eps, 2024)
                if bv > B * 1.15:
                    continue
                br = 100 * E.success_rate(BF, {"budget": int(bv)}, R_TEST, pmin, pmax, eps, 2024)
                margin = 100 * rv - br
                if best_margin is None or margin > best_margin[0]:
                    best_margin = (margin, 100 * rv, br, bv, me, mf)
        mg = best_margin
        print(f"  U({pmin:g},{pmax:g}), eps={eps:.0e}: best RE-lite {mg[1]:.1f}% vs brute {mg[2]:.1f}% "
              f"(margin {mg[0]:+.1f}pp) @budget {mg[3]:,.0f}  (m'={mg[4]}, m={mg[5]})")
        if overall is None or mg[0] > overall[0]:
            overall = (mg[0], pmin, pmax, eps, mg[1], mg[2], mg[3], mg[4], mg[5])
    print(f"\n  BEST RE-lite margin over brute anywhere: {overall[0]:+.1f}pp  "
          f"(U({overall[1]:g},{overall[2]:g}), eps={overall[3]:.0e}, {overall[4]:.1f}% vs {overall[5]:.1f}% @ {overall[6]:,.0f})")


if __name__ == "__main__":
    part1()
    part2()
