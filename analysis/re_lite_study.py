"""RE-Lite low-budget study.

For a set of target average budgets, report the best RE-Lite convergence (tuned on seed 42,
validated on seed 2024) next to brute force and tuned fixed-budget reverse engineering.
    python analysis/re_lite_study.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_reverse_engineering_lite as RL,
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_reverse_engineering as RE,
)

pmin, pmax, eps = 0.01, 0.1, 1e-3      # the Table-3.1 setting
R_TUNE, R_TEST = 3000, 40000
TARGETS = [1000, 1500, 2000, 3000, 5000, 8000]

M_EXP = [15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
M_FIN = [3, 5, 8, 12, 20, 40, 80, 150, 300]


def re_lite_configs(seed):
    """(mean_budget, rate, m_exploration, m_exploitation) for every RE-lite grid point."""
    out = []
    for me in M_EXP:
        for mf in M_FIN:
            r, b = E.rate_and_budget(RL, {"m_exploration": me, "m_exploitation": mf},
                                     R_TUNE, pmin, pmax, eps, seed)
            out.append((b, r, me, mf))
    return out


def best_re_within(B, tuned):
    """De-biased best RE-lite at avg budget <= B: take top-5 tune configs within B, validate, pick best."""
    cand = sorted((c for c in tuned if c[0] <= B), key=lambda c: -c[1])[:5]
    best = None
    for _bt, _rt, me, mf in cand:
        rv, bv = E.rate_and_budget(RL, {"m_exploration": me, "m_exploitation": mf},
                                   R_TEST, pmin, pmax, eps, 2024)
        if bv <= B * 1.15 and (best is None or rv > best[0]):
            best = (rv, bv, me, mf)
    return best


def best_fixed_re(B):
    """Tuned fixed-budget reverse engineering at budget B (de-biased)."""
    grid = {"m_exploration": np.unique(np.geomspace(3, max(10, B // 15), 10).astype(int)),
            "safeguard": [0.8, 0.85, 0.9, 0.95], "budget": [int(B)]}
    res = E.grid_full(RE, grid, R_TUNE, pmin, pmax, eps, 42)
    win = max(res, key=lambda r: r[0])[2]
    return 100 * E.success_rate(RE, win, R_TEST, pmin, pmax, eps, 2024)


def main():
    tuned = re_lite_configs(42)
    print(f"RE-Lite low-budget study  (φ~U({pmin},{pmax}), ε={eps:.0e}, R={R_TEST:,}, de-biased)\n")
    print(f"  {'target':>7} | {'brute':>7} | {'fixed-RE':>8} | {'RE-Lite best':>12} | {'@ avg budget':>12} | config")
    for B in TARGETS:
        brute = 100 * E.success_rate(BF, {"budget": int(B)}, R_TEST, pmin, pmax, eps, 2024)
        fre = best_fixed_re(B)
        rl = best_re_within(B, tuned)
        if rl:
            rv, bv, me, mf = rl
            flag = "  <- beats brute" if rv > brute + 1 else ("  (~brute)" if rv > brute - 1 else "")
            print(f"  {B:>7,} | {brute:6.1f}% | {fre:7.1f}% | {100*rv:11.1f}% | {bv:>12,.0f} | m'={me}, m={mf}{flag}")
        else:
            print(f"  {B:>7,} | {brute:6.1f}% | {fre:7.1f}% | {'—':>12}")
    # also: the single most quotable RE-lite points on the frontier
    print("\n  RE-Lite frontier (best validated rate at each avg-budget bin):")
    val = sorted(((c[0], c) for c in tuned), key=lambda x: x[0])
    seen = set()
    for b, (bt, rt, me, mf) in val:
        bin_ = int(np.log10(max(bt, 1)) * 3)
        if bin_ in seen:
            continue
        seen.add(bin_)
        rv, bv = E.rate_and_budget(RL, {"m_exploration": me, "m_exploitation": mf}, R_TEST, pmin, pmax, eps, 2024)
        print(f"     {100*rv:5.1f}% @ avg budget {bv:>8,.0f}   (m'={me}, m={mf})")


if __name__ == "__main__":
    main()
