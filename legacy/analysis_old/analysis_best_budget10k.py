"""Best *achievable* convergence at a true budget = 10,000 (narrow, U(0.01,0.1)).

Motivation: the published Table 3.1 understates the adaptive algorithms, and a
recent run suggested the windowed linear search reaches >50% at budget=10,000.
But the fixed-budget variants can OVERSPEND the cap (the budget check fires after
each exploration step), so a grid-max can "cheat" by using ~2x the budget.

So for each algorithm we report:
  * grid-max (unconstrained): best success over the grid, with its mean budget
  * budget-respecting best:    best success among configs with mean_budget <= 10,000
  * de-biased: the budget-respecting winner re-evaluated at R=50k with a fresh seed
Tuning seed 42 (R=600), test seed 2024 (R=50k).
"""
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force,
    find_phi_fixed_budget_binary_search,
    find_phi_fixed_budget_linear_search,
    find_phi_fixed_budget_linear_search_windowed,
    find_phi_fixed_budget_reverse_engineering,
)

PHI_MIN, PHI_MAX, EPS, BUD = 0.01, 0.1, 1e-3, 10_000
SEED_TUNE, SEED_TEST, R_TUNE, R_TEST = 42, 2024, 600, 50_000
CAP = BUD * 1.0   # require mean budget <= the nominal cap

m_exp = np.unique(np.logspace(1, 3, 20, dtype=int))
GRIDS = {
    "windowed_linear": (find_phi_fixed_budget_linear_search_windowed, {
        "m_exploration": m_exp, "budget": [BUD], "window": [2, 3, 4, 5],
        "lookback_window": [1, 2, 5], "safeguard": [1, 2]}),
    "plain_linear": (find_phi_fixed_budget_linear_search, {
        "m_exploration": m_exp, "budget": [BUD], "lookback_window": [1, 2, 5],
        "safeguard": [0, 1, 2], "inc": [1, 2, 5]}),
    "binary": (find_phi_fixed_budget_binary_search, {
        "m_exploration": m_exp, "budget": [BUD], "conf": [0.5, 0.8, 0.9, 0.95],
        "safeguard": [0, 1, 2]}),
    "reverse_eng": (find_phi_fixed_budget_reverse_engineering, {
        "m_exploration": m_exp, "budget": [BUD], "safeguard": [0.8, 0.85, 0.9, 0.95]}),
}


def main():
    print(f"Best achievable at TRUE budget={BUD} (narrow U(0.01,0.1), eps=1e-3)\n")
    rows = []
    for name, (fn, grid) in GRIDS.items():
        res = E.grid_full(fn, grid, R_TUNE, PHI_MIN, PHI_MAX, EPS, SEED_TUNE)
        ncfg = len(res)
        gmax_s, gmax_b, gmax_p = max(res, key=lambda r: r[0])
        respecting = [r for r in res if r[1] <= CAP]
        if respecting:
            bs, bb, bp = max(respecting, key=lambda r: r[0])
            deb, deb_b = E.rate_and_budget(fn, bp, R_TEST, PHI_MIN, PHI_MAX, EPS, SEED_TEST)
            lo, hi = E.wilson(deb, R_TEST)
        else:
            bs = bb = deb = deb_b = float("nan"); bp = {}; lo = hi = float("nan")
        rows.append((name, ncfg, gmax_s, gmax_b, bs, bb, deb, deb_b, lo, hi, bp))
        print(f"{name:16s} (ncfg={ncfg})")
        print(f"   grid-max (unconstrained) : {gmax_s*100:5.1f}%  @ mean budget {gmax_b:8.0f}  {gmax_p}")
        print(f"   budget-respecting best   : {bs*100:5.1f}%  @ mean budget {bb:8.0f}  {bp}")
        print(f"   de-biased (R=50k)        : {deb*100:5.1f}%  CI[{lo*100:.1f},{hi*100:.1f}]  @ mean budget {deb_b:8.0f}\n")

    # brute force anchor (param-free, always respects budget)
    br, bb = E.rate_and_budget(find_phi_fixed_budget_brute_force, {"budget": BUD},
                               R_TEST, PHI_MIN, PHI_MAX, EPS, SEED_TEST)
    print(f"{'brute_force':16s} param-free       : {br*100:5.1f}%  @ mean budget {bb:8.0f}  (paper 56.3%)")

    # markdown summary
    L = [f"# Best achievable at budget = {BUD} (narrow, eps=1e-3)\n",
         "Tuning R=600 (seed 42); de-biased R=50000 (seed 2024). 'budget-respecting' = best "
         "config whose **mean budget <= 10,000** (fixed-budget variants can overspend the cap "
         "by up to one exploration step).\n",
         "| Algorithm | grid-max (unconstrained) | its budget | budget-respecting best | de-biased | de-biased budget |",
         "|---|---|---|---|---|---|"]
    for name, ncfg, gs, gb, bs, bb_, deb, deb_b, lo, hi, bp in rows:
        L.append(f"| {name} | {gs*100:.1f}% | {gb:.0f} | {bs*100:.1f}% | {deb*100:.1f}% "
                 f"[{lo*100:.1f},{hi*100:.1f}] | {deb_b:.0f} |")
    L.append(f"| brute force | 56.3% | {BUD} | 56.3% | {br*100:.1f}% | {bb:.0f} |")
    L.append("\n_Paper Table 3.1: linear 14.0%, binary 11.5%, reverse-eng 61.3%, brute 56.3%._")

    # Overspend demonstration: the old "65.7%" windowed winner (m_exploration=615)
    L.append("\n## Why the old 'windowed 65.7% @ 10,000' was misleading\n")
    L.append("The fixed-budget variants check the budget *after* each exploration step, so a large "
             "`m_exploration` overspends the cap. Actual mean budget used (windowed linear, nominal cap 10,000):\n")
    L.append("| m_exploration | success | ACTUAL mean budget |")
    L.append("|---|---|---|")
    for m in [10, 200, 615, 1000]:
        p = {"m_exploration": m, "budget": BUD, "window": 5, "lookback_window": 1, "safeguard": 1}
        r, b = E.rate_and_budget(find_phi_fixed_budget_linear_search_windowed, p, 20000,
                                 PHI_MIN, PHI_MAX, EPS, SEED_TEST)
        note = " (overspends ~2x!)" if b > 1.5 * BUD else (" (overspends)" if b > 1.05 * BUD else "")
        L.append(f"| {m} | {r*100:.1f}% | {b:.0f}{note} |")
    L.append("\nThe recalled `m_exploration=615` config scores ~57% but **uses ~19,065 budget**, not 10,000. "
             "At a true 10,000 cap the honest windowed best is ~55% (m_exploration=10), essentially identical "
             "to plain linear — windowing gives no real gain here.\n")
    L.append("## Bottom line\n")
    L.append("- **Yes, the paper understates linear/binary at budget=10,000:** honest best is ~55% (linear) and "
             "~48% (binary), not 14% / 11.5%.\n"
             "- **Reverse engineering is the genuine winner (~63%)**, beating brute force (~56%). Linear ties "
             "brute force; binary is below it.\n"
             "- **Windowing does not help** at this budget; the 65.7% was a budget-overspend artifact.\n"
             "- Net: adaptive *can* beat brute force at budget=10,000 (RE does), but only modestly — consistent "
             "with the thesis's 'marginal for narrow distributions' claim, with corrected numbers.")
    print("\n".join(L))
    print("\n(optional deep-dive; the authoritative results live in results/RESULTS.md)")


if __name__ == "__main__":
    main()
