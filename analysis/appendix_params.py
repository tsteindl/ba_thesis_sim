"""Optimal parameters behind the reported results, written to results/optimal_params.md.

Table 3.1 winners are recomputed at budget 10,000; Table 3.2 winners are read from
results/story_winners.csv (the config at the budget nearest each 90% crossing).
    python analysis/appendix_params.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import config as C
from qmetrology.algorithms import FIXED_BUDGET
from qmetrology.experiments import grid_search_max

NICE = C.NICE
T32_COLS = [("ε=10⁻³, U(0.01,0.1)", "U(0.01,0.1), eps=1e-03"),
            ("ε=10⁻⁴, U(0.01,0.1)", "U(0.01,0.1), eps=1e-04"),
            ("ε=10⁻⁴, U(0.001,0.01)", "U(0.001,0.01), eps=1e-04"),
            ("ε=10⁻⁴, U(0.001,0.1)", "U(0.001,0.1), eps=1e-04")]


def fmt_params(p):
    p = {k: v for k, v in p.items() if k != "budget"}
    return ", ".join(f"{k}={v}" for k, v in p.items())


def main():
    winners = pd.read_csv("results/story_winners.csv")
    cube = pd.read_csv("results/story_cube.csv")
    L = ["# Appendix — optimal parameters for the reported results\n",
         "_Every adaptive number is the de-biased winner of a grid search (tuned seed 42, validated seed 2024). "
         "The full per-budget winners are in `results/story_winners.csv`; the tables below extract the ones "
         "behind the headline cells._\n"]

    # ---- Table 3.1 (recompute the budget-10,000 narrow winners) ----
    L.append("## Table 3.1 — winning parameters at fixed budget 10,000  (ε=10⁻³, U(0.01,0.1))\n")
    L.append("| Algorithm | optimal parameters |")
    L.append("|---|---|")
    for algo in ["linear", "binary", "reverse_eng"]:
        _gmax, arg, _ = grid_search_max(FIXED_BUDGET[algo], C.grids(10000)[algo],
                                        2000, C.PHI_MIN, 0.1, 1e-3, C.SEED_TUNE)
        L.append(f"| {NICE[algo]} | `{fmt_params(arg)}` |")
    L.append("")

    # ---- Table 3.2 (winner at the 90% crossing budget, per column) ----
    L.append("## Table 3.2 — winning parameters at the 90%-convergence budget\n")
    L.append("| Algorithm | " + " | ".join(lbl for lbl, _ in T32_COLS) + " |")
    L.append("|---|" + "---|" * len(T32_COLS))
    for algo in ["linear", "binary", "reverse_eng"]:
        cells = []
        for _lbl, setting in T32_COLS:
            cr = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == 90)]
            sub = winners[(winners.setting == setting) & (winners.algo == algo)]
            if not len(cr) or pd.isna(cr.budget_to_reach.iloc[0]) or not len(sub):
                cells.append("—")
                continue
            target = cr.budget_to_reach.iloc[0]
            row = sub.iloc[(sub.budget - target).abs().argmin()]
            cells.append(f"`{fmt_params(json.loads(row.params))}` (@budget {row.budget:,})")
        L.append(f"| {NICE[algo]} | " + " | ".join(cells) + " |")
    L.append("\n_Note the recurring pattern: the winning linear/RE configs use a **small exploration count** and "
             "commit the rest of the budget to exploitation at the inferred depth._")

    os.makedirs("results", exist_ok=True)
    with open("results/optimal_params.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/optimal_params.md")


if __name__ == "__main__":
    main()
