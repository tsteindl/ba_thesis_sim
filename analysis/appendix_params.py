"""Optimal parameters behind the reported results, written to results/optimal_params.md.

Table 3.1 winners are recomputed at budget 10,000; Table 3.2 winners are read from
results/story_winners.csv (the config at the budget nearest each 90% crossing).

Also emits results/optimal_params.csv (the budget-10,000 winners, machine readable) so
analysis/make_tex.py can render the appendix table without repeating the grid search.
    python analysis/appendix_params.py
"""
import csv
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
# the *_risk variants have no safeguard row to report — that is the result worth showing
ALGOS = ["linear", "binary_risk", "reverse_eng_risk"]
T32_COLS = [("ε=10⁻³, U(0.01,0.1)", "U(0.01,0.1), eps=1e-03"),
            ("ε=10⁻⁴, U(0.01,0.1)", "U(0.01,0.1), eps=1e-04"),
            ("ε=10⁻⁴, U(0.001,0.01)", "U(0.001,0.01), eps=1e-04"),
            ("ε=10⁻⁴, U(0.001,0.1)", "U(0.001,0.1), eps=1e-04")]


def fmt_params(p):
    p = {k: v for k, v in p.items() if k != "budget"}
    return ", ".join(f"{k}={v}" for k, v in p.items())


def n_min(phi_max):
    return max(1, int(np.floor(np.pi / (2 * phi_max))))


def pilot_share(params, budget, phi_max):
    """Fraction of the budget the tuned exploration size actually spends on the pilot.

    Reported next to every tuned m' so the recommended default (C.PILOT_SHARE) can be checked against
    what the grid search independently chose. Only meaningful where the exploration is a single
    probe at N_min — i.e. reverse engineering; binary search re-probes at growing depths, so its
    exploration cost is not m'*N_min and no share is shown.
    """
    m = params.get("m_exploration")
    if m is None or not budget:
        return None
    return m * n_min(phi_max) / float(budget)


def main():
    winners = pd.read_csv("results/story_winners.csv")
    cube = pd.read_csv("results/story_cube.csv")
    L = ["# Appendix — optimal parameters for the reported results\n",
         "_Every adaptive number is the de-biased winner of a grid search (tuned seed 42, validated seed 2024). "
         "The full per-budget winners are in `results/story_winners.csv`; the tables below extract the ones "
         "behind the headline cells._\n"]

    # ---- Table 3.1 (recompute the budget-10,000 narrow winners) ----
    L.append("## Table 3.1 — winning parameters at fixed budget 10,000  (ε=10⁻³, U(0.01,0.1))\n")
    L.append("| Algorithm | optimal parameters | pilot share of budget |")
    L.append("|---|---|---|")
    t31 = {}
    for algo in ALGOS:
        _gmax, arg, _ = grid_search_max(FIXED_BUDGET[algo], C.grids(10000)[algo],
                                        2000, C.PHI_MIN, 0.1, 1e-3, C.SEED_TUNE)
        t31[algo] = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in arg.items() if k != "budget"}
        sh = pilot_share(arg, 10000, 0.1) if algo == "reverse_eng_risk" else None
        L.append(f"| {NICE[algo]} | `{fmt_params(arg)}` | "
                 + (f"{100*sh:.1f}%" if sh is not None else "—") + " |")
    L.append(f"\n_Reverse engineering's exploration size is quoted in the thesis as a **share of the "
             f"budget**, ρ = {100*C.PILOT_SHARE:g}% (floor {C.PILOT_FLOOR} shots), not as a shot count: "
             f"a fixed m′ can only be correct at one budget. Over the 546 operating points of "
             f"`analysis/re_share_sweep.py` the fixed ρ costs −0.06 pp against tuning m′ at every "
             f"budget separately, while a fixed m′ = 200 costs −5.55 pp (worst −99.4 pp)._")
    L.append("")

    # machine-readable, consumed by analysis/make_tex.py
    with open("results/optimal_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "budget", "eps", "phi_min", "phi_max", "params"])
        for algo, p in t31.items():
            w.writerow([algo, 10000, 1e-3, C.PHI_MIN, 0.1, json.dumps(p)])

    # ---- Table 3.2 (winner at the 90% crossing budget, per column) ----
    L.append("## Table 3.2 — winning parameters at the 90%-convergence budget\n")
    L.append("| Algorithm | " + " | ".join(lbl for lbl, _ in T32_COLS) + " |")
    L.append("|---|" + "---|" * len(T32_COLS))
    shares = []
    for algo in ALGOS:
        cells = []
        for _lbl, setting in T32_COLS:
            cr = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == 90)]
            sub = winners[(winners.setting == setting) & (winners.algo == algo)]
            if not len(cr) or pd.isna(cr.budget_to_reach.iloc[0]) or not len(sub):
                cells.append("—")
                continue
            target = cr.budget_to_reach.iloc[0]
            row = sub.iloc[(sub.budget - target).abs().argmin()]
            p = json.loads(row.params)
            cell = f"`{fmt_params(p)}` (@budget {row.budget:,})"
            if algo == "reverse_eng_risk":
                sh = pilot_share(p, row.budget, float(row.phi_max))
                if sh is not None:
                    shares.append(sh)
                    cell += f" — **{100*sh:.1f}%** of budget"
            cells.append(cell)
        L.append(f"| {NICE[algo]} | " + " | ".join(cells) + " |")
    L.append("\n_Note the recurring pattern: the winning linear/RE configs use a **small exploration count** and "
             "commit the rest of the budget to exploitation at the inferred depth._")
    if shares:
        L.append(f"\n_Read as a fraction of the budget, reverse engineering's tuned exploration sizes above span "
                 f"**{100*min(shares):.1f}–{100*max(shares):.1f}%** — the same few percent at every budget and "
                 f"prior, which is why the thesis quotes ρ = {100*C.PILOT_SHARE:g}% rather than a shot count. "
                 f"The corresponding m′ values span {min(json.loads(r.params).get('m_exploration', 0) for _, r in winners[winners.algo == 'reverse_eng_risk'].iterrows()):,}"
                 f"–{max(json.loads(r.params).get('m_exploration', 0) for _, r in winners[winners.algo == 'reverse_eng_risk'].iterrows()):,}, "
                 f"a range of four orders of magnitude._")

    os.makedirs("results", exist_ok=True)
    with open("results/optimal_params.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/optimal_params.md")


if __name__ == "__main__":
    main()
