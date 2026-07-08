"""Estimator-error distribution vs budget, written to results/error_curves.csv.

Reuses the winning configs from extensive_sweep.py (results/story_winners.csv): for each budget it
re-evaluates the winner on the test seed and records quantiles of |phi_hat - phi| (median and the
25-75% band). Plotted by render_story.py as fig_error.png.
    python analysis/error_curves.py
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import experiments as E
from qmetrology import algorithms as A
import extensive_sweep as X   # for SEED_TEST, label, MAX flag (and fallback tuning)

QUICK = "--quick" in sys.argv
R_TEST = 5000 if QUICK else 40_000
QS = (0.25, 0.5, 0.75)
BF, SEP = A.find_phi_fixed_budget_brute_force, A.find_phi_fixed_budget_separable
ADAPT = ["linear", "binary", "reverse_eng"]


def error_settings():
    """Which U(0.01,0.1) precisions to draw the error curves for."""
    epslist = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8] if X.MAX else [1e-3, 1e-4]
    return [X.label(0.01, 0.1, e) for e in epslist]


def _q(fn, params, pmin, pmax, eps):
    q, _, _ = E.error_quantiles(fn, params, R_TEST, pmin, pmax, eps, X.SEED_TEST, quantiles=QS)
    return q[0.25], q[0.5], q[0.75]


def main():
    win = "results/story_winners.csv"
    df = pd.read_csv(win) if os.path.exists(win) else None
    rows = []
    for setting in error_settings():
        if df is None or not (df.setting == setting).any():
            print(f"(no winners for {setting} — skipping; run extensive_sweep first)")
            continue
        sub = df[df.setting == setting]
        pmin, pmax, eps = float(sub.phi_min.iloc[0]), float(sub.phi_max.iloc[0]), float(sub.eps.iloc[0])
        budgets = sorted(int(b) for b in sub.budget.unique())
        print(f"=== {setting}  ({len(budgets)} budgets) ===", flush=True)
        for algo, fn in [("brute", BF), ("separable", SEP)]:
            for b in budgets:
                p25, p50, p75 = _q(fn, {"budget": int(b)}, pmin, pmax, eps)
                rows.append([setting, pmin, pmax, eps, algo, int(b), p25, p50, p75])
        for algo in ADAPT:
            for _, r in sub[sub.algo == algo].iterrows():
                fn = getattr(A, r.fn)
                p25, p50, p75 = _q(fn, json.loads(r.params), pmin, pmax, eps)
                rows.append([setting, pmin, pmax, eps, algo, int(r.budget), p25, p50, p75])
    os.makedirs("results", exist_ok=True)
    with open("results/error_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "phi_min", "phi_max", "eps", "algo", "budget", "err_p25", "err_p50", "err_p75"])
        w.writerows(rows)
    print("wrote results/error_curves.csv")


if __name__ == "__main__":
    main()
