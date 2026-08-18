"""The one uncertainty the stored results cannot give you: how much does the *tuning* wobble?

analysis/uncertainty.py derives Monte-Carlo intervals from the convergence rates already on disk.
Those intervals condition on the parameter configuration the grid search happened to pick. Selecting
that configuration is itself a random experiment (R_TUNE = 2,000 trials on seed 42): de-biasing —
re-validating the winner on an independent seed — removes the resulting *bias*, but the *variance* of
which configuration wins is invisible in the stored output.

This script measures it directly, and cheaply, by repeating only the tuning step: at each headline
scenario, at the single budget nearest the 90% crossing, the grid search is re-run on K independent
tuning seeds and each winner is validated on the common test seed. The spread of those validated
rates is the tuning-selection variance. It is converted to a budget-equivalent using the local slope
of the convergence curve, so it can be read next to the bootstrap intervals in the same units.

    python analysis/tuning_stability.py [--seeds 5] [--quick]

Writes results/tuning_stability.csv and results/TUNING_STABILITY.md. Cost is roughly
K x (#scenarios) x (#algorithms) grid searches — minutes, not the hours a full re-sweep would take.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.uncertainty import crossing

import importlib
SWEEP = importlib.import_module("analysis.extensive_sweep") if os.path.isdir("analysis") else None
if SWEEP is None:  # pragma: no cover - executed only from an unusual cwd
    sys.exit("run from the repository root")

QUICK = "--quick" in sys.argv
K = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
R_TUNE = 500 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TEST = 2024
TUNE_SEEDS = [42, 7, 1234, 99, 20240, 555, 8, 31415][:K]
SCENARIOS = ["U(0.01,0.1), eps=1e-03", "U(0.01,0.1), eps=1e-04",
             "U(0.001,0.01), eps=1e-04", "U(0.001,0.1), eps=1e-04"]
ALGOS = ["linear", "binary", "reverse_eng", "binary_risk", "reverse_eng_risk"]


def local_slope(budgets, rates, T):
    """d(rate)/d(log budget) at the crossing — converts a rate wobble into a budget wobble."""
    b, r = np.asarray(budgets, float), np.asarray(rates, float)
    x = crossing(b, r, T)
    if not np.isfinite(x):
        return float("nan")
    i = int(np.searchsorted(b, x))
    i = min(max(i, 1), len(b) - 1)
    dr, dlb = r[i] - r[i - 1], np.log(b[i]) - np.log(b[i - 1])
    return dr / dlb if dlb else float("nan")


def main():
    curves = pd.read_csv("results/story_curves.csv")
    cube = pd.read_csv("results/story_cube.csv")
    rows = []
    for setting in SCENARIOS:
        d = curves[curves.setting == setting]
        if not len(d):
            print(f"!! {setting} not in the sweep, skipped"); continue
        pmin, pmax, eps = d.phi_min.iloc[0], d.phi_max.iloc[0], d.eps.iloc[0]
        ref = d[d.algo == "brute"].sort_values("budget")
        grids = SWEEP.grids(pmax, eps)
        print(f"\n=== {setting} ===", flush=True)
        for algo in ALGOS:
            cr = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == 90)]
            da = d[d.algo == algo].sort_values("budget")
            if not len(cr) or pd.isna(cr.budget_to_reach.iloc[0]) or not len(da):
                continue
            target = float(cr.budget_to_reach.iloc[0])
            budget = int(da.budget.to_numpy()[np.abs(da.budget.to_numpy() - target).argmin()])
            slope = local_slope(ref.budget.to_numpy(float), ref.rate.to_numpy(float), 0.90)

            vals, cfgs = [], []
            for s in TUNE_SEEDS:
                pool = []
                for fn, g in grids[algo]:
                    res = E.grid_full(fn, {**g, "budget": [budget]}, R_TUNE, pmin, pmax, eps, s)
                    pool += [(r, fn, cfg) for r, _b, cfg in res]
                _tr, fn, cfg = max(pool, key=lambda x: x[0])
                v = E.success_rate(fn, cfg, R_TEST, pmin, pmax, eps, SEED_TEST)
                vals.append(v); cfgs.append(SWEEP._clean(cfg))
            vals = np.array(vals)
            # a rate spread of sd translates into a budget spread of sd/slope in log-budget
            bud_pct = 100 * (vals.std(ddof=1) / slope) if np.isfinite(slope) and slope else np.nan
            rows.append({"setting": setting, "algo": algo, "budget": budget,
                         "n_seeds": len(vals), "rate_mean": round(vals.mean(), 5),
                         "rate_sd": round(vals.std(ddof=1), 5),
                         "rate_min": round(vals.min(), 5), "rate_max": round(vals.max(), 5),
                         "budget_sd_pct": None if not np.isfinite(bud_pct) else round(bud_pct, 2),
                         "n_distinct_winners": len({str(c) for c in cfgs})})
            print(f"   {algo:18s} @{budget:>12,}  rate {vals.mean():.4f} ± {vals.std(ddof=1):.4f} "
                  f"(range {vals.min():.4f}-{vals.max():.4f}, {rows[-1]['n_distinct_winners']}/{len(vals)} "
                  f"distinct winners)  ~±{bud_pct:.1f}% of budget", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/tuning_stability.csv", index=False)

    sd_pp = 100 * df.rate_sd.dropna()
    bud = df.budget_sd_pct.dropna()
    L = ["# Tuning-selection variance\n",
         f"_Generated by `analysis/tuning_stability.py`: the grid search repeated on {len(TUNE_SEEDS)} "
         f"independent tuning seeds ({', '.join(map(str, TUNE_SEEDS))}) at the 90%-crossing budget, "
         f"each winner validated on seed {SEED_TEST} at R = {R_TEST:,}._\n",
         "This is the uncertainty component that `analysis/uncertainty.py` cannot recover from the "
         "stored curves: not the noise in a measured rate, but the noise in *which parameter "
         "configuration the grid search picks*.\n",
         f"- Across the cells measured, the validated convergence rate has a standard deviation of "
         f"**{sd_pp.median():.2f} percentage points** (max {sd_pp.max():.2f} pp) across tuning seeds.",
         f"- In budget terms that is **±{bud.median():.1f}%** (max ±{bud.max():.1f}%), to be added to "
         "the Monte-Carlo interval in `results/UNCERTAINTY.md` rather than replacing it.",
         "- `n_distinct_winners` shows how often a different configuration wins: where it is greater "
         "than 1 but the rate spread is small, the top of the grid is flat and the choice does not "
         "matter — which is the reassuring case.\n",
         df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)]
    with open("results/TUNING_STABILITY.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nwrote results/tuning_stability.csv and results/TUNING_STABILITY.md")


if __name__ == "__main__":
    main()
