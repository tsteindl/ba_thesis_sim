"""Error bars for every number in the sweep — pure post-processing, no re-simulation.

Reads results/story_curves.csv (convergence rate per algorithm/budget, R = 40,000 paired trials) and
attaches to each reported quantity:

  * `rate_lo/rate_hi`     Wilson 95% interval on each convergence rate;
  * `budget_lo/budget_hi` 95% interval on the budget-to-reach-p*, by parametric bootstrap of the
                          curve the crossing is interpolated from;
  * `ratio_lo/ratio_hi`   95% interval on the budget ratio vs brute force (conservative — the two
                          curves are resampled independently although the trials are seed-paired);
  * `grid_dev`            the relative shift of the crossing when it is re-derived from the
                          half-density budget subgrids, i.e. how much the *discretisation* of the
                          budget axis contributes as opposed to Monte-Carlo noise.

    python analysis/uncertainty.py [--boot 2000] [--R 40000] [--quiet]

Writes results/story_cube_ci.csv, results/story_curves_ci.csv and results/UNCERTAINTY.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.uncertainty import crossing, crossing_ci, grid_sensitivity, ratio_ci, wilson

# must match the R the curves were evaluated at (analysis/extensive_sweep.py R_TEST); the intervals
# are meaningless if it does not, so it is a flag rather than a silent constant
R_TEST = int(sys.argv[sys.argv.index("--R") + 1]) if "--R" in sys.argv else 40_000
THRESHOLDS = [50, 80, 90, 95]
N_BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 2000
QUIET = "--quiet" in sys.argv
HEAD = ["U(0.01,0.1), eps=1e-03", "U(0.01,0.1), eps=1e-04",
        "U(0.001,0.01), eps=1e-04", "U(0.001,0.1), eps=1e-04"]


def curve_ci(curves):
    """Wilson interval on every individual convergence rate."""
    lo, hi = zip(*[wilson(p, R_TEST) for p in curves.rate])
    out = curves.copy()
    out["rate_lo"], out["rate_hi"] = np.round(lo, 5), np.round(hi, 5)
    return out


def cube_ci(curves):
    """Bootstrap intervals for every (setting, algo, threshold) crossing and its ratio vs brute."""
    rows = []
    for setting, d in curves.groupby("setting", sort=False):
        ref = d[d.algo == "brute"].sort_values("budget")
        if not len(ref):
            continue
        budgets, brute_rates = ref.budget.to_numpy(float), ref.rate.to_numpy(float)
        for algo, da in d.groupby("algo", sort=False):
            da = da.sort_values("budget")
            # curves need not share brute's abscissae — the ceilings get extra low-budget points
            # (analysis/oracle_curves.py), so each is crossed on its own grid
            b_algo = da.budget.to_numpy(float)
            rates = da.rate.to_numpy(float)
            for T in THRESHOLDS:
                t = T / 100
                b, blo, bhi, cov = crossing_ci(b_algo, rates, t, R_TEST, N_BOOT, seed=T)
                r, rlo, rhi, _ = ratio_ci(budgets, brute_rates, rates, t, R_TEST, N_BOOT, seed=T,
                                          budgets_alg=b_algo)
                # brute against itself is exactly 1 by construction, and the width the independent
                # bootstrap assigns it is a direct read-out of how much ignoring the seed pairing
                # inflates every other ratio interval. Keep the diagnostic, report the exact value.
                infl = (rhi - rlo) / 2 if algo == "brute" else None
                if algo == "brute":
                    r, rlo, rhi = 1.0, 1.0, 1.0
                rows.append({
                    "setting": setting, "phi_min": da.phi_min.iloc[0], "phi_max": da.phi_max.iloc[0],
                    "eps": da.eps.iloc[0], "algo": algo, "threshold_pct": T,
                    "budget_to_reach": None if not np.isfinite(b) else round(b, 1),
                    "budget_lo": None if not np.isfinite(blo) else round(blo, 1),
                    "budget_hi": None if not np.isfinite(bhi) else round(bhi, 1),
                    "ratio_vs_brute": None if not np.isfinite(r) else round(r, 3),
                    "ratio_lo": None if not np.isfinite(rlo) else round(rlo, 3),
                    "ratio_hi": None if not np.isfinite(rhi) else round(rhi, 3),
                    "grid_dev": round(grid_sensitivity(budgets, rates, t), 4),
                    "boot_coverage": round(cov, 3),
                    "ratio_ci_inflation": None if infl is None else round(infl, 4),
                })
        if not QUIET:
            print(f"  {setting}", flush=True)
    return pd.DataFrame(rows)


def report(cube, curves_ci):
    """A short prose summary — the numbers to quote in the methodology section."""
    at90 = cube[(cube.threshold_pct == 90)].dropna(subset=["budget_to_reach"])
    rel_b = ((at90.budget_hi - at90.budget_lo) / 2 / at90.budget_to_reach).dropna()
    rr = at90.dropna(subset=["ratio_vs_brute"])
    rel_r = ((rr.ratio_hi - rr.ratio_lo) / 2 / rr.ratio_vs_brute).dropna()
    gd = at90.grid_dev.dropna()
    infl = at90.ratio_ci_inflation.dropna()
    worst_rate = (curves_ci.rate_hi - curves_ci.rate_lo).max() / 2

    L = ["# Uncertainty of the reported numbers\n",
         "_Generated by `analysis/uncertainty.py` from `results/story_curves.csv`. No simulation is "
         "re-run: every interval below is derived from the convergence rates already on disk._\n",
         "## What each number's error bar is\n",
         f"- **Convergence rates** are binomial proportions over R = {R_TEST:,} independent trials. "
         f"The Wilson 95% interval is at most **±{100*worst_rate:.2f} percentage points** across the "
         "whole sweep (worst case, p = 0.5), and narrower wherever the rate is near 0 or 1.",
         "- **Budgets to reach p\\*** are *derived*: the convergence-vs-budget curve is log-interpolated "
         "to where it crosses p\\*. Their Monte-Carlo uncertainty is obtained by parametric bootstrap "
         f"(resample every curve point from Binomial(R, p̂)/R, re-interpolate, {N_BOOT:,} replicates). "
         f"At the 90% threshold the 95% half-width is a median **±{100*rel_b.median():.2f}%** of the "
         f"budget (90th percentile ±{100*rel_b.quantile(0.9):.2f}%).",
         f"- **Budget ratios** inherit both curves' noise: median half-width "
         f"**±{100*rel_r.median():.2f}%** (90th percentile ±{100*rel_r.quantile(0.9):.2f}%). This is "
         "**conservative**: all algorithms are evaluated on the same seed list, so trial *i* draws the "
         "same φ for every algorithm, and the true paired uncertainty is smaller than the independent "
         "resampling used here. How much smaller is directly measurable — brute force against itself "
         "has a ratio of exactly 1, yet the same independent bootstrap assigns it a half-width of "
         f"**±{100*infl.median():.2f}%** (median). Read the ratio intervals as an upper bound of "
         "roughly that magnitude.",
         f"- **Budget-grid discretisation** is a second, independent source. Re-deriving each crossing "
         f"from the two half-density subgrids moves it by a median {100*gd.median():.2f}% "
         f"(90th percentile {100*gd.quantile(0.9):.2f}%); log-linear interpolation error scales with "
         f"the square of the grid spacing, so on the full 40-point grid this contributes roughly "
         f"{100*gd.median()/4:.2f}%.\n",
         "**Rule of thumb for the thesis:** quote budgets to three significant figures and ratios to "
         "two decimals; treat differences below ~5% in a ratio as not resolved by this experiment.\n",
         "## Headline cells (90% convergence)\n",
         "| Scenario | Algorithm | Budget | 95% CI | Ratio vs brute | 95% CI |",
         "|---|---|---:|---|---:|---|"]
    for s in HEAD:
        for a in ["brute", "linear", "binary", "reverse_eng", "binary_risk", "reverse_eng_risk",
                  "oracle_alias", "oracle"]:
            r = at90[(at90.setting == s) & (at90.algo == a)]
            if not len(r):
                continue
            r = r.iloc[0]
            rat = "—" if pd.isna(r.ratio_vs_brute) else f"{r.ratio_vs_brute:.2f}×"
            rci = "—" if pd.isna(r.ratio_lo) else f"[{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]"
            L.append(f"| {s} | {a} | {r.budget_to_reach:,.0f} | "
                     f"[{r.budget_lo:,.0f}, {r.budget_hi:,.0f}] | {rat} | {rci} |")
    L.append("\n## Not covered by these intervals\n")
    L.append("The grid search that selects each adaptive algorithm's parameters is itself random "
             "(tuned at R = 2,000 on seed 42). De-biasing removes the *bias* this causes — the winning "
             "configuration is re-validated on an independent seed — but not the *variance* of which "
             "configuration wins. That component cannot be recovered from the stored results; "
             "`analysis/tuning_stability.py` measures it directly by repeating the tuning on several "
             "seeds at the headline cells only.")
    return "\n".join(L)


def main():
    curves = pd.read_csv("results/story_curves.csv")
    print(f"curves: {len(curves):,} rows, {curves.setting.nunique()} settings, "
          f"{N_BOOT:,} bootstrap replicates", flush=True)
    cc = curve_ci(curves)
    cube = cube_ci(curves)
    os.makedirs("results", exist_ok=True)
    cc.to_csv("results/story_curves_ci.csv", index=False)
    cube.to_csv("results/story_cube_ci.csv", index=False)
    with open("results/UNCERTAINTY.md", "w") as f:
        f.write(report(cube, cc) + "\n")
    print("\nwrote results/story_curves_ci.csv, results/story_cube_ci.csv, results/UNCERTAINTY.md")


if __name__ == "__main__":
    main()
