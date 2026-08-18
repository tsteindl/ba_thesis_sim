"""Add the omniscient ceilings to an existing sweep, without re-running it.

analysis/extensive_sweep.py now evaluates `oracle` and `oracle_alias` itself, so a fresh full run
needs nothing from here. This script exists so the ceilings can be added to the sweep that is
*already* on disk: both are parameter-free, so they need no grid search at all — only the R_TEST
evaluation on the budget grid the sweep already used, which is a small fraction of the sweep's cost.

The budget grid, R, and seed are read back from results/story_curves.csv, so the new rows land on
exactly the same abscissae as the existing curves and the ratios stay comparable.

    python analysis/oracle_curves.py [--quick] [--settings "U(0.01,0.1), eps=1e-03; U(...)"]
    python analysis/oracle_curves.py --hl-only    # only the analytic non-degenerate ceiling
    python analysis/oracle_curves.py --check      # re-verify an existing merge, compute nothing

(`--settings` takes a *semicolon*-separated list — the labels themselves contain commas.)

Rewrites results/story_curves.csv and results/story_cube.csv in place, replacing any oracle rows
already there (idempotent), leaving every other row untouched, and merging after each setting so an
interrupted run keeps what it has finished. The headline scenarios are computed first. Every run
ends with `check()`, which confirms against the data that no algorithm beats the oracle anywhere.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_oracle as ORA,
    find_phi_fixed_budget_oracle_alias as ORA_A,
)
from qmetrology.oracle import degeneracy_share, heisenberg_rate
from qmetrology.uncertainty import crossing

QUICK = "--quick" in sys.argv
# the non-degenerate ceiling is analytic, so it can be (re)computed on its own in seconds
HL_ONLY = "--hl-only" in sys.argv
R_TEST = 4000 if QUICK else 40_000          # matches extensive_sweep.py
SEED_TEST = 2024
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]
ORACLES = ["oracle", "oracle_alias", "oracle_hl"]
CURVES, CUBE = "results/story_curves.csv", "results/story_cube.csv"
# the cells that feed Tables 3.1/3.2 and the precision progression — computed first
HEADLINE = ["U(0.01,0.1), eps=1e-03", "U(0.01,0.1), eps=1e-04", "U(0.001,0.01), eps=1e-04",
            "U(0.001,0.1), eps=1e-04", "U(0.01,0.1), eps=1e-05", "U(0.01,0.1), eps=1e-06",
            "U(0.01,0.1), eps=1e-07", "U(0.01,0.1), eps=1e-08"]


def _wanted(all_settings):
    if "--settings" not in sys.argv:
        return list(all_settings)
    raw = sys.argv[sys.argv.index("--settings") + 1]
    want = [s.strip() for s in raw.split(";") if s.strip()]   # ';' — the labels contain commas
    missing = [s for s in want if s not in set(all_settings)]
    if missing:
        sys.exit(f"unknown setting(s): {missing}\navailable: {sorted(all_settings)}")
    return want


def check(tol_pp=0.6):
    """Verify empirically that the oracle bounds every algorithm, at every budget in the sweep.

    The bound is an argument, not a measurement: each protocol here ends with a single estimate taken
    at one depth N with m <= budget/N shots, and the oracle maximises the convergence probability over
    exactly that family with the full budget. The argument does assume more shots at a fixed depth
    never hurt, which the discreteness of the estimator does not strictly guarantee, so it is worth
    confirming against the data rather than asserting.

    Reports any budget where an algorithm's measured rate exceeds the oracle's by more than `tol_pp`
    percentage points — roughly twice the Monte-Carlo standard error at R = 40,000, so smaller
    excesses are noise rather than a violated bound.
    """
    curves = pd.read_csv(CURVES)
    viol = []
    for (setting, budget), d in curves.groupby(["setting", "budget"], sort=False):
        orc = d[d.algo == "oracle"].rate
        if not len(orc):
            continue
        others = d[~d.algo.isin(ORACLES)]
        excess = 100 * (others.rate - orc.iloc[0])
        for algo, e in zip(others.algo, excess):
            if e > tol_pp:
                viol.append((setting, int(budget), algo, e))
    n = curves[curves.algo == "oracle"].groupby("setting").ngroups
    print(f"\noracle bound check over {n} setting(s), tolerance {tol_pp} pp:")
    if not viol:
        print("  OK — no algorithm exceeds the oracle at any budget.")
    else:
        print(f"  !! {len(viol)} violation(s), largest first:")
        for s, b, a, e in sorted(viol, key=lambda v: -v[3])[:15]:
            print(f"     {s:28s} budget {b:>16,}  {a:18s} +{e:.2f} pp")
    return viol


def _merge(setting, curve_rows, cube_rows):
    """Fold one setting's oracle rows into the CSVs, replacing any already there (idempotent).

    Done per setting rather than once at the end so an interrupted run keeps everything it has
    finished — the same guarantee analysis/extensive_sweep.py gives.
    """
    touched = {r[4] for r in curve_rows} | {r[4] for r in cube_rows}
    for path, rows in [(CURVES, curve_rows), (CUBE, cube_rows)]:
        df = pd.read_csv(path)
        keep = ~(df.algo.isin(touched) & (df.setting == setting))
        pd.concat([df[keep], pd.DataFrame(rows, columns=df.columns)]).to_csv(path, index=False)


def main():
    if not os.path.exists(CURVES):
        sys.exit(f"{CURVES} not found — run analysis/extensive_sweep.py first")
    if "--check" in sys.argv:          # verify an existing merge, compute nothing
        check(); return
    curves = pd.read_csv(CURVES)
    settings = _wanted(curves.setting.unique())
    # the settings that feed the thesis tables first, so an interrupted run still leaves them done
    seen = {s: i for i, s in enumerate(settings)}
    settings.sort(key=lambda s: (HEADLINE.index(s) if s in HEADLINE else len(HEADLINE), seen[s]))

    done = 0
    for setting in settings:
        new_curve_rows, new_cube_rows = [], []
        ref = curves[(curves.setting == setting) & (curves.algo == "brute")].sort_values("budget")
        if not len(ref):
            print(f"!! {setting}: no brute curve to take the budget grid from, skipped")
            continue
        pmin, pmax, eps = ref.phi_min.iloc[0], ref.phi_max.iloc[0], ref.eps.iloc[0]
        budgets = ref.budget.to_numpy(np.int64)
        brute_rates = ref.rate.to_numpy(float)
        print(f"\n=== {setting}  ({len(budgets)} budgets, R={R_TEST:,}) ===", flush=True)

        for algo, fn in ([] if HL_ONLY else [("oracle", ORA), ("oracle_alias", ORA_A)]):
            def rate_at(b):
                params = {"budget": int(b)}
                if algo == "oracle":
                    params["eps_target"] = float(eps)
                return E.success_rate(fn, params, R_TEST, pmin, pmax, eps, SEED_TEST)

            bs = list(budgets)
            rates = [rate_at(b) for b in bs]
            # The sweep's budget grid is anchored on brute force (c/50 .. c*30 around brute's
            # analytic 90% budget), and the oracle can already be above 90% at its lowest point —
            # its crossing would then fall off the bottom of the grid and be reported as missing.
            # Extend downward until the curve drops below the lowest threshold. Cheap: the ceilings
            # are parameter-free, so extra budgets cost one evaluation each, not a grid search.
            step, guard = 4.0, 0
            while rates[0] > min(THRESHOLDS) and bs[0] > 2 and guard < 25:
                b = max(int(bs[0] / step), 1)
                if b == bs[0]:
                    break
                bs.insert(0, b); rates.insert(0, rate_at(b)); guard += 1
            if guard:
                print(f"   {algo:14s} grid extended down to {bs[0]:,} ({guard} extra point(s))",
                      flush=True)
            budgets_a, rates = np.array(bs, dtype=np.int64), np.array(rates)
            print(f"   {algo:14s} max={rates.max():.3f}", flush=True)
            for b, r in zip(budgets_a, rates):
                new_curve_rows.append([setting, pmin, pmax, eps, algo, int(b), round(float(r), 5)])
            for T in THRESHOLDS:
                # each curve is crossed on its own abscissae — brute's grid is unchanged
                bc, ac = crossing(budgets, brute_rates, T), crossing(budgets_a, rates, T)
                ratio = (bc / ac) if (np.isfinite(bc) and np.isfinite(ac)) else float("nan")
                new_cube_rows.append([setting, pmin, pmax, eps, algo, int(T * 100),
                                      None if not np.isfinite(ac) else round(ac, 1),
                                      None if not np.isfinite(ratio) else round(ratio, 3)])

        # The non-degenerate ceiling is analytic (quadrature over the prior, no simulation), so it
        # costs nothing and gets the same downward extension treatment.
        bs = list(budgets)
        rates = [heisenberg_rate(b, eps, pmin, pmax) for b in bs]
        guard = 0
        while rates[0] > min(THRESHOLDS) and bs[0] > 2 and guard < 25:
            b = max(int(bs[0] / 4), 1)
            if b == bs[0]:
                break
            bs.insert(0, b); rates.insert(0, heisenberg_rate(b, eps, pmin, pmax)); guard += 1
        budgets_h, rates = np.array(bs, dtype=np.int64), np.array(rates)
        print(f"   {'oracle_hl':14s} max={rates.max():.3f}   "
              f"(exact oracle degenerate over {100*degeneracy_share(eps, pmin, pmax):.0f}% of the prior)",
              flush=True)
        for b, r in zip(budgets_h, rates):
            new_curve_rows.append([setting, pmin, pmax, eps, "oracle_hl", int(b), round(float(r), 5)])
        for T in THRESHOLDS:
            bc, ac = crossing(budgets, brute_rates, T), crossing(budgets_h, rates, T)
            ratio = (bc / ac) if (np.isfinite(bc) and np.isfinite(ac)) else float("nan")
            new_cube_rows.append([setting, pmin, pmax, eps, "oracle_hl", int(T * 100),
                                  None if not np.isfinite(ac) else round(ac, 1),
                                  None if not np.isfinite(ratio) else round(ratio, 3)])

        _merge(setting, new_curve_rows, new_cube_rows)
        done += 1
        print(f"   [{done}/{len(settings)}] merged into {CURVES} / {CUBE}", flush=True)

    print(f"\ndone — {done} setting(s) updated")
    check()


if __name__ == "__main__":
    main()
