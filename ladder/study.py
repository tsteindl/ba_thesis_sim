"""De-biased budget study of the phase-unwrapping ladder.

Mirrors the methodology of analysis/extensive_sweep.py exactly:
  * per budget, grid-tune the ladder on the TUNE seed (42), take the argmax config,
    then VALIDATE that single config on the independent TEST seed (2024) at high R;
  * brute force is parameter-free and re-evaluated here (self-consistent denominator),
    then cross-checked against the existing results/story_cube.csv (read-only);
  * convergence-vs-budget curves -> log-interpolated budget at which each algorithm
    reaches p* in {50,80,90,95}%; report the ratio B_brute(p*)/B_algo(p*).

Also records, per scenario, the branch-miss rate of the validated winner at the budget
nearest the 90% crossing (miss := error > 10*eps, i.e. a wrong-branch catastrophe,
computed as 1 - success_rate at 10*eps).

    python ladder/study.py            # full run (~30-60 min on 20 cores)
    python ladder/study.py --quick    # smoke run

Writes ladder/results/{ladder_curves,ladder_cube,ladder_winners,ladder_t31}.csv.
Nothing under results/ (the existing study) is touched.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E  # noqa: E402
from qmetrology.algorithms import find_phi_fixed_budget_brute_force as BF  # noqa: E402
from ladder.algorithms import (  # noqa: E402
    find_phi_fixed_budget_ladder as LADU,
    find_phi_fixed_budget_ladder_capped as LADC,
)

QUICK = "--quick" in sys.argv
SEED_TUNE, SEED_TEST = 42, 2024
R_TUNE = 500 if QUICK else 2000
R_TEST = 4000 if QUICK else 50_000
N_BUDGETS = 12 if QUICK else 36
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
STORY_CUBE = os.path.join(HERE, os.pardir, "results", "story_cube.csv")

# (phi_min, phi_max, eps) — the RESULTS.md precision sweep plus one shifted range
SCENARIOS = [
    (0.01, 0.1, 1e-3),
    (0.01, 0.1, 1e-4),
    (0.01, 0.1, 1e-5),
    (0.01, 0.1, 1e-6),
    (0.01, 0.1, 1e-7),
    (0.01, 0.1, 1e-8),
    (0.001, 0.01, 1e-4),
]
if QUICK:
    SCENARIOS = [(0.01, 0.1, 1e-3), (0.01, 0.1, 1e-4)]

GRID = {
    "m_stage": [50, 100, 200, 400],
    "z": [2.0, 2.5, 3.0],
    "safety": [0.6, 0.75, 0.9],
    "final_frac": [0.3, 0.5, 0.7],
}
if QUICK:
    GRID = {"m_stage": [100, 200], "z": [2.0], "safety": [0.75], "final_frac": [0.5]}

LADDERS = [("ladder_capped", LADC), ("ladder_unbounded", LADU)]


def label(pmin, pmax, eps):
    return f"U({pmin:g},{pmax:g}), eps={eps:.0e}"


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    """Analytic brute-force ~90% budget (same anchor as extensive_sweep)."""
    return 0.6724 / (n_min(pmax) * eps ** 2)


def heis90(eps):
    """Rough Heisenberg-scaling anchor for the unbounded ladder's 90% budget."""
    return 20.0 / eps


def budget_grid(pmax, eps):
    c = brute90(pmax, eps)
    lo = min(c / 50, heis90(eps) / 30)
    hi = 5 * c
    return np.unique(np.geomspace(lo, hi, N_BUDGETS).astype(np.int64))


def crossing(budgets, rates, T):
    """Log-interpolated budget at which the curve first reaches rate T (same as sweep)."""
    b, r = np.asarray(budgets, float), np.asarray(rates, float)
    if T <= r[0] or T > r[-1]:
        return float("nan")
    for i in range(1, len(r)):
        if r[i] >= T:
            if r[i] == r[i - 1]:
                return float(b[i])
            f = (T - r[i - 1]) / (r[i] - r[i - 1])
            return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1]))))
    return float("nan")


def _clean(cfg):
    return {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in cfg.items()}


def cube_reference():
    """Existing de-biased crossings from results/story_cube.csv (read-only)."""
    ref = {}
    if not os.path.exists(STORY_CUBE):
        return ref
    with open(STORY_CUBE) as f:
        for row in csv.DictReader(f):
            key = (float(row["phi_min"]), float(row["phi_max"]), float(row["eps"]),
                   row["algo"], int(row["threshold_pct"]))
            ref[key] = float(row["budget_to_reach"]) if row["budget_to_reach"] else float("nan")
    return ref


def run_scenario(pmin, pmax, eps, ref):
    name = label(pmin, pmax, eps)
    budgets = budget_grid(pmax, eps)
    print(f"\n=== {name}   N_min={n_min(pmax)}  budgets {budgets[0]:,}..{budgets[-1]:,} "
          f"({len(budgets)}) ===", flush=True)

    curves, winners = {}, {}
    curves["brute"] = np.array([E.success_rate(BF, {"budget": int(b)}, R_TEST,
                                               pmin, pmax, eps, SEED_TEST) for b in budgets])
    bc90_ref = ref.get((pmin, pmax, eps, "brute", 90))
    bc90_here = crossing(budgets, curves["brute"], 0.90)
    if bc90_ref and np.isfinite(bc90_here):
        print(f"   brute 90% crossing: {bc90_here:,.0f} here vs {bc90_ref:,.0f} in story_cube "
              f"(x{bc90_here / bc90_ref:.3f})", flush=True)

    for algo, fn in LADDERS:
        rates, wins = [], []
        for b in budgets:
            res = E.grid_full(fn, {**GRID, "budget": [int(b)]}, R_TUNE, pmin, pmax, eps, SEED_TUNE)
            r_tune, _bu, cfg = max(res, key=lambda x: x[0])
            rate = E.success_rate(fn, cfg, R_TEST, pmin, pmax, eps, SEED_TEST) if r_tune > 0 else 0.0
            rates.append(rate)
            wins.append((int(b), fn.__name__, _clean(cfg)))
        curves[algo] = np.array(rates)
        winners[algo] = wins
        # branch-miss telemetry at the budget nearest the 90% crossing
        c90 = crossing(budgets, curves[algo], 0.90)
        miss = float("nan")
        if np.isfinite(c90):
            i = int(np.argmin(np.abs(np.log(budgets.astype(float)) - np.log(c90))))
            ok10 = E.success_rate(fn, wins[i][2] | {"budget": int(budgets[i])}, R_TEST,
                                  pmin, pmax, 10 * eps, SEED_TEST)
            miss = 1.0 - ok10
        print(f"   {algo:17s} max={curves[algo].max():.3f}  90%@{c90:,.0f}  "
              f"miss@90%={100 * miss:.3f}%", flush=True)
        winners[algo + "_miss90"] = miss
    return name, pmin, pmax, eps, budgets, curves, winners


def scenario_rows(name, pmin, pmax, eps, budgets, curves, winners, ref):
    crow, xrow, wrow = [], [], []
    algos = ["brute"] + [a for a, _ in LADDERS]
    for algo in algos:
        for b, r in zip(budgets, curves[algo]):
            crow.append([name, pmin, pmax, eps, algo, int(b), round(float(r), 5)])
    bc = {T: crossing(budgets, curves["brute"], T) for T in THRESHOLDS}
    for algo in algos:
        for T in THRESHOLDS:
            ac = crossing(budgets, curves[algo], T)
            ratio = (bc[T] / ac) if (np.isfinite(bc[T]) and np.isfinite(ac)) else float("nan")
            miss = winners.get(algo + "_miss90", "") if T == 0.90 else ""
            xrow.append([name, pmin, pmax, eps, algo, int(T * 100),
                         None if not np.isfinite(ac) else round(ac, 1),
                         None if not np.isfinite(ratio) else round(ratio, 3),
                         "this_study",
                         round(miss, 5) if isinstance(miss, float) and np.isfinite(miss) else ""])
        # existing de-biased numbers for context (read-only from story_cube)
    for algo in ["reverse_eng", "linear", "binary", "separable"]:
        for T in THRESHOLDS:
            ac = ref.get((pmin, pmax, eps, algo, int(T * 100)), float("nan"))
            ratio = (bc[T] / ac) if (np.isfinite(bc[T]) and np.isfinite(ac)) else float("nan")
            if np.isfinite(ac):
                xrow.append([name, pmin, pmax, eps, algo, int(T * 100), round(ac, 1),
                             None if not np.isfinite(ratio) else round(ratio, 3),
                             "story_cube", ""])
    for algo, _fn in LADDERS:
        for b, fnname, cfg in winners[algo]:
            wrow.append([name, pmin, pmax, eps, algo, b, fnname, json.dumps(cfg)])
    return crow, xrow, wrow


def table31_check():
    """Ladder rows for the Table-3.1 operating point (budget 10^4, eps 1e-3, narrow)."""
    rows = []
    for algo, fn in LADDERS:
        res = E.grid_full(fn, {**GRID, "budget": [10_000]}, R_TUNE, 0.01, 0.1, 1e-3, SEED_TUNE)
        r_tune, _b, cfg = max(res, key=lambda x: x[0])
        deb = E.success_rate(fn, cfg, R_TEST, 0.01, 0.1, 1e-3, SEED_TEST)
        lo, hi = E.wilson(deb, R_TEST)
        rows.append([algo, round(100 * deb, 2), round(100 * lo, 2), round(100 * hi, 2),
                     json.dumps(_clean(cfg))])
        print(f"   T3.1 point {algo}: {100 * deb:.2f}%  cfg={_clean(cfg)}", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = cube_reference()
    files = {
        f"{OUT}/ladder_curves.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "rate"],
        f"{OUT}/ladder_cube.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "threshold_pct",
                                   "budget_to_reach", "ratio_vs_brute", "source", "miss_rate_at_90"],
        f"{OUT}/ladder_winners.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "fn", "params"],
    }
    for path, header in files.items():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for i, (pmin, pmax, eps) in enumerate(SCENARIOS, 1):
        try:
            res = run_scenario(pmin, pmax, eps, ref)
            crow, xrow, wrow = scenario_rows(*res, ref)
            for path, rows in [(f"{OUT}/ladder_curves.csv", crow),
                               (f"{OUT}/ladder_cube.csv", xrow),
                               (f"{OUT}/ladder_winners.csv", wrow)]:
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerows(rows)
            print(f"   [{i}/{len(SCENARIOS)}] written.", flush=True)
        except Exception as ex:  # keep completed scenarios on a crash
            print(f"   !! scenario {label(pmin, pmax, eps)} FAILED: {type(ex).__name__}: {ex}",
                  flush=True)

    print("\n=== Table 3.1 operating point (budget 10,000, eps=1e-3, U(0.01,0.1)) ===", flush=True)
    rows = table31_check()
    with open(f"{OUT}/ladder_t31.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "debiased_pct", "wilson_lo", "wilson_hi", "params"])
        w.writerows(rows)
    print(f"\ndone — wrote {OUT}/ladder_{{curves,cube,winners,t31}}.csv")


if __name__ == "__main__":
    main()
