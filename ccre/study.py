"""De-biased budget study of CCRE (Confidence-Calibrated Reverse Engineering).

Mirrors ladder/study.py's methodology exactly (which mirrors analysis/extensive_sweep.py):
  * per budget, grid-tune CCRE on the TUNE seed (42), take the argmax config, then VALIDATE
    that single config on the independent TEST seed (2024) at high R;
  * brute force is parameter-free and re-evaluated here (self-consistent denominator), then
    cross-checked against the existing results/story_cube.csv (read-only);
  * convergence-vs-budget curves -> log-interpolated budget at which each algorithm reaches
    p* in {50,80,90,95}%; report the ratio B_brute(p*)/B_algo(p*).
  * reverse engineering is also recomputed (read from story_cube where available, else
    left blank) so Table C1 is a direct three-way comparison: brute / RE / CCRE.

Also records, at the budget nearest CCRE's own 90% crossing, the empirical overshoot rate
(N*phi >= pi/2 on the round that set the final depth) vs the predicted (1-conf) bound —
the "provable safety" claim's empirical check.

    python ccre/study.py            # full run
    python ccre/study.py --quick    # smoke run

Writes ccre/results/{ccre_curves,ccre_cube,ccre_winners,ccre_t31}.csv.
Nothing under qmetrology/, ladder/, analysis/, results/ is touched.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E  # noqa: E402
from qmetrology.algorithms import find_phi_fixed_budget_brute_force as BF  # noqa: E402
from ccre.algorithms import find_phi_fixed_budget_ccre as CCRE  # noqa: E402

QUICK = "--quick" in sys.argv
SEED_TUNE, SEED_TEST = 42, 2024
R_TUNE = 500 if QUICK else 2000
R_TEST = 4000 if QUICK else 50_000
N_BUDGETS = 12 if QUICK else 36
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
STORY_CUBE = os.path.join(HERE, os.pardir, "results", "story_cube.csv")

# Same operating points as ladder/results/LADDER.md, for a direct three-way comparison.
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
    "m_round": sorted(set(np.geomspace(10, 10_000, 10).astype(int).tolist())),
    "conf": [0.70, 0.85, 0.95, 0.99],
    "n_rounds": [1, 2, 3],
}
if QUICK:
    GRID = {"m_round": [50, 500], "conf": [0.9, 0.99], "n_rounds": [1, 2]}


def label(pmin, pmax, eps):
    return f"U({pmin:g},{pmax:g}), eps={eps:.0e}"


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def budget_grid(pmax, eps):
    c = brute90(pmax, eps)
    return np.unique(np.geomspace(c / 50, c * 5, N_BUDGETS).astype(np.int64))


def crossing(budgets, rates, T):
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
    """Existing de-biased crossings from results/story_cube.csv (read-only) -- gives us RE
    (and brute, for a sanity cross-check) at the same operating points without recomputation."""
    ref = {}
    if not os.path.exists(STORY_CUBE):
        return ref
    with open(STORY_CUBE) as f:
        for row in csv.DictReader(f):
            key = (float(row["phi_min"]), float(row["phi_max"]), float(row["eps"]),
                   row["algo"], int(row["threshold_pct"]))
            ref[key] = float(row["budget_to_reach"]) if row["budget_to_reach"] else float("nan")
    return ref


def overshoot_check(cfg, pmin, pmax, eps, R=4000, seed=SEED_TEST):
    """Empirical P(final round's N*phi >= pi/2) vs the predicted (1 - conf)."""
    seeds = np.random.default_rng(seed + 1).integers(0, 2**63, size=R)
    over = 0
    for s in seeds:
        rng = np.random.default_rng(int(s))
        phi = rng.uniform(pmin, pmax)
        tr = []
        CCRE(rng, phi, pmax, pmin, trace=tr, **cfg)
        if tr and tr[-1][0] * phi >= np.pi / 2:
            over += 1
    return over / R


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

    rates, wins = [], []
    for b in budgets:
        res = E.grid_full(CCRE, {**GRID, "budget": [int(b)]}, R_TUNE, pmin, pmax, eps, SEED_TUNE)
        r_tune, _bu, cfg = max(res, key=lambda x: x[0])
        rate = E.success_rate(CCRE, cfg, R_TEST, pmin, pmax, eps, SEED_TEST) if r_tune > 0 else 0.0
        rates.append(rate)
        wins.append((int(b), _clean(cfg)))
    curves["ccre"] = np.array(rates)
    winners["ccre"] = wins
    c90 = crossing(budgets, curves["ccre"], 0.90)
    miss_empirical, miss_predicted = float("nan"), float("nan")
    if np.isfinite(c90):
        i = int(np.argmin(np.abs(np.log(budgets.astype(float)) - np.log(c90))))
        cfg90 = dict(wins[i][1])
        miss_predicted = 1.0 - cfg90.get("conf", 0.95)
        miss_empirical = overshoot_check(cfg90, pmin, pmax, eps)
    print(f"   ccre  max={curves['ccre'].max():.3f}  90%@{c90:,.0f}  "
          f"overshoot empirical={100 * miss_empirical:.2f}% vs predicted={100 * miss_predicted:.2f}%",
          flush=True)
    winners["ccre_miss90_empirical"] = miss_empirical
    winners["ccre_miss90_predicted"] = miss_predicted
    return name, pmin, pmax, eps, budgets, curves, winners


def scenario_rows(name, pmin, pmax, eps, budgets, curves, winners, ref):
    crow, xrow, wrow = [], [], []
    for algo in ["brute", "ccre"]:
        for b, r in zip(budgets, curves[algo]):
            crow.append([name, pmin, pmax, eps, algo, int(b), round(float(r), 5)])
    bc = {T: crossing(budgets, curves["brute"], T) for T in THRESHOLDS}
    for T in THRESHOLDS:
        ac = crossing(budgets, curves["ccre"], T)
        ratio = (bc[T] / ac) if (np.isfinite(bc[T]) and np.isfinite(ac)) else float("nan")
        emp = winners["ccre_miss90_empirical"] if T == 0.90 else ""
        pred = winners["ccre_miss90_predicted"] if T == 0.90 else ""
        xrow.append([name, pmin, pmax, eps, "ccre", int(T * 100),
                     None if not np.isfinite(ac) else round(ac, 1),
                     None if not np.isfinite(ratio) else round(ratio, 3), "this_study",
                     round(emp, 5) if isinstance(emp, float) and np.isfinite(emp) else "",
                     round(pred, 5) if isinstance(pred, float) and np.isfinite(pred) else ""])
        xrow.append([name, pmin, pmax, eps, "brute", int(T * 100),
                     None if not np.isfinite(bc[T]) else round(bc[T], 1),
                     1.0 if np.isfinite(bc[T]) else None, "this_study", "", ""])
    for algo in ["reverse_eng", "linear", "binary", "separable"]:
        for T in THRESHOLDS:
            ac = ref.get((pmin, pmax, eps, algo, int(T * 100)), float("nan"))
            ratio = (bc[T] / ac) if (np.isfinite(bc[T]) and np.isfinite(ac)) else float("nan")
            if np.isfinite(ac):
                xrow.append([name, pmin, pmax, eps, algo, int(T * 100), round(ac, 1),
                             None if not np.isfinite(ratio) else round(ratio, 3),
                             "story_cube", "", ""])
    for b, cfg in winners["ccre"]:
        wrow.append([name, pmin, pmax, eps, "ccre", b, json.dumps(cfg)])
    return crow, xrow, wrow


def table31_check():
    """CCRE row for the Table-3.1 operating point (budget 10^4, eps=1e-3, narrow)."""
    res = E.grid_full(CCRE, {**GRID, "budget": [10_000]}, R_TUNE, 0.01, 0.1, 1e-3, SEED_TUNE)
    r_tune, _b, cfg = max(res, key=lambda x: x[0])
    deb = E.success_rate(CCRE, cfg, R_TEST, 0.01, 0.1, 1e-3, SEED_TEST)
    lo, hi = E.wilson(deb, R_TEST)
    print(f"   T3.1 point ccre: {100 * deb:.2f}%  cfg={_clean(cfg)}", flush=True)
    return [["ccre", round(100 * deb, 2), round(100 * lo, 2), round(100 * hi, 2),
             json.dumps(_clean(cfg))]]


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = cube_reference()
    files = {
        f"{OUT}/ccre_curves.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "rate"],
        f"{OUT}/ccre_cube.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "threshold_pct",
                                 "budget_to_reach", "ratio_vs_brute", "source",
                                 "overshoot_empirical_at_90", "overshoot_predicted_at_90"],
        f"{OUT}/ccre_winners.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "params"],
    }
    for path, header in files.items():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for i, (pmin, pmax, eps) in enumerate(SCENARIOS, 1):
        try:
            res = run_scenario(pmin, pmax, eps, ref)
            crow, xrow, wrow = scenario_rows(*res, ref)
            for path, rows in [(f"{OUT}/ccre_curves.csv", crow),
                               (f"{OUT}/ccre_cube.csv", xrow),
                               (f"{OUT}/ccre_winners.csv", wrow)]:
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerows(rows)
            print(f"   [{i}/{len(SCENARIOS)}] written.", flush=True)
        except Exception as ex:
            print(f"   !! scenario {label(pmin, pmax, eps)} FAILED: {type(ex).__name__}: {ex}",
                  flush=True)

    print("\n=== Table 3.1 operating point (budget 10,000, eps=1e-3, U(0.01,0.1)) ===", flush=True)
    rows = table31_check()
    with open(f"{OUT}/ccre_t31.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "debiased_pct", "wilson_lo", "wilson_hi", "params"])
        w.writerows(rows)
    print(f"\ndone — wrote {OUT}/ccre_{{curves,cube,winners,t31}}.csv")


if __name__ == "__main__":
    main()
