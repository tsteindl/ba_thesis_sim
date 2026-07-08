"""Budget-ratio sweep.

For each scenario (phi_min, phi_max, eps): sweep the budget; at every budget grid-tune each adaptive
algorithm on the tune seed (42) and re-evaluate the winner on the test seed (2024); build the
convergence-vs-budget curve and locate where it reaches p* in {50,80,90,95}%. Reports the budget
ratio B_brute(p*)/B_algo(p*). Brute and separable are parameter-free.

Writes results/story_{curves,cube,winners}.csv.
    python analysis/extensive_sweep.py [--quick | --fine | --max]
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_separable as SEP,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_binary_search as BIN,
    find_phi_fixed_budget_binary_search_anneal_m as BINA,
    find_phi_fixed_budget_reverse_engineering as RE,
)

QUICK = "--quick" in sys.argv
MAX = "--max" in sys.argv                                   # maxed grids + fineness + eps to 1e-8 + wild ranges
FINE = MAX or ("--fine" in sys.argv) or ("--heavy" in sys.argv)  # FINER grids (same R), finer budgets, +eps=1e-6
SEED_TUNE, SEED_TEST = 42, 2024
# R is deliberately NOT increased: the current R is already tight (SE ~0.25pp) and the de-biasing
# (validate on a fresh seed) is what controls bias, not R. --fine/--max only refine the SEARCH.
R_TUNE = 500 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
N_BUDGETS = 8 if QUICK else (40 if MAX else (30 if FINE else 18))
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]

# (phi_min, phi_max, eps)
SCENARIOS = [
    (0.01, 0.1, 1e-3),
    (0.01, 0.1, 1e-4),
    (0.01, 0.1, 1e-5),
    (0.001, 0.1, 1e-3),
    (0.001, 0.1, 1e-4),
    (0.001, 0.01, 1e-4),
    (0.005, 0.05, 1e-4),
]
if MAX:
    # full precision sweep (incl. 1e-7, 1e-8) at the two main ranges, plus WILD dynamic ranges
    SCENARIOS = []
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        SCENARIOS += [(0.01, 0.1, eps), (0.001, 0.01, eps)]
    for pmin, pmax in [(0.001, 0.1), (0.0001, 0.1), (0.0001, 0.01), (0.005, 0.05), (0.0001, 0.001)]:
        for eps in [1e-4, 1e-6]:
            SCENARIOS.append((pmin, pmax, eps))
    SCENARIOS.append((1e-5, 0.01, 1e-5))   # extreme downward dynamic range
elif FINE:  # add the eps=1e-6 saturation points (mainly to SHOW the advantage plateaus)
    SCENARIOS += [(0.01, 0.1, 1e-6), (0.001, 0.1, 1e-6)]
if QUICK:
    SCENARIOS = [(0.01, 0.1, 1e-3), (0.01, 0.1, 1e-4)]


def label(pmin, pmax, eps):
    hi = {round(np.pi / 2, 4): "pi/2", 0.1: "0.1", 0.05: "0.05", 0.01: "0.01"}.get(round(pmax, 4), f"{pmax:g}")
    return f"U({pmin:g},{hi}), eps={eps:.0e}"


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    """Analytic budget for brute force to reach ~90% convergence (z=1.64, err=1/(2 N_min sqrt(m)))."""
    return 0.6724 / (n_min(pmax) * eps ** 2)


def budget_grid(pmax, eps):
    c = brute90(pmax, eps)
    return np.unique(np.geomspace(c / 50, c * 30, N_BUDGETS).astype(np.int64))


def grids(pmax, eps):
    """Per-scenario parameter grids; m_exploration geomspace-scaled to the regime.
    --fine and --max densify the search grid; R is unchanged."""
    nm, nr, nb = (44, 52, 26) if MAX else ((32, 40, 22) if FINE else (16, 20, 14))
    c = brute90(pmax, eps)
    m_hi = int(np.clip(c, 200, 300_000))
    m = np.unique(np.geomspace(3, m_hi, nm).astype(int))
    m_b = np.unique(np.geomspace(20, m_hi, nb).astype(int))
    # --fine/--max densify mainly the CONTINUOUS m axis; --max also widens the discrete shape lists.
    lin = {"m_exploration": m, "lookback_window": [1, 2, 3, 5] if FINE else [1, 2, 5],
           "safeguard": [0, 1, 2, 3, 5] if MAX else ([0, 1, 2, 5]),
           "inc": [1, 2, 5]}
    re = {"m_exploration": np.unique(np.geomspace(20, m_hi, nr).astype(int)),
          "safeguard": [0.8, 0.85, 0.9, 0.95] if not FINE else [0.8, 0.85, 0.9, 0.925, 0.95, 0.975]}
    conf = [0.5, 0.65, 0.8, 0.9, 0.95] if MAX else [0.5, 0.8, 0.9]
    binv = [(BIN, {"m_exploration": m_b, "safeguard": [0, 1, 2], "conf": conf}),
            (BINA, {"m_exploration": m_b, "safeguard": [1, 2] if MAX else [1], "conf": [0.8]})]
    return {"linear": [(LIN, lin)], "reverse_eng": [(RE, re)], "binary": binv}


def tune_winner(variants, budget, pmin, pmax, eps):
    """Grid-tune over all variants on the TUNE seed; return the argmax (fn, cfg)."""
    pool = []  # (tune_rate, fn, cfg)
    for fn, g in variants:
        res = E.grid_full(fn, {**g, "budget": [int(budget)]}, R_TUNE, pmin, pmax, eps, SEED_TUNE)
        pool += [(r, fn, cfg) for r, _b, cfg in res]
    _tr, fn, cfg = max(pool, key=lambda x: x[0])
    return fn, cfg


def tune_validate(variants, budget, pmin, pmax, eps):
    """Tune on seed 42, then evaluate the winning config on seed 2024; return its success rate."""
    fn, cfg = tune_winner(variants, budget, pmin, pmax, eps)
    return E.success_rate(fn, cfg, R_TEST, pmin, pmax, eps, SEED_TEST)


def crossing(budgets, rates, T):
    """Log-interpolated budget at which the curve first reaches rate T (nan if uncrossed)."""
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
    """Make a param dict JSON-serialisable (numpy scalars -> python)."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def run_scenario(pmin, pmax, eps):
    name = label(pmin, pmax, eps)
    budgets = budget_grid(pmax, eps)
    G = grids(pmax, eps)
    print(f"\n=== {name}   N_min={n_min(pmax)}  budgets {budgets[0]:,}..{budgets[-1]:,} ({len(budgets)}) ===", flush=True)
    curves, winners = {}, {}
    # parameter-free baselines
    curves["brute"] = np.array([E.success_rate(BF, {"budget": int(b)}, R_TEST, pmin, pmax, eps, SEED_TEST) for b in budgets])
    curves["separable"] = np.array([E.success_rate(SEP, {"budget": int(b)}, R_TEST, pmin, pmax, eps, SEED_TEST) for b in budgets])
    # tuned adaptive algorithms — keep the winning (fn, cfg) per budget so error_curves needn't re-tune
    for algo, variants in G.items():
        rates, wins = [], []
        for b in budgets:
            fn, cfg = tune_winner(variants, b, pmin, pmax, eps)
            rates.append(E.success_rate(fn, cfg, R_TEST, pmin, pmax, eps, SEED_TEST))
            wins.append((int(b), fn.__name__, _clean(cfg)))
        curves[algo] = np.array(rates)
        winners[algo] = wins
        print(f"   {algo:12s} max={curves[algo].max():.3f}", flush=True)
    return name, pmin, pmax, eps, budgets, curves, winners


ORDER = ["brute", "separable", "linear", "binary", "reverse_eng"]
ADAPT = ["linear", "binary", "reverse_eng"]


def _scenario_rows(name, pmin, pmax, eps, budgets, curves, winners):
    import json
    crow, xrow, wrow = [], [], []
    for algo in ORDER:
        for b, r in zip(budgets, curves[algo]):
            crow.append([name, pmin, pmax, eps, algo, int(b), round(float(r), 5)])
    best_adapt = np.max([curves[a] for a in ADAPT], axis=0)
    bc = {T: crossing(budgets, curves["brute"], T) for T in THRESHOLDS}
    for algo in ORDER + ["best_adaptive"]:
        rates = best_adapt if algo == "best_adaptive" else curves[algo]
        for T in THRESHOLDS:
            ac = crossing(budgets, rates, T)
            ratio = (bc[T] / ac) if (np.isfinite(bc[T]) and np.isfinite(ac)) else float("nan")
            xrow.append([name, pmin, pmax, eps, algo, int(T * 100),
                         None if not np.isfinite(ac) else round(ac, 1),
                         None if not np.isfinite(ratio) else round(ratio, 3)])
    for algo, wins in winners.items():
        for b, fnname, cfg in wins:
            wrow.append([name, pmin, pmax, eps, algo, b, fnname, json.dumps(cfg)])
    return crow, xrow, wrow


def main():
    os.makedirs("results", exist_ok=True)
    # (re)create files with headers, then APPEND per scenario so a crash keeps completed scenarios
    files = {
        "results/story_curves.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "rate"],
        "results/story_cube.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "threshold_pct", "budget_to_reach", "ratio_vs_brute"],
        "results/story_winners.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "fn", "params"],
    }
    for path, header in files.items():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for i, (pmin, pmax, eps) in enumerate(SCENARIOS, 1):
        try:
            res = run_scenario(pmin, pmax, eps)
            crow, xrow, wrow = _scenario_rows(*res)
            for path, rows in [("results/story_curves.csv", crow),
                               ("results/story_cube.csv", xrow),
                               ("results/story_winners.csv", wrow)]:
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerows(rows)
            print(f"   [{i}/{len(SCENARIOS)}] written.", flush=True)
        except Exception as ex:  # never let one scenario kill the whole unattended run
            print(f"   !! scenario {label(pmin, pmax, eps)} FAILED: {type(ex).__name__}: {ex}", flush=True)
    print("\ndone — wrote results/story_curves.csv, story_cube.csv, story_winners.csv")


if __name__ == "__main__":
    main()
