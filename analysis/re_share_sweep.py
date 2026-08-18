"""Can ONE budget share replace the per-budget-tuned exploration size of reverse engineering?

Runs over every scenario and budget of analysis/extensive_sweep.py. The fixed-m' arm is NOT re-tuned:
its de-biased rate and winning m' are read back from results/story_{curves,winners}.csv, so the
comparison uses exactly the same scenarios, budgets, seeds and R as the reported numbers.

Four arms per operating point, all evaluated on the TEST seed:

  m'_tuned    the reported row -- m' grid-tuned at THIS budget (status quo, ~52 configs of tuning)
  rho_tuned   pilot_share grid-tuned at THIS budget (10 configs)
  rho_fixed   ONE share, never tuned -- the arm that matters
  m'_fixed    ONE shot count, never tuned -- the honest counterpart to rho_fixed

Writes results/re_share.csv.
    python analysis/re_share_sweep.py [--quick] [--rho 0.05] [--resume]
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_reverse_engineering_risk as RE_R,
    find_phi_fixed_budget_reverse_engineering_share as RE_S,
)

QUICK = "--quick" in sys.argv
RESUME = "--resume" in sys.argv
RHO_FIXED = float(sys.argv[sys.argv.index("--rho") + 1]) if "--rho" in sys.argv else 0.05
M_FIXED = 45                      # best single shot count found in analysis/exploration_study.py
RHO_GRID = [0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2, 0.3]
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
OUT = "results/re_share.csv"
# only compare where the race is live: outside this band every arm is pinned at 0 % or 100 %
LIVE_LO, LIVE_HI = 0.01, 0.995


def load_reported():
    """[(setting, pmin, pmax, eps, budget, tuned_m, tuned_rate), ...] for reverse_eng_risk."""
    wins = {}
    with open("results/story_winners.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "reverse_eng_risk":
                wins[(r["setting"], int(r["budget"]))] = json.loads(r["params"])["m_exploration"]
    out = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] != "reverse_eng_risk":
                continue
            key = (r["setting"], int(r["budget"]))
            if key not in wins:
                continue
            rate = float(r["rate"])
            if not (LIVE_LO < rate < LIVE_HI):
                continue
            out.append((r["setting"], float(r["phi_min"]), float(r["phi_max"]), float(r["eps"]),
                        int(r["budget"]), int(wins[key]), rate))
    return out


def main():
    pts = load_reported()
    if QUICK:
        pts = pts[::37]
    done = set()
    if RESUME and os.path.exists(OUT):
        with open(OUT, newline="") as f:
            done = {(r["setting"], int(r["budget"])) for r in csv.DictReader(f)}
        pts = [p for p in pts if (p[0], p[4]) not in done]
    print(f"{len(pts)} live operating points to evaluate "
          f"({len(done)} already done)" if done else f"{len(pts)} live operating points", flush=True)

    header = ["setting", "phi_min", "phi_max", "eps", "budget", "tuned_m", "m_tuned_rate",
              "rho_tuned_rate", "best_rho", "rho_fixed_rate", "m_fixed_rate"]
    if not (RESUME and os.path.exists(OUT)):
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(header)

    last = None
    for i, (name, pmin, pmax, eps, b, tuned_m, tuned_rate) in enumerate(pts, 1):
        if name != last:
            print(f"\n=== {name} ===", flush=True)
            last = name
        # tune the share on seed 42, validate the winner on seed 2024
        res = E.grid_full(RE_S, {"pilot_share": RHO_GRID, "eps_target": [eps], "budget": [b]},
                          R_TUNE, pmin, pmax, eps, SEED_TUNE)
        _r, _bud, cfg = max(res, key=lambda x: x[0])
        rho_tuned = E.success_rate(RE_S, cfg, R_TEST, pmin, pmax, eps, SEED_TEST)
        # the two untuned arms
        rho_fixed = E.success_rate(RE_S, {"pilot_share": RHO_FIXED, "eps_target": eps, "budget": b},
                                   R_TEST, pmin, pmax, eps, SEED_TEST)
        m_fixed = E.success_rate(RE_R, {"m_exploration": M_FIXED, "eps_target": eps, "budget": b},
                                 R_TEST, pmin, pmax, eps, SEED_TEST)
        row = [name, pmin, pmax, eps, b, tuned_m, round(100 * tuned_rate, 3),
               round(100 * rho_tuned, 3), cfg["pilot_share"], round(100 * rho_fixed, 3),
               round(100 * m_fixed, 3)]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  [{i}/{len(pts)}] B={b:>15,}  m'={tuned_m:>6} {100*tuned_rate:6.2f} | "
              f"rho* {cfg['pilot_share']:.3f} {100*rho_tuned:6.2f} | "
              f"rho={RHO_FIXED} {100*rho_fixed:6.2f} ({100*(rho_fixed-tuned_rate):+5.2f}) | "
              f"m'={M_FIXED} {100*m_fixed:6.2f} ({100*(m_fixed-tuned_rate):+5.2f})", flush=True)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
