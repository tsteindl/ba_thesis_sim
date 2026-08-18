"""Binary search: pilot = the opening probe phi_hat_0, or the deepest non-flagged probe?

Settles §12 of results/SAFEGUARD_DERIVATION.md over the WHOLE sweep instead of 3 scenarios.

These are two different algorithms, and the metric that decides between them is convergence, not the
overshoot rate:

  deep    pilot = deepest probe not flagged as an overshoot   (what qmetrology/algorithms.py does)
  first   pilot = the opening probe at N_min                   (sigma is larger, but A2 holds by
                                                                construction and there is no
                                                                selection bias)

The `deep` arm is NOT re-run: its de-biased rate and winning config are read back from
results/story_{curves,winners}.csv, so both arms use identical scenarios, budgets, grids, seeds and R.
The `first` arm is tuned over the same grid extensive_sweep.py used (--max: 26 exploration sizes x 5
confidences), on seed 42, and validated on seed 2024 at R = 40,000.

Also records the mean budget actually consumed by each arm, because Algorithm 5's exploration can
overspend (see results/budget_audit.csv), and a comparison is only meaningful at equal cost.

    python analysis/binary_pilot_sweep.py [--quick] [--resume] [--stride N]
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import experiments as E
from qmetrology.algorithms import find_phi_fixed_budget_binary_search_risk as BIN_DEEP
from binary_pilot_study import binary_pilot_first as BIN_FIRST

QUICK = "--quick" in sys.argv
RESUME = "--resume" in sys.argv
STRIDE = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 1
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
OUT = "results/binary_pilot_sweep.csv"
LIVE_LO, LIVE_HI = 0.01, 0.99


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def grid_for(pmax, eps):
    """The exact binary_risk grid of analysis/extensive_sweep.py under --max."""
    m_hi = int(np.clip(brute90(pmax, eps), 200, 300_000))
    m_b = np.unique(np.geomspace(20, m_hi, 26).astype(int))
    return {"m_exploration": m_b, "conf": [0.5, 0.65, 0.8, 0.9, 0.95], "eps_target": [eps]}


def load_points():
    """Live binary_risk points, with the deep arm's stored config and rate."""
    wins = {}
    with open("results/story_winners.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "binary_risk":
                wins[(r["setting"], int(r["budget"]))] = json.loads(r["params"])
    pts = []
    with open("results/story_curves.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["algo"] != "binary_risk":
                continue
            key = (r["setting"], int(r["budget"]))
            rate = float(r["rate"])
            if key not in wins or not (LIVE_LO < rate < LIVE_HI):
                continue
            pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                            phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                            budget=int(r["budget"]), deep_cfg=wins[key], deep_rate=100 * rate))
    return pts


HEADER = ["setting", "phi_min", "phi_max", "eps", "budget",
          "deep_rate", "first_rate", "delta",
          "deep_m", "deep_conf", "first_m", "first_conf",
          "deep_spent_ratio", "first_spent_ratio"]


def main():
    os.makedirs("results", exist_ok=True)
    pts = load_points()[:: STRIDE]
    if QUICK:
        pts = pts[::40]
    done = set()
    if RESUME and os.path.exists(OUT):
        with open(OUT, newline="") as f:
            done = {(r["setting"], int(r["budget"])) for r in csv.DictReader(f)}
        pts = [p for p in pts if (p["setting"], p["budget"]) not in done]
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    print(f"{len(pts)} live operating points to run ({len(done)} already done)", flush=True)

    last = None
    for i, p in enumerate(pts, 1):
        if p["setting"] != last:
            print(f"\n=== {p['setting']} ===", flush=True)
            last = p["setting"]
        g = {**grid_for(p["phi_max"], p["eps"]), "budget": [p["budget"]]}
        res = E.grid_full(BIN_FIRST, g, R_TUNE, p["phi_min"], p["phi_max"], p["eps"], SEED_TUNE)
        _r, _b, cfg = max(res, key=lambda x: x[0])
        f_rate, f_spent = E.rate_and_budget(BIN_FIRST, cfg, R_TEST, p["phi_min"], p["phi_max"],
                                            p["eps"], SEED_TEST)
        # the deep arm's rate is the reported one; re-measure only its SPEND, which the sweep
        # never recorded, so the two arms can be compared at equal cost
        dcfg = {**p["deep_cfg"], "budget": p["budget"]}
        _dr, d_spent = E.rate_and_budget(BIN_DEEP, dcfg, 4000, p["phi_min"], p["phi_max"],
                                         p["eps"], SEED_TEST)
        row = [p["setting"], p["phi_min"], p["phi_max"], p["eps"], p["budget"],
               round(p["deep_rate"], 3), round(100 * f_rate, 3),
               round(100 * f_rate - p["deep_rate"], 3),
               p["deep_cfg"].get("m_exploration"), p["deep_cfg"].get("conf"),
               int(cfg["m_exploration"]), cfg["conf"],
               round(d_spent / p["budget"], 4), round(f_spent / p["budget"], 4)]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows([row])
        print(f"  [{i}/{len(pts)}] B={p['budget']:>15,}  deep {p['deep_rate']:6.2f}  "
              f"first {100*f_rate:6.2f}  ({100*f_rate - p['deep_rate']:+5.2f})  "
              f"spend d{d_spent/p['budget']:.2f} f{f_spent/p['budget']:.2f}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
