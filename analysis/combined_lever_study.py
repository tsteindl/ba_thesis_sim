"""Combining the two levers: precision (eps) x dynamic range (phi_min / phi_max).

Maximum-chase sweep to find the single largest realizable budget-ratio advantage of the adaptive
strategies (reverse engineering) over brute force, and to locate both breaking points:
  - phi_min -> 0, where RE's single-shot depth inference fails on the smallest phases;
  - phi_max -> large, where the first estimate aliases (N_min gets too small).

Reuses the de-biased tune(seed 42) / validate(seed 2024) crossing-ratio engine from extensive_sweep,
so the ratios are directly comparable to Table 3.2. Writes its OWN results/lever_{curves,cube,winners}.csv
(does not touch story_*.csv).

    python analysis/combined_lever_study.py [--quick | --fine]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extensive_sweep as X  # de-biasing + crossing-ratio engine (run_scenario, _scenario_rows, grids, ...)

QUICK = "--quick" in sys.argv

# phi_max stays "small" (RE-safe: n_min*phi_max < pi/2 by construction) but is pushed up toward the
# aliasing edge; phi_min is pushed down toward the depth-inference breaking point.
PHI_MIN = [0.01, 1e-3, 1e-4, 3e-5, 1e-5]
PHI_MAX = [0.05, 0.1, 0.15, 0.2, 0.3]

# Layer 1: 2D (phi_min x phi_max) frontier at a near-saturated precision -> the ridge / optimum.
LAYER1 = [(pmin, pmax, 1e-6) for pmax in PHI_MAX for pmin in PHI_MIN]
# Layer 2: precision deepening to 1e-8 along three representative priors (1e-6 already in Layer 1).
LAYER2 = [(pmin, pmax, eps)
          for (pmin, pmax) in [(0.01, 0.1), (1e-4, 0.1), (1e-4, 0.2)]
          for eps in [1e-4, 1e-5, 1e-7, 1e-8]]
# Layer 3: breaking points -- phi_min past the edge, phi_max past the RE-safe edge, one deep-eps extreme.
LAYER3 = [(3e-6, 0.1, 1e-6), (1e-6, 0.1, 1e-6),
          (1e-3, 0.4, 1e-6), (1e-4, 0.4, 1e-6),
          (1e-5, 0.2, 1e-8)]

SCENARIOS = list(dict.fromkeys(LAYER1 + LAYER2 + LAYER3))  # dedup, preserve order

if QUICK:  # fast smoke: moderate eps -> small budgets; spans narrow / wide / raised-phi_max / edge
    SCENARIOS = [(0.01, 0.1, 1e-4), (1e-4, 0.1, 1e-4), (1e-4, 0.2, 1e-6), (1e-3, 0.4, 1e-6)]

FILES = {
    "results/lever_curves.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "rate"],
    "results/lever_cube.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "threshold_pct",
                               "budget_to_reach", "ratio_vs_brute"],
    "results/lever_winners.csv": ["setting", "phi_min", "phi_max", "eps", "algo", "budget", "fn", "params"],
}


def main():
    os.makedirs("results", exist_ok=True)
    for path, header in FILES.items():  # (re)create with headers, then append per scenario (crash-resilient)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    print(f"combined-lever study: {len(SCENARIOS)} scenarios, R_test={X.R_TEST}, "
          f"budgets/scenario={X.N_BUDGETS}{'  [QUICK]' if QUICK else ''}", flush=True)

    for i, (pmin, pmax, eps) in enumerate(SCENARIOS, 1):
        try:
            res = X.run_scenario(pmin, pmax, eps)
            crow, xrow, wrow = X._scenario_rows(*res)
            for path, rows in [("results/lever_curves.csv", crow),
                               ("results/lever_cube.csv", xrow),
                               ("results/lever_winners.csv", wrow)]:
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerows(rows)
            print(f"   [{i}/{len(SCENARIOS)}] written.", flush=True)
        except Exception as ex:  # never let one scenario kill the unattended run
            print(f"   !! scenario {X.label(pmin, pmax, eps)} FAILED: {type(ex).__name__}: {ex}", flush=True)

    print("\ndone -- wrote results/lever_{curves,cube,winners}.csv")


if __name__ == "__main__":
    main()
