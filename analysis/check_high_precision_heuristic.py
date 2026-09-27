"""Numerically check the high-precision budget-ratio heuristic in Section 4.4.

For a fixed phase-gate count N, Equations (4.2)--(4.5) predict that the
budget needed for a conditional success probability q is proportional to
1/N.  The thesis results, however, average the success probability over a
uniform phase prior.  This script therefore compares like with like:

* the fixed-N_min asymptotic baseline crossing;
* the prior-averaged asymptotic oracle crossing, using N_opt(phi); and
* the empirical Reverse Engineering crossing in budget_crossings.csv.

Run from the repository root with

    python analysis/check_high_precision_heuristic.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qmetrology.oracle import oracle_budget_for_rate, oracle_shots_for_rate


RESULTS = ROOT / "results" / "budget_crossings.csv"
TARGET = 0.90
PHI_MIN = 0.01
PHI_MAX = 0.1


def reported_crossings() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["scenario_id"].startswith("narrow_e")
                    and float(row["threshold_pct"]) == 100.0 * TARGET):
                out[(row["scenario_id"], row["algorithm"])] = float(row["budget_to_reach"])
    return out


def main() -> None:
    crossings = reported_crossings()
    n_min = max(int(np.floor(np.pi / (2.0 * PHI_MAX))), 1)

    header = (
        "scenario   eps       asymptotic baseline/oracle   "
        "reported baseline/RE   reported RE/oracle   RE above oracle"
    )
    print(f"N_min = {n_min}; target prior-averaged convergence = {TARGET:.0%}")
    print(header)

    for scenario_id in sorted({scenario for scenario, _ in crossings}):
        exponent = int(scenario_id.removeprefix("narrow_e"))
        eps = 10.0 ** (-exponent)

        baseline_asymptotic = n_min * oracle_shots_for_rate(TARGET, eps, n_min)
        oracle_asymptotic = oracle_budget_for_rate(
            TARGET, eps, PHI_MIN, PHI_MAX
        )
        predicted_ratio = baseline_asymptotic / oracle_asymptotic

        baseline_reported = crossings[(scenario_id, "brute")]
        reverse_reported = crossings[(scenario_id, "reverse_eng_risk")]
        oracle_reported = crossings[(scenario_id, "oracle_hl")]
        reverse_ratio = baseline_reported / reverse_reported
        reverse_over_oracle = reverse_reported / oracle_reported

        print(
            f"{scenario_id:<10} {eps:<9.0e} "
            f"{predicted_ratio:>10.4f}x"
            f"{reverse_ratio:>25.4f}x"
            f"{reverse_over_oracle:>22.4f}x"
            f"{100.0 * (reverse_over_oracle - 1.0):>17.2f}%"
        )

    scenario_id = "narrow_e8"
    baseline = crossings[(scenario_id, "brute")]
    reverse = crossings[(scenario_id, "reverse_eng_risk")]
    oracle = crossings[(scenario_id, "oracle_hl")]
    print("\nHigh-precision example (epsilon = 1e-8):")
    print(f"  baseline / Reverse Engineering = {baseline / reverse:.4f}x")
    print(f"  baseline / oracle              = {baseline / oracle:.4f}x")
    print(f"  Reverse Engineering / oracle   = {reverse / oracle:.4f}x")
    print(f"  excess RE budget over oracle   = {(reverse / oracle - 1.0):.2%}")
    print(
        "  effective N from Eq. (4.5): "
        f"RE {n_min * baseline / reverse:.2f}, "
        f"oracle {n_min * baseline / oracle:.2f}"
    )


if __name__ == "__main__":
    main()
