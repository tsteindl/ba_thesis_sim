"""Small parity check between the compact listings and the measured implementations."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binary_search import estimate_phi_binary_search
from brute_force import estimate_phi_brute_force
from linear_search import estimate_phi_linear_search
from reverse_engineering import estimate_phi_reverse_engineering
from qmetrology.algorithms import (
    find_phi_fixed_budget_binary_search_deep,
    find_phi_fixed_budget_brute_force,
    find_phi_fixed_budget_linear_search,
    find_phi_fixed_budget_reverse_engineering_risk,
)


def main():
    matches = {name: 0 for name in ("brute", "linear", "binary", "reverse")}
    for seed in range(200):
        phi = float(np.random.default_rng(seed + 10_000).uniform(0.01, 0.1))

        compact = estimate_phi_brute_force(np.random.default_rng(seed), phi, 0.1, 10_000)
        measured = find_phi_fixed_budget_brute_force(
            np.random.default_rng(seed), phi, 0.1, 0.01, 10_000)[0]
        matches["brute"] += int(np.isclose(compact, measured, equal_nan=True))

        compact = estimate_phi_linear_search(
            np.random.default_rng(seed), phi, 0.01, 0.1, 10_000, 2, 4, 1, 1)
        measured = find_phi_fixed_budget_linear_search(
            np.random.default_rng(seed), phi, 0.1, 0.01, 2, 10_000, 4, 1, 1)[0]
        matches["linear"] += int(np.isclose(compact, measured, equal_nan=True))

        compact = estimate_phi_binary_search(
            np.random.default_rng(seed), phi, 0.01, 0.1, 10_000, 114, 0.5, 1e-3)
        measured = find_phi_fixed_budget_binary_search_deep(
            np.random.default_rng(seed), phi, 0.1, 0.01, 114, 10_000, 1e-3, 0.5)[0]
        matches["binary"] += int(np.isclose(compact, measured, equal_nan=True))

        compact = estimate_phi_reverse_engineering(
            np.random.default_rng(seed), phi, 0.01, 0.1, 10_000, 44, 1e-3)
        measured = find_phi_fixed_budget_reverse_engineering_risk(
            np.random.default_rng(seed), phi, 0.1, 0.01, 44, 10_000, 1e-3)[0]
        matches["reverse"] += int(np.isclose(compact, measured, equal_nan=True))

    print(matches)
    if any(value != 200 for value in matches.values()):
        raise SystemExit("compact listing differs from a measured implementation")


if __name__ == "__main__":
    main()
