import argparse
import numpy as np
from grid_search import grid_search_algorithm
from binary_search import find_phi_binary_search
from linear_search import find_phi_linear_search
from reverse_engineering import find_phi_reverse_engineering
from brute_force import find_phi_brute_force

ALGORITHMS = {
    "binary_search": find_phi_binary_search,
    "linear_search": find_phi_linear_search,
    "reverse_engineering": find_phi_reverse_engineering,
    "brute_force": find_phi_brute_force,
}


def main():
    algorithm_name = "binary_search"
    find_phi_fn = ALGORITHMS[algorithm_name]

    param_grid = {
        "m_exploration": [500, 1000, 2000],
        "m_exploitation": [10_000, 50_000, 100_000, 500_000],
        "conf": [0.95, 0.975],
        "safeguard": [0, 1, 2],
    }

    grid_search_algorithm(
        find_phi_fn=find_phi_fn,
        param_grid=param_grid,
        R=R,
        eps=args.eps,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        success_threshold=args.success_threshold,
        algorithm_name=args.algorithm,
        # algorithm_name=algorithm_name
    )

if __name__ == "__main__":
    main()
