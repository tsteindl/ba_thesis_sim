import argparse
import numpy as np
from grid_search import grid_search_algorithm
from binary_search import find_phi_binary_search, find_phi_binary_search_anneal_m
from linear_search import find_phi_linear_search
from reverse_engineering import find_phi_reverse_engineering
from brute_force import find_phi_brute_force

ALGORITHMS = {
    "binary_search": find_phi_binary_search,
    "binary_search_anneal_m": find_phi_binary_search_anneal_m,
    "linear_search": find_phi_linear_search,
    "reverse_engineering": find_phi_reverse_engineering,
    "brute_force": find_phi_brute_force,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for phase estimation algorithms"
    )

    # parser.add_argument("--algorithm", type=str, required=False)
    # parser.add_argument("--eps", type=float, required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=ALGORITHMS.keys(),
        default="binary_search",
        help="Which algorithm to run grid search for",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Target precision",
    )
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--phi-min", type=float, required=True)
    parser.add_argument("--phi-max", type=float, required=True)
    parser.add_argument("--R", type=int, default=1000)

    return parser.parse_args()

def main():
    args = parse_args()
    
    algorithm_name = args.algorithm
    eps = args.eps
    
    find_phi_fn = ALGORITHMS[algorithm_name]
    
    if eps == 1e-3:
        m_exploration_vals = np.unique(np.logspace(2, 3, 50, dtype=int))
        m_exploitation_vals = np.unique(np.logspace(3, 4, 50, dtype=int))

        if algorithm_name == "binary_search":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "conf": [0.5, 0.75, 0.8, 0.95, 0.975],
                "safeguard": [0, 1, 2, 3, 4, 5, 8, 10, 15],
            }
        
        elif algorithm_name == "binary_search_anneal_m":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "conf": [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95],
                "safeguard": [1, 2, 3, 5],
                "max_b_steps_sub": [-10, 0, 1, 2, 3],
                "delta": [1, 2, 3, 4],
                # "annealing_factor": [1, 2, 3, 4]
            }

        elif algorithm_name == "linear_search":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "lookback_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],
                "safeguard": [0, 1, 2, 3, 4, 5, 8, 10, 15],
            }

        elif algorithm_name == "reverse_engineering":
            m_exploration_vals = np.unique(np.logspace(2, 4, 75, dtype=int))
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "safeguard": [0.5, 0.75, 0.8, 0.85, 0.90],
            }


    elif eps == 1e-4:
        m_exploration_vals = np.unique(np.logspace(2, 3, 60, dtype=int))
        m_exploitation_vals = np.unique(np.logspace(3, 5, 60, dtype=int))

        if algorithm_name == "binary_search":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "conf": [0.4, 0.5, 0.6, 0.7, 0.8, 0.95],
                "safeguard": [2, 5, 10, 15, 20, 30],
            }

        elif algorithm_name == "linear_search":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "lookback_window": [5, 10, 15, 20, 30, 50],
                "safeguard": [2, 5, 10, 15, 20, 30],
            }

        elif algorithm_name == "reverse_engineering":
            param_grid = {
                "m_exploration": m_exploration_vals,
                "m_exploitation": m_exploitation_vals,
                "safeguard": [0.7, 0.8, 0.9],
            }

            

    grid_search_algorithm(
        find_phi_fn=find_phi_fn,
        param_grid=param_grid,
        R=args.R,
        eps=eps,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        success_threshold=args.success_threshold,
        algorithm_name=algorithm_name,
    )


    # # Small precision
    # eps = 1e-3
    # m_exploration_vals = np.unique(np.logspace(2, 4, 50, dtype=int))
    # m_exploitation_vals = np.unique(np.logspace(2, 5, 50, dtype=int))
    
    # # Binary search
    # algorithm_name = "binary_search"
    # find_phi_fn = ALGORITHMS[algorithm_name]
    
    # param_grid = {
    #     "m_exploration": m_exploration_vals,
    #     "m_exploitation": m_exploitation_vals,
    #     "conf": [0.5, 0.75, 0.8, 0.95, 0.975, 0.99],
    #     "safeguard": [0, 1, 2, 3, 4, 5, 8, 10, 15],
    # }
    
    # grid_search_algorithm(
    #     find_phi_fn=find_phi_fn,
    #     param_grid=param_grid,
    #     R=args.R,
    #     eps=eps,
    #     phi_min=args.phi_min,
    #     phi_max=args.phi_max,
    #     success_threshold=args.success_threshold,
    #     algorithm_name=algorithm_name,
    # )
    
    # Linear search
    # algorithm_name = "linear_search"
    # find_phi_fn = ALGORITHMS[algorithm_name]
    
    # param_grid = {
    #     "m_exploration": m_exploration_vals,
    #     "m_exploitation": m_exploitation_vals,
    #     "lookback_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],
    #     "safeguard": [0, 1, 2, 3, 4, 5, 8, 10, 15, 25],
    # }
    
    # grid_search_algorithm(
    #     find_phi_fn=find_phi_fn,
    #     param_grid=param_grid,
    #     R=args.R,
    #     eps=eps,
    #     phi_min=args.phi_min,
    #     phi_max=args.phi_max,
    #     success_threshold=args.success_threshold,
    #     algorithm_name=algorithm_name,
    # )
    
    

if __name__ == "__main__":
    main()
