"""Seeds, settings, canonical parameters, grids, and the published table values."""
import numpy as np

SEED_TUNE = 42
SEED_TEST = 2024
SEED_UNBIASED = 12345
R_TUNE = 1000
R_TEST = 50000
EPS = 1e-3
PHI_MIN = 0.01

# (key, phi_max, latex label)
SETTINGS = [
    ("narrow", 0.1, r"$\mathcal{U}(0.01, 0.1)$"),
    ("pi/2", np.pi / 2, r"$\mathcal{U}(0.01, \pi/2)$"),
    ("pi/4", np.pi / 4, r"$\mathcal{U}(0.01, \pi/4)$"),
    ("pi/8", np.pi / 8, r"$\mathcal{U}(0.01, \pi/8)$"),
    ("pi/16", np.pi / 16, r"$\mathcal{U}(0.01, \pi/16)$"),
]

NICE = {
    "separable": "Separable (N=1)", "brute": "Brute force", "linear": "Linear search",
    "binary": "Binary search", "reverse_eng": "Reverse Engineering",
}

# parameters stated in the thesis text
CANON = {
    "linear": {"m_exploration": 10, "lookback_window": 5, "safeguard": 1, "inc": 1},
    "binary": {"m_exploration": 100, "conf": 0.95, "safeguard": 1},
    "reverse_eng": {"m_exploration": 200, "safeguard": 0.9},
}


def grids(budget=10_000):
    m_exp = np.logspace(1, 4, num=50, dtype=int)
    return {
        "linear": {"m_exploration": m_exp, "lookback_window": [1, 2, 5],
                   "safeguard": [0, 1, 2, 5], "budget": [budget], "inc": [1, 2, 5, 10]},
        "binary": {"m_exploration": m_exp, "safeguard": [0, 1, 2, 5],
                   "budget": [budget], "conf": [0.5, 0.8, 0.9]},
        "reverse_eng": {"m_exploration": m_exp, "safeguard": [0.8, 0.9, 0.95],
                        "budget": [budget]},
    }


# published values (percent) for the paper-vs-this-work comparison
PAPER_T31 = {"separable": 15.9, "brute": 56.30, "linear": 14.00, "binary": 11.5, "reverse_eng": 61.30}
PAPER_T33 = {
    "pi/2": {"brute": 15.9, "linear": 20.8, "binary": 16.3, "reverse_eng": 17.4},
    "pi/4": {"brute": 20.4, "linear": 24.4, "binary": 21.3, "reverse_eng": 22.3},
    "pi/8": {"brute": 31.3, "linear": 34.3, "binary": 30.4, "reverse_eng": 39.7},
    "pi/16": {"brute": 43.3, "linear": 46.6, "binary": 39.3, "reverse_eng": 51.8},
}
