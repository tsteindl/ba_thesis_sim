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
    # the statistical safeguard REPLACES the tuned constant, so the *_risk keys carry the plain
    # algorithm names; the constant-C originals are kept only for the published-vs-this-work delta
    "binary_risk": "Binary search", "reverse_eng_risk": "Reverse Engineering",
    "linear_risk": "Linear search + stat. safeguard (diagnostic, not reported)",
    "reverse_eng_share": "Reverse Engineering (pilot = 2% of budget)",
    "binary": "Binary search (tuned s, superseded)",
    "reverse_eng": "Reverse Engineering (tuned C, superseded)",
    "oracle": "Oracle (N_opt, knows phi)", "oracle_alias": "Oracle (no safeguard)",
    "oracle_hl": "Attainable ceiling (Eq. 3.4 at N_opt)",
}

# parameters stated in the thesis text (the *_risk variants have no tuned safeguard: the depth
# follows from the asymptotic law, so only the exploration size is left to choose)
CANON = {
    "linear": {"m_exploration": 10, "lookback_window": 5, "safeguard": 1, "inc": 1},
    "binary": {"m_exploration": 100, "conf": 0.95, "safeguard": 1},
    "reverse_eng": {"m_exploration": 200, "safeguard": 0.9},
    "binary_risk": {"m_exploration": 100, "conf": 0.95, "eps_target": EPS},
    # the stated exploration size for reverse engineering is now a BUDGET SHARE, not a shot count:
    # a fixed m' can only be right at one budget, and m'=200 (the published value) is -5.55 pp on
    # average over the 546 operating points of analysis/re_share_sweep.py, -99.4 pp at worst.
    # rho = 2% is -0.06 pp, i.e. indistinguishable from tuning m' at every budget separately.
    "reverse_eng_risk": {"pilot_share": 0.02, "eps_target": EPS},
    "linear_risk": {"m_exploration": 10, "lookback_window": 5, "inc": 1, "eps_target": EPS},
    # pilot_share = 2% of the budget is the transferable default (analysis/re_share_sweep.py):
    # -0.06 pp against the per-budget-tuned m' over all 23 scenarios, i.e. within the 0.35 pp SE
    "reverse_eng_share": {"pilot_share": 0.02, "eps_target": EPS},
}


def grids(budget=10_000, eps=EPS):
    m_exp = np.logspace(1, 4, num=50, dtype=int)
    return {
        "linear": {"m_exploration": m_exp, "lookback_window": [1, 2, 5],
                   "safeguard": [0, 1, 2, 5], "budget": [budget], "inc": [1, 2, 5, 10]},
        "binary": {"m_exploration": m_exp, "safeguard": [0, 1, 2, 5],
                   "budget": [budget], "conf": [0.5, 0.8, 0.9]},
        "reverse_eng": {"m_exploration": m_exp, "safeguard": [0.8, 0.9, 0.95],
                        "budget": [budget]},
        # safeguard axis removed — that is the point of the statistical rule
        "binary_risk": {"m_exploration": m_exp, "budget": [budget],
                        "conf": [0.5, 0.8, 0.9], "eps_target": [eps]},
        "reverse_eng_risk": {"m_exploration": m_exp, "budget": [budget], "eps_target": [eps]},
        "linear_risk": {"m_exploration": m_exp, "lookback_window": [1, 2, 3, 5],
                        "budget": [budget], "inc": [1, 2, 5], "eps_target": [eps]},
        "reverse_eng_share": {"pilot_share": [0.005, 0.01, 0.02, 0.035, 0.05, 0.08],
                              "budget": [budget], "eps_target": [eps]},
    }


# Some canonical parameter sets are expressed in a different parameterisation than the algorithm
# they describe. CANON_FN says which implementation to evaluate them with; everything else (the
# grid-tuned and de-biased numbers) is unaffected -- this is a REPORTING change only.
CANON_FN = {"reverse_eng_risk": "reverse_eng_share"}

# The recommended default, quoted in the thesis text and Appendix C.
PILOT_SHARE = 0.02
PILOT_FLOOR = 20


# published values (percent) for the paper-vs-this-work comparison
PAPER_T31 = {"separable": 15.9, "brute": 56.30, "linear": 14.00, "binary": 11.5, "reverse_eng": 61.30}
PAPER_T33 = {
    "pi/2": {"brute": 15.9, "linear": 20.8, "binary": 16.3, "reverse_eng": 17.4},
    "pi/4": {"brute": 20.4, "linear": 24.4, "binary": 21.3, "reverse_eng": 22.3},
    "pi/8": {"brute": 31.3, "linear": 34.3, "binary": 30.4, "reverse_eng": 39.7},
    "pi/16": {"brute": 43.3, "linear": 46.6, "binary": 39.3, "reverse_eng": 51.8},
}
