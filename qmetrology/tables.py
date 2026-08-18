"""Fixed-budget %-converged numbers for Table 3.1 (narrow) and Table 3.3 (broad).

compute() returns a dict keyed by (setting, algo). Table 3.2 comes from analysis/extensive_sweep.py;
RESULTS.md is assembled by analysis/make_results.py.
"""
import numpy as np

from . import config as C
from .algorithms import FIXED_BUDGET, RISK_OF
from .experiments import grid_search_max, success_rate, wilson
from .oracle import heisenberg_rate

ALGOS = ["separable", "brute", "linear", "binary", "reverse_eng"]
# What the tables actually report: the statistical safeguard replaces the tuned constant, so
# binary_risk / reverse_eng_risk ARE binary search / reverse engineering. PARENT maps them back to
# the published row so the "paper vs this work" column still lines up.
REPORT = ["separable", "brute", "linear", "binary_risk", "reverse_eng_risk"]
PARENT = {"binary_risk": "binary", "reverse_eng_risk": "reverse_eng"}
# the statistical-safeguard variants and the omniscient ceilings have no published counterpart,
# so `paper` is None for them
ORACLES = ["oracle_hl"]
ALGOS_RISK = ALGOS + ["binary_risk", "reverse_eng_risk"]
ALGOS_ALL = REPORT + ORACLES   # what the tables report
# evaluated directly rather than grid-tuned: nothing to search over
PARAM_FREE = ["brute", "separable"] + ORACLES


def _r1000_ok(p_true_pct, paper_pct):
    p = p_true_pct / 100
    se = 100 * np.sqrt(p * (1 - p) / C.R_TUNE)
    return abs(paper_pct - p_true_pct) <= 1.96 * se


def compute(R_test=None, R_tune=None, n_jobs=None, verbose=True, algos=None):
    """Fixed-budget (=10,000) '% converged' experiments. Returns dict keyed by (setting, algo).
    Param-free algorithms (brute, separable) are evaluated directly; the adaptive ones are
    grid-tuned on seed 42 and re-validated on seed 2024."""
    R_test = R_test or C.R_TEST
    R_tune = R_tune or C.R_TUNE
    kw = {} if n_jobs is None else {"n_jobs": n_jobs}
    res = {}
    for key, pmax, label in C.SETTINGS:
        paper = C.PAPER_T31 if key == "narrow" else C.PAPER_T33[key]
        if verbose:
            print(f"[{key}] N_min={max(np.pi//(2*pmax),1):.0f}")
        for algo in (algos or ALGOS):
            if algo not in paper and algo not in RISK_OF.values() and algo not in ORACLES:
                continue
            d = {"paper": paper.get(PARENT.get(algo, algo)), "label": label}
            if algo == "oracle_hl":
                # analytic (quadrature over the prior): no simulation, hence no sampling interval
                r = heisenberg_rate(10000, C.EPS, C.PHI_MIN, pmax)
                d.update(unbiased=100 * r, ci=(100 * r, 100 * r), ok=None)
                res[(key, algo)] = d
                continue
            if algo in PARAM_FREE:
                params = {"budget": 10000}
                if algo == "oracle":
                    params["eps_target"] = C.EPS
                r = success_rate(FIXED_BUDGET[algo], params, R_test,
                                 C.PHI_MIN, pmax, C.EPS, C.SEED_UNBIASED, **kw)
                d["unbiased"] = 100 * r
                d["ci"] = tuple(100 * x for x in wilson(r, R_test))
                d["ok"] = None if d["paper"] is None else _r1000_ok(d["unbiased"], d["paper"])
            else:
                gmax, arg, _ = grid_search_max(FIXED_BUDGET[algo], C.grids(10000)[algo],
                                               R_tune, C.PHI_MIN, pmax, C.EPS, C.SEED_TUNE, **kw)
                deb = success_rate(FIXED_BUDGET[algo], arg, R_test, C.PHI_MIN, pmax,
                                   C.EPS, C.SEED_TEST, **kw)
                d.update(grid_max=100 * gmax, debiased=100 * deb, argmax=arg,
                         bias=100 * (gmax - deb), ci=tuple(100 * x for x in wilson(deb, R_test)))
                if key == "narrow":
                    cp = dict(C.CANON[algo]); cp["budget"] = 10000
                    # the canonical set may be written in a different parameterisation than the
                    # tuned one (reverse engineering states a budget share, not a shot count)
                    fn = FIXED_BUDGET[C.CANON_FN.get(algo, algo)]
                    d["canon"] = 100 * success_rate(fn, cp, R_test,
                                                    C.PHI_MIN, pmax, C.EPS, C.SEED_UNBIASED, **kw)
            res[(key, algo)] = d
    return res
