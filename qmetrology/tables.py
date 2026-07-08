"""Fixed-budget %-converged numbers for Table 3.1 (narrow) and Table 3.3 (broad).

compute() returns a dict keyed by (setting, algo). Table 3.2 comes from analysis/extensive_sweep.py;
RESULTS.md is assembled by analysis/make_results.py.
"""
import numpy as np

from . import config as C
from .algorithms import FIXED_BUDGET
from .experiments import grid_search_max, success_rate, wilson

ALGOS = ["separable", "brute", "linear", "binary", "reverse_eng"]


def _r1000_ok(p_true_pct, paper_pct):
    p = p_true_pct / 100
    se = 100 * np.sqrt(p * (1 - p) / C.R_TUNE)
    return abs(paper_pct - p_true_pct) <= 1.96 * se


def compute(R_test=None, R_tune=None, n_jobs=None, verbose=True):
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
        for algo in ALGOS:
            if algo not in paper:
                continue
            d = {"paper": paper[algo], "label": label}
            if algo in ("brute", "separable"):
                r = success_rate(FIXED_BUDGET[algo], {"budget": 10000}, R_test,
                                 C.PHI_MIN, pmax, C.EPS, C.SEED_UNBIASED, **kw)
                d["unbiased"] = 100 * r
                d["ci"] = tuple(100 * x for x in wilson(r, R_test))
                d["ok"] = _r1000_ok(d["unbiased"], d["paper"])
            else:
                gmax, arg, _ = grid_search_max(FIXED_BUDGET[algo], C.grids(10000)[algo],
                                               R_tune, C.PHI_MIN, pmax, C.EPS, C.SEED_TUNE, **kw)
                deb = success_rate(FIXED_BUDGET[algo], arg, R_test, C.PHI_MIN, pmax,
                                   C.EPS, C.SEED_TEST, **kw)
                d.update(grid_max=100 * gmax, debiased=100 * deb, argmax=arg,
                         bias=100 * (gmax - deb), ci=tuple(100 * x for x in wilson(deb, R_test)))
                if key == "narrow":
                    cp = dict(C.CANON[algo]); cp["budget"] = 10000
                    d["canon"] = 100 * success_rate(FIXED_BUDGET[algo], cp, R_test,
                                                    C.PHI_MIN, pmax, C.EPS, C.SEED_UNBIASED, **kw)
            res[(key, algo)] = d
    return res
