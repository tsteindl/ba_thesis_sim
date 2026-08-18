"""Parallel Monte-Carlo evaluation of the phase-search algorithms.

Each experiment derives its per-trial seeds from a single integer `seed`, so results are
reproducible and independent of execution order.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

N_JOBS = min(20, (os.cpu_count() or 2))

# Use 'fork' so the pool works from a script, a notebook cell, or `python -m`
# (the forkserver default re-imports the launching module and breaks top-level pools).
try:
    _CTX = mp.get_context("fork")
except ValueError:  # non-fork platform (Windows)
    _CTX = mp.get_context()


def _pool(n_jobs):
    return ProcessPoolExecutor(max_workers=n_jobs, mp_context=_CTX)


def _draw_phi(rng, phi_min, phi_max, phi_dist):
    """Sample the true phase. 'uniform' = U(phi_min, phi_max); 'loguniform' spreads it across decades."""
    if phi_dist == "loguniform":
        return float(np.exp(rng.uniform(np.log(phi_min), np.log(phi_max))))
    return float(rng.uniform(phi_min, phi_max))


def _eval_tuple(task):
    """Worker: count converged trials for one config. Top-level so it pickles."""
    fn, phi_min, phi_max, eps, params, seeds, phi_dist = task
    succ = 0
    with np.errstate(invalid="ignore", divide="ignore"):  # m=0 -> nan -> non-convergence
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = _draw_phi(rng, phi_min, phi_max, phi_dist)
            phi_hat, _ = fn(rng, phi, phi_max, phi_min, **params)
            if abs(phi_hat - phi) < eps:
                succ += 1
    return succ


def _eval_tuple_budget(task):
    fn, phi_min, phi_max, eps, params, seeds, phi_dist = task
    succ = 0
    budget_sum = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = _draw_phi(rng, phi_min, phi_max, phi_dist)
            phi_hat, budget = fn(rng, phi, phi_max, phi_min, **params)
            budget_sum += budget
            if abs(phi_hat - phi) < eps:
                succ += 1
    return succ, budget_sum


def _seeds(seed, R):
    return np.random.default_rng(seed).integers(0, 2**63, size=R)


def success_rate(fn, params, R, phi_min, phi_max, eps, seed, n_jobs=N_JOBS, phi_dist="uniform"):
    """Fraction of R trials with |phi_hat - phi| < eps for a single config."""
    seeds = _seeds(seed, R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 3) if len(c)]
    tasks = [(fn, phi_min, phi_max, eps, params, ch, phi_dist) for ch in chunks]
    with _pool(n_jobs) as ex:
        return sum(ex.map(_eval_tuple, tasks)) / R


def rate_and_budget(fn, params, R, phi_min, phi_max, eps, seed, n_jobs=N_JOBS, phi_dist="uniform"):
    """(success_rate, mean budget) for a single config."""
    seeds = _seeds(seed, R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 3) if len(c)]
    tasks = [(fn, phi_min, phi_max, eps, params, ch, phi_dist) for ch in chunks]
    with _pool(n_jobs) as ex:
        out = list(ex.map(_eval_tuple_budget, tasks))
    return sum(o[0] for o in out) / R, sum(o[1] for o in out) / R


def _eval_tuple_err(task):
    """Worker: return the per-trial estimator error |phi_hat - phi| for one config."""
    fn, phi_min, phi_max, eps, params, seeds, phi_dist = task
    errs = np.empty(len(seeds))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            phi = _draw_phi(rng, phi_min, phi_max, phi_dist)
            phi_hat, _ = fn(rng, phi, phi_max, phi_min, **params)
            errs[i] = abs(phi_hat - phi)
    return errs


def error_quantiles(fn, params, R, phi_min, phi_max, eps, seed, quantiles=(0.25, 0.5, 0.75),
                    n_jobs=N_JOBS, phi_dist="uniform"):
    """Quantiles (+ mean, std) of the estimator error |phi_hat - phi| over R trials.
    Non-finite estimates (e.g. m=0) are dropped."""
    seeds = _seeds(seed, R)
    chunks = [c for c in np.array_split(seeds, n_jobs * 3) if len(c)]
    tasks = [(fn, phi_min, phi_max, eps, params, ch, phi_dist) for ch in chunks]
    with _pool(n_jobs) as ex:
        errs = np.concatenate(list(ex.map(_eval_tuple_err, tasks)))
    errs = errs[np.isfinite(errs)]
    q = {qq: float(np.quantile(errs, qq)) for qq in quantiles}
    return q, float(errs.mean()), float(errs.std())


def grid_search_max(fn, grid, R, phi_min, phi_max, eps, seed, n_jobs=N_JOBS, phi_dist="uniform"):
    """Evaluate every config in `grid` (paired seeds) and return
    (max_rate, argmax_params, [(rate, params), ...])."""
    seeds = _seeds(seed, R)
    names = list(grid.keys())
    configs = [dict(zip(names, vals)) for vals in product(*grid.values())]
    tasks = [(fn, phi_min, phi_max, eps, cfg, seeds, phi_dist) for cfg in configs]
    with _pool(n_jobs) as ex:
        rates = [s / R for s in ex.map(_eval_tuple, tasks, chunksize=4)]
    i = int(np.argmax(rates))
    return rates[i], configs[i], list(zip(rates, configs))


def grid_full(fn, grid, R, phi_min, phi_max, eps, seed, n_jobs=N_JOBS, phi_dist="uniform"):
    """Like grid_search_max but returns (success_rate, mean_budget, params) per config,
    so callers can enforce a true budget cap (fixed-budget variants may overspend)."""
    seeds = _seeds(seed, R)
    names = list(grid.keys())
    configs = [dict(zip(names, vals)) for vals in product(*grid.values())]
    tasks = [(fn, phi_min, phi_max, eps, cfg, seeds, phi_dist) for cfg in configs]
    with _pool(n_jobs) as ex:
        out = list(ex.map(_eval_tuple_budget, tasks, chunksize=4))
    return [(s / R, b / R, cfg) for (s, b), cfg in zip(out, configs)]


# re-exported so callers keep importing it from here; the single definition lives with the rest of
# the error-bar machinery in qmetrology/uncertainty.py
from .uncertainty import wilson  # noqa: E402,F401
