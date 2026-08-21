"""Tuning and held-out evaluation for the consolidated pipeline.

The rule the handoff insists on: **the reported values are never maxima evaluated on the trials used
to pick the parameters.**

    1. every candidate configuration is scored on each tuning block (seeds in SEED_TUNE_BLOCKS);
    2. the winner is the argmax of the MEAN block rate -- two blocks, so one accidental draw cannot
       decide it -- with ties broken toward the smaller exploration size;
    3. the winner is frozen, together with its per-block evidence;
    4. it is re-evaluated on the disjoint held-out seed SEED_TEST;
    5. performance AND diagnostics both come from that single held-out evaluation, so no reported
       number and its telemetry can refer to different runs.

The held-out pass runs the algorithms with a trace attached and returns per-run scalar arrays
(qmetrology.trace.run_record). Full probe lists are kept only when `keep_traces` is set, for the
handful of points the report inspects in detail.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from .experiments import N_JOBS, _CTX, _draw_phi
from .manifest import (ALGORITHMS, MODES, SEED_TEST, SEED_TUNE_BLOCKS, coarse_discrete,
                       m_bounds, stage1_m, stage2_m, stage3_m, stage2_discrete, strided_discrete)
from .trace import AlgorithmTrace, run_record, RUN_FIELDS

_NUMERIC = [f for f in RUN_FIELDS if f != "status"]


def seeds_for(seed, R):
    """The trial seeds of one evaluation block. Same construction as qmetrology.experiments, so a
    trial index draws the same phi for every algorithm (the comparisons are paired)."""
    return np.random.default_rng(seed).integers(0, 2 ** 63, size=R)


def _call_params(algo, scen, budget, cfg):
    p = dict(cfg)
    for k, v in ALGORITHMS[algo]["fixed_params"].items():
        p[k] = scen[v] if isinstance(v, str) else v
    p["budget"] = int(budget)
    return p


# ------------------------------------------------------------------------------------ workers
def _task(t):
    """One chunk of trials. Returns (successes, n) for tuning, or per-run arrays for the held-out
    pass. Top-level so it pickles into the process pool."""
    algo, scen, budget, cfg, seeds, want_trace, keep_traces = t
    fn = ALGORITHMS[algo]["fn"]
    has_det = ALGORITHMS[algo]["has_detector"]
    pmin, pmax, eps, dist = scen["phi_min"], scen["phi_max"], scen["eps"], scen["phi_dist"]
    params = _call_params(algo, scen, budget, cfg)

    if not want_trace:
        succ = 0
        with np.errstate(invalid="ignore", divide="ignore"):
            for s in seeds:
                rng = np.random.default_rng(int(s))
                phi = _draw_phi(rng, pmin, pmax, dist)
                phi_hat, _ = fn(rng, phi, pmax, pmin, **params)
                if abs(phi_hat - phi) < eps:
                    succ += 1
        return succ, len(seeds)

    cols = {f: np.empty(len(seeds)) for f in _NUMERIC}
    status = []
    traces = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            phi = _draw_phi(rng, pmin, pmax, dist)
            tr = AlgorithmTrace(algorithm=algo, implementation_variant=ALGORITHMS[algo]["variant"],
                                nominal_budget=float(budget))
            phi_hat, used = fn(rng, phi, pmax, pmin, trace=tr, **params)
            tr.finalize(phi, phi_hat, used, eps)
            rec = run_record(tr, has_det)
            for f in _NUMERIC:
                cols[f][i] = rec[f]
            status.append(rec["status"])
            if keep_traces:
                traces.append(tr.to_dict())
    return cols, status, traces


def _chunks(seeds, n_jobs):
    return [c for c in np.array_split(seeds, n_jobs * 3) if len(c)]


# ------------------------------------------------------------------------------------- tuning
def _expand(algo, ms, discrete):
    axes = {"m_exploration": [int(m) for m in ms]}
    axes.update({k: list(v) for k, v in discrete.items()})
    names = list(axes)
    return [dict(zip(names, vals)) for vals in product(*axes.values())]


def _score_configs(algo, scen, budget, cfgs, R, seed, n_jobs):
    seeds = seeds_for(seed, R)
    ch = _chunks(seeds, n_jobs)
    tasks = [(algo, scen, budget, c, k, False, False) for c in cfgs for k in ch]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_CTX) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    n = len(ch)
    return [sum(o[0] for o in out[i * n:(i + 1) * n]) / sum(o[1] for o in out[i * n:(i + 1) * n])
            for i in range(len(cfgs))]


def _select(cfgs, block_rates):
    """argmax of the mean block rate; ties -> smaller exploration size, then earlier candidate."""
    mean = np.mean(block_rates, axis=0)
    best = mean.max()
    tied = [i for i in range(len(cfgs)) if mean[i] >= best - 1e-12]
    return min(tied, key=lambda i: (cfgs[i].get("m_exploration", 0), i)), mean


def tune(algo, scen, budget, mode, n_jobs=N_JOBS):
    """Three-stage grid tuning on the tuning blocks only. Returns (winner_cfg, evidence).

    A  locate the exploration size: m' across the whole feasible box [1, B // N_min], crossed with a
       3-point-per-axis skeleton of the discrete grid.
    B  search the FULL discrete grid at m' within a factor of four of A's winner.
    C  refine: m' within a factor of two of B's winner, over the neighbouring discrete values, at the
       much larger R_tune2 -- this is the stage that decides, so it is the one that gets the trials.

    Splitting A from B is what makes a large discrete grid affordable: linear search has 330 discrete
    combinations, and crossing those with a 24-point m' grid would be 7,920 configurations per stage.
    """
    spec = ALGORITHMS[algo]
    cfgm = MODES[mode]
    if not spec["adaptive"]:
        return {}, {
            "stage1_configs": 0, "stage2_configs": 0, "stage3_configs": 0, "block_rates": {},
            "tuned": False, "at_axis_bound": {}, "axis_grids": {},
            "m_lo": None, "m_hi": None, "at_m_min": 0, "at_m_max": 0,
            "runner_up": None, "margin_pp": None}

    # stride 1 in the production modes; the cheap modes subsample (see strided_discrete)
    disc_full = strided_discrete(algo, cfgm.get("disc_stride", 1))

    # ---- A: locate the exploration size ----------------------------------------------------
    msA = stage1_m(scen, budget, cfgm["n_m1"])
    cA = _expand(algo, msA, coarse_discrete(algo))
    rA = [_score_configs(algo, scen, budget, cA, cfgm["R_tune"], sd, n_jobs)
          for sd in SEED_TUNE_BLOCKS]
    wA = cA[_select(cA, rA)[0]]

    # ---- B: the full discrete grid, at a sensible exploration size --------------------------
    msB = stage2_m(scen, budget, wA["m_exploration"], cfgm["n_m2"])
    cB = _expand(algo, msB, disc_full)
    rB = [_score_configs(algo, scen, budget, cB, cfgm["R_tune"], sd, n_jobs)
          for sd in SEED_TUNE_BLOCKS]
    wB = cB[_select(cB, rB)[0]]

    # ---- C: refinement, where the trials go -------------------------------------------------
    msC = stage3_m(scen, budget, wB["m_exploration"], cfgm["n_m3"])
    cC = _expand(algo, msC, stage2_discrete(algo, wB))
    rC = [_score_configs(algo, scen, budget, cC, cfgm["R_tune2"], sd, n_jobs)
          for sd in SEED_TUNE_BLOCKS]
    iC, meanC = _select(cC, rC)
    win = cC[iC]

    order = np.argsort(-meanC)
    runner = cC[int(order[1])] if len(order) > 1 else None
    margin = float(100 * (meanC[iC] - meanC[int(order[1])])) if len(order) > 1 else float("nan")

    # Boundary reporting against the FULL grids, not the per-stage sub-grids: a winner pinned at an
    # endpoint means the optimum may lie outside the search box, and that is as invalidating on a
    # discrete axis as it is on the exploration size.
    lo, hi = m_bounds(scen, budget)
    axis_bounds = {}
    for ax, vals in disc_full.items():
        if win[ax] == vals[0]:
            axis_bounds[ax] = "min"
        elif win[ax] == vals[-1]:
            axis_bounds[ax] = "max"
    if win["m_exploration"] <= lo:
        axis_bounds["m_exploration"] = "min"
    elif win["m_exploration"] >= hi:
        axis_bounds["m_exploration"] = "max"

    ev = {
        "stage1_configs": len(cA), "stage2_configs": len(cB), "stage3_configs": len(cC),
        "tuned": True,
        "block_rates": {int(sd): float(rC[k][iC]) for k, sd in enumerate(SEED_TUNE_BLOCKS)},
        "block_mean": float(meanC[iC]),
        "at_axis_bound": axis_bounds,
        "axis_grids": {k: list(v) for k, v in disc_full.items()},
        "m_lo": int(lo), "m_hi": int(hi),
        "at_m_min": int(win["m_exploration"] <= lo), "at_m_max": int(win["m_exploration"] >= hi),
        "runner_up": runner, "margin_pp": margin,
        "R_tune_stage1": cfgm["R_tune"], "R_tune_stage2": cfgm["R_tune2"],
    }
    return win, ev


# ------------------------------------------------------------------------------- held-out eval
def heldout(algo, scen, budget, cfg, mode, n_jobs=N_JOBS, keep_traces=False, trace_R=None):
    """Evaluate the frozen configuration on the held-out seed, WITH tracing.

    Returns (arrays, statuses, traces): `arrays` maps each per-run scalar of
    qmetrology.trace.RUN_FIELDS to a length-R array, so performance and every diagnostic are
    computed from the same R runs by construction.

    `trace_R` truncates the seed list when full probe lists are being kept. The trial seeds are a
    deterministic prefix of the held-out list, so the dumped runs ARE the first `trace_R` runs of the
    evaluation above -- a dump of the same trials, not a second, differently seeded experiment.
    Keeping all R full traces is deliberately not the default: every one has to be pickled out of a
    worker, which costs far more than the simulation itself.
    """
    R = MODES[mode]["R_test"]
    seeds = seeds_for(SEED_TEST, R)
    if keep_traces and trace_R:
        seeds = seeds[:int(trace_R)]
    ch = _chunks(seeds, n_jobs)
    tasks = [(algo, scen, budget, cfg, k, True, keep_traces) for k in ch]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_CTX) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    arrays = {f: np.concatenate([o[0][f] for o in out]) for f in _NUMERIC}
    status = [s for o in out for s in o[1]]
    traces = [t for o in out for t in o[2]]
    return arrays, status, traces
