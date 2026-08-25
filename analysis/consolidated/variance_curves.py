"""Signed estimator-error variance versus budget for the headline figure.

The protocols are re-evaluated with their frozen parameters, held-out seed and recorded trial
count.  Their value is the ordinary sample variance (ddof=1) of phi_hat-phi.  The Oracle value is
the analytic prior average of 1/(4 m N_opt^2), with m=floor(B/N_opt); it has no trial count or
sampling interval.  Bias and MSE are retained beside the requested variance because a biased
estimator can have variance below the ordinary QCRB without violating it.

    python analysis/consolidated/variance_curves.py
    python analysis/consolidated/variance_curves.py --scenarios narrow_e3,wide_e4
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import manifest as M
from qmetrology import pipeline as P
from qmetrology.experiments import N_JOBS, _CTX, _draw_phi
from qmetrology.oracle import oracle_error_variance
from pipeline_io import Table, path

FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "eps", "algorithm",
          "implementation_variant", "budget", "params", "R", "seed_test",
          "error_variance", "error_bias", "error_mse", "n_finite"]


def _moment_task(task):
    """One seed chunk; return sufficient statistics instead of all per-trial traces."""
    key, algorithm, scen, budget, cfg, seeds = task
    fn = M.ALGORITHMS[algorithm]["fn"]
    params = P._call_params(algorithm, scen, budget, cfg)
    n, total, total2 = 0, 0.0, 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            phi = _draw_phi(rng, scen["phi_min"], scen["phi_max"], scen["phi_dist"])
            phi_hat, _used = fn(rng, phi, scen["phi_max"], scen["phi_min"], **params)
            error = phi_hat - phi
            if np.isfinite(error):
                n += 1
                total += error
                total2 += error * error
    return key, n, total, total2


def main():
    mode = json.load(open(path("experiment_manifest.json")))["mode"]
    want = {"narrow_e3"}
    if "--scenarios" in sys.argv:
        want = set(sys.argv[sys.argv.index("--scenarios") + 1].split(","))
    with open(path("winners.csv"), newline="") as f:
        wins = list(csv.DictReader(f))
    with open(path("performance_curves.csv"), newline="") as f:
        perf = list(csv.DictReader(f))
    scen_by_id = {s["id"]: s for s in M.SCENARIOS}

    rows, seen, t0 = [], set(), time.time()
    metadata, tasks = {}, []
    seeds = P.seeds_for(M.SEED_TEST, M.MODES[mode]["R_test"])
    chunks = P._chunks(seeds, N_JOBS)
    for w in wins:
        sid, algorithm, budget = w["scenario_id"], w["algorithm"], int(w["budget"])
        key = (sid, budget, algorithm)
        if sid not in want or key in seen:
            continue
        seen.add(key)
        scen = scen_by_id[sid]
        cfg = json.loads(w["params"])
        metadata[key] = (scen, w)
        tasks.extend((key, algorithm, scen, budget, cfg, chunk) for chunk in chunks)

    moments = {key: [0, 0.0, 0.0] for key in metadata}
    with ProcessPoolExecutor(max_workers=N_JOBS, mp_context=_CTX) as executor:
        for key, n, total, total2 in executor.map(_moment_task, tasks, chunksize=1):
            moments[key][0] += n
            moments[key][1] += total
            moments[key][2] += total2
    for key, (scen, w) in metadata.items():
        sid, budget, algorithm = key
        n, total, total2 = moments[key]
        if n < 2:
            continue
        bias, mse = total / n, total2 / n
        variance = max((total2 - total * total / n) / (n - 1), 0.0)
        rows.append({
            "setting": scen["label"], "scenario_id": sid,
            "phi_min": scen["phi_min"], "phi_max": scen["phi_max"], "eps": scen["eps"],
            "algorithm": algorithm, "implementation_variant": w["implementation_variant"],
            "budget": budget, "params": w["params"], "R": M.MODES[mode]["R_test"],
            "seed_test": M.SEED_TEST, "error_variance": variance,
            "error_bias": bias, "error_mse": mse, "n_finite": n})
        print(f"   {sid:12s} {algorithm:18s} B={budget:>12,d}", flush=True)

    for r in perf:
        sid = r["scenario_id"]
        if sid not in want or r["algorithm"] != "oracle_hl":
            continue
        scen, budget = scen_by_id[sid], int(r["budget"])
        variance = oracle_error_variance(budget, scen["phi_min"], scen["phi_max"])
        rows.append({
            "setting": scen["label"], "scenario_id": sid,
            "phi_min": scen["phi_min"], "phi_max": scen["phi_max"], "eps": scen["eps"],
            "algorithm": "oracle_hl", "implementation_variant": r["implementation_variant"],
            "budget": budget, "params": "{}", "R": "", "seed_test": "",
            "error_variance": variance, "error_bias": 0.0, "error_mse": variance,
            "n_finite": ""})

    Table("variance_curves.csv", FIELDS, reset=True).rows(rows)
    print(f"wrote {path('variance_curves.csv')} -- {len(rows)} rows "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
