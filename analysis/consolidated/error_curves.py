"""Estimator-error distribution vs budget, from the frozen winners -> error_curves.csv.

`fig_error` needs quantiles of |phi_hat - phi|, which the aggregate diagnostics do not carry. This
re-evaluates each frozen winner on the SAME held-out seed and at the SAME R as the sweep, so the
error curves and the convergence curves describe the same runs and the same configurations.

    python analysis/consolidated/error_curves.py [--scenarios a,b,c]
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import manifest as M
from qmetrology import pipeline as P
from pipeline_io import Table, path

FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "eps", "algorithm",
          "implementation_variant", "budget", "params", "R", "seed_test",
          "err_p25", "err_p50", "err_p75", "err_p90", "err_mean", "n_finite"]
QS = (0.25, 0.50, 0.75, 0.90)


def main():
    mode = json.load(open(path("experiment_manifest.json")))["mode"]
    want = None
    if "--scenarios" in sys.argv:
        want = set(sys.argv[sys.argv.index("--scenarios") + 1].split(","))
    with open(path("winners.csv"), newline="") as f:
        wins = list(csv.DictReader(f))
    scen_by_id = {s["id"]: s for s in M.SCENARIOS}
    rows, t0 = [], time.time()
    seen = set()
    for w in wins:
        sid = w["scenario_id"]
        if want and sid not in want:
            continue
        key = (sid, int(w["budget"]), w["algorithm"])
        if key in seen:
            continue
        seen.add(key)
        scen = scen_by_id[sid]
        cfg = json.loads(w["params"])
        A, _st, _ = P.heldout(w["algorithm"], scen, int(w["budget"]), cfg, mode)
        err = np.abs(A["phi_hat_final"] - A["phi"])
        err = err[np.isfinite(err)]
        if not err.size:
            continue
        q = np.quantile(err, QS)
        rows.append({
            "setting": scen["label"], "scenario_id": sid,
            "phi_min": scen["phi_min"], "phi_max": scen["phi_max"], "eps": scen["eps"],
            "algorithm": w["algorithm"], "implementation_variant": w["implementation_variant"],
            "budget": int(w["budget"]), "params": w["params"],
            "R": M.MODES[mode]["R_test"], "seed_test": M.SEED_TEST,
            "err_p25": q[0], "err_p50": q[1], "err_p75": q[2], "err_p90": q[3],
            "err_mean": float(err.mean()), "n_finite": int(err.size)})
        if len(rows) % 60 == 0:
            print(f"   {len(rows)} rows, {(time.time()-t0)/60:.1f} min", flush=True)
    Table("error_curves.csv", FIELDS, reset=True).rows(rows)
    print(f"wrote {path('error_curves.csv')} — {len(rows)} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
