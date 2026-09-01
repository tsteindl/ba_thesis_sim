"""Append the thesis tables' reference rows to an existing sweep, without re-running it.

Two rows the thesis reports for context but which are not protocols under comparison:

  separable   N = 1, the whole budget on shots. A real simulation, but the cheapest algorithm in the
              study (~6 us/trial), so it is evaluated at every operating point on the same held-out
              seed and at the same R as everything else, with tracing, exactly like the protocols.
  oracle_hl   told N = N_opt instead of searching for it, with m = floor(B/N) whole shots. Its
              convergence rate is evaluated deterministically from the asymptotic normal law that
              saturates the QCRB. No artificial estimator errors are drawn.

Neither enters the live/saturated regime rule, the "points won" count or any diagnostic aggregate:
those refer to the four protocols under test (qmetrology.manifest.PROTOCOLS). An oracle that
saturates would otherwise reclassify operating points for no algorithmic reason.

    python analysis/add_baselines.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qmetrology import diagnostics as D
from qmetrology import manifest as M
from qmetrology import pipeline as P
from qmetrology.oracle import oracle_rate
from pipeline_io import path, scenario_keys
from run import PERF_FIELDS, WIN_FIELDS, AUDIT_FIELDS, PHASE_FIELDS, diag_fields


def _read(name):
    with open(path(name), newline="") as f:
        return list(csv.DictReader(f))


def _append(name, fields, rows):
    if not rows:
        return
    with open(path(name), "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def _rewrite(name, fields, rows):
    with open(path(name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def main():
    mode = json.load(open(path("experiment_manifest.json")))["mode"]
    cfgm = M.MODES[mode]
    perf = _read("performance_curves.csv")
    have = {(r["scenario_id"], int(r["budget"]), r["algorithm"]) for r in perf}
    by_scen = {}
    for r in perf:
        by_scen.setdefault(r["scenario_id"], set()).add(int(r["budget"]))
    scen_by_id = {s["id"]: s for s in M.SCENARIOS}

    perf_rows, oracle_rows, win_rows, diag_rows, audit_rows, phase_rows = [], [], [], [], [], []
    t0 = time.time()
    for sid, budgets in sorted(by_scen.items()):
        scen = scen_by_id[sid]
        keys = scenario_keys(scen)
        keys["N_min"], keys["N_max"] = M.n_min_of(scen), M.n_max_of(scen)
        keys["seed_test"] = M.SEED_TEST
        keys["seed_tune_blocks"] = ",".join(str(x) for x in M.SEED_TUNE_BLOCKS)
        print(f"  {sid:14s} {len(budgets)} budgets", flush=True)

        for b in sorted(budgets):
            # ---- separable: an ordinary simulated row ---------------------------------
            if (sid, int(b), "separable") not in have:
                spec = M.ALGORITHMS["separable"]
                A, st, _ = P.heldout("separable", scen, b, {}, mode)
                d = D.point_diagnostics(A, st, spec, cfgm["n_boot"], M.SEED_BOOT)
                row = dict(d)
                row.update(keys)
                row.update(algorithm="separable", implementation_variant=spec["variant"],
                           budget=int(b), params="{}", ci_method="wilson")
                perf_rows.append(row); diag_rows.append(row); audit_rows.append(row)
                win_rows.append(dict(keys, algorithm="separable",
                                     implementation_variant=spec["variant"], budget=int(b),
                                     params="{}", tuned=False, stage1_configs=0,
                                     stage2_configs=0, stage3_configs=0, block_rates="{}",
                                     at_axis_bound="{}", axis_grids="{}",
                                     R=d["R"], heldout_rate=d["rate"]))
                for pr in D.phase_strata(A, spec):
                    phase_rows.append(dict(keys, algorithm="separable", budget=int(b),
                                           R=d["R"], **pr))

            # ---- oracle: deterministic, no synthetic error samples -------------------
            rate = oracle_rate(int(b), scen["eps"], scen["phi_min"], scen["phi_max"])
            oracle_rows.append(dict(
                keys, algorithm="oracle_hl",
                implementation_variant="qmetrology.oracle.oracle_rate (analytic QCRB rate at N_opt)",
                budget=int(b), params="{}", rate=round(float(rate), 12),
                rate_lo="", rate_hi="", rate_k="", R="",
                ci_method="analytic asymptotic QCRB probability; no Monte-Carlo error",
                ci_level=""))

    # Always replace the oracle rows. This makes the script idempotent and removes any rows left by
    # the former simulated-Gaussian implementation without touching the expensive protocol results.
    _rewrite("performance_curves.csv", PERF_FIELDS,
             [r for r in perf if r["algorithm"] != "oracle_hl"] + perf_rows + oracle_rows)
    _append("winners.csv", WIN_FIELDS, win_rows)
    _append("diagnostics_by_point.csv", diag_fields(), diag_rows)
    _append("budget_audit.csv", AUDIT_FIELDS, audit_rows)
    _append("diagnostics_by_phase.csv", PHASE_FIELDS, phase_rows)
    # Analytic oracle rows have no winner, trace, budget audit or phase diagnostics. Remove stale
    # records created by the former simulated implementation while preserving every protocol row.
    for name, fields in (("winners.csv", WIN_FIELDS),
                         ("diagnostics_by_point.csv", diag_fields()),
                         ("budget_audit.csv", AUDIT_FIELDS),
                         ("diagnostics_by_phase.csv", PHASE_FIELDS)):
        _rewrite(name, fields, [r for r in _read(name) if r["algorithm"] != "oracle_hl"])
    print(f"\nupdated {len(perf_rows) + len(oracle_rows)} performance rows "
          f"({sum(1 for r in perf_rows if r['algorithm']=='separable')} separable, "
          f"{len(oracle_rows)} analytic oracle) "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
