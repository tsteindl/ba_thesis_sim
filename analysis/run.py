"""Consolidated results + algorithm diagnostics — the production pipeline.

    python analysis/run.py --quick          # ~1 min smoke run, same code path
    python analysis/run.py --full           # production sweep
    python analysis/run.py --full --resume  # continue an interrupted sweep
    python analysis/run.py --report-only    # rebuild the derived files

One manifest (`qmetrology/manifest.py`) defines every scenario, budget grid, algorithm, tuning grid,
seed and trial count. For each (scenario, budget, algorithm) the pipeline

    tunes on the tuning blocks -> freezes the winner -> evaluates it ONCE on the held-out seed with
    tracing on -> emits the convergence rate, the exploration/safeguard diagnostics, the detector
    confusion and the budget audit from those same runs.

So no performance number and its telemetry can refer to different trials.
"""
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
from pipeline_io import Table, ensure_dirs, path, scenario_keys, write_json


PERF_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "phi_distribution", "eps",
               "algorithm", "implementation_variant", "budget", "params", "R", "seed_test",
               "seed_tune_blocks", "rate", "rate_lo", "rate_hi", "rate_k", "ci_method",
               "ci_level", "N_min", "N_max"]

WIN_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "phi_distribution", "eps",
              "algorithm", "implementation_variant", "budget", "params", "tuned",
              "seed_tune_blocks", "R_tune_stage1", "R_tune_stage2", "stage1_configs",
              "stage2_configs", "stage3_configs", "block_rates", "block_mean", "runner_up", "margin_pp",
              "m_lo", "m_hi", "at_m_min", "at_m_max", "at_axis_bound", "axis_grids",
              "seed_test", "R", "heldout_rate"]

DET_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "phi_distribution", "eps",
              "algorithm", "implementation_variant", "budget", "R", "seed_test",
              "detector_eligible_n", "false_alarm_rate", "false_alarm_rate_lo",
              "false_alarm_rate_hi", "false_alarm_rate_n", "false_alarm_rate_k",
              "miss_rate", "miss_rate_lo", "miss_rate_hi", "miss_rate_n", "miss_rate_k",
              "probe_tp", "probe_fp", "probe_tn", "probe_fn", "ci_method_proportion", "ci_level"]

AUDIT_FIELDS = ["setting", "scenario_id", "algorithm", "implementation_variant", "budget", "R",
                "seed_test", "budget_util_mean", "budget_util_median", "budget_util_p90",
                "budget_util_max", "budget_unused_mean", "budget_violations",
                "no_exploitation_budget", "no_exploitation_budget_n", "termination_reasons"]

PHASE_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "eps", "algorithm", "budget", "R",
                "phase_bin", "phi_lo", "phi_hi", "n", "rate", "guess_ratio_median",
                "guess_overshoot", "guess_n", "star_ratio_median", "star_overshoot", "star_n",
                "exploration_share_median"]

ID_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "phi_distribution", "eps",
             "algorithm", "implementation_variant", "budget", "params", "seed_test",
             "seed_tune_blocks", "N_min", "N_max"]


def diag_fields():
    """Stable header for diagnostics_by_point.csv: identifiers first, then every diagnostic."""
    import numpy as _np
    R = 32
    A = {k: _np.zeros(R) for k in P._NUMERIC}
    A["nominal_budget"] = _np.ones(R)
    probe = D.point_diagnostics(A, ["ok"] * R, M.ALGORITHMS["binary_deep"], 8, 0)
    rest = [k for k in sorted(probe) if k not in ID_FIELDS]
    return ID_FIELDS + rest



def run_scenario(scen, mode, tables, algos=None):
    algos = list(algos) if algos else list(M.ORDER)
    cfgm = M.MODES[mode]
    budgets = M.budgets_for(scen, cfgm["n_budgets"])
    keys = scenario_keys(scen)
    keys["N_min"], keys["N_max"] = M.n_min_of(scen), M.n_max_of(scen)
    keys["seed_test"] = M.SEED_TEST
    keys["seed_tune_blocks"] = ",".join(str(s) for s in M.SEED_TUNE_BLOCKS)
    print(f"\n=== {scen['label']}  [{scen['id']}]  N_min={keys['N_min']} N_max={keys['N_max']}  "
          f"{len(budgets)} budgets {budgets[0]:,}..{budgets[-1]:,} ===", flush=True)

    perf, wins, diags, dets, audits, phases = [], [], [], [], [], []
    rates_by_algo = {a: [] for a in algos}
    frozen = {}          # (algo, budget) -> cfg, needed for the second (trace-keeping) pass
    for b in budgets:
        line = f"   B={b:>18,}"
        for algo in algos:
            spec = M.ALGORITHMS[algo]
            t0 = time.time()
            cfg, ev = P.tune(algo, scen, b, mode)
            A, statuses, _ = P.heldout(algo, scen, b, cfg, mode)
            d = D.point_diagnostics(A, statuses, spec, cfgm["n_boot"], M.SEED_BOOT)
            frozen[(algo, int(b))] = cfg
            row = dict(d)
            row.update(keys)
            row.update(algorithm=algo, implementation_variant=spec["variant"], budget=int(b),
                       params=json.dumps(cfg, sort_keys=True), ci_method="wilson")
            perf.append(row)
            diags.append(row)
            dets.append(row)
            audits.append(row)
            wins.append(dict(keys, algorithm=algo, implementation_variant=spec["variant"],
                             budget=int(b), params=json.dumps(cfg, sort_keys=True),
                             tuned=ev["tuned"],
                             R_tune_stage1=ev.get("R_tune_stage1", ""),
                             R_tune_stage2=ev.get("R_tune_stage2", ""),
                             stage1_configs=ev["stage1_configs"],
                             stage2_configs=ev["stage2_configs"],
                             stage3_configs=ev.get("stage3_configs", 0),
                             block_rates=json.dumps(ev["block_rates"]),
                             block_mean=ev.get("block_mean", ""),
                             runner_up=json.dumps(ev["runner_up"]) if ev["runner_up"] else "",
                             margin_pp=ev.get("margin_pp", ""),
                             m_lo=ev["m_lo"], m_hi=ev["m_hi"], at_m_min=ev["at_m_min"],
                             at_m_max=ev["at_m_max"],
                             at_axis_bound=json.dumps(ev.get("at_axis_bound", {}), sort_keys=True),
                             axis_grids=json.dumps(ev.get("axis_grids", {}), sort_keys=True),
                             R=d["R"], heldout_rate=d["rate"]))
            for pr in D.phase_strata(A, spec):
                phases.append(dict(keys, algorithm=algo, budget=int(b), R=d["R"], **pr))
            rates_by_algo[algo].append(d["rate"])
            line += f"  {algo[:3]} {100*d['rate']:6.2f}"
            del A
            _ = time.time() - t0
        print(line, flush=True)

    tables["perf"].rows(perf)
    tables["winners"].rows(wins)
    tables["diag"].rows(diags)
    tables["det"].rows([r for r in dets if M.ALGORITHMS[r["algorithm"]]["has_detector"]])
    tables["audit"].rows(audits)
    tables["phase"].rows(phases)



def completed_scenarios(algos, mode):
    """Scenario ids whose EVERY selected algorithm already has a full budget grid of rows.

    Deliberately not "the scenario_id appears in the table": a scenario interrupted midway
    through its algorithms would otherwise count as complete and silently skip real work.
    """
    rows = Table("performance_curves.csv", PERF_FIELDS, reset=False).read()
    seen = {}
    for r in rows:
        seen.setdefault((r["scenario_id"], r["algorithm"]), set()).add(r["budget"])
    done = set()
    for scen in M.SCENARIOS:
        want = len(M.budgets_for(scen, M.MODES[mode]["n_budgets"]))
        if all(len(seen.get((scen["id"], a), ())) >= want for a in algos):
            done.add(scen["id"])
    return done


def main():
    argv = sys.argv[1:]
    mode = "quick" if ("--quick" in argv or not argv) else "full"
    if "--smoke" in argv:
        mode = "smoke"
    if "--full" in argv:
        mode = "full"
    if "--max" in argv:
        mode = "max"
    resume = "--resume" in argv
    report_only = "--report-only" in argv
    algos, scens = list(M.ORDER), list(M.SCENARIOS)
    ensure_dirs()

    man = M.to_json(mode)
    write_json("experiment_manifest.json", man)
    if not report_only:
        # preflight: which implementation is which thesis algorithm, and what the resolved
        # binary-search pilot mismatch actually costs -- checked BEFORE the production sweep
        import audit
        print(audit.build(run_comparison=True, R=M.MODES[mode]["R_audit"]), flush=True)

        done = set()
        if resume:
            done = completed_scenarios(algos, mode)
            print(f"--resume: {len(done)} scenario(s) already complete for {algos}", flush=True)
        reset = not resume
        tables = {
            "perf": Table("performance_curves.csv", PERF_FIELDS, reset),
            "winners": Table("winners.csv", WIN_FIELDS, reset),
            "diag": Table("diagnostics_by_point.csv", diag_fields(), reset),
            "det": Table("detector_confusion.csv", DET_FIELDS, reset),
            "audit": Table("budget_audit.csv", AUDIT_FIELDS, reset),
            "phase": Table("diagnostics_by_phase.csv", PHASE_FIELDS, reset),
        }
        t0 = time.time()
        for scen in scens:
            if scen["id"] in done:
                continue
            try:
                run_scenario(scen, mode, tables, algos)
            except Exception as ex:     # never let one scenario kill an unattended sweep
                import traceback
                print(f"   !! scenario {scen['id']} FAILED: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()
        print(f"\nsweep finished in {(time.time()-t0)/60:.1f} min", flush=True)

    # Add the cheap reference rows after the expensive protocol sweep. The oracle is analytic and
    # separable needs no tuning, so this remains fast and makes the production command complete.
    import add_baselines
    add_baselines.main()
    import finalize
    finalize.main(mode)
    return mode


if __name__ == "__main__":
    main()
