"""Derived outputs: budget crossings, headline tables, appendix parameters.

Reads only what the sweep already wrote (results/*.csv) -- no simulation happens here,
so the report can be rebuilt after the fact with `--report-only`.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qmetrology import manifest as M
from qmetrology.oracle import oracle_budget_for_rate
from qmetrology.uncertainty import crossing, crossing_ci, ratio_ci, grid_sensitivity
from pipeline_io import Table, path

CROSS_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "phi_distribution", "eps",
                "algorithm", "implementation_variant", "threshold_pct", "budget_to_reach",
                "budget_to_reach_lo", "budget_to_reach_hi", "ratio_vs_brute", "ratio_lo",
                "ratio_hi", "grid_sensitivity_rel", "crossing_rule", "R", "n_boot", "boot_seed",
                "ci_level", "coverage", "n_budget_points"]

PARAM_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "eps", "algorithm",
                "implementation_variant", "reported_for", "budget", "budget_note", "params",
                "R_tune_stage1", "R_tune_stage2", "seed_tune_blocks", "block_rates", "margin_pp",
                "at_m_min", "at_m_max", "m_lo", "m_hi", "heldout_rate", "R", "seed_test"]

HEAD_FIELDS = ["setting", "scenario_id", "eps", "algorithm", "budget", "R", "rate", "rate_lo",
               "rate_hi", "guess_ratio_median", "guess_ratio_q1", "guess_ratio_q3",
               "guess_ratio_median_lo", "guess_ratio_median_hi", "guess_exact", "guess_within10",
               "guess_overshoot", "guess_eligible_n", "exploration_share_median",
               "exploration_share_p90", "n_probes_mean", "star_ratio_median", "star_ratio_q1",
               "star_ratio_q3", "star_overshoot", "star_eligible_n", "conv_given_safe_depth",
               "rescue_share", "rescue_share_n", "false_alarm_rate", "miss_rate",
               "detector_eligible_n", "no_exploitation", "budget_util_max", "budget_violations",
               "regime"]


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def sig(x, n=3):
    """Round to n significant figures (budgets are reported to ~3 s.f. per the handoff)."""
    if not np.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{n}g}")


def load(name):
    p = path(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


OP_FIELDS = ["setting", "scenario_id", "phi_min", "phi_max", "eps", "budget", "R",
             "best_rate", "best_algorithm", "worst_rate", "spread_pp", "regime", "n_algorithms"]

# A point is only informative if the algorithms can actually be told apart there. Above `SAT` every
# configuration converges and the tuner's argmax -- and therefore every diagnostic measured at that
# argmax -- is Monte-Carlo noise; below `FLOOR` nothing converges at any depth.
SAT, FLOOR = 0.99, 0.03


def operating_points():
    """Classify every (scenario, budget) as floored / live / saturated, once, for everything else
    to join against. Written as its own tidy file rather than mutating the sweep's output."""
    perf = load("performance_curves.csv")
    by = {}
    for r in perf:
        by.setdefault((r["scenario_id"], int(f(r["budget"]))), []).append(r)
    rows = []
    for (sid, b), rs in sorted(by.items()):
        rates = {r["algorithm"]: f(r["rate"]) for r in rs
                 if M.ALGORITHMS.get(r["algorithm"], {}).get("role", "protocol") == "protocol"}
        if not rates:
            continue
        best = max(rates.values())
        worst = min(rates.values())
        regime = "saturated" if best > SAT else ("floored" if best < FLOOR else "live")
        rows.append({
            "setting": rs[0]["setting"], "scenario_id": sid,
            "phi_min": rs[0]["phi_min"], "phi_max": rs[0]["phi_max"], "eps": rs[0]["eps"],
            "budget": b, "R": rs[0]["R"],
            "best_rate": round(best, 5),
            "best_algorithm": max(rates, key=lambda k: rates[k]),
            "worst_rate": round(worst, 5), "spread_pp": round(100 * (best - worst), 2),
            "regime": regime, "n_algorithms": len(rs)})
    Table("operating_points.csv", OP_FIELDS, reset=True).rows(rows)
    return {(r["scenario_id"], int(r["budget"])): r["regime"] for r in rows}


# --------------------------------------------------------------------------------- crossings
def crossings(mode):
    cfgm = M.MODES[mode]
    perf = load("performance_curves.csv")
    by = {}
    for r in perf:
        by.setdefault((r["scenario_id"], r["algorithm"]), []).append(r)
    rows = []
    scen_by_id = {s["id"]: s for s in M.SCENARIOS}
    for (sid, algo), rs in sorted(by.items()):
        rs.sort(key=lambda r: int(r["budget"]))
        b = np.array([int(r["budget"]) for r in rs], float)
        p = np.array([f(r["rate"]) for r in rs])
        Rs = {int(f(r["R"])) for r in rs if str(r.get("R", "")).strip() not in ("", "nan")}
        # The oracle is analytic -- closed-form probability averaged over the prior, nothing sampled.
        # It has no R, so it gets its crossing as a point value with no bootstrap interval, rather
        # than a bootstrap over a trial count it does not have.
        exact = not Rs
        R = min(Rs) if Rs else 0
        ref = by.get((sid, "brute"))
        if ref:
            ref = sorted(ref, key=lambda r: int(r["budget"]))
            b_ref = np.array([int(r["budget"]) for r in ref], float)
            p_ref = np.array([f(r["rate"]) for r in ref])
        scen = scen_by_id.get(sid, {})
        for T in M.THRESHOLDS:
            analytic_oracle = algo == "oracle_hl"
            if analytic_oracle:
                pt = float(oracle_budget_for_rate(
                    T, f(scen["eps"]), f(scen["phi_min"]), f(scen["phi_max"])))
                lo, hi, cov = float("nan"), float("nan"), 1.0
            elif exact:
                pt, lo, hi, cov = crossing(b, p, T), float("nan"), float("nan"), 1.0
            else:
                pt, lo, hi, cov = crossing_ci(b, p, T, R, n_boot=cfgm["n_boot"],
                                              seed=M.SEED_BOOT, ci=M.CI_LEVEL)
            rat = rl = rh = float("nan")
            if ref is not None and algo != "brute":
                if exact:
                    ca = crossing(b_ref, p_ref, T)
                    cb = pt if analytic_oracle else crossing(b, p, T)
                    rat = (ca / cb) if (np.isfinite(ca) and np.isfinite(cb) and cb) else float("nan")
                else:
                    rat, rl, rh, _c = ratio_ci(b_ref, p_ref, p, T, R, n_boot=cfgm["n_boot"],
                                               seed=M.SEED_BOOT, ci=M.CI_LEVEL, budgets_alg=b)
            rows.append({
                "setting": rs[0]["setting"], "scenario_id": sid,
                "phi_min": rs[0]["phi_min"], "phi_max": rs[0]["phi_max"],
                "phi_distribution": rs[0]["phi_distribution"], "eps": rs[0]["eps"],
                "algorithm": algo, "implementation_variant": rs[0]["implementation_variant"],
                "threshold_pct": int(100 * T),
                "budget_to_reach": int(pt) if analytic_oracle else sig(pt),
                "budget_to_reach_lo": sig(lo),
                "budget_to_reach_hi": sig(hi),
                "ratio_vs_brute": round(rat, 2) if np.isfinite(rat) else "",
                "ratio_lo": round(rl, 2) if np.isfinite(rl) else "",
                "ratio_hi": round(rh, 2) if np.isfinite(rh) else "",
                "grid_sensitivity_rel": "" if analytic_oracle else
                    (round(grid_sensitivity(b, p, T), 4)
                     if np.isfinite(grid_sensitivity(b, p, T)) else ""),
                "crossing_rule": ("smallest integer budget with analytic oracle rate >= threshold"
                                  if analytic_oracle else M.CROSSING_RULE),
                "R": R, "n_boot": cfgm["n_boot"],
                "boot_seed": M.SEED_BOOT, "ci_level": M.CI_LEVEL,
                "coverage": round(cov, 3), "n_budget_points": len(b),
            })
    t = Table("budget_crossings.csv", CROSS_FIELDS, reset=True)
    t.rows(rows)
    return rows


# ------------------------------------------------------------------------- appendix parameters
def optimal_params(cross):
    """Appendix-ready frozen configurations.

    Two kinds of row, distinguished by `reported_for`:
      `tested_budget`  the frozen winner at a budget that was actually evaluated;
      `B90_nearest`    the winner at the tested budget NEAREST the interpolated B_90 crossing;
      `B90_bracket_lo` / `B90_bracket_hi`  the two winners bracketing it.
    A budget crossing is interpolated between tested budgets; parameter dictionaries are NOT
    interpolated, which is exactly why the bracketing rows exist.
    """
    wins = load("winners.csv")
    rows = []
    for w in wins:
        rows.append({k: w.get(k, "") for k in PARAM_FIELDS} |
                    {"reported_for": "tested_budget", "budget_note": "evaluated directly",
                     "setting": w["setting"], "scenario_id": w["scenario_id"],
                     "phi_min": w["phi_min"], "phi_max": w["phi_max"], "eps": w["eps"],
                     "algorithm": w["algorithm"],
                     "implementation_variant": w["implementation_variant"],
                     "budget": w["budget"], "params": w["params"]})
    b90 = {(c["scenario_id"], c["algorithm"]): f(c["budget_to_reach"])
           for c in cross if int(c["threshold_pct"]) == 90}
    by = {}
    for w in wins:
        by.setdefault((w["scenario_id"], w["algorithm"]), []).append(w)
    for k, target in b90.items():
        cand = sorted(by.get(k, []), key=lambda w: int(w["budget"]))
        if not cand or not np.isfinite(target):
            continue
        bs = np.array([int(w["budget"]) for w in cand], float)
        i_near = int(np.argmin(np.abs(np.log(bs) - np.log(target))))
        lo = cand[max(0, int(np.searchsorted(bs, target)) - 1)]
        hi = cand[min(len(cand) - 1, int(np.searchsorted(bs, target)))]
        for tag, w, note in (("B90_nearest", cand[i_near],
                              f"NEAREST TESTED budget to the interpolated B_90 = {sig(target):,.0f}"
                              "; not an exact optimum at B_90"),
                             ("B90_bracket_lo", lo,
                              f"lower bracket of the interpolated B_90 = {sig(target):,.0f}"),
                             ("B90_bracket_hi", hi,
                              f"upper bracket of the interpolated B_90 = {sig(target):,.0f}")):
            rows.append({kk: w.get(kk, "") for kk in PARAM_FIELDS} |
                        {"reported_for": tag, "budget_note": note,
                         "setting": w["setting"], "scenario_id": w["scenario_id"],
                         "phi_min": w["phi_min"], "phi_max": w["phi_max"], "eps": w["eps"],
                         "algorithm": w["algorithm"],
                         "implementation_variant": w["implementation_variant"],
                         "budget": w["budget"], "params": w["params"]})
    t = Table("optimal_params.csv", PARAM_FIELDS, reset=True)
    t.rows(rows)
    return rows


# ------------------------------------------------------------------------------- headline set
def headline(regime):
    diag = load("diagnostics_by_point.csv")
    by_scen = {}
    for r in diag:
        by_scen.setdefault(r["scenario_id"], []).append(r)
    rows = []
    for sid, rs in by_scen.items():
        scen = next((s for s in M.SCENARIOS if s["id"] == sid), None)
        budgets = sorted({int(r["budget"]) for r in rs})
        ref = {int(r["budget"]): f(r["rate"]) for r in rs if r["algorithm"] == "reverse_eng_risk"}
        pins = set(int(b) for b in (scen.get("pin_budgets", ()) if scen else ()))
        if ref:
            bb = np.array(sorted(ref)); rr = np.array([ref[int(b)] for b in bb])
            for t in (0.5, 0.8, 0.9):
                pins.add(int(bb[int(np.argmin(np.abs(rr - t)))]))
        else:
            pins.add(budgets[len(budgets) // 2])
        for r in rs:
            if int(f(r["budget"])) in pins:
                rows.append({k: r.get(k, "") for k in HEAD_FIELDS} |
                            {"regime": regime.get((r["scenario_id"], int(f(r["budget"]))), "")})
    sid_order = {sc["id"]: i for i, sc in enumerate(M.SCENARIOS)}
    rows.sort(key=lambda r: (sid_order.get(r["scenario_id"], 99), int(f(r["budget"])),
                             M.ORDER.index(r["algorithm"]) if r["algorithm"] in M.ORDER else 9))
    t = Table("diagnostics_headline.csv", HEAD_FIELDS, reset=True)
    t.rows(rows)
    return rows


# ------------------------------------------------------------------------------------- LaTeX
NICE = {a: M.ALGORITHMS[a]["label"] for a in M.ALGORITHMS}
SHORT = {"brute": "Brute force", "linear": "Linear search", "binary_deep": "Binary search",
         "reverse_eng_risk": "Reverse engineering"}


def main(mode="full"):
    # each of these writes its CSV as a side effect; the return values are no longer needed
    regime = operating_points()
    optimal_params(crossings(mode))
    headline(regime)
    try:
        import thesis_figures
        thesis_figures.main(only={"diagnostics_vs_budget", "guess_vs_final_depth"})
    except Exception as ex:      # a missing matplotlib must not invalidate the numbers
        print(f"(figures skipped: {type(ex).__name__}: {ex})")
    print("wrote budget_crossings.csv, optimal_params.csv, diagnostics_headline.csv")


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
