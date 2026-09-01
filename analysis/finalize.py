"""Derived outputs: budget crossings, headline tables, appendix parameters, REPORT.md, tex/.

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


def _tex(name, body):
    with open(path("tex", name), "w") as fh:
        fh.write(body)


def pct(x, d=1):
    return f"{100*f(x):.{d}f}" if np.isfinite(f(x)) else "--"


def num(x, d=2):
    return f"{f(x):.{d}f}" if np.isfinite(f(x)) else "--"


def bud(x):
    """Budgets to ~3 significant figures, in LaTeX scientific notation when large."""
    v = f(x)
    if not np.isfinite(v):
        return "--"
    if v >= 1e6:
        m, e = f"{v:.2e}".split("e")
        return rf"${m}\times 10^{{{int(e)}}}$"
    return f"{v:,.0f}"


def tex_tables(head, cross, params):
    # 1. exploration table -----------------------------------------------------------------
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Scenario & $B$ & Algorithm & \makecell{median\\$N_{\rm guess}/N_{\rm opt}$ (IQR)} & "
         r"exact & $\pm10\%$ & \makecell{guess\\overshoot} & "
         r"\makecell{median expl.\\budget share} & probes \\", r"\midrule"]
    for r in head:
        if r["algorithm"] == "brute":
            continue
        L.append(f"{r['setting']} & {bud(r['budget'])} & "
                 f"{SHORT.get(r['algorithm'], r['algorithm'])} & "
                 f"{num(r['guess_ratio_median'])} ({num(r['guess_ratio_q1'])}--"
                 f"{num(r['guess_ratio_q3'])}) & {pct(r['guess_exact'])}\\% & "
                 f"{pct(r['guess_within10'])}\\% & {pct(r['guess_overshoot'])}\\% & "
                 f"{pct(r['exploration_share_median'], 2)}\\% & {num(r['n_probes_mean'],1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _tex("exploration_table.tex", "\n".join(L) + "\n")

    # 2. guess -> final depth ---------------------------------------------------------------
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Scenario & $B$ & Algorithm & \makecell{median\\$N_{\rm guess}/N_{\rm opt}$} & "
         r"\makecell{median\\$N_*/N_{\rm opt}$} & \makecell{final\\overshoot} & "
         r"\makecell{converged\\$\mid$ safe depth} & converged \\", r"\midrule"]
    for r in head:
        L.append(f"{r['setting']} & {bud(r['budget'])} & "
                 f"{SHORT.get(r['algorithm'], r['algorithm'])} & "
                 f"{num(r['guess_ratio_median'])} & {num(r['star_ratio_median'])} & "
                 f"{pct(r['star_overshoot'],2)}\\% & {pct(r['conv_given_safe_depth'])}\\% & "
                 f"{pct(r['rate'])}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _tex("downstream_table.tex", "\n".join(L) + "\n")

    # 3. detector table ----------------------------------------------------------------------
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Scenario & $B$ & Algorithm & \makecell{false-alarm\\rate} & \makecell{miss\\rate} & "
         r"probes/trial & eligible runs \\", r"\midrule"]
    for r in head:
        if not np.isfinite(f(r["false_alarm_rate"])):
            continue
        L.append(f"{r['setting']} & {bud(r['budget'])} & "
                 f"{SHORT.get(r['algorithm'], r['algorithm'])} & "
                 f"{pct(r['false_alarm_rate'])}\\% & {pct(r['miss_rate'])}\\% & "
                 f"{num(r['n_probes_mean'],1)} & {r['detector_eligible_n']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _tex("detector_table.tex", "\n".join(L) + "\n")

    # 4. budget crossings ---------------------------------------------------------------------
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Scenario & Algorithm & $p^*$ & $B(p^*)$ [95\% CI] & ratio vs.\ brute [95\% CI] \\",
         r"\midrule"]
    sid_order = {sc["id"]: i for i, sc in enumerate(M.SCENARIOS)}
    for c in sorted(cross, key=lambda c: (sid_order.get(c["scenario_id"], 99),
                                          M.ORDER.index(c["algorithm"])
                                          if c["algorithm"] in M.ORDER else 9)):
        if int(c["threshold_pct"]) != 90 or c["budget_to_reach"] in ("", "nan"):
            continue
        lo, hi = c["budget_to_reach_lo"], c["budget_to_reach_hi"]
        rr = (f"{c['ratio_vs_brute']} [{c['ratio_lo']}, {c['ratio_hi']}]"
              if c["ratio_vs_brute"] != "" else "--")
        L.append(f"{c['setting']} & {SHORT.get(c['algorithm'], c['algorithm'])} & 90\\% & "
                 f"{f(c['budget_to_reach']):.3g} [{f(lo):.3g}, {f(hi):.3g}] & {rr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _tex("budget_crossings.tex", "\n".join(L) + "\n")

    # 5. appendix parameter table --------------------------------------------------------------
    L = [r"\begin{tabular}{llrll}", r"\toprule",
         r"Scenario & Algorithm & Budget & Frozen configuration & Note \\", r"\midrule"]
    for p in params:
        if p["reported_for"] == "tested_budget" or p["algorithm"] == "brute":
            continue
        cfg = p["params"].replace("_", r"\_").replace("{", "").replace("}", "").replace('"', "")
        L.append(f"{p['setting']} & {SHORT.get(p['algorithm'], p['algorithm'])} & "
                 f"{int(f(p['budget'])):,} & {cfg} & {p['reported_for'].replace('_', ' ')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _tex("optimal_params.tex", "\n".join(L) + "\n")


# ------------------------------------------------------------------------------------ REPORT
def report(mode, head, cross, params, regime):
    diag_all = load("diagnostics_by_point.csv")
    # every aggregate below is over LIVE points only: see operating_points.csv
    diag = [r for r in diag_all
            if regime.get((r["scenario_id"], int(f(r["budget"])))) == "live"]
    diag_protocols = [r for r in diag if r["algorithm"] in M.PROTOCOLS]
    perf = load("performance_curves.csv")
    wins = load("winners.csv")
    audit = load("budget_audit.csv")
    cfgm = M.MODES[mode]
    A = lambda algo, key: np.array([f(r[key]) for r in diag if r["algorithm"] == algo])
    fin = lambda v: v[np.isfinite(v)]
    L = []
    L.append("# Consolidated results and algorithm diagnostics\n")
    L.append(f"Mode `{mode}`. Generated by `python analysis/run.py --{mode}`. "
             f"Every number below comes from the held-out evaluation (seed {M.SEED_TEST}, "
             f"R = {cfgm['R_test']:,} trials per operating point) of a configuration frozen on the "
             f"disjoint tuning blocks {list(M.SEED_TUNE_BLOCKS)}.\n")
    n_live = len({(r["scenario_id"], r["budget"]) for r in diag})
    L.append(f"* scenarios: {len(M.SCENARIOS)}  * operating points: "
             f"{len({(r['scenario_id'], r['budget']) for r in perf})} "
             f"({n_live} live, see below)  "
             f"* evaluated (point, algorithm) cells: {len(perf)}\n")
    L.append("Reported algorithms: " + ", ".join(f"`{a}` ({M.ALGORITHMS[a]['label']})"
                                                 for a in M.ORDER) + ".\n")

    # 1 -------------------------------------------------------------------------------------
    L.append("## 1. Performance findings\n")
    L.append("| algorithm | mean convergence over all points | median | points won |")
    L.append("|---|---:|---:|---:|")
    rate = {a: A(a, "rate") for a in M.ORDER}
    by_pt = {}
    for r in diag:
        # only the protocols under comparison can "win" a point; the separable baseline and the
        # oracle are reference rows and are excluded by construction
        if r["algorithm"] not in M.PROTOCOLS:
            continue
        by_pt.setdefault((r["scenario_id"], r["budget"]), {})[r["algorithm"]] = f(r["rate"])
    live = by_pt      # `diag` is already restricted to the live points
    wins_count = {a: 0 for a in M.ORDER}
    for v in live.values():
        wins_count[max(v, key=lambda k: v[k])] += 1
    for a in M.ORDER:
        v = fin(rate[a])
        if not v.size:
            continue
        L.append(f"| {M.ALGORITHMS[a]['label']} | {100*v.mean():.2f}% | "
                 f"{100*np.median(v):.2f}% | {wins_count[a]}/{len(live)} |")
    L.append("")
    n_all = len({(r["scenario_id"], r["budget"]) for r in diag_all})
    L.append(f"**Every aggregate in this report is over the {len(live)} LIVE operating points** of "
             f"{n_all} — those where the best algorithm converges between 3% and 99%. At a "
             "saturated point every configuration ties, so the tuner's argmax, and therefore every "
             "diagnostic measured at that argmax, is Monte-Carlo noise rather than a property of "
             "the algorithm; at a floored point nothing converges at any depth. The per-point CSVs "
             "keep all 161 points and `operating_points.csv` carries the `regime` label to join on. "
             "Means and medians here mix scenarios and budgets, so they summarise the table rather "
             "than making a headline claim.\n")
    c90 = [c for c in cross if int(c["threshold_pct"]) == 90 and c["ratio_vs_brute"] != ""]
    if c90:
        L.append("Budget to reach 90% convergence, relative to brute force "
                 "(>1 means the algorithm needs less budget):\n")
        L.append("| algorithm | median ratio | min | max | scenarios with a 90% crossing |")
        L.append("|---|---:|---:|---:|---:|")
        for a in M.ORDER[1:]:
            v = np.array([f(c["ratio_vs_brute"]) for c in c90 if c["algorithm"] == a])
            v = fin(v)
            if v.size:
                L.append(f"| {M.ALGORITHMS[a]['label']} | {np.median(v):.2f} | {v.min():.2f} | "
                         f"{v.max():.2f} | {v.size} |")
        L.append("")
    L.append(f"Crossing rule: {M.CROSSING_RULE}. Intervals are {M.CI_LEVEL}% percentile intervals "
             f"from a parametric bootstrap ({cfgm['n_boot']} replicates, seed {M.SEED_BOOT}) of the "
             "convergence curve; full rows in `budget_crossings.csv`.\n")

    # 2 -------------------------------------------------------------------------------------
    L.append("## 2. Exploration phase — `N_guess / N_opt`\n")
    L.append("| algorithm | median ratio (over points) | mean abs. rel. error | signed rel. error "
             "| exact hit | within 10% | guess overshoot | median expl. budget share | "
             "mean probes |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        if not M.ALGORITHMS[a]["has_guess"]:
            continue
        g = fin(A(a, "guess_ratio_median"))
        if not g.size:
            continue
        L.append(f"| {M.ALGORITHMS[a]['label']} | {np.median(g):.2f} | "
                 f"{fin(A(a,'guess_abs_rel_mean')).mean():.2f} | "
                 f"{fin(A(a,'guess_signed_rel_mean')).mean():+.2f} | "
                 f"{100*fin(A(a,'guess_exact')).mean():.1f}% | "
                 f"{100*fin(A(a,'guess_within10')).mean():.1f}% | "
                 f"{100*fin(A(a,'guess_overshoot')).mean():.1f}% | "
                 f"{100*np.median(fin(A(a,'exploration_share_median'))):.2f}% | "
                 f"{fin(A(a,'n_probes_mean')).mean():.1f} |")
    L.append("")
    L.append("Averages are over operating points (each point already averages over its "
             f"R = {cfgm['R_test']:,} runs); per-point rows with intervals and eligible "
             "denominators are in `diagnostics_by_point.csv`, and the compact per-scenario subset "
             "is `diagnostics_headline.csv`. `brute` has no exploration phase, so its `N_guess` "
             "columns are empty by construction rather than zero.\n")
    for a in M.ORDER:
        if not M.ALGORITHMS[a]["has_guess"]:
            continue
        sp = fin(A(a, "single_probe"))
        ne = fin(A(a, "no_exploitation"))
        if sp.size:
            L.append(f"* **{M.ALGORITHMS[a]['label']}**: {100*sp.mean():.1f}% of runs take only the "
                     f"opening probe (no search step); {100*ne.mean():.2f}% never reach an "
                     f"exploitation phase.")
    L.append("")
    L.append("### Trend with the target precision\n")
    L.append("The precision-sweep scenarios (`U(0.01, 0.1)`, eps from 1e-3 to 1e-8) hold the prior "
             "fixed, so the only thing changing is eps and the budget scale it forces. Live points "
             "only.\n")
    L.append("| algorithm | eps | median expl. share | median `N_guess/N_opt` | "
             "median `N*/N_opt` | final overshoot | mean convergence |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    sweep_ids = [sc["id"] for sc in M.SCENARIOS if "precision_sweep" in sc["families"]]
    for a in M.ORDER:
        for sid in sweep_ids:
            rs = [r for r in diag if r["algorithm"] == a and r["scenario_id"] == sid]
            if not rs:
                continue
            col = lambda k: fin(np.array([f(r[k]) for r in rs]))
            e = rs[0]["eps"]
            gm = col("guess_ratio_median")
            L.append(f"| {M.ALGORITHMS[a]['label']} | {float(e):.0e} | "
                     f"{100*np.median(col('exploration_share_median')):.2f}% | "
                     f"{(f'{np.median(gm):.2f}' if gm.size else '--')} | "
                     f"{np.median(col('star_ratio_median')):.2f} | "
                     f"{100*col('star_overshoot').mean():.2f}% | "
                     f"{100*col('rate').mean():.1f}% |")
    L.append("")
    L.append("### Trend with the available budget\n")
    L.append("Live operating points split into terciles of budget *within each scenario*, so the "
             "comparison is not confounded by the scale differences between scenarios. The "
             "saturated high-budget points are excluded: there every configuration ties and the "
             "tuned exploration size drifts to an arbitrary value, which would otherwise show up "
             "here as a spurious collapse in depth quality.\n")
    L.append("| algorithm | budget tercile | median expl. share | median `N_guess/N_opt` | "
             "guess overshoot | mean probes | only the opening probe |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    ranks = {}
    for sid in {r["scenario_id"] for r in diag}:
        bs = sorted({int(f(r["budget"])) for r in diag if r["scenario_id"] == sid})
        for i, b in enumerate(bs):
            ranks[(sid, b)] = "low" if i < len(bs) / 3 else ("mid" if i < 2 * len(bs) / 3
                                                            else "high")
    for a in M.ORDER:
        if not M.ALGORITHMS[a]["has_guess"]:
            continue
        for band in ("low", "mid", "high"):
            rs = [r for r in diag if r["algorithm"] == a
                  and ranks.get((r["scenario_id"], int(f(r["budget"])))) == band]
            if not rs:
                continue
            col = lambda k: fin(np.array([f(r[k]) for r in rs]))
            L.append(f"| {M.ALGORITHMS[a]['label']} | {band} | "
                     f"{100*np.median(col('exploration_share_median')):.2f}% | "
                     f"{np.median(col('guess_ratio_median')):.2f} | "
                     f"{100*col('guess_overshoot').mean():.1f}% | "
                     f"{col('n_probes_mean').mean():.1f} | "
                     f"{100*col('single_probe').mean():.1f}% |")
    L.append("")

    # 3 -------------------------------------------------------------------------------------
    L.append("## 3. Safeguard and final depth — `N_star / N_opt`\n")
    L.append("| algorithm | median `N*/N_opt` | final overshoot | converged given a safe depth | "
             "unsafe-guess rescue | median `N*/N_guess` |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        s = fin(A(a, "star_ratio_median"))
        if not s.size:
            continue
        rs = fin(A(a, "rescue_share"))
        og = fin(A(a, "star_over_guess_median"))
        L.append(f"| {M.ALGORITHMS[a]['label']} | {np.median(s):.2f} | "
                 f"{100*fin(A(a,'star_overshoot')).mean():.2f}% | "
                 f"{100*fin(A(a,'conv_given_safe_depth')).mean():.1f}% | "
                 f"{(f'{100*rs.mean():.1f}%' if rs.size else '--')} | "
                 f"{(f'{np.median(og):.2f}' if og.size else '--')} |")
    L.append("")
    L.append("The safeguard's target is `N_opt`, and it deliberately backs off from the aliasing "
             "cliff, so a median below 1 is the intended behaviour rather than a miss. Early "
             "stopping, final overshoot and ordinary shot noise are reported here as overlapping "
             "stage flags and conditional rates, **not** as a mutually exclusive failure "
             "decomposition — no priority or counterfactual rule is defined that would justify "
             "one.\n")

    # detector
    det = load("detector_confusion.csv")
    if det:
        L.append("### Detector quality (linear and binary search only)\n")
        L.append("| algorithm | trial-level false alarm | trial-level miss | probe TP | FP | TN | "
                 "FN | eligible runs |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for a in ("linear", "binary_deep"):
            rs = [r for r in det if r["algorithm"] == a]
            if not rs:
                continue
            fa = fin(np.array([f(r["false_alarm_rate"]) for r in rs]))
            ms = fin(np.array([f(r["miss_rate"]) for r in rs]))
            L.append(f"| {M.ALGORITHMS[a]['label']} | {100*fa.mean():.1f}% | {100*ms.mean():.1f}% |"
                     f" {sum(int(f(r['probe_tp'],0)) for r in rs):,} | "
                     f"{sum(int(f(r['probe_fp'],0)) for r in rs):,} | "
                     f"{sum(int(f(r['probe_tn'],0)) for r in rs):,} | "
                     f"{sum(int(f(r['probe_fn'],0)) for r in rs):,} | "
                     f"{sum(int(f(r['detector_eligible_n'],0)) for r in rs):,} |")
        L.append("")
        L.append("The rates are trial-level (at least one safe probe declared an overshoot / at "
                 "least one overshooting probe accepted) with Wilson intervals at the row's own R, "
                 "because probes inside one run are dependent. The probe-level confusion counts are "
                 "supporting telemetry only. Reverse engineering and brute force have no detector "
                 "and are recorded as N/A, not as zero.\n")

    # 4 -------------------------------------------------------------------------------------
    L.append("## 4. Statistical uncertainty and grid sensitivity\n")
    L.append(f"* Convergence proportions: Wilson score intervals at the row's own R "
             f"(R = {cfgm['R_test']:,} here; the value is stored per row and read from it).\n"
             f"* Diagnostic proportions: Wilson at the metric's own eligible denominator, which is "
             f"stored alongside every share as `*_n` / `*_k`.\n"
             f"* Continuous diagnostics: median with an exact order-statistic bootstrap percentile "
             f"interval, plus the mean with a run-level percentile bootstrap "
             f"({cfgm['n_boot']} replicates, seed {M.SEED_BOOT}).\n"
             f"* Budget crossings and ratios: parametric bootstrap of the convergence curve, same "
             f"replicate count and seed.\n")
    gs = np.array([f(c["grid_sensitivity_rel"]) for c in cross])
    gs = fin(gs)
    if gs.size:
        L.append(f"Budget-grid discretisation: re-deriving each crossing from the two half-density "
                 f"subgrids moves it by a median of {100*np.median(gs):.2f}% and at most "
                 f"{100*gs.max():.2f}%. Log-linear interpolation error is O(h^2) in the grid "
                 f"log-spacing, so the full-grid contribution is about a quarter of that.\n")
    viol = sum(int(f(r["budget_violations"], 0)) for r in audit)
    mx = max([f(r["budget_util_max"]) for r in audit] + [float("nan")])
    L.append(f"**Budget compliance.** Over every held-out run of every point, "
             f"{viol} trial(s) spent more than the nominal budget; the worst per-trial spend "
             f"observed is {mx:.4f}x the cap. Per-point rows are in `budget_audit.csv` "
             "(mean, median, p90, max and unused share). A mean spend of 1.0000x does not by "
             "itself certify compliance, which is why the per-trial maximum and the violation "
             "count are reported.\n")
    bnd = [w for w in wins if str(w.get("at_m_min")) == "1" or str(w.get("at_m_max")) == "1"]
    tuned = [w for w in wins if str(w.get("tuned")).lower() in ("true", "1")]
    if tuned:
        L.append(f"**Tuning provenance.** {len(tuned)} tuned cells; every held-out row points to "
                 f"exactly one frozen winner in `winners.csv`, with its per-block tuning rates, "
                 f"the runner-up and the selection margin. The exploration-size grid spans "
                 f"[1, B // N_min], which cannot bind: "
                 f"{len(bnd)}/{len(tuned)} winners sit on an endpoint and each such row carries "
                 f"`at_m_min` / `at_m_max` so the case is visible rather than silent.\n")

    # 5 -------------------------------------------------------------------------------------
    L.append("## 5. Code/thesis mismatches and limitations\n")
    L.append("1. **Binary-search pilot (resolved, and it changes the reported algorithm).** The "
             "thesis pseudocode feeds the deepest non-overshooting probe to the statistical "
             "safeguard; `find_phi_fixed_budget_binary_search_risk` fed the opening probe. A new "
             "function `find_phi_fixed_budget_binary_search_deep` implements the pseudocode and is "
             "what this pipeline reports; the old function is untouched. `algorithm_code_audit.md` "
             "measures the difference on common seeds.\n")
    L.append("2. **Selection bias in that pilot.** The deepest accepted estimate is accepted "
             "*because it passed the overshoot test*, so treating it afterwards as an unbiased "
             "Gaussian pilot is outside the safeguard derivation. It is not assumed away here: the "
             "detector false-alarm/miss rates and the `N_guess/N_opt` distribution measure the "
             "consequence directly.\n")
    L.append("3. **Linear search has no per-probe detector.** Its stopping rule is a trial-level "
             "verdict, so the probe-level classification scored above is the algorithm's own "
             "backtracking decision (the last `lookback_window` probes are the ones it discards). "
             "This is stated rather than hidden because a different mapping would give different "
             "false-alarm numbers.\n")
    L.append("4. **`../thesis/*.tex` is not present in this checkout**, so the audit is written "
             "against the pseudocode as reproduced in `thesis_code/`, not against the .tex "
             "sources. No thesis file is read or written by this pipeline.\n")
    L.append("## Figures\n")
    L.append("`fig_diagnostics_vs_budget.png` — exploration budget share, median "
             "`N_guess/N_opt` and final overshoot against the available budget (normalised per "
             "scenario). `fig_guess_vs_final_depth.png` — what the safeguard does to the "
             "exploration's guess, and how the final depth maps onto convergence. Both are "
             "regenerated by `python analysis/figures.py` from "
             "`diagnostics_by_point.csv` alone.\n")
    L.append("## Files\n")
    L.append("This pipeline's own outputs, in `results/` (the side studies -- ladder, posterior, "
             "broad priors -- write their own files into the same directory):\n")
    L.append("```\n" + "\n".join(f for f in PIPELINE_OUTPUTS
                                  if os.path.exists(path(f))) + "\n```\n")
    with open(path("REPORT.md"), "w") as fh:
        fh.write("\n".join(L))


# What this pipeline writes. Listed explicitly rather than by scanning the directory, because
# results/ also holds the side studies' output and a raw listing would present theirs as ours.
PIPELINE_OUTPUTS = (
    "experiment_manifest.json", "algorithm_code_audit.md",
    "performance_curves.csv", "winners.csv", "operating_points.csv",
    "diagnostics_by_point.csv", "diagnostics_by_phase.csv", "diagnostics_headline.csv",
    "detector_confusion.csv", "budget_audit.csv", "budget_crossings.csv", "optimal_params.csv",
    "error_curves.csv", "variance_curves.csv",
    "REPORT.md", "FULL_RESULTS.md",
    "fig_story.png", "fig_story_small.png", "fig_pareto.png", "fig_precision.png",
    "fig_error.png", "fig_variance.png", "fig_error_variance.png", "fig_error_density.png",
    "fig_signed_error_density.png", "fig_phi_hat_density.png", "fig_algorithm_diagnostics.png",
    "fig_diagnostics_vs_budget.png", "fig_guess_vs_final_depth.png",
    "tex", "tex_long", "traces",
)


def main(mode="full"):
    regime = operating_points()
    cross = crossings(mode)
    params = optimal_params(cross)
    head = headline(regime)
    tex_tables(head, cross, params)
    report(mode, head, cross, params, regime)
    try:
        import figures
        figures.main()
    except Exception as ex:      # a missing matplotlib must not invalidate the numbers
        print(f"(figures skipped: {type(ex).__name__}: {ex})")
    print("wrote", path("REPORT.md"), "+ budget_crossings.csv, optimal_params.csv, "
          "diagnostics_headline.csv, tex/")


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
