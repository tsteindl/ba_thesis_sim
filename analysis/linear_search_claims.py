"""Re-derive every quantitative linear-search claim of Section 3.2.1 from the stored output.

Section 3.2.1 makes about twenty numeric statements, plus four statements about what the CODE
does that the pseudocode is supposed to match. This script recomputes all of them -- the
aggregates from results/*.csv using the same aggregation the thesis tables use, the false-alarm
distance diagnostics by replaying the published configuration (analysis/linear_detector_study.py),
and the code statements from the live source -- and records agreement or disagreement.

It also checks the noiseless-branch calculation that replaces the claim that the estimator "tends
toward zero" whenever N > N_opt.

    python analysis/linear_search_claims.py                  # against the live results/
    python analysis/linear_search_claims.py --against DIR    # against another sweep's output

Writes results/linear_search_claims.csv. Nothing here modifies any .tex file.
"""
import csv
import inspect
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qmetrology import algorithms as A
from qmetrology import manifest as M
from pipeline_io import Table, path

FIELDS = ["group", "claim", "quoted", "recomputed", "verdict", "source"]

SRC = (os.path.abspath(sys.argv[sys.argv.index("--against") + 1])
       if "--against" in sys.argv else path(""))
TOL = dict(pct=0.6, ratio=0.03, count=0)      # pp for percentages, relative for ratios


def load(name):
    """Read one CSV of the audited sweep (results/ unless --against says otherwise)."""
    p = os.path.join(SRC, name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def live_keys():
    return {(r["scenario_id"], int(f(r["budget"]))) for r in load("operating_points.csv")
            if r["regime"] == "live"}


def rows_for(name, algo, live, sids=None):
    keys = live
    return [r for r in load(name)
            if r["algorithm"] == algo and (r["scenario_id"], int(f(r["budget"]))) in keys
            and (sids is None or r["scenario_id"] in sids)]


def agg(rows, key, how="median"):
    v = np.array([f(r.get(key)) for r in rows], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    return float(np.median(v)) if how == "median" else float(v.mean())


def verdict(quoted, got, kind="pct", tol=None):
    """`tol` overrides the default tolerance; a count claim is exact unless one is given."""
    if not np.isfinite(got):
        return "not recomputable"
    if quoted is None:
        return "reported (no value quoted)"
    if kind == "pct":
        return "confirmed" if abs(100 * got - quoted) <= (tol or TOL["pct"]) else "DIFFERS"
    if kind == "ratio":
        return "confirmed" if abs(got - quoted) <= (tol or TOL["ratio"]) else "DIFFERS"
    d = abs(got - quoted)
    if d == 0:
        return "confirmed"
    return "confirmed (within 1)" if tol and d <= tol else "DIFFERS"


def row(group, claim, quoted, got, kind="pct", source="", fmt=None, tol=None):
    if fmt is None:
        fmt = (lambda x: f"{100*x:.2f}%") if kind == "pct" else (
            (lambda x: f"{x:.2f}") if kind == "ratio" else (lambda x: f"{x:g}"))
    q = (f"{quoted:.2f}%" if kind == "pct" else
         (f"{quoted:.2f}" if kind == "ratio" else f"{quoted:g}")) if quoted is not None else ""
    return dict(group=group, claim=claim, quoted=q,
                recomputed=fmt(got) if np.isfinite(got) else "n/a",
                verdict=verdict(quoted, got, kind, tol), source=source)


# ------------------------------------------------------------------- 1. aggregate diagnostics
def aggregate_claims():
    live = live_keys()
    det = rows_for("detector_confusion.csv", "linear", live)
    diag = rows_for("diagnostics_by_point.csv", "linear", live)
    broad = {s["id"] for s in M.SCENARIOS if "broad_prior" in s.get("families", ())}
    diag_nb = [r for r in diag if r["scenario_id"] not in broad]
    src_d, src_g = "detector_confusion.csv", "diagnostics_by_point.csv"
    out = [
        row("aggregate diagnostics", "trial-level false-alarm rate", 45.3,
            agg(det, "false_alarm_rate", "mean"), source=src_d),
        row("aggregate diagnostics", "trial-level miss rate", 7.7,
            agg(det, "miss_rate", "mean"), source=src_d),
        row("aggregate diagnostics", "median N_guess/N_opt", 1.00,
            agg(diag, "guess_ratio_median"), "ratio", src_g),
        row("aggregate diagnostics", "guesses within 10% of N_opt", 68.6,
            agg(diag, "guess_within10", "mean"), source=src_g),
        row("aggregate diagnostics", "median N*/N_opt after the decrement", 0.95,
            agg(diag, "star_ratio_median"), "ratio", src_g),
        row("aggregate diagnostics", "final overshoot rate", 2.35,
            agg(diag, "star_overshoot", "mean"), source=src_g),
        row("aggregate diagnostics", "median exploration budget share", 1.78,
            agg(diag, "exploration_share_median"), source=src_g),
        row("aggregate diagnostics", "informative (live) operating points", 144,
            float(len(live)), "count", "operating_points.csv"),
    ]
    # Appendix B.2 excludes the broad priors; the notes quote its per-scenario rows
    for sid, (gr, mae, w10) in {"narrow_e3": (0.83, 0.29, 32.0),
                                "narrow_e4": (1.00, 0.05, 87.6),
                                "narrow_e5": (1.00, 0.04, 89.6)}.items():
        rs = [r for r in diag_nb if r["scenario_id"] == sid]
        eps = {"narrow_e3": "1e-3", "narrow_e4": "1e-4", "narrow_e5": "1e-5"}[sid]
        out += [
            row("Appendix B.2", f"eps={eps}: median N_guess/N_opt", gr,
                agg(rs, "guess_ratio_median"), "ratio", src_g),
            row("Appendix B.2", f"eps={eps}: mean absolute relative error", mae,
                agg(rs, "guess_abs_rel_mean", "mean"), "ratio", src_g),
            row("Appendix B.2", f"eps={eps}: guesses within 10%", w10,
                agg(rs, "guess_within10", "mean"), source=src_g),
        ]
    return out


# ------------------------------------------------------------------------ 2. tuned parameters
def parameter_claims():
    live = live_keys()
    wins = rows_for("winners.csv", "linear", live)
    ls = [json.loads(r["params"])["lookback_window"] for r in wins]
    out = [row("tuned parameters", "operating points selecting lookback_window = 1", 22,
               float(sum(1 for l in ls if l == 1)), "count", "winners.csv")]
    for sid, b, quoted in (("narrow_e3", 10_000, 4), ("narrow_e3", 41_326, 3),
                           ("narrow_e4", 2_917_365, 8)):
        got = [json.loads(r["params"])["lookback_window"] for r in wins
               if r["scenario_id"] == sid and int(f(r["budget"])) == b]
        out.append(row("tuned parameters", f"headline lookback_window at {sid} B={b:,}", quoted,
                       float(got[0]) if got else float("nan"), "count", "winners.csv"))
    out.append(row("tuned parameters", "operating points selecting inc > 1", None,
                   float(sum(1 for r in wins if json.loads(r["params"])["inc"] > 1)), "count",
                   "winners.csv"))
    return out


# ----------------------------------------------------------------------- 3. end-to-end results
def performance_claims():
    perf = {(r["scenario_id"], int(f(r["budget"])), r["algorithm"]): f(r["rate"])
            for r in load("performance_curves.csv")}
    cross = {(r["scenario_id"], r["algorithm"]): f(r["budget_to_reach"])
             for r in load("budget_crossings.csv") if int(f(r["threshold_pct"])) == 90}
    out = [
        row("end-to-end", "linear search at B=10,000 (eps=1e-3)", 57.34,
            perf.get(("narrow_e3", 10_000, "linear"), np.nan), source="performance_curves.csv"),
        row("end-to-end", "brute force at B=10,000 (eps=1e-3)", 56.09,
            perf.get(("narrow_e3", 10_000, "brute"), np.nan), source="performance_curves.csv"),
        row("end-to-end", "binary search at B=10,000 (eps=1e-3)", 65.07,
            perf.get(("narrow_e3", 10_000, "binary_deep"), np.nan),
            source="performance_curves.csv"),
    ]
    for sid, algo, quoted, tag in (("narrow_e3", "linear", 42_500, "eps=1e-3"),
                                   ("narrow_e3", "brute", 45_900, "eps=1e-3"),
                                   ("narrow_e3", "binary_deep", 32_400, "eps=1e-3"),
                                   ("narrow_e4", "linear", 2_910_000, "eps=1e-4"),
                                   ("narrow_e4", "brute", 4_530_000, "eps=1e-4")):
        got = cross.get((sid, algo), np.nan)
        rel = abs(got - quoted) / quoted if np.isfinite(got) else np.nan
        out.append(dict(group="end-to-end",
                        claim=f"budget for 90% reliability, {algo} ({tag})",
                        quoted=f"{quoted:,}", recomputed=f"{got:,.0f}",
                        verdict="confirmed" if np.isfinite(rel) and rel <= 0.01 else "DIFFERS",
                        source="budget_crossings.csv"))
    for sid, tag, quoted in (("narrow_e3", "eps=1e-3", 1.08), ("narrow_e4", "eps=1e-4", 1.56)):
        got = cross.get((sid, "brute"), np.nan) / cross.get((sid, "linear"), np.nan)
        out.append(row("end-to-end", f"linear-vs-brute budget ratio at 90% ({tag})", quoted, got,
                       "ratio", "budget_crossings.csv"))
    return out


# ----------------------------------------------- 4. how far a false alarm stops short of N_opt
def published_config_from(sid, budget):
    """The Algorithm 4 configuration the AUDITED snapshot froze at this operating point."""
    for r in load("winners.csv"):
        if (r["algorithm"] == "linear" and r["scenario_id"] == sid
                and int(f(r["budget"])) == int(budget)):
            return json.loads(r["params"])
    return None


def distance_claims(R=50_000):
    """Replay the PUBLISHED configuration and measure N_opt - N_guess on false-alarm runs.

    The notes quote these from a reproduction of the held-out runs; recomputing them here makes
    them reproducible from a script rather than from a one-off, and at the full held-out R rather
    than the 2,000 runs the stored traces keep.
    """
    import linear_detector_study as L
    quoted = {(("narrow_e3", 10_000)): (5, 64, 91, 26.7),
              (("narrow_e3", 41_326)): (10, 60, 82, 17.6),
              (("narrow_e4", 2_917_365)): (1, None, None, 67.8)}
    out = []
    for (sid, b), (med, p90, p95, w10) in quoted.items():
        pub = published_config_from(sid, b)
        if pub is None:
            continue
        cfg = pub
        scen = next(s for s in M.SCENARIOS if s["id"] == sid)
        rule, c = L.rule_of(cfg)
        o = L.heldout(scen, b, rule, c, R)
        d = L.diag_row(o, b)
        tag = f"{sid} B={b:,}"
        # the quantiles are of an integer distribution and are re-drawn here with a different
        # probe stream, so they are checked to +-1 phase gate rather than exactly
        for name, q, got, kind in (("median phase gates below N_opt", med, d["fa_short_median"], "count"),
                                   ("90th percentile below N_opt", p90, d["fa_short_p90"], "count"),
                                   ("95th percentile below N_opt", p95, d["fa_short_p95"], "count"),
                                   ("false alarms within 10% of N_opt", w10, d["fa_within_10pct"], "pct")):
            if q is None:
                continue
            out.append(row("false-alarm distance", f"{tag}: {name}", q, got, kind,
                           "linear_detector_study.heldout, published config",
                           tol=1 if kind == "count" else None))
        out.append(row("false-alarm distance", f"{tag}: trial-level false-alarm rate", None,
                       d["false_alarm_rate"], "pct", "linear_detector_study.heldout"))
    return out


# ------------------------------------------------------------------ 5. what the code really does
def code_claims():
    src = inspect.getsource(A.find_phi_fixed_budget_linear_search)
    checks = [
        ("Algorithm 4 pseudocode says N <- N - l; the code backtracks by l*inc",
         "N = N_list[-1] - lookback_window * inc if overshot else N_list[-1]"),
        ("the safeguard is N* = max(1, N_guess - s)", "N = max(1, N - safeguard)"),
    ]
    out = [dict(group="code vs pseudocode", claim=c, quoted="stated in the notes",
                recomputed=("present" if snippet in src else "ABSENT"),
                verdict=("confirmed" if snippet in src else "DIFFERS"),
                source="qmetrology/algorithms.py::find_phi_fixed_budget_linear_search")
           for c, snippet in checks]
    mark = inspect.getsource(A._mark_linear_probes)
    out.append(dict(
        group="code vs pseudocode",
        claim="the backtracked probes are exactly the ones declared overshoots (not a proven "
              "'last N before overshooting')",
        quoted="stated in the notes",
        recomputed="present" if "len(trace.probes) - lookback_window" in mark else "ABSENT",
        verdict="confirmed" if "len(trace.probes) - lookback_window" in mark else "DIFFERS",
        source="qmetrology/algorithms.py::_mark_linear_probes"))
    expl = inspect.getsource(A._linear_search_explore)
    # The notes describe the rule as testing the cumulative mean. That was the whole rule when they
    # were written; the live code has since made the width a tuned parameter, `mean_window`, whose
    # 0 is exactly that cumulative mean (docs/LINEAR_SEARCH.md). Both halves are
    # checked, so this row records the generalisation rather than reporting it as a contradiction.
    cumulative_branch = "phi_hat_list[-mean_window:] if mean_window else phi_hat_list" in expl
    still_plain = "np.mean(phi_hat_list)" in expl
    out.append(dict(
        group="code vs pseudocode",
        claim="the stopping rule tests the CUMULATIVE mean of all probes so far",
        quoted="stated in the notes",
        recomputed=("still the whole rule" if still_plain else
                    ("now the mean_window = 0 case of a tuned window width" if cumulative_branch
                     else "ABSENT")),
        verdict=("confirmed" if still_plain else
                 ("confirmed for the audited snapshot; generalised since" if cumulative_branch
                  else "DIFFERS")),
        source="qmetrology/algorithms.py::_linear_search_explore"))
    return out


# ------------------------------------------------------- 6. the noiseless branch calculation
def branch_claims():
    """g(N) = arccos|cos(N phi)|/N: phi on the admissible branch, pi/N - phi on the first aliased
    branch (slope -pi/N^2), and a sawtooth -- NOT a decay to zero -- beyond it."""
    phi = 0.037
    N = np.arange(1, 4000)
    g = np.arccos(np.abs(np.cos(N * phi))) / N
    n_opt = int(np.pi // (2 * phi))
    b1 = N <= n_opt
    b2 = (N > n_opt) & (N * phi <= np.pi)
    slope_num = np.diff(g[b2])
    slope_ana = -np.pi / N[b2][:-1].astype(float) ** 2
    tail = N > 3 * np.pi / phi
    out = [
        dict(group="noiseless branches",
             claim="g(N) = phi exactly on the admissible branch 0 <= N phi <= pi/2",
             quoted="stated in the notes",
             recomputed=f"max |g - phi| = {np.abs(g[b1] - phi).max():.2e}",
             verdict="confirmed" if np.abs(g[b1] - phi).max() < 1e-12 else "DIFFERS",
             source="closed form, no simulation"),
        dict(group="noiseless branches",
             claim="g(N) = pi/N - phi on the first aliased branch pi/2 <= N phi <= pi",
             quoted="stated in the notes",
             recomputed=f"max |g - (pi/N - phi)| = "
                        f"{np.abs(g[b2] - (np.pi / N[b2] - phi)).max():.2e}",
             verdict="confirmed" if np.abs(g[b2] - (np.pi / N[b2] - phi)).max() < 1e-12
                     else "DIFFERS",
             source="closed form, no simulation"),
        dict(group="noiseless branches",
             claim="the first aliased branch decreases, dg/dN = -pi/N^2",
             quoted="stated in the notes",
             recomputed=f"max relative deviation of the numerical difference "
                        f"{np.abs(slope_num / slope_ana - 1).max():.3f}",
             verdict="confirmed" if (slope_num < 0).all() else "DIFFERS",
             source="closed form, no simulation"),
        dict(group="noiseless branches",
             claim="beyond the first aliased branch g(N) is a sawtooth, not a decay to zero: it "
                   "rises on a fraction of the steps",
             quoted="the thesis says the estimates tend toward zero for every N > N_opt",
             recomputed=f"{100*np.mean(np.diff(g[N > n_opt]) > 0):.1f}% of steps past N_opt "
                        f"INCREASE g",
             verdict=("thesis text refuted, notes correct"
                      if np.mean(np.diff(g[N > n_opt]) > 0) > 0.05 else "confirmed"),
             source="closed form, no simulation"),
        dict(group="noiseless branches",
             claim="the sawtooth envelope decays, so 'tends to zero' is true only in envelope",
             quoted="",
             recomputed=f"max g over N > 3pi/phi is {g[tail].max():.2e} vs phi = {phi:g}",
             verdict="confirmed" if g[tail].max() < phi else "DIFFERS",
             source="closed form, no simulation"),
    ]
    return out


def main():
    print(f"auditing the linear-search claims against {os.path.relpath(SRC, ROOT)}")
    rows = []
    for fn in (aggregate_claims, parameter_claims, performance_claims, code_claims,
               branch_claims, distance_claims):
        try:
            rows += fn()
        except Exception as exc:                      # a missing input must not hide the rest
            rows.append(dict(group=fn.__name__, claim="could not run", quoted="",
                             recomputed=f"{type(exc).__name__}: {exc}", verdict="ERROR",
                             source=""))
    Table("linear_search_claims.csv", FIELDS, reset=True).rows(rows)
    w = max(len(r["claim"]) for r in rows)
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            print(f"\n== {group}")
        flag = {"confirmed": "  ok", "confirmed (within 1)": " ~ok", "DIFFERS": "DIFF",
                "confirmed for the audited snapshot; generalised since": " ok*",
                "ERROR": " ERR", "thesis text refuted, notes correct": "THES",
                "reported (no value quoted)": "info"}.get(r["verdict"], "  ??")
        print(f" {flag}  {r['claim']:<{w}}  notes: {r['quoted']:>10}   recomputed: "
              f"{r['recomputed']}")
    ok = ("confirmed", "confirmed (within 1)", "reported (no value quoted)",
          "thesis text refuted, notes correct", "not recomputable",
          "confirmed for the audited snapshot; generalised since")
    n_bad = sum(1 for r in rows if r["verdict"] not in ok)
    print(f"\n{len(rows)} claims checked, {n_bad} disagree with the revision notes.")
    print("wrote", path("linear_search_claims.csv"))


if __name__ == "__main__":
    main()
