"""Thesis-shaped LaTeX tables, regenerated from the consolidated results.

Emits updated versions of the Chapter-4 tables in the SAME shape and style as the originals in
results/tex/ (which are stale -- they come from the pre-consolidation sweep), plus compact
diagnostics tables for the new algorithm telemetry.

Output: results/consolidated/tex/thesis/*.tex  and  all_thesis_tables.tex

    python analysis/consolidated/thesis_tables.py            # SHORT (default) -- thesis-sized
    python analysis/consolidated/thesis_tables.py --long     # per-scenario detail, for an appendix
    python analysis/consolidated/thesis_tables.py --out DIR  # write somewhere else

SHORT is what fits a thesis page: the diagnostics collapse to one row per algorithm, the
budget table shows the four selected prior/eps settings, and the broad-prior tables list only
budgets in the informative band. LONG expands each of those to every scenario / every budget and is
meant for an appendix. Both are generated from the same CSVs, so the numbers agree by construction.

Nothing here simulates. Every number is read from results/consolidated/*.csv, so these tables cannot
drift from the data or from FULL_RESULTS.md.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import manifest as M
from pipeline_io import path

LONG = "--long" in sys.argv
OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else path("tex", "thesis" + ("_long" if LONG else "")))

# Two label registers, split on TABLE WIDTH -- the only criterion that actually matters.
#
#   LABEL         full names. For tables with few columns, where the first column has room for
#                 "A. 3: Brute force baseline" without forcing a linebreak.
#   LABEL_NARROW  abbreviations. For wide tables -- the 6-8 column diagnostics tables, and any
#                 table that puts algorithms in COLUMN HEADERS, where a full name overflows.
#
# Edit either one and every table using that register follows. `--names full|short` overrides the
# per-table default globally if you want one register everywhere.
LABEL = {
    "brute": r"A.~\ref{alg:brute-force}: Brute force baseline",
    "linear": r"A.~\ref{alg:linear-search}: Linear search",
    "binary_deep": r"A.~\ref{alg:binary-search}: Binary search",
    "reverse_eng_risk": r"A.~\ref{alg:reverse-engineering}: Reverse engineering",
    "separable": r"Separable protocol ($N=1$)",
    "oracle_hl": r"Oracle ($N_\text{opt}$)",
}
LABEL_NARROW = {
    "brute": r"A.~\ref{alg:brute-force}: BF",
    "linear": r"A.~\ref{alg:linear-search}: LS",
    "binary_deep": r"A.~\ref{alg:binary-search}: BS",
    "reverse_eng_risk": r"A.~\ref{alg:reverse-engineering}: RE",
    "separable": r"Sep. ($N=1$)",
    "oracle_hl": r"Oracle ($N_\text{opt}$)",
}

_FORCE = (sys.argv[sys.argv.index("--names") + 1] if "--names" in sys.argv else "auto")


def lab(a, narrow=False):
    """Row label. `narrow=True` asks for the abbreviated register; --names overrides."""
    if _FORCE == "full":
        narrow = False
    elif _FORCE == "short":
        narrow = True
    return (LABEL_NARROW if narrow else LABEL)[a]


def head_label(a):
    """Column header: the abbreviated name with the `A.~\ref{alg:...}:` prefix stripped.

    Column headers always use the narrow register and never carry a cross-reference -- a \ref in a
    header is both ugly and redundant with the row labels elsewhere in the document.
    """
    if a == "oracle_hl":
        return r"Oracle ($N_\text{opt}$)"
    t = LABEL[a]
    return t.split(": ", 1)[1] if ": " in t else t

def head_label_narrow(a):
    if a == "oracle_hl":
        return r"Oracle ($N_\text{opt}$)"
    t = LABEL_NARROW[a]
    return t.split(": ", 1)[1] if ": " in t else t


# Backwards-compatible aliases.
ALG_REF, SHORT = LABEL, LABEL_NARROW

ROWS = ["brute", "linear", "binary_deep", "reverse_eng_risk", "separable", "oracle_hl"]


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def load(name):
    p = path(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def tex_int(x, decimal_places=None):
    """Format numbers for LaTeX with thousands separators."""
    if not np.isfinite(x):
        return "--"

    if x >= 1e9:
        e = int(np.floor(np.log10(x)))
        return rf"${x/10**e:.2f}\times10^{{{e}}}$"

    if decimal_places is not None:
        s = f"{x:,.{decimal_places}f}"
        s = s.rstrip("0").rstrip(".")
        return s.replace(",", "{,}")

    return f"{int(round(x)):,}".replace(",", "{,}")


def eps_tex(e):
    return rf"$10^{{{int(round(np.log10(f(e))))}}}$"


def prior_tex(s):
    hi = {"1.5708": r"\pi/2", "0.7854": r"\pi/4"}.get(f"{f(s['phi_max']):.4f}",
                                                      f"{f(s['phi_max']):g}")
    return rf"$\phi \sim \mathcal{{U}}({f(s['phi_min']):g},{hi})$"


FOOT = (r"\footnotesize Adaptive entries are de-biased: parameters are grid-tuned on seeds "
        r"42 and 43 and the frozen winner is re-validated on the independent seed 2024 at "
        r"$R=%s$ trials. The oracle is exact and can be calculated analytically for each $\phi$."
        r" Thus it carries no confidence interval.")


def wrap(body, caption, label, size=None, foot=None):
    L = [r"\begin{table}[ht]", r"\centering"]
    if size:
        L.append("\\" + size)
    body = [b for b in body if b != ""] if body and body[-1] == "" else body
    L += [rf"\caption{{{caption}}}", rf"\label{{{label}}}"] + body + [r"\bottomrule",
                                                                     r"\end{tabular}"]
    if foot:
        L += [r"\\[2pt]", foot]
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)
    return name


class Data:
    def __init__(self):
        self.perf = load("performance_curves.csv")
        self.cross = load("budget_crossings.csv")
        self.diag = load("diagnostics_by_point.csv")
        self.det = load("detector_confusion.csv")
        self.wins = load("winners.csv")
        self.ops = load("operating_points.csv")
        self.man = json.load(open(path("experiment_manifest.json")))
        self.R = self.man["trials"]["R_test"]
        self.regime = {(r["scenario_id"], int(f(r["budget"]))): r["regime"] for r in self.ops}
        self.scen = {s["id"]: s for s in self.man["scenarios"]}

    def rate(self, sid, budget, algo):
        for r in self.perf:
            if (r["scenario_id"] == sid and int(f(r["budget"])) == int(budget)
                    and r["algorithm"] == algo):
                return f(r["rate"]), f(r.get("rate_lo")), f(r.get("rate_hi"))
        return float("nan"), float("nan"), float("nan")

    def crossing(self, sid, algo, thr=90):
        for c in self.cross:
            if (c["scenario_id"] == sid and c["algorithm"] == algo
                    and int(c["threshold_pct"]) == thr):
                return (f(c["budget_to_reach"]), f(c.get("budget_to_reach_lo")),
                        f(c.get("budget_to_reach_hi")), f(c.get("ratio_vs_brute")),
                        f(c.get("ratio_lo")), f(c.get("ratio_hi")))
        return (float("nan"),) * 6

    def budgets(self, sid):
        return sorted({int(f(r["budget"])) for r in self.perf if r["scenario_id"] == sid})

    def winner(self, sid, budget, algo):
        for x in self.wins:
            if (x["scenario_id"] == sid and int(f(x["budget"])) == int(budget)
                    and x["algorithm"] == algo):
                return json.loads(x["params"]), x
        return {}, {}

    def live(self, rows):
        return [r for r in rows
                if self.regime.get((r["scenario_id"], int(f(r["budget"])))) == "live"]


# ============================================================ Table 1: low-precision summary
def tab_summary_low_prec(D, ci=False):
    """Raw values only -- no inline improvement factors.

    The x-factors used to live in these cells, which made every entry carry two numbers and forced
    the reader to hold a baseline in their head. They now have their own table
    (`tab_ratios`), where brute force and the oracle can both be shown as reference columns.
    """
    sid, B = "narrow_e3", 10_000
    body = [r"\begin{tabular}{lcc}", r"\toprule", "Algorithm",
            r"& \makecell{Avg. \% converged \\ (budget = 10{,}000)}",
            r"& \makecell{Budget for $>90\%$ \\ convergence} \\",
            r"\midrule"]
    for a in ROWS:
        r, lo, hi = D.rate(sid, B, a)
        x, xlo, xhi, _rat, _rl, _rh = D.crossing(sid, a)
        if not np.isfinite(r):
            continue
        if a in ("separable", "oracle_hl"):
            body.append(r"\midrule")
        c1 = f"{100*r:.2f}\\%"
        if ci and np.isfinite(lo):
            c1 += f" \\newline {{\\scriptsize [{100*lo:.2f}, {100*hi:.2f}]}}"
        c2 = tex_int(x) if np.isfinite(x) else "--"
        if ci and np.isfinite(xlo):
            c2 += f" \\newline {{\\scriptsize [{tex_int(xlo)}, {tex_int(xhi)}]}}"
        body += [lab(a), f"& {c1}", f"& {c2} \\\\", ""]
    cap = (r"Algorithm performance under low-precision constraints $\epsilon=10^{-3}$ for "
           r"$\phi\sim\mathcal{U}(0.01,0.1)$: the convergence rate at fixed budget $C=10{,}000$ "
           r"and the budget required to achieve \(>90\%\) convergence. Improvement factors "
           r"relative to the baseline are collected in Table~\ref{tab:ratios}. The "
           r"Oracle ($N_\text{opt}$) row is not an implementable protocol: it is given the true "
           r"$N_\text{opt}$ and is evaluated analytically using the asymptotic QCRB law.")
    if ci:
        cap += (r" Brackets give 95\% intervals (Wilson for rates, bootstrap for budgets); the "
                r"oracle is exact and carries none.")
    return wrap(body, cap, "tab:summary-low-prec" + ("-ci" if ci else ""),
                foot=FOOT % f"{D.R:,}".replace(",", "{,}"))


def tab_ratios(D):
    """Improvement over the baseline, and distance from the oracle -- both as budget ratios.

    A bare ratio vs brute force cannot be read on its own: the same factor can be close to or far
    from the oracle reference. Each cell therefore carries two numbers, and both are
    ratios of the SAME quantity (the budget to reach 90%), so no new unit is introduced:

        top     B_brute / B_alg       the factor less budget than the baseline (>1 is better)
        bottom  B_alg / B_oracle     budget relative to the oracle reference
                                      (1.00 = equal oracle budget)

    They are self-consistent by construction: (B_brute/B_alg) x (B_alg/B_oracle) = B_brute/B_oracle,
    the oracle's own factor in the last row. A reader can multiply across and check.

    REJECTED ALTERNATIVE: rate_alg / rate_oracle ("share of the oracle"). It is the simplest thing
    to say, but it compresses everything into 92-99.5% and gives brute force 92%, which makes the
    baseline look near-optimal and the adaptive gains look like rounding error. It is also a rate
    share inside a budget table.

    Restricted to the four settings of Table~\ref{tab:summary-all}. The precision sweep has its own
    ratio table (Table~\ref{tab:scaling-with-prec}) and the broad priors theirs
    (Table~\ref{tab:broad-pi2}).
    """
    cols = [c for c, _ in ALL_COLS if c in D.scen]
    algs = ["linear", "binary_deep", "reverse_eng_risk"]
    body = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
            r"\makecell[l]{Algorithm \\ {\scriptsize $B_{\mathrm{brute}}/B_{\mathrm{alg}}$} "
            r"\\ {\scriptsize ($B_{\mathrm{alg}}/B_{\mathrm{oracle}}$)}}"]
    for sid in cols:
        sc = D.scen[sid]
        e = int(round(np.log10(f(sc["eps"]))))
        body.append(rf"& \makecell{{{prior_tex(sc)} \\ $\epsilon = 10^{{{e}}}$}}")
    body[-1] += r" \\"
    body.append(r"\midrule")
    for a in algs:
        cells = []
        for sid in cols:
            Bb = D.crossing(sid, "brute")[0]
            Bc = D.crossing(sid, "oracle_hl")[0]
            Ba = D.crossing(sid, a)[0]
            if not all(np.isfinite(x) for x in (Bb, Bc, Ba)) or Bc <= 0:
                cells.append("& --")
                continue
            cells.append(rf"& \makecell{{{Bb/Ba:.2f}\texttimes"
                         rf"{{\scriptsize ({Ba/Bc:.2f}\texttimes\ oracle)}}}}")
        cells[-1] += r" \\"
        body += [lab(a, narrow=not LONG)] + cells
    body.append(r"\midrule")
    cells = []
    for sid in cols:
        Bb, Bc = D.crossing(sid, "brute")[0], D.crossing(sid, "oracle_hl")[0]
        cells.append(rf"& \makecell{{{Bb/Bc:.2f}\texttimes"
                     rf"{{\scriptsize (1.00\texttimes\ oracle)}}}}"
                     if np.isfinite(Bb) and np.isfinite(Bc) else "& --")
    cells[-1] += r" \\"
    body += [r"Oracle ($N_\text{opt}$)"] + cells
    cap = (r"Budget to reach $90\%$ convergence, expressed two ways. \textbf{First number:} "
           r"$B_{\mathrm{brute}}/B_{\mathrm{alg}}$, the factor less budget than the baseline "
           r"($>1$ is better). \textbf{Second number:} "
           r"$B_{\mathrm{alg}}/B_{\mathrm{oracle}}$, the algorithm's budget relative to the "
           r"oracle reference ($1.00\texttimes$ means equal budget). The second number is what "
           r"makes the first interpretable: it shows directly how far the algorithm remains from "
           r"the oracle reference. The two "
           r"multiply to the oracle's own factor in the last row. The separable protocol is "
           r"omitted -- it is slower than the baseline; its raw budget is in "
           r"Table~\ref{tab:summary-all-ci}.")
    return wrap(body, cap, "tab:ratios", size="small",
                foot=FOOT % f"{D.R:,}".replace(",", "{,}"))


# ============================================================ Table 2: budgets across settings
ALL_COLS = [("narrow_e3", None), ("narrow_e4", None), ("small_e4", None), ("wide_e4", None)]


def tab_summary_all(D, ci=False):
    cols = ([s["id"] for s in M.SCENARIOS if s["id"] in D.scen] if LONG
            else [c for c, _ in ALL_COLS if c in D.scen])
    head = ["Algorithm"]
    for sid in cols:
        s = D.scen[sid]
        head.append(rf"& \makecell{{Budget in thousands\\ ($>90\%$ conv.) \\ "
                    rf"$\epsilon={{10^{{{int(round(np.log10(f(s['eps']))))}}}}}$ \\ {prior_tex(s)}}}")
    head[-1] += r" \\"
    body = [r"\begin{tabular}{l" + " c" * len(cols) + "}", r"\toprule"] + head + [r"\midrule"]
    base = {sid: D.crossing(sid, "brute")[0] for sid in cols}
    for a in ROWS:
        cells = []
        for sid in cols:
            x, xlo, xhi, _r, _rl, _rh = D.crossing(sid, a)
            x, xlo, xhi = x/1000, xlo/1000, xhi/1000
            if not np.isfinite(x):
                cells.append("& --")
                continue
            # c = f"& {tex_int(x)} (\\texttimes {base[sid]/x:.2f})"
            c = f"& {tex_int(x, decimal_places=1)}"
            if ci and np.isfinite(xlo):
                c += f" \\newline {{\\scriptsize [{tex_int(xlo, decimal_places=1)}, {tex_int(xhi, decimal_places=1)}]}}"
            cells.append(c)
        if not any(c != "& --" for c in cells):
            continue
        if a in ("separable", "oracle_hl"):
            body.append(r"\midrule")
        cells[-1] += r" \\"
        body += [lab(a, narrow=True)] + cells + [""]
    cap = (r"Budget required to achieve $>90\%$ convergence for selected precision settings and "
           r"prior intervals. Improvement factors are relative to the brute-force baseline; values "
           r"greater than one indicate a lower required budget.")
    if ci:
        cap += r" Brackets give 95\% bootstrap intervals."
    return wrap(body, cap, "tab:summary-all" + ("-ci" if ci else ""), size="scriptsize",
                foot=FOOT % f"{D.R:,}".replace(",", "{,}"))


# ======================================================= Table 3: scaling with precision
def tab_scaling_with_prec(D, ci=False):
    sweep = [s["id"] for s in M.SCENARIOS if "precision_sweep" in s["families"]]
    sweep = [s for s in sweep if s in D.scen]
    cols = ["linear", "binary_deep", "reverse_eng_risk", "oracle_hl"]
    body = ([r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r"Precision $\epsilon$"]
            + [f"& {head_label(a)}" for a in cols[:-1]]
            + [f"& {head_label(cols[-1])} \\\\", r"\midrule"])
    for sid in sorted(sweep, key=lambda k: -f(D.scen[k]["eps"])):
        cells = []
        for a in cols:
            _x, _lo, _hi, rat, rl, rh = D.crossing(sid, a)
            # Bb = D.crossing(sid, "brute")[0]
            # Bc = D.crossing(sid, "oracle_hl")[0]
            # Ba = D.crossing(sid, a)[0]
            if not np.isfinite(rat):
                cells.append("& --")
                continue
            c = f"& {rat:.2f}\\texttimes"
            # cells.append(rf"& \makecell{{{Bb/Ba:.2f}\texttimes"
                                    #  rf"{{\scriptsize ({Ba/Bc:.2f}\texttimes\ opt.)}}}}")
            if ci and np.isfinite(rl):
                c += f" \\newline {{\\scriptsize [{rl:.2f}, {rh:.2f}]}}"
            cells.append(c)
        cells[-1] += r" \\"
        body += [eps_tex(D.scen[sid]["eps"])] + cells
    cap = (r"Improvement factor over the brute-force baseline as the precision requirement tightens, "
           r"at $\phi\sim\mathcal{U}(0.01,0.1)$ and $90\%$ convergence. The advantage grows with "
           r"precision and then saturates; the oracle column bounds how much of it is attainable "
           r"at all.")
    if ci:
        cap += r" Brackets give 95\% bootstrap intervals; these are conservative (the two curves are resampled independently despite being seed-paired)."
    return wrap(body, cap, "tab:scaling-with-prec" + ("-ci" if ci else ""), size="small",
                foot=FOOT % f"{D.R:,}".replace(",", "{,}"))


# ================================================================= Table 4: broad priors
def tab_broad(D, sid, tag):
    cols = ["brute", "linear", "binary_deep", "reverse_eng_risk", "oracle_hl"]
    body = ([r"\begin{tabular}{r" + "c" * len(cols) + "}", r"\toprule",
             r"Budget $C=N\cdot m$"]
            + [f"& {head_label(a)}" for a in cols[:-1]]
            + [f"& {head_label(cols[-1])} \\\\", r"\midrule"])
    buds = D.budgets(sid) if LONG else [b for b in D.budgets(sid)
                                        if D.regime.get((sid, b)) == "live"]
    for b in buds:
        vals = {a: D.rate(sid, b, a)[0] for a in cols}
        impl = {a: v for a, v in vals.items() if a != "oracle_hl" and np.isfinite(v)}
        best = max(impl.values()) if impl else float("nan")
        cells = []
        for a in cols:
            v = vals[a]
            if not np.isfinite(v):
                cells.append("& --")
            elif a != "oracle_hl" and np.isfinite(best) and abs(v - best) < 1e-12:
                cells.append(f"& \\textbf{{{100*v:.1f}\\%}}")
            else:
                cells.append(f"& {100*v:.1f}\\%")
        cells[-1] += r" \\"
        body += [tex_int(b)] + cells
    cap = (rf"Convergence under a broad uniform prior $\phi\sim\mathcal{{U}}(0.01,{tag})$ at "
           r"$\epsilon=10^{-3}$: average share of simulations that converge, by budget. The best "
           r"implementable entry in each row is set in bold; the oracle is an analytic reference, "
           r"not a competitor. Only budgets in the informative band are listed (see "
           r"\texttt{operating\_points.csv}).")
    lab = "tab:broad-" + ("pi2" if "pi/2" in tag else "pi4")
    return wrap(body, cap, lab, size="small", foot=FOOT % f"{D.R:,}".replace(",", "{,}"))


# =============================================== Appendix: threshold robustness
def tab_robustness_across_thresholds(D):
    """Best adaptive budget ratio at each reliability threshold.

    The winning algorithm is selected separately in every cell. This avoids the old table's
    ambiguity: its final "Winner" column named only the winner at 90%, even though the four ratios
    could come from different algorithms.
    """
    thresholds = [int(t) for t in D.man["thresholds_pct"]]
    scenarios = D.man["scenarios"]
    adaptive = ["linear", "binary_deep", "reverse_eng_risk"]
    short = {"linear": "LS", "binary_deep": "BS", "reverse_eng_risk": "RE"}

    body = ([r"\begin{tabular}{l" + "c" * len(thresholds) + "}", r"\toprule", "Scenario"]
            + [rf"& \makecell{{${t}\%$ convergence}}" for t in thresholds])
    body[-1] += r" \\"
    body.append(r"\midrule")

    for s in scenarios:
        cells = []
        for threshold in thresholds:
            candidates = []
            for algo in adaptive:
                _b, _blo, _bhi, ratio, lo, hi = D.crossing(s["id"], algo, threshold)
                if np.isfinite(ratio):
                    candidates.append((ratio, algo, lo, hi))
            if not candidates:
                cells.append("& --")
                continue
            ratio, algo, lo, hi = max(candidates, key=lambda x: x[0])
            interval = (rf" \\ {{\scriptsize [{lo:.2f}, {hi:.2f}]}}"
                        if np.isfinite(lo) and np.isfinite(hi) else "")
            cells.append(rf"& \makecell{{{ratio:.2f}\texttimes{interval}"
                         rf" \\ {{\scriptsize {short[algo]}}}}}")
        cells[-1] += r" \\"
        scenario = rf"\makecell[l]{{{prior_tex(s)} \\ $\epsilon={eps_tex(s['eps'])[1:-1]}$}}"
        body += [scenario] + cells + [""]

    cap = (r"Robustness of the adaptive advantage across target convergence probabilities. Each "
           r"cell reports $\max_a B_{\mathrm{BF}}(p^\ast)/B_a(p^\ast)$ over the three adaptive "
           r"algorithms, followed by its pointwise 95\% bootstrap interval and the algorithm "
           r"attaining the maximum. Values greater than one mean that the selected adaptive "
           r"algorithm requires less budget than brute force. The maximizing algorithm is selected "
           r"separately at each threshold.")
    tune = ", ".join(str(x) for x in D.man["seeds"]["tune_blocks"])
    test = D.man["seeds"]["test"]
    R_tex = f"{D.R:,}".replace(",", "{,}")
    foot = (rf"\footnotesize LS = linear search, BS = binary search, RE = reverse engineering. "
            rf"Parameters were tuned on seeds {tune}; frozen configurations were evaluated on "
            rf"held-out seed {test} with $R={R_tex}$ trials per point. Budget crossings use "
            r"log-linear interpolation between tested budgets.")
    return wrap(body, cap, "tab:robustness-across-thresholds", size="scriptsize", foot=foot)


# ======================================================= Appendix: selected parameters
def _parameter_items(params):
    p = {k: v for k, v in params.items() if k not in ("eps_target", "budget")}
    if not p:
        return []
    order = ["m_exploration", "lookback_window", "safeguard", "inc", "conf"]
    items = sorted(p.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    return [k.replace("_", "\\_") + "=" + str(v) for k, v in items]


def _params_cell(params, budget=None, shot_fraction=None):
    items = _parameter_items(params)
    if not items:
        return "--"
    lines = [r"\texttt{" + x + "}" for x in items]
    if budget is not None:
        B = f"{int(budget):,}".replace(",", "{,}")
        lines.append(r"{\scriptsize $B=" + B + "$}")
    if shot_fraction is not None:
        lines.append(r"{\scriptsize $m'/B="
                     + f"{100*shot_fraction:.2f}" + r"\%$}")
    return r"\makecell{" + r", \\ ".join(lines) + "}"


def _params_inline(params, shot_fraction=None):
    items = _parameter_items(params)
    out = r"\texttt{" + ", ".join(items) + "}" if items else "--"
    if shot_fraction is not None:
        out += (r", $m'/B="
                + f"{100*shot_fraction:.2f}" + r"\%$")
    return out


def _re_shot_fraction(budget, params):
    """Planned RE pilot shots as a fraction of the total resource budget."""
    return float(params["m_exploration"] / budget)


def tab_opt_param_first(D):
    sid, B = "narrow_e3", 10_000
    body = [r"\begin{tabular}{ll}", r"\toprule", "Algorithm",
            r"& Selected parameters \\", r"\midrule"]
    for a in ["linear", "binary_deep", "reverse_eng_risk"]:
        p, _ = D.winner(sid, B, a)
        shot_fraction = _re_shot_fraction(B, p) if a == "reverse_eng_risk" else None
        body += [head_label(a), f"& {_params_inline(p, shot_fraction)} \\\\"]
    cap = (r"Parameter configurations selected for "
           r"Table~\ref{tab:summary-low-prec-ci} at fixed budget $10{,}000$ "
           r"($\epsilon=10^{-3}$, $\phi\sim\mathcal{U}(0.01,0.1)$). For Reverse Engineering, "
           r"the planned exploration-shot count is additionally reported as the fraction $m'/B$. "
           r"Exploration resource shares are reported in Table~\ref{tab:diag-downstream}.")
    return wrap(body, cap, "tab:opt-param-first-tab")


def tab_opt_param_second(D):
    """Selected parameters near B90, with the interpolation caveat made explicit."""
    cols = [c for c, _ in ALL_COLS if c in D.scen]
    body = [r"\setlength{\tabcolsep}{2pt}",
            r"\begin{tabular}{l" + "l" * len(cols) + "}", r"\toprule", "Algorithm"]
    for sid in cols:
        eps = eps_tex(D.scen[sid]["eps"])[1:-1]
        body.append(rf"& \makecell{{{prior_tex(D.scen[sid])}, \\ $\epsilon={eps}$}}")
    body[-1] += r" \\"
    body.append(r"\midrule")
    rows = ["linear", "binary_deep", "reverse_eng_risk"]
    for a in rows:
        cells = []
        for sid in cols:
            b90 = D.crossing(sid, a)[0]
            buds = D.budgets(sid)
            if not np.isfinite(b90) or not buds:
                cells.append("& --")
                continue
            bb = np.array(buds, float)
            near = int(bb[int(np.argmin(np.abs(np.log(bb) - np.log(b90))))])
            p, _ = D.winner(sid, near, a)
            shot_fraction = _re_shot_fraction(near, p) if a == "reverse_eng_risk" else None
            cells.append(f"& {_params_cell(p, near, shot_fraction)}")
        cells[-1] += r" \\"
        body += [head_label(a) if LONG else head_label_narrow(a)] + cells
        if a != rows[-1]:
            body.append(r"\midrule")

    cap = (r"Parameter configurations selected near the $90\%$-convergence budget. "
           r"A budget crossing is "
           r"interpolated between tested budgets; parameter dictionaries are not. Each entry is "
           r"therefore the selected configuration at the nearest tested budget, "
           r"not a configuration evaluated at the interpolated $B_{90}$. For Reverse Engineering, "
           r"the planned exploration-shot count is additionally reported as $m'/B$ at that tested "
           r"budget. Exploration resource shares are reported in Table~\ref{tab:diag-downstream}.")
    return wrap(body, cap, "tab:opt-param-second-tab", size="scriptsize")


# ================================================= NEW: compact diagnostics tables
def _agg(D, algo, key, how="median", sid=None, allowed_sids=None):
    rs = [r for r in D.live(D.diag) if r["algorithm"] == algo
          and (sid is None or r["scenario_id"] == sid)
          and (allowed_sids is None or r["scenario_id"] in allowed_sids)]
    v = np.array([f(r.get(key)) for r in rs], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    return float(np.median(v)) if how == "median" else float(v.mean())


def tab_diag_search(D):
    """Main-text summary of exploration quality, cost, and detector errors."""
    algs = ["linear", "binary_deep", "reverse_eng_risk"]
    body = [r"\begin{tabular}{lcccccc}", r"\toprule", "Algorithm",
            r"& \makecell{Median \\ $N_{\mathrm{guess}}/N_{\mathrm{opt}}$}",
            r"& \makecell{Within \\ $10\%$}",
            r"& \makecell{Guess \\ overshoot}",
            r"& \makecell{Probes \\ per trial}",
            r"& \makecell{Median expl. \\ budget share}",
            r"& \makecell{False alarm \\ / miss} \\", r"\midrule"]
    for a in algs:
        rs = [r for r in D.live(D.det) if r["algorithm"] == a]
        fa = np.mean([f(r["false_alarm_rate"]) for r in rs]) if rs else float("nan")
        ms = np.mean([f(r["miss_rate"]) for r in rs]) if rs else float("nan")
        detector = f"{100*fa:.1f}\\% / {100*ms:.1f}\\%" if np.isfinite(fa) else "--"
        body += [lab(a, narrow=True),
                 f"& {_agg(D,a,'guess_ratio_median'):.2f}",
                 f"& {100*_agg(D,a,'guess_within10','mean'):.1f}\\%",
                 f"& {100*_agg(D,a,'guess_overshoot','mean'):.1f}\\%",
                 f"& {_agg(D,a,'n_probes_mean','mean'):.1f}",
                 f"& {100*_agg(D,a,'exploration_share_median'):.2f}\\%",
                 f"& {detector} \\\\"]
    n_live = sum(1 for r in D.ops if r["regime"] == "live")
    cap = (r"Exploration quality, cost, and overshoot-detector errors, aggregated over the "
           rf"{n_live} informative operating points. $N_{{\mathrm{{guess}}}}$ is recorded before "
           r"the safeguard and $N_{\mathrm{opt}}=\left\lfloor\pi/(2\phi)\right\rfloor$ is the "
           r"largest non-aliasing phase-gate count. False-alarm and miss rates are trial-level: "
           r"they indicate whether at least one such event occurred during a run. Reverse "
           r"engineering has no overshoot detector, so these rates are not applicable.")
    return wrap(body, cap, "tab:diag-search", size="small")


def tab_diag_exploration(D):
    algs = ["linear", "binary_deep", "reverse_eng_risk"]
    broad = {s["id"] for s in M.SCENARIOS if "broad_prior" in s.get("families", ())}
    allowed = {sid for sid in D.scen if sid not in broad}
    if LONG:
        return tab_diag_exploration_long(D, algs, allowed)
    body = [r"\begin{tabular}{lcccccc}", r"\toprule", "Algorithm",
            r"& \makecell{Median \\ $N_{\mathrm{guess}}/N_{\mathrm{opt}}$}",
            r"& \makecell{Mean abs.\ \\ rel.\ error}",
            r"& \makecell{Exact \\ hit}", r"& \makecell{Within \\ $10\%$}",
            r"& \makecell{Guess \\ overshoot}",
            r"& \makecell{Probes \\ per trial} \\", r"\midrule"]
    scen_ids = [x["id"] for x in M.SCENARIOS if x["id"] in allowed] if LONG else [None]
    for sid in scen_ids:
        if LONG and sid != scen_ids[0]:
            body.append(r"\midrule")
        if LONG:
            body.append(r"\multicolumn{7}{l}{\itshape " +
                        D.scen[sid]["label"].replace("_", r"\_") + r"} \\")
        for a in algs:
            _s = sid
            body += [lab(a, narrow=True),
                     f"& {_agg(D,a,'guess_ratio_median',sid=_s,allowed_sids=allowed):.2f}",
                     f"& {_agg(D,a,'guess_abs_rel_mean','mean',sid=_s,allowed_sids=allowed):.2f}",
                     f"& {100*_agg(D,a,'guess_exact','mean',sid=_s,allowed_sids=allowed):.1f}\\%",
                     f"& {100*_agg(D,a,'guess_within10','mean',sid=_s,allowed_sids=allowed):.1f}\\%",
                     f"& {100*_agg(D,a,'guess_overshoot','mean',sid=_s,allowed_sids=allowed):.1f}\\%",
                     f"& {_agg(D,a,'n_probes_mean','mean',sid=_s,allowed_sids=allowed):.1f} \\\\"]
    n_live = sum(1 for r in D.ops
                 if r["regime"] == "live" and r["scenario_id"] in allowed)
    cap = (r"Quality of the value of $N$ returned by exploration, before any safeguard is applied. "
           r"$N_{\mathrm{guess}}$ is each algorithm's own answer to \emph{which value of $N$ "
           r"should be used for exploitation}; "
           r"$N_{\mathrm{opt}}=\left\lfloor\pi/(2\phi)\right\rfloor$ is the "
           r"largest non-aliasing value. Brute force is absent because it does not search: its "
           r"$N_{\mathrm{guess}}$ is undefined, not zero. "
           rf"Broad-prior scenarios are excluded; the values aggregate the remaining {n_live} "
           r"informative operating points.")
    return wrap(body, cap, "tab:diag-exploration", size="small")


def tab_diag_exploration_long(D, algs, allowed):
    """Split scenario-level diagnostics into page-sized appendix tables."""
    def make_body(scen_ids):
        body = [r"\begin{tabular}{lcccccc}", r"\toprule", "Algorithm",
                r"& \makecell{Median \\ $N_{\mathrm{guess}}/N_{\mathrm{opt}}$}",
                r"& \makecell{Mean abs.\ \\ rel.\ error}",
                r"& \makecell{Exact \\ hit}", r"& \makecell{Within \\ $10\%$}",
                r"& \makecell{Guess \\ overshoot}",
                r"& \makecell{Probes \\ per trial} \\", r"\midrule"]
        for i, sid in enumerate(scen_ids):
            if i:
                body.append(r"\midrule")
            body.append(r"\multicolumn{7}{l}{\itshape " +
                        D.scen[sid]["label"].replace("_", r"\_") + r"} \\")
            for a in algs:
                body += [lab(a, narrow=True),
                         f"& {_agg(D,a,'guess_ratio_median',sid=sid,allowed_sids=allowed):.2f}",
                         f"& {_agg(D,a,'guess_abs_rel_mean','mean',sid=sid,allowed_sids=allowed):.2f}",
                         f"& {100*_agg(D,a,'guess_exact','mean',sid=sid,allowed_sids=allowed):.1f}\\%",
                         f"& {100*_agg(D,a,'guess_within10','mean',sid=sid,allowed_sids=allowed):.1f}\\%",
                         f"& {100*_agg(D,a,'guess_overshoot','mean',sid=sid,allowed_sids=allowed):.1f}\\%",
                         f"& {_agg(D,a,'n_probes_mean','mean',sid=sid,allowed_sids=allowed):.1f} \\\\"]
        return body

    scen_ids = [x["id"] for x in M.SCENARIOS if x["id"] in allowed]
    sweep_ids = [x["id"] for x in M.SCENARIOS
                 if x["id"] in allowed and "precision_sweep" in x.get("families", ())]
    variant_ids = [sid for sid in scen_ids if sid not in sweep_ids]
    n_sweep = sum(1 for r in D.ops
                  if r["regime"] == "live" and r["scenario_id"] in sweep_ids)
    sweep_cap = (r"Quality of the value of $N$ returned by exploration, before any safeguard is "
                 r"applied, for the precision sweep. $N_{\mathrm{guess}}$ is each algorithm's "
                 r"proposed exploitation value and "
                 r"$N_{\mathrm{opt}}=\left\lfloor\pi/(2\phi)\right\rfloor$ is the largest "
                 r"non-aliasing value. Brute force is absent because it does not search. Each "
                 rf"scenario group aggregates its informative tested budgets ({n_sweep} operating "
                 r"points in total).")
    text = wrap(make_body(sweep_ids), sweep_cap, "tab:diag-exploration", size="small")
    if variant_ids:
        n_variants = sum(1 for r in D.ops
                         if r["regime"] == "live" and r["scenario_id"] in variant_ids)
        variant_cap = (r"Exploration-quality diagnostics for the prior-range variants at "
                       r"$\epsilon=10^{-4}$. Definitions are as in "
                       r"Table~\ref{tab:diag-exploration}; the scenario groups comprise "
                       rf"{n_variants} informative operating points.")
        text += "\n" + wrap(make_body(variant_ids), variant_cap,
                            "tab:diag-exploration-priors", size="small")
    return text


def tab_diag_downstream(D):
    algs = ["brute", "linear", "binary_deep", "reverse_eng_risk"]
    body = [r"\begin{tabular}{lccccc}", r"\toprule", "Algorithm",
            r"& \makecell{Median \\ $N_{\mathrm{guess}}/N_{\mathrm{opt}}$}",
            r"& \makecell{Median \\ $N^*/N_{\mathrm{opt}}$}",
            r"& \makecell{Final \\ overshoot}",
            r"& \makecell{Unsafe-guess \\ rescue}",
            r"& \makecell{Converged \\ given safe $N^*$} \\", r"\midrule"]
    for a in algs:
        # Brute force does not search, but its fixed pre-exploitation choice is N_min.
        g = _agg(D, a, "star_ratio_median" if a == "brute" else "guess_ratio_median")
        rc = _agg(D, a, "rescue_share", "mean")
        body += [lab(a, narrow=True),
                 f"& {g:.2f}" if np.isfinite(g) else "& --",
                 f"& {_agg(D,a,'star_ratio_median'):.2f}",
                 f"& {100*_agg(D,a,'star_overshoot','mean'):.2f}\\%",
                 f"& {100*rc:.1f}\\%" if np.isfinite(rc) else "& --",
                 f"& {100*_agg(D,a,'conv_given_safe_depth','mean'):.1f}\\% \\\\"]
    n_live = sum(1 for r in D.ops if r["regime"] == "live")
    cap = (r"From the exploration's guess to the value of $N$ actually used, aggregated over the "
           rf"{n_live} informative operating points. $N^*$ is the "
           r"exploitation value after the safeguard; \emph{unsafe-guess rescue} is "
           r"$P(N^* \le N_{\mathrm{opt}} \mid N_{\mathrm{guess}} > N_{\mathrm{opt}})$, i.e.\ how "
           r"often the safeguard pulls an aliasing guess back to safety. For the statistical "
           r"safeguard, a selected $N^*/N_{\mathrm{opt}}$ below one may be intentional: it maximizes "
           r"$P(\text{no overshoot})\times P(\text{converge})$ and deliberately backs off from the "
           r"aliasing cliff. \emph{Converged given safe $N^*$} is "
           r"$P(|\hat\phi-\phi|<\epsilon\mid N^*\leq N_{\mathrm{opt}})$. "
           r"Brute force performs no search, so its fixed "
           r"$N_{\mathrm{guess}}=N_{\min}$ is reported.")
    return wrap(body, cap, "tab:diag-downstream", size="small")


def tab_diag_detector(D):
    body = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule", "Algorithm",
            r"& \makecell{False-alarm \\ rate}", r"& \makecell{Miss \\ rate}",
            r"& \makecell{Probes \\ per trial}",
            r"& \makecell{Probe-level \\ TP / FP}", r"& \makecell{Probe-level \\ TN / FN} \\",
            r"\midrule"]
    for a in ["linear", "binary_deep"]:
        rs = [r for r in D.live(D.det) if r["algorithm"] == a]
        fa = np.mean([f(r["false_alarm_rate"]) for r in rs])
        ms = np.mean([f(r["miss_rate"]) for r in rs])
        tp = sum(int(f(r["probe_tp"], 0)) for r in rs)
        fp = sum(int(f(r["probe_fp"], 0)) for r in rs)
        tn = sum(int(f(r["probe_tn"], 0)) for r in rs)
        fn = sum(int(f(r["probe_fn"], 0)) for r in rs)
        body += [lab(a, narrow=True), f"& {100*fa:.1f}\\%", f"& {100*ms:.1f}\\%",
                 f"& {_agg(D,a,'n_probes_mean','mean'):.1f}",
                 f"& {tex_int(tp)} / {tex_int(fp)}",
                 f"& {tex_int(tn)} / {tex_int(fn)} \\\\"]
    n_live = sum(1 for r in D.ops if r["regime"] == "live")
    cap = (r"How well each search's own overshoot decision matches the simulation truth "
           r"$N_i > N_{\mathrm{opt}}$, aggregated over the "
           rf"{n_live} informative operating points. Rates are \emph{{trial-level}}: a run is a "
           r"false alarm if at "
           r"least one safe probe was declared an overshoot, and a miss if at least one "
           r"overshooting probe was accepted. Trial-level is the correct unit because probes within "
           r"a run are dependent; the probe-level counts are supporting telemetry only. Reverse "
           r"engineering and brute force have no detector and are omitted rather than scored as "
           r"zero. Reverse engineering's $N_{\mathrm{guess}}$ and the safeguard's correction to "
           r"$N^*$ are instead evaluated in Tables~\ref{tab:diag-exploration}--"
           r"\ref{tab:diag-exploration-priors} and \ref{tab:diag-downstream}. Linear search makes "
           r"no per-probe declaration, so its classification is the "
           r"backtracking decision of its own stopping rule.")
    return wrap(body, cap, "tab:diag-detector", size="small")



def tab_linear_detector(D):
    """Linear search under alternative stopping rules (from linear_detector_study.py).

    Restored after being lost in an edit; reconstructed from linear_detector_bakeoff.csv and
    verified to reproduce the previously rendered table exactly, numbers and bold marks alike.
    """
    rows = load("linear_detector_bakeoff.csv")
    if not rows:
        return None
    COLS = [("narrow_e3", "10000", r"$\epsilon = 10^{-3}$,\ $B = 10{,}000$"),
            ("narrow_e3", "41326", r"$\epsilon = 10^{-3}$,\ $B = 41{,}326$"),
            ("narrow_e4", "2917365", r"$\epsilon = 10^{-4}$,\ $B = 2{,}917{,}365$")]
    ORDER = [("published", "cumulative mean, as published"),
             ("cumulative", "cumulative mean, re-tuned"),
             None,
             ("window", r"moving window of width $w$"),
             ("prepost", "pre/post windows, noise threshold"),
             ("threshold", r"Eq.~(\ref{eq:overshoot-threshold}) at lag $w$"),
             ("pooled", r"Eq.~(\ref{eq:overshoot-threshold}) vs.\ pooled reference"),
             ("cusum", "CUSUM on the pooled reference")]
    cells = {}
    for sid, budget, _ in COLS:
        cells[(sid, budget)] = {r["rule"]: r for r in rows
                                if r["scenario_id"] == sid and r["budget"] == budget}
    body = [r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
            "Stopping rule"]
    for _, _, head in COLS:
        body.append(rf"& \multicolumn{{2}}{{c}}{{{head}}}")
    body.append(r"\\")
    body.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    body.append(r"& Converged & False alarm & Converged & False alarm & Converged "
                r"& False alarm \\")
    body.append(r"\midrule")
    for entry in ORDER:
        if entry is None:
            body.append(r"\midrule")
            continue
        rule, label = entry
        line = [label]
        for sid, budget, _ in COLS:
            sub = cells[(sid, budget)]
            if rule not in sub:
                line += ["& --", "& --"]
                continue
            r = sub[rule]
            rate, fa = f(r["rate"]), f(r["false_alarm_rate"])
            base = max(f(sub["published"]["rate"]), f(sub["cumulative"]["rate"]))
            bold = (rule not in ("published", "cumulative")
                    and 100 * (rate - base) > 2 * f(r["delta_se_pp"]))
            txt = rf"\textbf{{{100*rate:.2f}\%}}" if bold else rf"{100*rate:.2f}\%"
            line += [f"& {txt}", rf"& {100*fa:.1f}\%"]
        body.append(" ".join(line) + r" \\")
    cap = (r"Linear search under alternative stopping rules. Every rule replaces only the "
           r"\emph{overshoot verdict} of Algorithm~\ref{alg:linear-search}; the scan, the "
           r"backtracking and the safeguard are unchanged, and each rule is tuned over the same "
           r"$(m', \mathrm{inc}, s)$ grids on the same two tuning blocks before being scored on "
           r"the same 50{,}000 held-out trials. The first row is the configuration the sweep "
           r"froze, and the second is the same cumulative-mean rule re-tuned here, so the gap "
           r"between them is what better tuning alone is worth and the rows below it are what "
           r"changing the rule is worth. Convergence is the exact probability "
           r"$P(|\hat\phi-\phi|<\epsilon)$ of the depth each run selects, averaged over trials, "
           r"which removes the exploitation coin flip from the comparison. Bold marks an "
           r"improvement over the better of those two cumulative-mean rows by more than two "
           r"standard errors of the paired difference. False-alarm rates are trial-level, as in "
           r"Table~\ref{tab:diag-detector}.")
    return wrap(body, cap, "tab:linear-detector", size="small")


def tab_overshoot_operating(D):
    """How well the normal-placed threshold is justified, two ways: a bound and a measurement.

    Analytic, from analysis/consolidated/overshoot_criterion.py.
    """
    rows = load("overshoot_size.csv")
    if not rows:
        return None
    confs = sorted({f(r["conf"]) for r in rows})
    ms = sorted({int(f(r["m_exploration"])) for r in rows})
    by = {(int(f(r["m_exploration"])), f(r["conf"])): r for r in rows}
    head = ["$m'$", r"& \makecell{Berry--Esseen \\ bound}", r"& \makecell{True distance \\ to normal}"]
    for c in confs:
        head.append(rf"& \makecell{{$\mathrm{{conf}} = {c:g}$ \\ ($\alpha = {1-c:g}$)}}")
    body = [r"\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{r cc " + "c" * len(confs) + "}", r"\toprule",
            " ".join(head) + r" \\",
            rf"\cmidrule(lr){{2-3}}\cmidrule(lr){{4-{3+len(confs)}}}",
            r"& \multicolumn{2}{c}{normal approximation to $K$}"
            rf" & \multicolumn{{{len(confs)}}}{{c}}{{largest deviation of the achieved size}} \\",
            r"\midrule"]
    for m in ms:
        anchor = by.get((m, confs[0]), {})
        line = [str(m), rf"& {f(anchor.get('be_bound_max')):.3f}",
                rf"& {f(anchor.get('true_ks_max')):.3f}"]
        for c in confs:
            line.append(rf"& {f(by[(m, c)]['max_deviation']):.3f}")
        body.append(" ".join(line) + r" \\")
    cap = (r"Justification of the normal quantile in Equation~(\ref{eq:overshoot-threshold}), "
           r"restricted to the two-sided regularity region $m'\min(p_0,1-p_0)\geq 10$. Because "
           r"$\hat\phi_N$ is strictly decreasing in the count $K$, the rule $\hat\phi_N < \phi_1$ "
           r"is \emph{identical} to the cut $K > m'\cos^2(N\phi_1)$, so the only quantity that has "
           r"to be Gaussian is $K$ itself, and $K$ is a sum of $m'$ i.i.d.\ Bernoulli variables. "
           r"The Berry--Esseen theorem then bounds the error of its normal approximation by "
           r"$C(p_0^2+(1-p_0)^2)/\sqrt{m'p_0(1-p_0)}$ with $C \leq 0.4748$, non-asymptotically and "
           r"for every $m'$; the second column reports the worst case of that bound and the third "
           r"the distance actually attained. The remaining columns give the largest gap between "
           r"the size the rule achieves and its nominal level $\alpha = 1-\mathrm{conf}$, computed "
           r"from the exact binomial over safe depths. The residual gap is the discreteness of "
           r"$K$, not a central-limit error, which is why it does not shrink with $m'$.")
    return wrap(body, cap, "tab:overshoot-operating", size="small")


def main():
    D = Data()
    have = {r["algorithm"] for r in D.perf}
    missing = [a for a in ("separable", "oracle_hl") if a not in have]
    if missing:
        print(f"!! {missing} absent from performance_curves.csv — "
              f"run `python analysis/consolidated/add_baselines.py` first; "
              f"those rows will render as '--'.")

    made = []
    made.append(write("tab_summary_low_prec.tex", tab_summary_low_prec(D, ci=False)))
    made.append(write("tab_summary_low_prec_ci.tex", tab_summary_low_prec(D, ci=True)))
    made.append(write("tab_ratios.tex", tab_ratios(D)))
    made.append(write("tab_summary_all.tex", tab_summary_all(D, ci=False)))
    made.append(write("tab_summary_all_ci.tex", tab_summary_all(D, ci=True)))
    made.append(write("tab_scaling_with_prec.tex", tab_scaling_with_prec(D, ci=False)))
    made.append(write("tab_scaling_with_prec_ci.tex", tab_scaling_with_prec(D, ci=True)))
    if "broad_pi2_e3" in D.scen:
        made.append(write("tab_broad_pi2.tex", tab_broad(D, "broad_pi2_e3", r"\pi/2")))
    if "broad_pi4_e3" in D.scen:
        made.append(write("tab_broad_pi4.tex", tab_broad(D, "broad_pi4_e3", r"\pi/4")))
    made.append(write("tab_robustness_across_thresholds.tex",
                      tab_robustness_across_thresholds(D)))
    made.append(write("tab_opt_param_first.tex", tab_opt_param_first(D)))
    made.append(write("tab_opt_param_second.tex", tab_opt_param_second(D)))
    made.append(write("tab_diag_search.tex", tab_diag_search(D)))
    made.append(write("tab_diag_exploration.tex", tab_diag_exploration(D)))
    made.append(write("tab_diag_downstream.tex", tab_diag_downstream(D)))
    made.append(write("tab_diag_detector.tex", tab_diag_detector(D)))
    _ld = tab_linear_detector(D)
    if _ld:
        made.append(write("tab_linear_detector.tex", _ld))
    else:
        print("!! linear_detector_bakeoff.csv absent — run "
              "`python analysis/consolidated/linear_detector_study.py` first.")

    _oc = tab_overshoot_operating(D)
    if _oc:
        made.append(write("tab_overshoot_operating.tex", _oc))
    else:
        print("!! overshoot_operating.csv absent — run "
              "`python analysis/consolidated/overshoot_criterion.py` first.")

    bundle = [
        "% Updated Chapter-4 tables, regenerated from results/consolidated/*.csv",
        f"% sweep mode: {D.man['mode']}   R_test = {D.R:,}   "
        f"seeds: tune {D.man['seeds']['tune_blocks']}, test {D.man['seeds']['test']}",
        "% Generated by analysis/consolidated/thesis_tables.py -- do not edit by hand.",
        "%",
        "% NOTE: results/tex/ contains the OLD tables from the pre-consolidation sweep.",
        "%       These supersede them.",
        "",
    ]
    for n in made:
        bundle += [f"% ---- {n} " + "-" * (60 - len(n)),
                   open(os.path.join(OUT, n)).read(), ""]
    write("all_thesis_tables.tex", "\n".join(bundle))
    print(f"wrote {len(made)+1} files to {OUT}/")
    for n in made:
        print(f"   {n}")


if __name__ == "__main__":
    main()
