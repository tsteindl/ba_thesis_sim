"""Generate every thesis table as paste-ready LaTeX — one .tex file per table.

Renders only; it reads the result CSVs and writes LaTeX, so it runs in a second and can be re-run
after any simulation without re-deriving anything. Each table is written twice where it makes sense:

    results/tex/<name>.tex        the table exactly as it appears in the manuscript
    results/tex/<name>_ci.tex     the same table with 95% confidence intervals added

so the plain files stay drop-in replacements while the _ci files are available if the error bars are
to be reported. `results/tex/all_tables.tex` \\input{}s every plain table; `results/TEX.md` collects
all of them in fenced blocks for copy-paste.

Inputs (all optional — a table whose inputs are missing is skipped with a note):
    results/all_numbers.csv       % converged at budget 10,000     (analysis/make_results.py)
    results/story_cube.csv        budgets to reach p* and ratios   (analysis/extensive_sweep.py)
    results/story_cube_ci.csv     their confidence intervals       (analysis/uncertainty.py)
    results/story_winners.csv     winning parameters per budget    (analysis/extensive_sweep.py)
    results/optimal_params.csv    winners at fixed budget 10,000   (analysis/appendix_params.py)
    results/broad_dist.csv        broad-prior convergence          (analysis/broad_dist_study.py)

    python analysis/make_tex.py [--outdir results/tex]

To change a caption, a row order, or which algorithms appear, edit the builder for that table below —
the formatting primitives live in qmetrology/latex.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.latex import (DASH, MU, PLAIN, REF, SHORT, bold, ci, eps_math, factor, num, pct,
                              stack, table, texttt)
from qmetrology.oracle import degeneracy_share

OUTDIR = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "results/tex"

# row order used by the summary tables; the ceilings are appended last, below a rule
MAIN_ROWS = ["brute", "linear", "binary_risk", "reverse_eng_risk"]
# ordered: the meaningful ceiling first, the exact (sometimes degenerate) one last
CEILINGS = ["oracle_hl"]
ADAPT = ["linear", "binary_risk", "reverse_eng_risk"]
HEAD = "U(0.01,0.1), eps=1e-03"
# Table 3.2's four columns: (header eps, header prior, setting label in the cube)
T32 = [(r"\epsilon=10^{-3}", "U(0.01,0.1)"), (r"\epsilon=10^{-4}", "U(0.01,0.1)"),
       (r"\epsilon=10^{-4}", "U(0.001,0.01)"), (r"\epsilon=10^{-4}", "U(0.001,0.1)")]
T32_SETTINGS = ["U(0.01,0.1), eps=1e-03", "U(0.01,0.1), eps=1e-04",
                "U(0.001,0.01), eps=1e-04", "U(0.001,0.1), eps=1e-04"]
ALL_EPS = ["1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]
PARAM_ORDER = {"linear": ["m_exploration", "lookback_window", "safeguard", "inc"],
               "binary": ["m_exploration", "safeguard", "conf"],
               "reverse_eng": ["m_exploration", "safeguard"],
               "binary_risk": ["m_exploration", "conf"],
               "reverse_eng_risk": ["m_exploration"]}
DEBIAS_NOTE = ("Every adaptive entry is de-biased: parameters are grid-tuned on seed 42 and the "
               "winning configuration is re-validated on the independent seed 2024 at "
               "$R=40{,}000$ trials.")


# ---------------------------------------------------------------- data access helpers
def _load(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


class Data:
    """Thin accessor over the result CSVs, so each builder reads one clean call."""

    def __init__(self):
        self.numbers = _load("results/all_numbers.csv")
        self.cube = _load("results/story_cube.csv")
        self.cube_ci = _load("results/story_cube_ci.csv")
        self.winners = _load("results/story_winners.csv")
        self.opt = _load("results/optimal_params.csv")
        self.broad = _load("results/broad_dist.csv")
        self.pilot = _load("results/pilot_share_summary.csv")
        # mean budget actually consumed / nominal budget, per (setting, algo, budget).
        # Algorithm 4/5 do not cap their exploration phase: the loop checks the budget only AFTER
        # taking a probe, so at budgets too small to fit the scan they overspend. A cell whose
        # winner overspends is not an equal-budget comparison and must not be read as one.
        self.spend = _load("results/budget_audit.csv")

    def conv10k(self, algo, setting="narrow"):
        """% converged at fixed budget 10,000."""
        if self.numbers is None:
            return None
        r = self.numbers[(self.numbers.setting == setting) & (self.numbers.algorithm == algo)
                         & (self.numbers.metric == "pct_converged_10k")]
        return float(r.value.iloc[0]) if len(r) else None

    def cell(self, setting, algo, thr=90):
        """(budget_to_reach, ratio_vs_brute) or (None, None)."""
        if self.cube is None:
            return None, None
        r = self.cube[(self.cube.setting == setting) & (self.cube.algo == algo)
                      & (self.cube.threshold_pct == thr)]
        if not len(r) or pd.isna(r.budget_to_reach.iloc[0]):
            return None, None
        return float(r.budget_to_reach.iloc[0]), float(r.ratio_vs_brute.iloc[0])

    def cell_ci(self, setting, algo, thr=90):
        """(budget_lo, budget_hi, ratio_lo, ratio_hi) or four Nones."""
        if self.cube_ci is None:
            return (None,) * 4
        r = self.cube_ci[(self.cube_ci.setting == setting) & (self.cube_ci.algo == algo)
                         & (self.cube_ci.threshold_pct == thr)]
        if not len(r):
            return (None,) * 4
        r = r.iloc[0]
        g = lambda k: None if pd.isna(r[k]) else float(r[k])  # noqa: E731
        return g("budget_lo"), g("budget_hi"), g("ratio_lo"), g("ratio_hi")

    def settings(self):
        return [] if self.cube is None else list(self.cube.setting.unique())

    def degeneracy(self, setting):
        """Share of this prior where the exact oracle can answer from the aliasing constant alone."""
        if self.cube is None:
            return float("nan")
        r = self.cube[self.cube.setting == setting]
        if not len(r):
            return float("nan")
        return degeneracy_share(r.eps.iloc[0], r.phi_min.iloc[0], r.phi_max.iloc[0])

    def spend_ratio(self, setting, algo, thr=90):
        """Budget actually consumed / nominal, at the budget nearest this crossing. None if unknown."""
        if self.spend is None:
            return None
        b, _ = self.cell(setting, algo, thr)
        sub = self.spend[(self.spend.setting == setting) & (self.spend.algo == algo)]
        if b is None or not len(sub):
            return None
        return float(sub.iloc[(sub.budget - b).abs().argmin()].ratio)

    def winner_params(self, setting, algo, thr=90):
        """The winning configuration at the budget nearest the p* crossing, and that crossing."""
        if self.winners is None or self.cube is None:
            return None, None
        b, _ = self.cell(setting, algo, thr)
        sub = self.winners[(self.winners.setting == setting) & (self.winners.algo == algo)]
        if b is None or not len(sub):
            return None, None
        row = sub.iloc[(sub.budget - b).abs().argmin()]
        return json.loads(row.params), int(round(b))


def _params_cell(algo, params, budget=None):
    """One parameter per line, trailing commas inside the \\texttt as in the manuscript."""
    keys = [k for k in PARAM_ORDER.get(algo, sorted(params)) if k in params]
    lines = [texttt(**{k: f"{params[k]}{',' if i < len(keys) - 1 else ''}"})
             for i, k in enumerate(keys)]
    if budget is not None:
        lines.append(f"(@budget {num(budget)})")
    return stack(*lines)


# ---------------------------------------------------------------- table builders
def tab_summary_low_prec(D, with_ci=False):
    """tab:summary-low-prec — % converged at budget 10,000 and budget for >90%, at eps=1e-3."""
    base = D.conv10k("brute")
    if base is None or not D.cell(HEAD, "brute")[0]:
        return None
    head = ["Algorithm",
            stack(r"Avg. \% converged", r"(budget = 10{,}000)", r"(\texttimes factor vs baseline)"),
            stack(r"Budget for $>90\%$", "convergence", r"(\texttimes factor vs baseline)")]
    rows, rules = [], set()
    for algo in MAIN_ROWS + ["separable"] + CEILINGS:
        conv = D.conv10k(algo)
        bud, ratio = D.cell(HEAD, algo)
        if conv is None and bud is None:
            continue
        if algo in CEILINGS and CEILINGS[0] == algo:
            rules.add(len(rows))
        c1 = DASH if conv is None else f"{pct(conv)} ({factor(conv / base, 3)})"
        c2 = DASH if bud is None else f"{num(bud)} ({factor(ratio)})"
        if with_ci:
            blo, bhi, rlo, rhi = D.cell_ci(HEAD, algo)
            c2 = stack(c2, ci(rlo, rhi, lambda v: f"{v:.2f}")) if rlo is not None else c2
        rows.append([REF[algo], c1, c2])
    cap = (r"Algorithm performance under low-precision constraints $\epsilon=10^{-3}$ for "
           r"$\phi\sim\mathcal{U}(0.01,0.1)$. The table reports both the convergence rate at fixed "
           r"budget $C=10{,}000$ and the budget required to achieve \(>90\%\) convergence. "
           r"Improvement factors are computed relative to the brute force baseline. The two "
           r"\textit{oracle} rows are not implementable protocols but upper bounds: they select the "
           r"exploitation depth knowing the true $\phi$.")
    if with_ci:
        cap += (r" Bracketed values are $95\%$ confidence intervals on the improvement factor "
                r"(parametric bootstrap of the convergence curve; see Appendix).")
    return table("lcc", head, rows, cap, "tab:summary-low-prec",
                 note=DEBIAS_NOTE, blank_between_rows=True, midrules=rules)


def tab_summary_all(D, with_ci=False):
    """tab:summary-all — budget for >90% convergence across four precision/prior columns."""
    if D.cube is None:
        return None
    head = ["Algorithm"] + [
        stack("Budget", r"($>90\%$ conv.)", r"(\texttimes factor vs baseline)", f"${e}$",
              f"$\\phi \\sim {MU[p]}$") for e, p in T32]
    rows, rules = [], set()
    for algo in MAIN_ROWS + CEILINGS:
        cells = []
        for s in T32_SETTINGS:
            bud, ratio = D.cell(s, algo)
            if bud is None:
                cells.append(DASH); continue
            cell = f"{num(bud)} ({factor(ratio)})"
            if with_ci:
                _bl, _bh, rlo, rhi = D.cell_ci(s, algo)
                if rlo is not None:
                    cell = stack(cell, ci(rlo, rhi, lambda v: f"{v:.2f}"))
            cells.append(cell)
        if all(c == DASH for c in cells):
            continue
        if algo == CEILINGS[0]:
            rules.add(len(rows))
        rows.append([REF[algo]] + cells)
    cap = (r"Budget required to achieve $>90\%$ convergence for selected precision settings and "
           r"prior intervals. Improvement factors are computed relative to the brute force baseline, "
           r"with values larger than one indicating lower required budget.")
    if with_ci:
        cap += r" Bracketed values are $95\%$ confidence intervals on the improvement factor."
    return table("l c c c c", head, rows, cap, "tab:summary-all",
                 size="scriptsize", note=DEBIAS_NOTE, blank_between_rows=True, midrules=rules)


def tab_scaling_with_prec(D, with_ci=False):
    """tab:scaling-with-prec — improvement factor vs precision, at U(0.01,0.1), 90%."""
    if D.cube is None:
        return None
    cols = [a for a in ADAPT + CEILINGS
            if any(D.cell(f"U(0.01,0.1), eps={t}", a)[1] is not None for t in ALL_EPS)]
    tags = [t for t in ALL_EPS if f"U(0.01,0.1), eps={t}" in set(D.settings())]
    if not tags or not cols:
        return None
    # seven columns is already wide: stack each name so the header does not force a landscape page
    wrap = {"oracle_hl": (r"\textit{Attainable}", "ceiling"),
            "linear": ("Linear", "search"),
            "binary_risk": ("Binary", "search"),
            "reverse_eng_risk": ("Reverse", "engineering")}
    head = [r"Precision $\epsilon$"] + [stack(*wrap.get(a, (PLAIN[a],))) for a in cols]
    rows = []
    for t in tags:
        s = f"U(0.01,0.1), eps={t}"
        cells = []
        for a in cols:
            _b, r = D.cell(s, a)
            if r is None:
                cells.append(DASH); continue
            cell = f"{r:.2f}\\texttimes"
            if with_ci:
                _bl, _bh, rlo, rhi = D.cell_ci(s, a)
                if rlo is not None:
                    cell = stack(cell, ci(rlo, rhi, lambda v: f"{v:.2f}"))
            cells.append(cell)
        rows.append([eps_math(float(t))] + cells)
    return table("l" + "c" * len(cols), head, rows,
                 r"Improvement factor over the brute-force baseline as the precision requirement "
                 r"tightens, at $\phi\sim\mathcal{U}(0.01,0.1)$ and $90\%$ convergence. The "
                 r"advantage grows with precision and then saturates; the oracle column bounds how "
                 r"much of it is attainable at all.",
                 "tab:scaling-with-prec", size="small", note=DEBIAS_NOTE)


def tab_broad(D, tag="pi/2", lo=15_000, hi=900_000):
    """tab:broad-pi2 — convergence by budget under a broad prior."""
    if D.broad is None:
        return None
    sub = D.broad[(D.broad.phi_max == tag) & (D.broad.budget >= lo) & (D.broad.budget <= hi)]
    if not len(sub):
        return None
    # the *_risk columns ARE binary search / reverse engineering; the constant-C arms stay in
    # results/broad_dist.csv for the published-vs-this-work delta but are not reported here.
    # broad_dist.csv names the analytic ceiling `oracle` (it is heisenberg_rate, i.e. oracle_hl).
    series = [c for c in ["brute", "linear", "binary_risk", "reverse_eng_risk", "oracle"]
              if c in sub.columns]
    ceil = CEILINGS + ["oracle"]
    head = ([r"Budget $C=N\cdot m$"]
            + [PLAIN["oracle_hl"] if c == "oracle" else PLAIN[c] for c in series])
    rows = []
    for _, r in sub.iterrows():
        vals = [float(r[c]) for c in series]
        # the ceiling is shown but never competes for the bold "best" mark
        best = max(v for c, v in zip(series, vals) if c not in ceil)
        rows.append([num(r.budget)] + [bold(pct(v, 1)) if (v == best and c not in ceil)
                                       else pct(v, 1) for c, v in zip(series, vals)])
    ltag = tag.replace("pi", r"\pi")
    return table("r" + "c" * len(series), head, rows,
                 rf"Convergence under a broad uniform prior $\phi\sim\mathcal{{U}}(0.01,{ltag})$ at "
                 r"$\epsilon=10^{-3}$: average share of simulations that converge, by budget. "
                 r"The best entry in each row is set in bold.",
                 f"tab:broad-{tag.replace('/', '')}", size="small", note=DEBIAS_NOTE)


def tab_robustness(D):
    """tab:robustness-across-thresholds — best adaptive ratio at every convergence threshold."""
    if D.cube is None:
        return None
    ranges = ["U(0.01,0.1)", "U(0.001,0.01)", "U(0.001,0.1)"]
    order = {r: i for i, r in enumerate(ranges)}
    rows_s = [s for s in D.settings() if s.split(", eps")[0] in ranges]
    rows_s.sort(key=lambda s: (order[s.split(", eps")[0]], s.split("eps=")[1]))
    rows, any_flagged = [], []
    for s in rows_s:
        cells = []
        for t in (50, 80, 90, 95):
            _b, r = D.cell(s, "best_adaptive", t)
            cells.append(DASH if r is None else f"{r:.2f}\\texttimes")
        cand = [(D.cell(s, a, 90)[1], a) for a in ADAPT if D.cell(s, a, 90)[1] is not None]
        win, flagged = DASH, False
        if cand:
            best = max(cand)[1]
            sr = D.spend_ratio(s, best, 90)
            # the winner is only a winner if it stayed inside the budget it is credited with
            flagged = sr is not None and sr > 1.02
            win = PLAIN[best] + (r"$^{\dagger}$" if flagged else "")
        rng, eps = s.split(", eps=")
        rows.append([f"${MU[rng]}$, $\\epsilon={eps_math(float(eps))[1:-1]}$"] + cells + [win])
        any_flagged.append(flagged)
    if not rows:
        return None
    note = DEBIAS_NOTE
    if any(any_flagged):
        note += (r" $^{\dagger}$This algorithm's exploration phase exceeded the budget it is "
                 r"credited with (Algorithm 4/5 test the budget only after taking a probe, so a "
                 r"scan that does not fit overspends); the cell is not an equal-budget comparison. "
                 r"Per-point spend ratios are in \texttt{results/budget\_audit.csv}.")
    return table("lccccl", ["Scenario", r"50\%", r"80\%", r"90\%", r"95\%", "Winner"], rows,
                 r"Best adaptive-vs-baseline budget ratio across convergence thresholds (higher is "
                 r"better). Winner is the algorithm achieving the highest ratio at $90\%$.",
                 "tab:robustness-across-thresholds", note=note)


def tab_pilot_share(D):
    """tab:pilot-share — cost of never tuning reverse engineering's exploration size."""
    if D.pilot is None:
        return None
    rows = []
    for _, r in D.pilot.iterrows():
        # the first four columns are percentage POINTS of convergence lost, not percentages;
        # only the last column is a share of operating points
        cells = [f'{r["mean"]:.2f}', f'{r["median"]:.2f}', f'{r["p05"]:.2f}', f'{r["worst"]:.2f}',
                 pct(r["pct_worse_1pp"], 1)]
        if int(r["recommended"]):
            cells = [bold(c) for c in cells]
            rows.append([bold(r["tex_label"]) + r" $\;\leftarrow$"] + cells)
        else:
            rows.append([r["tex_label"]] + cells)
    if not rows:
        return None
    return table("lccccc",
                 ["Fixed default", stack("mean", "regret (pp)"), stack("median", "(pp)"),
                  stack("5th pct.", "(pp)"), stack("worst", "(pp)"),
                  stack("share of points", r"$>1$ pp worse")], rows,
                 r"Cost of fixing reverse engineering's exploration size instead of grid-tuning it at "
                 r"every budget. Regret is the loss in convergence (percentage points) against the "
                 r"per-budget tuned configuration, over all $546$ live operating points of the "
                 r"$23$-scenario sweep. A budget share $\rho$ transfers; a shot count $m'$ does not, "
                 r"because the exploration size must grow with the budget.",
                 "tab:pilot-share", size="small", note=DEBIAS_NOTE)


def _pilot_share_cell(algo, params, budget, phi_max):
    """The tuned exploration size read as a fraction of the budget.

    Only meaningful where the exploration is a single probe at N_min -- reverse engineering. Binary
    search re-probes at growing depths, so its cost is not m'*N_min and no share is shown.
    """
    if algo != "reverse_eng_risk" or not budget:
        return DASH
    m = params.get("m_exploration")
    if m is None:
        return DASH
    nmin = max(1, int(np.floor(np.pi / (2 * float(phi_max)))))
    return pct(100.0 * m * nmin / float(budget), 1)


def tab_opt_param_first(D):
    """tab:opt-param-first-tab — winning parameters at fixed budget 10,000."""
    if D.opt is None:
        return None
    rows = []
    for _, r in D.opt.iterrows():
        if r.algorithm not in PARAM_ORDER:
            continue
        rows.append([PLAIN[r.algorithm], _params_cell(r.algorithm, json.loads(r.params)),
                     _pilot_share_cell(r.algorithm, json.loads(r.params), r.budget, r.phi_max)])
    if not rows:
        return None
    return table("llc", ["Algorithm", "Optimal parameters", stack("Pilot cost,", "\\% of budget")], rows,
                 r"Winning parameters for Table \ref{tab:summary-low-prec}, fixed budget "
                 r"$10{,}000$ ($\epsilon=10^{-3}$, $\mathcal{U}(0.01,0.1)$). Reverse engineering's "
                 r"exploration size is quoted in the text as a share of the budget "
                 r"(Table~\ref{tab:pilot-share}), so its tuned value is also shown that way.",
                 "tab:opt-param-first-tab")


def tab_opt_param_second(D):
    """tab:opt-param-second-tab — winning parameters at the 90% convergence budget."""
    if D.winners is None:
        return None
    head = ["Algorithm"] + [f"${e}$, ${MU[p]}$" for e, p in T32]
    rows, rules = [], set()
    for algo in ADAPT:
        cells = []
        for s in T32_SETTINGS:
            params, budget = D.winner_params(s, algo)
            if params is None:
                cells.append(stack(DASH))
                continue
            cell = _params_cell(algo, params, budget)
            if algo == "reverse_eng_risk":
                pm = D.cube[D.cube.setting == s]
                if len(pm):
                    sh = _pilot_share_cell(algo, params, budget, pm.phi_max.iloc[0])
                    if sh != DASH:
                        cell = stack(*(cell.replace("\\makecell{", "").rstrip("}").split(" \\\\ ")
                                       + [f"({sh} of budget)"]))
            cells.append(cell)
        if all(c == stack(DASH) for c in cells):
            continue
        if rows:
            rules.add(len(rows))
        rows.append([SHORT[algo]] + cells)
    if not rows:
        return None
    return table("lllll", head, rows,
                 r"Winning parameters at the $90\%$ convergence budget. The variants using "
                 r"Eq.~\eqref{eq:risk-depth} have no safeguard to tune: the exploitation depth "
                 r"follows from the pilot's own spread.",
                 "tab:opt-param-second-tab", size="scriptsize",
                 blank_between_rows=True, midrules=rules)


def tab_oracle_gap(D):
    """tab:oracle-gap — how much of the attainable advantage each protocol actually captures.

    The denominator is `oracle_hl`, not the exact oracle: with exact knowledge of phi the aliasing
    constant pi/(2 floor(pi/2phi)) is itself within ~2 phi^2/pi of the answer, so wherever
    phi < sqrt(pi eps/2) the exact oracle converges from the choice of depth alone and its ratio
    stops meaning anything (it reaches ~400x for U(0.001,0.01) at eps=1e-4). The share of each prior
    in that regime is reported in the last row so the reader can judge the exact oracle's cells.
    """
    if D.cube is None or D.cell(HEAD, "oracle_hl")[1] is None:
        return None
    settings = [s for s in T32_SETTINGS if D.cell(s, "oracle_hl")[1] is not None]
    if not settings:
        return None
    head = ["Algorithm"] + [stack(f"${e}$", f"${MU[p]}$") for (e, p), s
                            in zip(T32, T32_SETTINGS) if s in settings]
    rows, rules = [], set()
    for algo in MAIN_ROWS[1:] + CEILINGS:
        cells = []
        for s in settings:
            _b, r = D.cell(s, algo)
            _b2, ceil_r = D.cell(s, "oracle_hl")
            if r is None or ceil_r is None or ceil_r <= 1:
                cells.append(DASH); continue
            # the share is only meaningful once the protocol is at least at parity with the baseline
            share = f"{100 * (r - 1) / (ceil_r - 1):.0f}\\%" if r > 1 else DASH
            if algo == "oracle_hl":
                share = "100\\%"
            elif algo in ("oracle", "oracle_alias") and D.degeneracy(s) >= 0.25:
                # over a quarter of this prior can be answered from the aliasing constant rather
                # than from the data, so a "share of attainable" here would be nonsense
                share = r"\emph{degen.}"
            cells.append(f"{r:.2f}\\texttimes\\ ({share})")
        if all(c == DASH for c in cells):
            continue
        if algo == CEILINGS[0]:
            rules.add(len(rows))
        rows.append([REF[algo]] + cells)
    rules.add(len(rows))
    rows.append([r"\emph{prior share where the exact oracle degenerates}"]
                + [f"{100 * D.degeneracy(s):.0f}\\%" for s in settings])
    return table("l" + "c" * len(settings), head, rows,
                 r"Budget improvement over the brute-force baseline at $90\%$ convergence, and in "
                 r"parentheses the share of the \emph{attainable} improvement it realises, "
                 r"$(\rho-1)/(\rho_{\mathrm{ceil}}-1)$. The ceiling $\rho_{\mathrm{ceil}}$ grants "
                 r"perfect knowledge of the optimal depth $N=\lfloor\pi/2\phi\rfloor$ but still "
                 r"requires the estimate to be resolved by the sampling law of Eq.~(3.4), giving "
                 r"$\Pr(\text{converge})=2\Phi(2\epsilon\sqrt{NC})-1$. The exact oracle "
                 r"$\arg\max_N\Pr(|\hat\phi-\phi|<\epsilon)$ is a strictly tighter bound but a "
                 r"degenerate one: for $\phi<\sqrt{\pi\epsilon/2}$ the aliasing constant "
                 r"$\pi/(2\lfloor\pi/2\phi\rfloor)$ is already within $\epsilon$ of $\phi$, so it "
                 r"answers from the choice of depth rather than from the data. The last row gives "
                 r"the share of each prior in that regime. The third ceiling "
                 r"uses $N=\lfloor\pi/2\phi\rfloor$ with the exact estimator: perfect knowledge of "
                 r"$\phi$, but no backoff from the aliasing bound.",
                 "tab:oracle-gap", size="small", note=DEBIAS_NOTE)


def tab_uncertainty(D):
    """tab:uncertainty — the appendix table of confidence intervals for the headline budgets."""
    if D.cube_ci is None:
        return None
    settings = [s for s in T32_SETTINGS if s in set(D.cube_ci.setting)]
    if not settings:
        return None
    rows = []
    for s in settings:
        for algo in MAIN_ROWS + CEILINGS:
            bud, ratio = D.cell(s, algo)
            blo, bhi, rlo, rhi = D.cell_ci(s, algo)
            if bud is None or blo is None:
                continue
            rng, eps = s.split(", eps=")
            rows.append([f"${MU[rng]}$, $\\epsilon={eps_math(float(eps))[1:-1]}$", SHORT[algo], num(bud),
                         ci(blo, bhi), f"{ratio:.2f}", ci(rlo, rhi, lambda v: f"{v:.2f}")])
    if not rows:
        return None
    return table("llrrrr",
                 ["Scenario", "Algorithm", "Budget", r"$95\%$ CI", "Ratio", r"$95\%$ CI"], rows,
                 r"Monte-Carlo uncertainty of the budgets reported for $90\%$ convergence. Intervals "
                 r"are $95\%$ percentile bootstrap intervals obtained by resampling every point of "
                 r"the convergence-vs-budget curve from $\mathrm{Binomial}(R,\hat p)/R$ with "
                 r"$R=40{,}000$ and re-deriving the crossing. Ratio intervals resample the two "
                 r"curves independently and are therefore conservative: all algorithms share the "
                 r"same seed list, so the trials are paired in $\phi$.",
                 "tab:uncertainty", size="scriptsize")


# ---------------------------------------------------------------- self-check
def validate(tex, name):
    """Structural checks a paste-ready table must pass: brace balance and column counts.

    Not a LaTeX parser — it catches the two mistakes that actually happen when a table is generated
    (a row with the wrong number of `&`, and an unbalanced `\\makecell{`), both of which fail the
    thesis build rather than this script.
    """
    problems = []
    if tex.count("{") != tex.count("}"):
        problems.append(f"unbalanced braces ({tex.count('{')} open, {tex.count('}')} close)")

    spec = tex.split(r"\begin{tabular}{", 1)[1].split("}", 1)[0]
    ncol = sum(1 for ch in spec if ch in "lcr")
    body = tex.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]
    for line in _split_top_level(body, r"\\"):
        line = line.replace(r"\midrule", "").strip()
        if not line or line.startswith("%"):
            continue
        cells = 1 + len(_split_top_level(line, "&")) - 1
        if cells != ncol:
            problems.append(f"row has {cells} cells, tabular declares {ncol}: {line[:70]!r}")
    return [f"{name}: {p}" for p in problems]


def _split_top_level(text, sep):
    r"""Split on `sep` only outside braces — `\\` and `&` inside a \makecell{...} belong to the cell,
    not to the row, so a naive split would report every stacked cell as a malformed row."""
    out, cur, depth, i = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            out.append("".join(cur)); cur = []; i += len(sep); continue
        cur.append(ch); i += 1
    out.append("".join(cur))
    return out


# ---------------------------------------------------------------- driver
BUILDERS = [
    ("tab_summary_low_prec", tab_summary_low_prec, True),
    ("tab_summary_all", tab_summary_all, True),
    ("tab_scaling_with_prec", tab_scaling_with_prec, True),
    ("tab_broad_pi2", tab_broad, False),
    ("tab_robustness_across_thresholds", tab_robustness, False),
    ("tab_oracle_gap", tab_oracle_gap, False),
    ("tab_pilot_share", tab_pilot_share, False),
    ("tab_opt_param_first", tab_opt_param_first, False),
    ("tab_opt_param_second", tab_opt_param_second, False),
    ("tab_uncertainty", tab_uncertainty, False),
]


def main():
    D = Data()
    os.makedirs(OUTDIR, exist_ok=True)
    written, skipped, md, bad = [], [], [], []
    for name, fn, has_ci in BUILDERS:
        variants = [(name, {})] + ([(name + "_ci", {"with_ci": True})] if has_ci else [])
        for fname, kw in variants:
            try:
                tex = fn(D, **kw)
            except Exception as ex:
                skipped.append(f"{fname} ({type(ex).__name__}: {ex})")
                continue
            if tex is None:
                skipped.append(f"{fname} (inputs missing)")
                continue
            bad += validate(tex, fname)
            path = os.path.join(OUTDIR, fname + ".tex")
            with open(path, "w") as f:
                f.write(tex + "\n")
            written.append(path)
            if not fname.endswith("_ci"):
                md.append(f"## `{fname}`\n\n```latex\n{tex}\n```\n")

    with open(os.path.join(OUTDIR, "all_tables.tex"), "w") as f:
        f.write("% Generated by analysis/make_tex.py — do not edit by hand.\n"
                "% Requires \\usepackage{booktabs} and \\usepackage{makecell}.\n"
                "% \\input paths are resolved relative to the MAIN document, not to this file, so if\n"
                f"% you copy {OUTDIR}/ elsewhere in the thesis tree, adjust the prefix below (or\n"
                f"% re-run with --outdir <the path as seen from the main .tex>).\n")
        for p in written:
            if not p.endswith("_ci.tex"):
                f.write(f"\\input{{{p[:-4]}}}\n")

    with open("results/TEX.md", "w") as f:
        f.write("# Paste-ready LaTeX for every thesis table\n\n"
                "_Generated by `analysis/make_tex.py`. Individual files are in "
                f"`{OUTDIR}/`; `{OUTDIR}/all_tables.tex` inputs them all. Variants ending in `_ci` "
                "carry 95% confidence intervals. Preamble needs `booktabs` and `makecell`._\n\n"
                + "\n".join(md))

    print(f"wrote {len(written)} table(s) to {OUTDIR}/ and results/TEX.md")
    for p in written:
        print("  " + p)
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print("  " + s)
    if bad:
        print("\n!! structural problems — these would not compile:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("\nall tables pass the structural check (brace balance, column counts)")


if __name__ == "__main__":
    main()
