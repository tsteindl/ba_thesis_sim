"""Paste-ready LaTeX for the two "extra info" tables, written to results/story_tables_latex.md:
  * robustness across thresholds (best-adaptive ratio), filtered to the paper-relevant ranges only
  * winning parameters at the 90%-convergence budget (\\makecell format)
    python analysis/story_latex.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANGES = ["U(0.01,0.1)", "U(0.001,0.01)", "U(0.001,0.1)"]   # only ranges that appear in Tables 3.1-3.3
MU = {"U(0.01,0.1)": r"\mathcal{U}(0.01,0.1)", "U(0.001,0.01)": r"\mathcal{U}(0.001,0.01)",
      "U(0.001,0.1)": r"\mathcal{U}(0.001,0.1)"}
FULL = {"linear": "Linear search", "binary": "Binary search", "reverse_eng": "Reverse engineering"}
T32_COLS = [("$\\epsilon=10^{-3}$, U(0.01,0.1)", "U(0.01,0.1), eps=1e-03"),
            ("$\\epsilon=10^{-4}$, U(0.01,0.1)", "U(0.01,0.1), eps=1e-04"),
            ("$\\epsilon=10^{-4}$, U(0.001,0.01)", "U(0.001,0.01), eps=1e-04"),
            ("$\\epsilon=10^{-4}$, U(0.001,0.1)", "U(0.001,0.1), eps=1e-04")]
PARAM_ORDER = {"linear": ["m_exploration", "lookback_window", "safeguard", "inc"],
               "binary": ["m_exploration", "safeguard", "conf"],
               "reverse_eng": ["m_exploration", "safeguard"]}
SHORT = {"linear": "L-search", "binary": "B-search", "reverse_eng": "Rev Eng"}


def _rng(s):
    return s.split(", eps")[0]


def _winner(cube, setting):
    sub = cube[(cube.setting == setting) & (cube.threshold_pct == 90)
               & cube.algo.isin(["linear", "binary", "reverse_eng"])].dropna(subset=["ratio_vs_brute"])
    return FULL[sub.loc[sub.ratio_vs_brute.idxmax(), "algo"]] if len(sub) else "-"


def robustness_latex(cube):
    rows = [s for s in cube.setting.unique() if _rng(s) in RANGES]
    order = {r: i for i, r in enumerate(RANGES)}
    rows.sort(key=lambda s: (order[_rng(s)], s.split("eps=")[1]))
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Best adaptive-vs-baseline budget ratio across convergence thresholds "
         r"(higher is better). Winner is the algorithm achieving the highest ratio.}",
         r"\label{tab:robustness-across-thresholds}", r"\begin{tabular}{lccccl}", r"\toprule",
         r"Scenario & 50\% & 80\% & 90\% & 95\% & Winner \\", r"\midrule"]
    for s in rows:
        ba = cube[(cube.setting == s) & (cube.algo == "best_adaptive")]
        cells = [f"{ba[ba.threshold_pct == t].ratio_vs_brute.iloc[0]:.2f}\\texttimes" for t in (50, 80, 90, 95)]
        eps = s.split("eps=")[1]
        L.append(f"${MU[_rng(s)]}$, $\\epsilon=${eps}  & " + " & ".join(cells) + f" & {_winner(cube, s)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _makecell(algo, params, budget):
    keys = PARAM_ORDER[algo]
    parts = []
    for i, k in enumerate(keys):
        v = params[k]
        esc = k.replace("_", r"\_")
        parts.append(f"\\texttt{{{esc}={v}{',' if i < len(keys) - 1 else ''}}}")
    return r"\makecell{" + " \\\\ ".join(parts) + f" \\\\ (@budget {budget:,})}}"


def winners_latex(cube, winners):
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Winning parameters at the 90\% convergence budget.}",
         r"\label{tab:opt-param-second-tab}", r"\begin{tabular}{lllll}", r"\toprule",
         "Algorithm\n& " + "\n& ".join(h for h, _ in T32_COLS) + r" \\", r"\midrule", ""]
    for algo in ["linear", "binary", "reverse_eng"]:
        cells = []
        for _h, setting in T32_COLS:
            cr = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == 90)]
            sub = winners[(winners.setting == setting) & (winners.algo == algo)]
            if not len(cr) or pd.isna(cr.budget_to_reach.iloc[0]) or not len(sub):
                cells.append(r"\makecell{--}")
                continue
            # params come from the nearest evaluated budget; the label uses the Table 3.2 crossing
            # budget so the two tables are consistent.
            crossing = int(round(cr.budget_to_reach.iloc[0]))
            row = sub.iloc[(sub.budget - cr.budget_to_reach.iloc[0]).abs().argmin()]
            cells.append(_makecell(algo, json.loads(row.params), crossing))
        L.append(f"{SHORT[algo]}\n& " + "\n& ".join(cells) + r" \\")
        L.append("")
        L.append(r"\midrule" if algo != "reverse_eng" else r"\bottomrule")
        L.append("")
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    cube = pd.read_csv("results/story_cube.csv")
    winners = pd.read_csv("results/story_winners.csv")
    rob = robustness_latex(cube)
    win = winners_latex(cube, winners)
    with open("results/story_tables_latex.md", "w") as f:
        f.write("# Paste-ready LaTeX for the extra-info tables\n\n"
                "## Robustness across thresholds (paper-relevant ranges only)\n\n```latex\n"
                + rob + "\n```\n\n## Winning parameters at the 90% budget\n\n```latex\n" + win + "\n```\n")
    print(rob + "\n\n" + win + "\n\nwrote results/story_tables_latex.md")


if __name__ == "__main__":
    main()
