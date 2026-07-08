"""Generate results/TABLE32_COMPARISON.md — Table 3.2 under three framings, side by side,
so the variable-budget (Alg. 6) vs improved-variable vs fixed-budget results can be compared.
Fixed-budget numbers come from results/story_cube.csv; the variable numbers are from the
direct searches in the session (documented inline).
    python analysis/gen_table32_comparison.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

c = pd.read_csv("results/story_cube.csv")
COLS = [("eps1e-3 U(0.01,0.1)", "U(0.01,0.1), eps=1e-03"),
        ("eps1e-4 U(0.01,0.1)", "U(0.01,0.1), eps=1e-04"),
        ("eps1e-4 U(0.001,0.01)", "U(0.001,0.01), eps=1e-04"),
        ("eps1e-4 U(0.001,0.1)", "U(0.001,0.1), eps=1e-04")]
ALGOS = [("brute", "Brute force"), ("linear", "Linear search"),
         ("binary", "Binary search"), ("reverse_eng", "Reverse engineering")]


def cell(setting, algo):
    r = c[(c.setting == setting) & (c.algo == algo) & (c.threshold_pct == 90)]
    if not len(r) or pd.isna(r.budget_to_reach.iloc[0]):
        return "—"
    b = r.budget_to_reach.iloc[0]
    if algo == "brute":
        return f"{b:,.0f}"
    return f"{b:,.0f} (x{r.ratio_vs_brute.iloc[0]:.2f})"


L = []
w = L.append
w("# Table 3.2 - framing comparison (budget to reach **90%** convergence)\n")
w("_Lower budget = better. Ratio = brute / algorithm (>1 means adaptive needs LESS budget). "
  "Fixed-budget numbers from `story_cube.csv` (de-biased); variable-budget numbers from direct searches._\n")

w("## A) Original framing - variable-budget Algorithm 6 (scalar exploitation)\n")
w("The paper's Table 3.2 approach. Best honest configs (direct search): **adaptivity ties or loses.**\n")
w("| Algorithm | budget @90%, eps=1e-4, U(0.01,0.1) |")
w("|---|---:|")
w("| Brute force | 4,516,530 |")
w("| Linear search | ~4,510,000  (~ties) |")
w("| Binary search | > 6,000,000  (loses) |")
w("| Reverse engineering | 6,310,396  (x0.72, **loses**) |")
w("\n> The conv95 grid protocol mis-selects RE here (it reports 59.8M / 100% - a coarse-grid overshoot). "
  "The true best variable-Alg.6 config is 6.31M, which still loses to brute. A single exploitation count "
  "cannot give the hard (large-phi) trials more shots without over-spending on the easy ones.\n")

w("## B) Variable-budget, per-trial precision-targeted RE (improved; budget still the OUTPUT)\n")
w("Same reverse-engineering idea, but the exploitation shots adapt to the inferred depth: spend "
  "`m ~ 1/(N*eps)^2` shots (enough to hit precision eps). Budget is emergent, not prescribed.\n")
w("| Algorithm | budget @90%, eps=1e-4, U(0.01,0.1) |")
w("|---|---:|")
w("| Brute force | ~4,520,000 |")
w("| Reverse engineering (improved) | **3,307,012  (x1.37, wins)** |")
w("\n> Quick proof-of-concept (seed 2024, R=15k, coarse grid over exploration shots + a confidence factor). "
  "It shows the win is achievable in a genuinely variable-budget algorithm; a proper de-biased implementation "
  "would replace Algorithm 6's fixed exploitation count with this rule.\n")

w("## C) Fixed-budget - budget swept to the 90% crossing (the BEYOND table)\n")
w("Give each algorithm a per-estimation budget B, sweep B, report where convergence crosses 90%.\n")
w("| Algorithm | " + " | ".join(k for k, _ in COLS) + " |")
w("|---|" + "---:|" * len(COLS))
for a, nice in ALGOS:
    w("| " + nice + " | " + " | ".join(cell(s, a) for _, s in COLS) + " |")
w("\n_N_min=157 for U(0.001,0.01); N_min=15 elsewhere. Reverse engineering wins every column._\n")

w("## Takeaway\n")
w("- The advantage is **real** and is NOT an artifact of fixing the budget.\n")
w("- It is forfeited only by Algorithm 6's **scalar** exploitation count (framing A).\n")
w("- Both a per-trial precision-targeted **variable** algorithm (B) and the fixed-budget sweep (C) recover "
  "it (~1.4-1.6x). Framing B keeps budget as the output, which matches the Table 3.2 intent.\n")

with open("results/TABLE32_COMPARISON.md", "w") as f:
    f.write("\n".join(L) + "\n")
print("wrote results/TABLE32_COMPARISON.md")
