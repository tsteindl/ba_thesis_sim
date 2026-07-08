"""Paste-ready LaTeX for the broad-prior convergence table (U(0.01, pi/2)),
from results/broad_dist.csv. Writes results/broad_dist_latex.md.
    python analysis/broad_latex.py
"""
import csv

TAG, LO, HI = "pi/2", 15000, 900000
rows = [r for r in csv.DictReader(open("results/broad_dist.csv"))
        if r["phi_max"] == TAG and LO <= int(r["budget"]) <= HI]

L = [r"\begin{table}[ht]", r"\centering",
     r"\caption{Convergence under a broad uniform prior $\phi\sim\mathcal{U}(0.01,\pi/2)$ at "
     r"$\epsilon=10^{-3}$: average share of simulations that converge, by budget. Linear search "
     r"dominates at every budget while reverse engineering plateaus near $38\%$.}",
     r"\label{tab:broad-pi2}", r"\begin{tabular}{rcccc}", r"\toprule",
     r"Budget $C=N\cdot m$ & Brute force & Linear search & Binary search & Reverse engineering \\",
     r"\midrule"]
for r in rows:
    b = f"{int(r['budget']):,}".replace(",", "{,}")
    L.append(f"{b} & {float(r['brute']):.1f}\\% & \\textbf{{{float(r['linear']):.1f}\\%}} "
             f"& {float(r['binary']):.1f}\\% & {float(r['reverse_eng']):.1f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = "\n".join(L)
open("results/broad_dist_latex.md", "w").write("# Broad-distribution LaTeX table\n\n```latex\n" + out + "\n```\n")
print(out)
