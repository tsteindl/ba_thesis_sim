"""Build the result figures and results/STORY_TABLES.md from the sweep CSVs.

Figures: fig_story (convergence vs budget with the 90% budget-ratio arrow), fig_error
(estimator error + uncertainty), fig_pareto (budget-convergence frontier + dominance strip),
fig_precision (budget ratio at 90% vs precision). Style is centralised in PAPER_RC.
    python analysis/render_story.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# paper-adaptable style: change these and re-run
PAPER_RC = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 4,
}
plt.rcParams.update(PAPER_RC)

# consistent per-algorithm identity across every figure/table
ALGOS = ["brute", "separable", "linear", "binary", "reverse_eng"]
ADAPT = ["linear", "binary", "reverse_eng"]
NICE = {"brute": "Brute force (baseline)", "separable": "Separable (N=1)",
        "linear": "Linear search", "binary": "Binary search",
        "reverse_eng": "Reverse engineering", "best_adaptive": "Best adaptive"}
COL = {"brute": "#7f7f7f", "separable": "#8c564b", "linear": "#2ca02c",
       "binary": "#ff7f0e", "reverse_eng": "#1f77b4"}
MARK = {"brute": "o", "separable": "v", "linear": "^", "binary": "D", "reverse_eng": "s"}
HEAD = "U(0.01,0.1), eps=1e-04"        # headline operating point
OP = 90                                # headline threshold (%)
EPS_TAG = {"1e-03": "10⁻³", "1e-04": "10⁻⁴", "1e-05": "10⁻⁵",
           "1e-06": "10⁻⁶", "1e-07": "10⁻⁷", "1e-08": "10⁻⁸"}
ALL_EPS = ["1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]


def _eps_tags_present(df):
    """The eps tags actually swept for the headline U(0.01,0.1) prior, in order."""
    return [t for t in ALL_EPS if f"U(0.01,0.1), eps={t}" in set(df.setting)]


# tables
def _r(x):  # ratio cell
    return "—" if pd.isna(x) else (f"**{x:.2f}×**" if x >= 1.05 else (f"{x:.2f}×" if x > 0.95 else f"_{x:.2f}×_"))


def _b(x):  # budget cell
    return "—" if pd.isna(x) else f"{x:,.0f}"


def _winner(cube, setting, thr=OP):
    s = cube[(cube.setting == setting) & (cube.threshold_pct == thr) & cube.algo.isin(ADAPT)].dropna(subset=["ratio_vs_brute"])
    return "—" if not len(s) else NICE[s.loc[s.ratio_vs_brute.idxmax(), "algo"]]


def table_A(cube, out):
    print(f"\n## Table A — the field at one operating point ($\\phi\\sim U(0.01,0.1)$, ε=10⁻⁴, reach {OP}% convergence)\n", file=out)
    print("| Algorithm | Budget to reach 90% | Ratio vs brute |\n|---|---:|---:|", file=out)
    sub = cube[(cube.setting == HEAD) & (cube.threshold_pct == OP)]
    for a in ALGOS:
        r = sub[sub.algo == a]
        if len(r):
            print(f"| {NICE[a]} | {_b(r.budget_to_reach.iloc[0])} | {_r(r.ratio_vs_brute.iloc[0])} |", file=out)
    print("\n_Ratio > 1 = needs that many times **less** budget than brute for the same 90% convergence "
          "(**bold** wins, _italic_ loses). De-biased: tuned on seed 42, validated on seed 2024._\n", file=out)


def table_B(cube, out):
    print("\n## Table B — the main lever is precision ($\\phi\\sim U(0.01,0.1)$, reach 90%)\n", file=out)
    print("| Precision ε | Linear | Binary | Reverse eng. |\n|---|---:|---:|---:|", file=out)
    for tag in _eps_tags_present(cube):
        s = f"U(0.01,0.1), eps={tag}"
        cells = []
        for a in ADAPT:
            row = cube[(cube.setting == s) & (cube.algo == a) & (cube.threshold_pct == OP)]
            cells.append(_r(row.ratio_vs_brute.iloc[0]) if len(row) else "—")
        if any(c != "—" for c in cells):
            print(f"| {EPS_TAG[tag]} | " + " | ".join(cells) + " |", file=out)
    print("\n_The advantage grows with required precision — a tighter tolerance rewards the larger circuit "
          "depth N that the adaptive search selects._\n", file=out)


def table_C(cube, out):
    print("\n## Table C — robustness across thresholds (best adaptive)\n", file=out)
    thr = sorted(cube.threshold_pct.unique())
    print("| Scenario | " + " | ".join(f"{t}%" for t in thr) + " | winner |", file=out)
    print("|---|" + "---:|" * len(thr) + "---|", file=out)
    for s in cube.setting.unique():
        cells = [_r(cube[(cube.setting == s) & (cube.algo == "best_adaptive") & (cube.threshold_pct == t)].ratio_vs_brute.iloc[0])
                 for t in thr]
        print(f"| {s} | " + " | ".join(cells) + f" | {_winner(cube, s)} |", file=out)
    print("\n_Ratio ≥ 1 in every cell ⇒ the budget advantage survives at every reliability target._\n", file=out)


# figures
def _curve(curves, setting, algo):
    d = curves[(curves.setting == setting) & (curves.algo == algo)].sort_values("budget")
    return d.budget.to_numpy(float), 100 * d.rate.to_numpy(float)


def fig_story(cube, curves):
    """S-curves for U(0.01,0.1) across precisions, in chronological ε order (10^-3 -> ...),
    each with the 90% budget-ratio arrow. Budget stays on a log axis (see caption)."""
    order = [f"U(0.01,0.1), eps=1e-0{k}" for k in (3, 4, 5, 6)]
    settings = [s for s in order if s in set(curves.setting)]
    fig, axes = plt.subplots(1, len(settings), figsize=(5.2 * len(settings), 4.6), squeeze=False)
    for i, (ax, s) in enumerate(zip(axes[0], settings)):
        for a in ALGOS:
            x, y = _curve(curves, s, a)
            if len(x):
                ax.plot(x, y, MARK[a] + "-", color=COL[a], label=NICE[a],
                        alpha=0.95 if a in ("brute", "reverse_eng") else 0.55)
        ax.axhline(OP, color="k", ls=":", lw=0.8)
        # horizontal budget-ratio arrow at the 90% line: reverse-eng crossing <-> brute crossing
        bb = cube[(cube.setting == s) & (cube.algo == "brute") & (cube.threshold_pct == OP)].budget_to_reach
        ba = cube[(cube.setting == s) & (cube.algo == "reverse_eng") & (cube.threshold_pct == OP)].budget_to_reach
        if len(bb) and len(ba) and bb.notna().all() and ba.notna().all():
            b0, a0 = float(bb.iloc[0]), float(ba.iloc[0])
            ax.annotate("", xy=(a0, OP), xytext=(b0, OP),
                        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
            ax.text(np.sqrt(a0 * b0), OP + 4.5, f"{b0 / a0:.1f}× less budget",
                    color="crimson", ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        eps_tag = EPS_TAG[s.split("eps=")[1]]
        ax.set(xscale="log", ylim=(28, 103), xlabel="budget  $C = N\\cdot m$  (log scale)",
               title=f"$\\phi \\sim U(0.01, 0.1)$,  ε = {eps_tag}")
        if i == 0:
            ax.set_ylabel("% of trials converged  ($|\\hat\\phi-\\phi|<\\varepsilon$)")
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Adaptive estimation reaches the same reliability at lower budget — the advantage opens up "
                 "as precision tightens (de-biased, R = 40,000 trials/point; arrow = budget ratio at 90%)", fontsize=10.5)
    fig.text(0.5, 0.005, "Budget is log-scaled, so the small horizontal shift between curves is a "
             "multiplicative factor (the labelled arrow), not a small additive gap.",
             ha="center", fontsize=8.5, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("results/fig_story.png"); plt.close(fig)
    print("wrote results/fig_story.png")

def fig_story_small(cube, curves):
    order = [f"U(0.01,0.1), eps=1e-0{k}" for k in (3, 4, 5)]
    settings = [s for s in order if s in set(curves.setting)]
    fig, axes = plt.subplots(1, len(settings), figsize=(5.2 * len(settings), 4.6), squeeze=False)
    for i, (ax, s) in enumerate(zip(axes[0], settings)):
        for a in ALGOS:
            x, y = _curve(curves, s, a)
            if len(x):
                ax.plot(x, y, MARK[a] + "-", color=COL[a], label=NICE[a],
                        alpha=0.95 if a in ("brute", "reverse_eng") else 0.55)
        ax.axhline(OP, color="k", ls=":", lw=0.8)
        # horizontal budget-ratio arrow at the 90% line: reverse-eng crossing <-> brute crossing
        bb = cube[(cube.setting == s) & (cube.algo == "brute") & (cube.threshold_pct == OP)].budget_to_reach
        ba = cube[(cube.setting == s) & (cube.algo == "reverse_eng") & (cube.threshold_pct == OP)].budget_to_reach
        if len(bb) and len(ba) and bb.notna().all() and ba.notna().all():
            b0, a0 = float(bb.iloc[0]), float(ba.iloc[0])
            ax.annotate("", xy=(a0, OP), xytext=(b0, OP),
                        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
            ax.text(np.sqrt(a0 * b0), OP + 4.5, f"{b0 / a0:.1f}× less budget",
                    color="crimson", ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        eps_tag = EPS_TAG[s.split("eps=")[1]]
        ax.set(xscale="log", ylim=(28, 103), xlabel="budget  $C = N\\cdot m$  (log scale)",
               title=f"$\\phi\\sim U(0.01, 0.1)$,  ε = {eps_tag}")
        if i == 0:
            ax.set_ylabel("% of trials converged  ($|\\hat\\phi-\\phi|<\\varepsilon$)")
        ax.legend(fontsize=8.5, loc="lower right")
    # fig.suptitle("Adaptive estimation reaches the same reliability at lower budget — the advantage opens up "
                #  "as precision tightens (de-biased, R = 40,000 trials/point; arrow = budget ratio at 90%)", fontsize=10.5)
    fig.suptitle("Budet comparison (log-scaled) between adaptive and baseline strategies for increasing precision $\\epsilon$"
                 " (R = 40,000 trials/point)", fontsize=16)
    # fig.text(0.5, 0.005, "Budget is log-scaled, so the small horizontal shift between curves is a "
            #  "multiplicative factor (the labelled arrow), not a small additive gap.",
            #  ha="center", fontsize=8.5, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("results/fig_story_small.png"); plt.close(fig)
    print("wrote results/fig_story_small.png")

def fig_pareto(curves, setting=HEAD):
    """Budget-convergence Pareto frontier + a strip of which algorithm is Pareto-optimal."""
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(7.2, 5.4), height_ratios=[6, 1], sharex=True)
    grid = np.geomspace(*(_curve(curves, setting, "brute")[0][[0, -1]]), 300)
    interp = {}
    for a in ALGOS:
        x, y = _curve(curves, setting, a)
        if len(x):
            ax.plot(x, y, MARK[a] + "-", color=COL[a], label=NICE[a], alpha=0.85)
            interp[a] = np.interp(np.log(grid), np.log(x), y, left=y[0], right=y[-1])
    # combined frontier = best convergence achievable at each budget, and who owns it
    stack = np.vstack([interp[a] for a in ALGOS])
    front = stack.max(0)
    owner = np.array(ALGOS)[stack.argmax(0)]
    ax.plot(grid, front, "k--", lw=1.4, alpha=0.7, label="Pareto frontier (best)")
    ax.set(xscale="log", ylim=(28, 103), ylabel="% converged",
           title=f"Budget–convergence Pareto frontier   ({setting.replace('eps=1e-04','ε=10⁻⁴')}, R=40,000/point)")
    ax.legend(fontsize=8, loc="lower right")
    # dominance strip — only where the frontier is not yet saturated (>=99.5% is a numerical tie)
    meaningful = front < 99.5
    for a in ALGOS:
        m = (owner == a) & meaningful
        if m.any():
            axd.fill_between(grid, 0, 1, where=m, color=COL[a], step="mid")
    axd.axvspan(grid[meaningful][-1] if meaningful.any() else grid[-1], grid[-1],
                color="0.9", label="saturated (all ~100%)")
    axd.set(yticks=[], xlabel="budget  $C = N\\cdot m$", ylim=(0, 1))
    axd.set_ylabel("owns\nfrontier", fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig("results/fig_pareto.png"); plt.close(fig)
    print("wrote results/fig_pareto.png")


def fig_error():
    """Estimator error vs budget: median line + 25–75% uncertainty band per algorithm, across the
    full precision progression (down to ε=10⁻⁸). Auto-lays-out in a grid. Reads error_curves.csv."""
    if not os.path.exists("results/error_curves.csv"):
        print("(no error_curves.csv — run analysis/error_curves.py to add fig_error.png)")
        return
    ec = pd.read_csv("results/error_curves.csv")
    order = [f"U(0.01,0.1), eps={t}" for t in ALL_EPS]
    settings = [s for s in order if s in set(ec.setting)]
    if not settings:
        return
    ncol = min(3, len(settings))
    nrow = int(np.ceil(len(settings) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow), squeeze=False)
    axflat = axes.flatten()
    for i, s in enumerate(settings):
        ax = axflat[i]
        d0 = ec[ec.setting == s]
        eps = float(s.split("eps=")[1])
        for a in ALGOS:
            d = d0[d0.algo == a].sort_values("budget")
            if not len(d):
                continue
            ax.plot(d.budget, d.err_p50, MARK[a] + "-", color=COL[a], label=NICE[a],
                    alpha=0.95 if a in ("brute", "reverse_eng") else 0.6)
            ax.fill_between(d.budget, d.err_p25, d.err_p75, color=COL[a], alpha=0.16, linewidth=0)
        ax.axhline(eps, color="k", ls="--", lw=1.0)
        ax.text(d0.budget.min(), eps * 1.15, "ε (converged below)", fontsize=7.5, va="bottom")
        ax.set(xscale="log", yscale="log", xlabel="budget  $C = N\\cdot m$",
               title=f"$\\phi \\sim U(0.01, 0.1)$,  ε = {EPS_TAG[s.split('eps=')[1]]}")
        if i % ncol == 0:
            ax.set_ylabel("error $|\\hat\\phi-\\phi|$  (median, 25–75%)")
        ax.legend(fontsize=7.5, loc="upper right")
    for j in range(len(settings), len(axflat)):
        axflat[j].axis("off")
    fig.suptitle("Estimator error and its uncertainty vs budget. "
                 "(band = 25–75% over $\\phi$ draws, de-biased, R = 40,000 trials/point)", fontsize=16)
    fig.tight_layout()
    fig.savefig("results/fig_error.png"); plt.close(fig)
    print("wrote results/fig_error.png")

def fig_error_small():
    """Estimator error vs budget: median line + 25–75% uncertainty band per algorithm, across the
    full precision progression (down to ε=10⁻⁸). Auto-lays-out in a grid. Reads error_curves.csv."""
    if not os.path.exists("results/error_curves.csv"):
        print("(no error_curves.csv — run analysis/error_curves.py to add fig_error.png)")
        return
    ec = pd.read_csv("results/error_curves.csv")
    # order = [f"U(0.01,0.1), eps={t}" for t in ALL_EPS]
    order = [f"U(0.01,0.1), eps=1e-0{k}" for k in (3, 4, 5)]
    settings = [s for s in order if s in set(ec.setting)]
    if not settings:
        return
    ncol = min(3, len(settings))
    nrow = int(np.ceil(len(settings) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow), squeeze=False)
    axflat = axes.flatten()
    for i, s in enumerate(settings):
        ax = axflat[i]
        d0 = ec[ec.setting == s]
        eps = float(s.split("eps=")[1])
        for a in ALGOS:
            d = d0[d0.algo == a].sort_values("budget")
            if not len(d):
                continue
            ax.plot(d.budget, d.err_p50, MARK[a] + "-", color=COL[a], label=NICE[a],
                    alpha=0.95 if a in ("brute", "reverse_eng") else 0.6)
            ax.fill_between(d.budget, d.err_p25, d.err_p75, color=COL[a], alpha=0.16, linewidth=0)
        ax.axhline(eps, color="k", ls="--", lw=1.0)
        ax.text(d0.budget.min(), eps * 1.15, "ε (converged below)", fontsize=7.5, va="bottom")
        ax.set(xscale="log", yscale="log", xlabel="budget  $C = N\\cdot m$",
               title=f"$\\phi \\sim U(0.01, 0.1)$,  ε = {EPS_TAG[s.split('eps=')[1]]}")
        if i % ncol == 0:
            ax.set_ylabel("error $|\\hat\\phi-\\phi|$  (median, 25–75%)")
        ax.legend(fontsize=7.5, loc="upper right")
    for j in range(len(settings), len(axflat)):
        axflat[j].axis("off")
    fig.suptitle("Estimator error and its uncertainty vs budget. "
                 "(band = 25–75% over $\\phi$ draws, R = 40,000 trials/point)", fontsize=16)
    fig.tight_layout()
    fig.savefig("results/fig_error_small.png"); plt.close(fig)
    print("wrote results/fig_error_small.png")


def fig_precision(cube):
    """Budget ratio @90% vs precision — the 'advantage grows with precision' lever."""
    tags = _eps_tags_present(cube)
    x = [int(t.split("e-")[1]) for t in tags]  # |exponent| so precision increases left -> right
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for a in ADAPT:
        y = []
        for tag in tags:
            row = cube[(cube.setting == f"U(0.01,0.1), eps={tag}") & (cube.algo == a) & (cube.threshold_pct == OP)]
            y.append(row.ratio_vs_brute.iloc[0] if len(row) and row.ratio_vs_brute.notna().all() else np.nan)
        ax.plot(x, y, MARK[a] + "-", color=COL[a], label=NICE[a])
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.text(x[0], 1.005, "brute-force parity", fontsize=8, color="k", va="bottom")
    ax.set(xticks=x, xticklabels=[EPS_TAG[t] for t in tags], xlabel="target precision ε  (→ finer)",
           ylabel="budget ratio vs brute  @90%")
    ax.set_title("Adaptive advantage grows with precision\n$\\phi\\sim U(0.01, 0.1)$  (R = 40,000/point)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/fig_precision.png"); plt.close(fig)
    print("wrote results/fig_precision.png")


def main():
    cube = pd.read_csv("results/story_cube.csv")
    curves = pd.read_csv("results/story_curves.csv")
    with open("results/STORY_TABLES.md", "w") as out:
        print("# Budget-ratio results — extensive de-biased sweep\n", file=out)
        print("_Generated by `analysis/extensive_sweep.py` + `analysis/render_story.py`. "
              "Metric = **budget ratio** `B_brute(p*)/B_algo(p*)`: how much less budget an algorithm needs to "
              "first reach convergence reliability p*. Every adaptive number is de-biased (grid-tuned on seed 42, "
              "winning config re-validated on independent seed 2024, R=40,000). Raw data: `story_curves.csv`, "
              "`story_cube.csv`._\n", file=out)
        table_A(cube, out)
        table_B(cube, out)
        table_C(cube, out)
        print("\n## Figures\n", file=out)
        print("![convergence-vs-budget S-curves with budget-ratio arrows](fig_story.png)\n", file=out)
        print("![estimator error and uncertainty vs budget](fig_error.png)\n", file=out)
        print("![budget-convergence Pareto frontier and dominance strip](fig_pareto.png)\n", file=out)
        print("![budget ratio at 90% vs precision](fig_precision.png)\n", file=out)
    print(open("results/STORY_TABLES.md").read())
    fig_story(cube, curves)
    fig_story_small(cube, curves)
    fig_error()
    fig_error_small()
    fig_pareto(curves)
    fig_precision(cube)


if __name__ == "__main__":
    main()
