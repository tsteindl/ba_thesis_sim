"""Diagnostic figures for results/ (main-thesis reporting candidate 4).

Reads diagnostics_by_point.csv only -- no simulation. Two panels per figure row: how much budget the
exploration phase costs, and how well the depth it returns tracks N_opt, both as the available budget
grows. Budgets are normalised per scenario (B / B_median) so scenarios of wildly different scale sit
on one axis.

    python analysis/figures.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qmetrology import manifest as M
from pipeline_io import path

COL = {"brute": "#7f7f7f", "linear": "#2ca02c", "binary_deep": "#ff7f0e",
       "reverse_eng_risk": "#1f77b4"}
MARK = {"brute": "o", "linear": "^", "binary_deep": "D", "reverse_eng_risk": "s"}
RC = {"figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
      "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    p = path("diagnostics_by_point.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = load()
    if not rows:
        print("no diagnostics_by_point.csv — run the sweep first")
        return
    sids = [s["id"] for s in M.SCENARIOS if any(r["scenario_id"] == s["id"] for r in rows)]
    with plt.rc_context(RC):
        # ---- 1. trends, pooled over scenarios on a normalised budget axis --------------------
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for algo in M.ORDER:
            xs, sh, gr, ov = [], [], [], []
            for sid in sids:
                rs = sorted([r for r in rows if r["scenario_id"] == sid
                             and r["algorithm"] == algo], key=lambda r: f(r["budget"]))
                if not rs:
                    continue
                b = np.array([f(r["budget"]) for r in rs])
                xs += list(b / np.median(b))
                sh += [f(r["exploration_share_median"]) for r in rs]
                gr += [f(r["guess_ratio_median"]) for r in rs]
                ov += [f(r["star_overshoot"]) for r in rs]
            if not xs:
                continue
            o = np.argsort(xs)
            x = np.array(xs)[o]
            for ax, y, lab in ((axes[0], np.array(sh)[o], "median exploration budget share"),
                               (axes[1], np.array(gr)[o], r"median $N_{guess}/N_{opt}$"),
                               (axes[2], np.array(ov)[o], r"final overshoot $P(N_*>N_{opt})$")):
                m = np.isfinite(y)
                if not m.any():
                    continue
                # bin on the normalised budget axis so the panel shows a trend, not a scatter
                edges = np.geomspace(max(x[m].min(), 1e-3), x[m].max(), 9)
                idx = np.clip(np.digitize(x[m], edges) - 1, 0, len(edges) - 2)
                cx = np.sqrt(edges[:-1] * edges[1:])
                cy = np.array([np.nanmedian(y[m][idx == i]) if (idx == i).any() else np.nan
                               for i in range(len(cx))])
                ax.plot(cx, cy, MARK[algo] + "-", color=COL[algo], ms=4, lw=1.4,
                        label=M.ALGORITHMS[algo]["label"])
                ax.set_ylabel(lab)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel(r"budget / scenario median budget")
        axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle("Exploration cost and depth quality vs. available budget "
                     f"({len(sids)} scenarios, held-out seed {M.SEED_TEST})", fontsize=9)
        fig.tight_layout()
        fig.savefig(path("fig_diagnostics_vs_budget.png"), bbox_inches="tight")
        plt.close(fig)

        # ---- 2. N_guess vs N_star, per algorithm -------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for algo in M.ORDER:
            g = np.array([f(r["guess_ratio_median"]) for r in rows if r["algorithm"] == algo])
            s = np.array([f(r["star_ratio_median"]) for r in rows if r["algorithm"] == algo])
            m = np.isfinite(g) & np.isfinite(s)
            if m.any():
                axes[0].scatter(g[m], s[m], s=12, alpha=0.55, color=COL[algo],
                                marker=MARK[algo], label=M.ALGORITHMS[algo]["label"])
            rate = np.array([f(r["rate"]) for r in rows if r["algorithm"] == algo])
            ms = np.isfinite(s) & np.isfinite(rate)
            if ms.any():
                axes[1].scatter(s[ms], 100 * rate[ms], s=12, alpha=0.55, color=COL[algo],
                                marker=MARK[algo])
        lim = axes[0].get_xlim()
        axes[0].plot(lim, lim, "k:", lw=0.8)
        axes[0].axhline(1.0, color="k", lw=0.6, ls="--")
        axes[0].axvline(1.0, color="k", lw=0.6, ls="--")
        axes[0].set_xlabel(r"median $N_{guess}/N_{opt}$")
        axes[0].set_ylabel(r"median $N_*/N_{opt}$")
        axes[0].legend(fontsize=7, loc="best")
        axes[1].set_xlabel(r"median $N_*/N_{opt}$")
        axes[1].set_ylabel("convergence (%)")
        fig.suptitle("What the safeguard does to the exploration's guess", fontsize=9)
        fig.tight_layout()
        fig.savefig(path("fig_guess_vs_final_depth.png"), bbox_inches="tight")
        plt.close(fig)
    print("wrote", path("fig_diagnostics_vs_budget.png"), "and",
          path("fig_guess_vs_final_depth.png"))


if __name__ == "__main__":
    main()
