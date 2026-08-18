"""Figure + summary table for the exploration-size parameterisation (reverse engineering).

Reads what analysis/re_share_sweep.py and its rho scan produced and turns them into the two objects
the thesis needs:

  results/fig_pilot_share.png     left : the tuned exploration size, read as a share of the budget,
                                        against the budget -- the reason a share is quoted instead
                                        of a shot count.
                                  right: regret of each *fixed* default against per-budget tuning,
                                        by budget regime -- the reason the share is 2%.
  results/pilot_share_summary.csv one row per candidate default, consumed by analysis/make_tex.py.

    python analysis/pilot_share_report.py
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import config as C

SHARE = "results/re_share.csv"
RHO = "results/re_share_rho.csv"
M200 = "results/re_share_m200.csv"


def n_min(phi_max):
    return max(1, int(np.floor(np.pi / (2 * phi_max))))


def load():
    base = pd.read_csv(SHARE)
    rho = pd.read_csv(RHO)
    m200 = pd.read_csv(M200)
    d = base.merge(rho, on=["setting", "budget"]).merge(m200, on=["setting", "budget"])
    d["N_min"] = d.phi_max.map(n_min)
    d["x"] = d.budget / (400.0 * d.N_min)          # <1 : the shot floor governs, not rho
    return d


def candidates(d):
    """(label, rate column, is_the_recommended_one) for every fixed default under test."""
    out = [(r"$m'=200$ (published)", d.m200_rate, False),
           ("$m'=45$ (best fixed $m'$)", d.m_fixed_rate, False)]
    for c in sorted([c for c in d.columns if c.startswith("rho_0")], key=lambda c: float(c[4:])):
        r = float(c[4:])
        out.append((rf"$\rho={r:g}$", d[c], abs(r - C.PILOT_SHARE) < 1e-9))
    out.append((r"$\rho=0.05$", d.rho_fixed_rate, abs(0.05 - C.PILOT_SHARE) < 1e-9))
    return out


def main():
    d = load()
    ref = d.m_tuned_rate

    # ---------- summary table ----------
    rows = []
    for label, col, rec in candidates(d):
        g = (col - ref).values
        rows.append({"default": label.replace("$", "").replace("\\rho", "rho").replace("m'", "m"),
                     "tex_label": label, "recommended": int(rec),
                     "mean": g.mean(), "median": float(np.median(g)),
                     "p05": float(np.percentile(g, 5)), "worst": g.min(),
                     "pct_worse_1pp": 100 * (g < -1).mean(), "pct_worse_5pp": 100 * (g < -5).mean()})
    rows.sort(key=lambda r: -r["mean"])
    with open("results/pilot_share_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}])
    print("wrote results/pilot_share_summary.csv")
    for r in rows:
        print(f"   {r['default']:26s} mean {r['mean']:+6.2f}  worst {r['worst']:+7.2f}  "
              f">1pp {r['pct_worse_1pp']:5.1f}%")

    # ---------- figure ----------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False, "font.size": 10})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # left: the exploration size itself, against the budget per unit depth (B/N_min), so that the
    # constant-share rule is a single straight line rather than one per scenario.
    # Restricted to the SAME live operating points as the right panel: outside them every arm is
    # pinned at 0 % or 100 %, the objective is flat, and the winning m' is arbitrary.
    #
    # The point is NOT that a fixed share matches the tuned size -- it does not, the tuned m' grows
    # roughly as (B/N_min)^0.3 while the share rule grows linearly. It is that the two error
    # directions are not symmetric: rho=2% ends up ABOVE the tuned size at large budget (paying a
    # bounded 1-2% of the budget for an over-precise pilot) whereas a fixed shot count falls ever
    # further BELOW it (an under-precise pilot, which the depth rule answers by backing off).
    win = pd.read_csv("results/story_winners.csv")
    win = win[win.algo == "reverse_eng_risk"].copy()
    live = set(zip(d.setting, d.budget))
    win = win[[(s, b) in live for s, b in zip(win.setting, win.budget)]]
    win["m"] = win.params.map(lambda p: json.loads(p).get("m_exploration", np.nan))
    win["N_min"] = win.phi_max.map(n_min)
    win["bn"] = win.budget / win.N_min
    axL.scatter(win.bn, win.m, s=9, color="#1f77b4", alpha=0.55, zorder=3,
                label="grid-tuned $m'$ (per budget)")
    xs = np.logspace(np.log10(win.bn.min()), np.log10(win.bn.max()), 50)
    axL.plot(xs, np.maximum(C.PILOT_FLOOR, C.PILOT_SHARE * xs), color="#d62728", lw=2.4, zorder=5,
             label=rf"$\rho={100*C.PILOT_SHARE:g}\%$ rule (floor {C.PILOT_FLOOR})")
    axL.axhline(200, color="#2ca02c", lw=1.8, ls="--", zorder=4, label="$m'=200$ (published)")
    axL.set(xscale="log", yscale="log", xlabel=r"budget per unit depth,  $B/N_{\min}$  (log)",
            ylabel="exploration size $m'$ (log)",
            title="The exploration size has to grow with the budget\n"
                  f"({win.setting.nunique()} scenarios, {len(win)} live points; "
                  f"tuned $m'$ spans {int(win.m.min()):,}–{int(win.m.max()):,})")
    axL.legend(loc="upper left", fontsize=8)

    # right: regret of each fixed default, by budget regime
    bins = np.logspace(-1, 11, 13)
    d["bin"] = np.digitize(d.x, bins)
    for label, col, rec in candidates(d):
        g = (col - ref)
        med = [g[d.bin == b].mean() for b in range(1, len(bins))]
        ctr = np.sqrt(bins[:-1] * bins[1:])
        style = dict(lw=2.6, color="#d62728", zorder=6) if rec else dict(lw=1.2, alpha=0.75)
        axR.plot(ctr, med, marker="o", ms=3, label=label, **style)
    axR.axhline(0, color="k", lw=0.8)
    axR.set(xscale="log", ylim=(-12, 2), xlabel=r"$B\,/\,(400\,N_{\min})$   (<1: the shot floor governs)",
            ylabel="mean regret vs per-budget tuning (pp)",
            title="Cost of never tuning the exploration size\n(546 operating points, 23 scenarios)")
    axR.legend(fontsize=8, ncol=2, loc="lower right")

    fig.suptitle("Reverse engineering: a budget share errs by over-funding the pilot, a shot count by under-funding it",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig("results/fig_pilot_share.png", dpi=140)
    print("wrote results/fig_pilot_share.png")


if __name__ == "__main__":
    main()
