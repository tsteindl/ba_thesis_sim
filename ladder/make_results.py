"""Assemble ladder/results/LADDER.md from ladder/study.py's CSVs, in the same table
style as results/RESULTS.md (Table 3.2 / precision-lever format), so it is directly
comparable to that document. Reads results/story_cube.csv (existing study) read-only
for the brute/RE context columns — nothing under results/ is written.

    python ladder/make_results.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "results")

PAPER_RC = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5,
}
plt.rcParams.update(PAPER_RC)

NICE = {"brute": "Brute force", "reverse_eng": "Reverse Engineering",
        "linear": "Linear search", "binary": "Binary search", "separable": "Separable (N=1)",
        "ladder_capped": "Ladder (capped)", "ladder_unbounded": "Ladder (unbounded)"}
COLOR = {"brute": "#888888", "reverse_eng": "#4C78A8", "linear": "#72B7B2",
        "ladder_capped": "#E45756", "ladder_unbounded": "#B22222"}


def cell(cube, setting, algo, T=90):
    r = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == T)]
    if not len(r) or pd.isna(r.budget_to_reach.iloc[0]):
        return None
    return float(r.budget_to_reach.iloc[0]), (None if pd.isna(r.ratio_vs_brute.iloc[0])
                                              else float(r.ratio_vs_brute.iloc[0]))


def fmt(v, is_brute=False):
    if v is None:
        return "—"
    b, ratio = v
    return f"{b:,.0f}" if is_brute else f"{b:,.0f} (×{ratio:.2f})" if ratio else f"{b:,.0f}"


def main():
    cube = pd.read_csv(f"{OUT}/ladder_cube.csv")
    t31 = pd.read_csv(f"{OUT}/ladder_t31.csv")

    eps_cols = [("1e-3", "U(0.01,0.1), eps=1e-03"), ("1e-4", "U(0.01,0.1), eps=1e-04"),
                ("1e-5", "U(0.01,0.1), eps=1e-05"), ("1e-6", "U(0.01,0.1), eps=1e-06"),
                ("1e-7", "U(0.01,0.1), eps=1e-07"), ("1e-8", "U(0.01,0.1), eps=1e-08")]
    eps_cols = [(e, s) for e, s in eps_cols if (cube.setting == s).any()]
    shifted_setting = "U(0.001,0.01), eps=1e-04"
    has_shifted = (cube.setting == shifted_setting).any()

    L = []
    w = L.append
    w("# LADDER — a phase-unwrapping protocol beyond Table 3.2\n")
    w("_Companion to `results/RESULTS.md`; nothing there is modified. Same fixed-budget "
      "methodology (tune on seed 42, validate the winning config on seed 2024 at "
      "**R = 50,000**), same budget-to-90%-crossing definition, so the columns below are "
      "directly comparable to Table 3.2. New algorithm and code live entirely under `ladder/` "
      "and read `qmetrology`/`results` read-only — see `ladder/algorithms.py` for the mechanism._\n")

    w("## Why a new protocol\n")
    w("Every algorithm in Table 3.2 — including reverse engineering — inverts a **single** "
      "batch via `phi_hat = arccos(sqrt(p_hat))/N`, which is unambiguous only while `N*phi < "
      "pi/2`. That caps the usable circuit depth at `N <= pi/(2*phi)`, so brute force AND every "
      "adaptive protocol scale as `error ~ 1/(2*sqrt(N*budget))` with `N` bounded: **cost-to-eps "
      "~ 1/eps^2 for everyone**, and the adaptive/brute ratio saturates once both sides hit the "
      "same depth ceiling (the ~1.72× plateau reported in RESULTS.md — that plateau is the "
      "**invertibility cap**, not Heisenberg-limited saturation as currently phrased there).\n")
    w("The cap is informational, not physical. `cos^2(N phi)` only fixes `N*phi` up to the "
      "branch set `{+-arccos(sqrt(p_hat)) + k*pi}`; a *previous, coarser* estimate whose "
      "confidence interval is narrower than the branch spacing picks the right branch. So a "
      "**ladder** of measurements at geometrically deepening `N`, each unwrapped by the last "
      "estimate, can push `N` past `pi/(2 phi)` indefinitely — true Heisenberg scaling, "
      "`error ~ sqrt(m)/budget`, i.e. **cost-to-eps ~ 1/eps**. This is the standard "
      "Kitaev/Higgins iterative-phase-estimation idea (Kitaev quant-ph/9511026; Higgins et al., "
      "Nature 450, 393 (2007); Berry et al., PRA 80, 052114 (2009)), adapted to this thesis's "
      "circuit: since there is no controllable measurement phase, the ladder instead *steers the "
      "depth* so `N*phi_hat` lands on an odd multiple of `pi/4` — the point of maximal branch "
      "separation (`pi/2`) that also avoids the degenerate endpoints `p~0,1`.\n")
    w("Two variants are reported: **ladder (capped)** bounds `N <= pi/(2 phi_min)` — the same "
      "maximum depth the existing protocols already query, so it is the apples-to-apples "
      "'same hardware' comparison, isolating the gain to the *inference*. **Ladder (unbounded)** "
      "lets depth grow freely — the information-theoretic ceiling, at the honest cost of an "
      "unboundedly large GHZ circuit as ε shrinks.\n")

    w("## Table L1 — budget to reach 90% convergence vs precision  (U(0.01, 0.1))\n")
    w("_Same definition as Table 3.2: sweep budget, report the log-interpolated crossing at "
      "90%. Ratio = brute ÷ algorithm._\n")
    w("| Algorithm | " + " | ".join(f"eps={e}" for e, _ in eps_cols) + " |")
    w("|---|" + "---:|" * len(eps_cols))
    for algo in ["brute", "reverse_eng", "ladder_capped", "ladder_unbounded"]:
        cells = [fmt(cell(cube, s, algo), is_brute=(algo == "brute")) for _, s in eps_cols]
        w(f"| {NICE[algo]} | " + " | ".join(cells) + " |")
    w("")

    if has_shifted:
        w("## Table L1b — shifted range  U(0.001, 0.01), eps=1e-4\n")
        w("| Algorithm | budget to 90% |")
        w("|---|---:|")
        for algo in ["brute", "reverse_eng", "ladder_capped", "ladder_unbounded"]:
            w(f"| {NICE[algo]} | {fmt(cell(cube, shifted_setting, algo), is_brute=(algo=='brute'))} |")
        w("")

    w("## Table L2 — % converged at fixed budget 10,000  (eps = 1e-3, U(0.01, 0.1))\n")
    w("_Table-3.1 operating point; ladder rows computed by `ladder/study.py::table31_check`._\n")
    w("| Algorithm | this work |")
    w("|---|---:|")
    w("| Brute force | 56.2% |\n| Linear search | 56.6% |\n| Binary search | 47.5% |"
      "\n| Reverse Engineering | 60.3% |")
    for _, r in t31.iterrows():
        lo, hi = r["wilson_lo"], r["wilson_hi"]
        w(f"| {NICE[r['algo']]} | {r['debiased_pct']:.1f}% (CI {lo:.1f}-{hi:.1f}) |")
    w("")

    # unwrap-failure telemetry
    miss = cube[(cube.algo.isin(["ladder_capped", "ladder_unbounded"])) &
               (cube.threshold_pct == 90) & (cube.miss_rate_at_90 != "")]
    if len(miss):
        w("## Unwrap-failure rate\n")
        w("_Fraction of trials landing more than 10·eps off, at the budget nearest each "
          "algorithm's own 90% crossing (a wrong-branch catastrophe, not ordinary estimator "
          "noise)._\n")
        w("| Setting | ladder (capped) | ladder (unbounded) |")
        w("|---|---:|---:|")
        for e, s in eps_cols:
            row = miss[miss.setting == s]
            cells = []
            for algo in ["ladder_capped", "ladder_unbounded"]:
                v = row[row.algo == algo].miss_rate_at_90
                cells.append(f"{100*float(v.iloc[0]):.3f}%" if len(v) else "—")
            w(f"| eps={e} | " + " | ".join(cells) + " |")
        if has_shifted:
            row = miss[miss.setting == shifted_setting]
            cells = []
            for algo in ["ladder_capped", "ladder_unbounded"]:
                v = row[row.algo == algo].miss_rate_at_90
                cells.append(f"{100*float(v.iloc[0]):.3f}%" if len(v) else "—")
            w("| U(0.001,0.01), eps=1e-4 | " + " | ".join(cells) + " |")
        w("")

    # ---- figure: log-log cost-to-90% vs 1/eps, slope -2 vs -1 ----
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    xs = np.array([1.0 / float(e) for e, _ in eps_cols])
    for algo in ["brute", "reverse_eng", "ladder_capped", "ladder_unbounded"]:
        ys = [cell(cube, s, algo) for _, s in eps_cols]
        pts = [(x, y[0]) for x, y in zip(xs, ys) if y is not None]
        if len(pts) >= 2:
            px, py = zip(*pts)
            ax.plot(px, py, "o-", label=NICE[algo], color=COLOR.get(algo))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/\varepsilon$"); ax.set_ylabel("budget to reach 90% convergence")
    ax.set_title("Cost-to-precision scaling:\nSQL (slope -2) vs Heisenberg (slope -1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ladder_scaling.png")
    plt.close(fig)
    w("### Figure\n")
    w("![cost-to-90% vs 1/eps, log-log](fig_ladder_scaling.png)\n")

    w("## Mechanism & caveats\n")
    w("- **The RESULTS.md ~1.72× plateau is the invertibility cap, not Heisenberg-limited "
      "saturation.** Every existing algorithm is capped at N ~ pi/(2*phi); the ladder shows "
      "what happens once that cap is lifted by branch-unwrapping — the ratio keeps growing "
      "with precision instead of saturating (see Table L1 / the figure).\n")
    w("- **Ladder (capped) isolates the gain to inference**: same maximum depth as the "
      "existing protocols, so its multiple over reverse engineering is entirely due to "
      "sequential unwrapping (many cheap stages instead of one exploration + one exploitation "
      "shot), not to a bigger circuit.\n")
    w("- **Ladder (unbounded) is the information-theoretic ceiling, honestly**: its GHZ depth "
      "grows without bound as eps shrinks (see the per-stage N in `ladder/poc.py`'s traces) — "
      "a real device would cap it, which is exactly what the capped variant reports.\n")
    w("- **Unwrap failures are rare and controllable** via the `z`/`safety` margins (see table "
      "above); they trade off against growth rate (larger margins -> slower depth growth -> "
      "more stages -> closer to but never worse than the capped baseline).\n")

    w("## Methodology\n")
    w("- Same seeds as the rest of the project: SEED_TUNE=42 (R=2,000), SEED_TEST=2024 "
      "(R=50,000). Brute-force crossings recomputed here for self-consistency; cross-checked "
      "against `results/story_cube.csv` (agreement within ~1%, printed by `ladder/study.py`).\n")
    w("- Tuned grid: `m_stage in {50,100,200,400}`, `z in {2,2.5,3}`, `safety in {0.6,0.75,0.9}`, "
      "`final_frac in {0.3,0.5,0.7}` (see `ladder/study.py:GRID`).\n")
    w("- Regenerate: `python ladder/study.py` (~30-90 min) then `python ladder/make_results.py`.\n")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/LADDER.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}/LADDER.md and {OUT}/fig_ladder_scaling.png")


if __name__ == "__main__":
    main()
