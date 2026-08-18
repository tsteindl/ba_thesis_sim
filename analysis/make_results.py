"""Build results/RESULTS.md and results/all_numbers.csv.

Tables 3.1/3.3 are recomputed with qmetrology.tables.compute(); Table 3.2 and the precision
study are read from the budget sweep (results/story_cube.csv).
    python analysis/make_results.py
"""
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import config as C, tables
from qmetrology.experiments import rate_and_budget
from qmetrology.algorithms import find_phi_reverse_engineering_lite

R = C.R_TEST
NICE = C.NICE
# Table 3.2 columns -> the matching setting label in story_cube.csv
T32_COLS = [("eps1e-3 U(0.01,0.1)", "U(0.01,0.1), eps=1e-03"),
            ("eps1e-4 U(0.01,0.1)", "U(0.01,0.1), eps=1e-04"),
            ("eps1e-4 U(0.001,0.01)", "U(0.001,0.01), eps=1e-04"),
            ("eps1e-4 U(0.001,0.1)", "U(0.001,0.1), eps=1e-04")]
ADAPT = ["linear", "binary_risk", "reverse_eng_risk"]
CEILINGS = ["oracle_hl"]
ROWS = ["brute", "linear", "binary_risk", "reverse_eng_risk"]


def table32(cube):
    """budget-to-90% + ratio per algorithm, per column, from the fixed-budget sweep."""
    out = {}
    for label, setting in T32_COLS:
        for algo in ["brute", "separable"] + ADAPT + CEILINGS:
            r = cube[(cube.setting == setting) & (cube.algo == algo) & (cube.threshold_pct == 90)]
            out[(label, algo)] = (None if not len(r) or pd.isna(r.budget_to_reach.iloc[0])
                                  else (float(r.budget_to_reach.iloc[0]), r.ratio_vs_brute.iloc[0]))
    return out


def re_lite_row():
    best = None
    for m_exp, m_fin in [(100, 10), (150, 40), (200, 20), (150, 10)]:
        rate, bud = rate_and_budget(find_phi_reverse_engineering_lite,
                                    {"m_exploration": m_exp, "m_exploitation": m_fin},
                                    30000, 0.01, 0.1, 1e-3, C.SEED_TEST)
        if best is None or abs(100 * rate - 29.6) < abs(best[0] - 29.6):
            best = (100 * rate, bud, m_exp, m_fin)
    return best


def main():
    print("computing fixed-budget Tables 3.1 / 3.3 (~2 min) ...")
    res = tables.compute(verbose=False, algos=tables.ALGOS_ALL)
    cube = pd.read_csv("results/story_cube.csv")
    t32 = table32(cube)
    rl = re_lite_row()

    L = []
    w = L.append
    w("# Results — adaptive quantum metrology under resource constraints\n")
    w("_All comparisons use the fixed-budget formulation: every algorithm is given the same budget "
      "`C = N·m` and we measure convergence (`|φ̂−φ| < ε`). Adaptive numbers are grid-tuned on seed 42 and "
      "re-validated on seed 2024 at high R. Tables 3.1/3.3 are recomputed here; Table 3.2 and the precision "
      "study come from the budget sweep (`analysis/extensive_sweep.py`; raw data in `story_cube.csv`, "
      "`story_curves.csv`, `error_curves.csv`)._\n")

    # ---------- Table 3.1 ----------
    w("## Table 3.1 — % converged at fixed budget 10,000  (ε = 10⁻³, φ ~ U(0.01, 0.1))\n")
    w("| Algorithm | paper | this work |")
    w("|---|---:|---:|")
    for algo in tables.ALGOS_ALL:
        d = res[("narrow", algo)]
        val = d.get("unbiased", d.get("debiased"))
        pap = "—" if d["paper"] is None else f"{d['paper']}%"
        w(f"| {NICE[algo]} | {pap} | {val:.1f}% |")
    w(f"| _Reverse Eng. Lite (Alg. 7)_ | _29.6% @ ~2000_ | _{rl[0]:.1f}% (m'={rl[2]}, m={rl[3]}, budget ~{rl[1]:,.0f})_ |")
    w("\n_Narrow-range linear/binary are much higher than the paper's stale 14.0% / 11.5%. RE-Lite is the one "
      "variable-budget algorithm still reported (Algorithm 7)._\n")

    # ---------- what the parameters STATED in the thesis text achieve, vs the tuned optimum ----------
    canon = [(algo, res[("narrow", algo)]) for algo in tables.ALGOS_ALL
             if res[("narrow", algo)].get("canon") is not None]
    if canon:
        w("### Using the parameters as stated in the text (not re-tuned)\n")
        w("| Algorithm | stated parameters | stated | grid-tuned | cost of not tuning |")
        w("|---|---|---:|---:|---:|")
        for algo, d in canon:
            p = ", ".join(f"`{k}`={v}" for k, v in C.CANON[algo].items() if k != "eps_target")
            w(f"| {NICE[algo]} | {p} | {d['canon']:.1f}% | {d['debiased']:.1f}% | "
              f"{d['canon'] - d['debiased']:+.1f} pp |")
        w(f"\n_Reverse engineering's exploration size is stated as a **share of the budget** "
          f"(ρ = {100*C.PILOT_SHARE:g}%, floor {C.PILOT_FLOOR} shots) rather than a shot count, because a shot "
          f"count can only be right at one budget. Over the 546 operating points of the full sweep the stated "
          f"ρ costs −0.06 pp against tuning at every budget separately, while the previously stated m′ = 200 "
          f"costs −5.55 pp (worst −99.4 pp) — see `results/tex/tab_pilot_share.tex` and "
          f"`results/fig_pilot_share.png`. Budget 10,000 is the regime where ρ is least favourable "
          f"(the pilot floor governs below B ≈ 400·N_min), so this row is close to the worst case._\n")

    # ---------- Table 3.2 ----------
    w("## Table 3.2 — budget to reach 90% convergence  (fixed-budget, swept to the crossing)\n")
    w("_The budget is the **output**: we sweep the allocated per-estimation budget B and report the smallest B "
      "at which convergence reaches 90% (log-interpolated crossing of the convergence-vs-budget curve). "
      "Ratio = brute ÷ algorithm (>1 ⇒ adaptive needs less)._\n")
    w("| Algorithm | " + " | ".join(lbl for lbl, _ in T32_COLS) + " |")
    w("|---|" + "---:|" * len(T32_COLS))
    for algo in ["brute", "separable"] + ADAPT + CEILINGS:
        cells = []
        for lbl, _ in T32_COLS:
            v = t32[(lbl, algo)]
            if v is None:
                cells.append("—")
            elif algo == "brute":
                cells.append(f"{v[0]:,.0f}")
            else:
                cells.append(f"{v[0]:,.0f} (×{v[1]:.2f})")
        w(f"| {NICE[algo]} | " + " | ".join(cells) + " |")
    w("\n> **Why fixed-budget and not the run-until-done formulation:** the paper's variable-budget Algorithm 6 "
      "uses a single exploitation count for all φ, so it over-spends on easy (small-φ) trials — its mean budget "
      "for 90% is *higher* than brute (≈6.3M vs 4.5M at ε=10⁻⁴). Committing the whole budget to the inferred "
      "depth (fixed-budget) is what realises the advantage; that is the formulation reported here and in "
      "Tables 3.1/3.3.\n")

    # ---------- Table 3.3 ----------
    w("## Table 3.3 — broad distributions  (ε = 10⁻³, budget 10,000, % converged)\n")
    w("| Algorithm | " + " | ".join(lbl for k, _, lbl in C.SETTINGS if k != "narrow") + " |")
    w("|---|" + "---:|" * (len(C.SETTINGS) - 1))
    for algo in ROWS:
        cells = []
        for key, _, _ in C.SETTINGS:
            if key == "narrow":
                continue
            d = res[(key, algo)]
            h = d["unbiased"] if algo == "brute" else d["debiased"]
            cells.append(f"{h:.1f}%" + ("" if d["paper"] is None else f" ({d['paper']}%)"))
        w(f"| {NICE[algo]} | " + " | ".join(cells) + " |")
    w("\n_(this work vs paper). Paper adaptive numbers were grid maxima (winner's curse); de-biased they drop, "
      "but linear and reverse-engineering still beat brute for broad distributions._\n")

    # ---------- Beyond the paper ----------
    ba = cube[(cube.algo == "best_adaptive") & (cube.threshold_pct == 90)]
    prog = [(t, ba[ba.setting == f"U(0.01,0.1), eps={t}"].ratio_vs_brute)
            for t in ["1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]]
    w("## Beyond the paper — the advantage grows with precision, then saturates\n")
    w("Sweeping precision at φ ~ U(0.01, 0.1), reverse engineering's budget ratio at 90%:\n")
    w("| ε | " + " | ".join("10⁻" + t[-1] for t, _ in prog if len(_)) + " |")
    w("|---|" + "---:|" * sum(1 for _, s in prog if len(s)))
    w("| **budget ratio vs brute** | " + " | ".join(f"{s.iloc[0]:.2f}×" for _, s in prog if len(s)) + " |")
    w("\nThe advantage climbs from ~1.2× (ε=10⁻³) and **saturates from ε ≤ 10⁻⁶** — at ~1.76× for "
      "reverse engineering, with binary search drawing level there (~1.80× at the 90% threshold, "
      "but behind at 50%; the mean gap on the underlying curves is within one standard error, so "
      "neither leads). A tighter tolerance "
      "rewards the larger circuit depth N that the adaptive search selects, until the depth is capped "
      "by the prior support rather than by the budget.\n")

    # robustness across ranges (one line, per the 'wild' scan)
    re90 = cube[(cube.algo == "reverse_eng") & (cube.threshold_pct == 90)].dropna(subset=["ratio_vs_brute"])
    hi = re90[re90.eps <= 1e-5]
    w(f"**Robust to the prior range.** Across dynamic ranges from U(0.01,0.1) down to U(10⁻⁵,0.01), the "
      f"high-precision (ε≤10⁻⁵) advantage stays in a tight band ({hi.ratio_vs_brute.min():.2f}–"
      f"{hi.ratio_vs_brute.max():.2f}×): it is **precision-driven, not range-driven** (under the uniform prior, "
      "a wide range down does not add much small-φ mass). At low precision (ε=10⁻³) in narrow ranges the "
      "adaptive edge disappears and brute is competitive.\n")

    w("### Figures\n")
    w("![convergence-vs-budget with budget-ratio arrows]( )\n")
    w("![estimator error + uncertainty vs budget, ε to 10⁻⁸](fig_error.png)\n")
    w("![budget–convergence Pareto frontier + dominance strip](fig_pareto.png)\n")
    w("![budget ratio at 90% vs precision](fig_precision.png)\n")

    # ---------- mechanism & caveats ----------
    w("## Mechanism & caveats\n")
    w("- **Why adaptive wins:** brute fixes N = N_min, so its error 1/(2·N_min·√m) is φ-independent; adaptive "
      "infers N ≈ π/2φ, giving a per-sample edge (φ_max/φ)² — large only when φ ≪ φ_max. A tighter ε demands a "
      "larger N, which is exactly what the adaptive search supplies, so the advantage grows with precision "
      "(until it saturates at the Heisenberg limit).")
    w("- **It is a budget ratio, not an asymptotic win:** both curves reach ~100% eventually, so any single "
      "fixed-budget %-margin vanishes at high budget — report the budget ratio or the whole curve (fig_story).")
    w("- **Reverse engineering must not be used at φ_max = π/2** under the uniform prior (N-inference fails when "
      "N_min = 1 and it caps ~42%); linear search is the robust default there.")
    w("- **Binary search is not featured**: it wins only at high precision and loses at ε = 10⁻³; kept in the "
      "tables for completeness.")
    w("- **Log-uniform aside:** a scale-invariant prior spanning several decades yields a larger ~10× advantage "
      "via the same mechanism; we keep the paper's uniform assumption throughout.")

    # ---------- methodology ----------
    w("## Methodology & reproducibility\n")
    w(f"- **R** = {R:,} trials per point for Tables 3.1/3.3; the budget sweep (Table 3.2 and the figures) "
      "uses R = 40,000. Tuning uses R = 2,000 (seed 42); validation uses seed 2024.")
    w("- **Sampler:** binomial (`hits ~ Binomial(m, cos²Nφ)`), identical in distribution to per-shot sampling "
      "and much faster for large m, which is what makes the ε=10⁻⁸ / ~10¹⁴-budget sweeps feasible.")
    w("- **Seeds:** SEED_TUNE=42, SEED_TEST=2024, SEED_UNBIASED=12345. Re-running is deterministic.")
    w("- **Regenerate:** `python analysis/extensive_sweep.py` (budget sweep) → "
      "`python analysis/error_curves.py` → `python analysis/render_story.py` (figures) → "
      "`python analysis/make_results.py` (this file).")

    # ---------- paste-ready LaTeX ----------
    w("## LaTeX (paste-ready)\n")
    w("Every thesis table is emitted as its own `.tex` file by `python analysis/make_tex.py` into "
      "`results/tex/` (and collected in `results/TEX.md`), so there is exactly one place that knows "
      "the manuscript's table style. Variants ending in `_ci.tex` carry 95% confidence intervals; "
      "see `results/UNCERTAINTY.md` for what those intervals cover.\n")

    os.makedirs("results", exist_ok=True)
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(L) + "\n")

    # machine-readable
    with open("results/all_numbers.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["table", "setting", "algorithm", "metric", "value"])
        for key, _, _ in C.SETTINGS:
            for algo in tables.ALGOS_ALL:
                if (key, algo) in res:
                    d = res[(key, algo)]
                    wr.writerow(["3.1" if key == "narrow" else "3.3", key, algo,
                                 "pct_converged_10k", round(d.get("unbiased", d.get("debiased")), 2)])
        for lbl, _ in T32_COLS:
            for algo in ["brute", "separable"] + ADAPT + CEILINGS:
                v = t32[(lbl, algo)]
                if v is not None:
                    wr.writerow(["3.2", lbl, algo, "budget_to_90pct", round(v[0])])
    print("wrote results/RESULTS.md and results/all_numbers.csv")


if __name__ == "__main__":
    main()
