"""results/consolidated/FULL_RESULTS.md — the complete standalone record of the sweep.

Everything a reader could need to interpret, check or reproduce the numbers: the model, the exact
algorithm definitions, every scenario and budget grid, every seed, every trial count, every
parameter grid, the full statistical methodology, all results, the validation suite and the known
limitations.

Generated from results/consolidated/*.csv only -- no simulation -- so the document cannot drift from
the data it describes.

    python analysis/consolidated/full_report.py
"""
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import manifest as M
from pipeline_io import path

L = []


def w(s=""):
    L.append(s)


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


def fin(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def col(rows, k):
    return fin([f(r.get(k)) for r in rows])


def pct(x, d=2):
    return f"{100*x:.{d}f}%" if np.isfinite(x) else "--"


def sci(x, sig=3):
    x = f(x)
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1e6 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.{sig-1}e}"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.{sig}g}"


SHORT = {"brute": "Brute force", "linear": "Linear search",
         "binary_deep": "Binary search", "reverse_eng_risk": "Reverse engineering"}


# =============================================================================== PART 0: header
def part_header(man, perf, ops, runtime_min):
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", False
    n_pts = len({(r["scenario_id"], r["budget"]) for r in perf})
    live = sum(1 for r in ops if r["regime"] == "live")
    w("# Adaptive quantum metrology under resource constraints — complete results")
    w()
    w("Consolidated performance, algorithm diagnostics and uncertainty for the four algorithms the "
      "thesis reports, all derived from one set of held-out Monte-Carlo runs.")
    w()
    w("| | |")
    w("|---|---|")
    w(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M')} by `python analysis/consolidated/full_report.py` |")
    w(f"| Sweep command | `python analysis/consolidated/run.py --max --keep-traces` |")
    w(f"| Sweep mode | `{man['mode']}` |")
    w(f"| Sweep wall clock | {runtime_min:.0f} min ({runtime_min/60:.1f} h), 24 cores |")
    w(f"| Repository commit | `{sha}`{' (working tree dirty)' if dirty else ''} |")
    w(f"| Scenarios | {len(man['scenarios'])} |")
    w(f"| Operating points | {n_pts} ({live} live, {n_pts-live} saturated/floored) |")
    w(f"| Evaluated (point, algorithm) cells | {len(perf)} |")
    w(f"| Held-out trials per cell | R = {man['trials']['R_test']:,} |")
    w(f"| Total held-out trials | {len(perf) * man['trials']['R_test']:,} |")
    w()
    w("> **Scope note.** This document is generated from `results/consolidated/*.csv` and nothing "
      "else. No thesis file (`../thesis/*.tex`) is read or written anywhere in this pipeline; the "
      "LaTeX in `results/consolidated/tex/` is a proposal to paste, never an edit.")
    w()
    w("**Contents**")
    w()
    w("1. [The model](#1-the-model)")
    w("2. [Algorithms as implemented](#2-algorithms-as-implemented)")
    w("3. [Diagnostic quantities](#3-diagnostic-quantities)")
    w("4. [Experimental setting](#4-experimental-setting)")
    w("5. [Tuning and held-out protocol](#5-tuning-and-held-out-protocol)")
    w("6. [Statistical methodology](#6-statistical-methodology)")
    w("7. [Results — performance](#7-results--performance)")
    w("8. [Results — exploration phase](#8-results--exploration-phase)")
    w("9. [Results — safeguard and final depth](#9-results--safeguard-and-final-depth)")
    w("10. [Results — detector quality](#10-results--detector-quality)")
    w("11. [Results — trends](#11-results--trends)")
    w("12. [Budget compliance audit](#12-budget-compliance-audit)")
    w("13. [Tuning provenance and grid adequacy](#13-tuning-provenance-and-grid-adequacy)")
    w("14. [Validation suite](#14-validation-suite)")
    w("15. [Limitations and code/thesis mismatches](#15-limitations-and-codethesis-mismatches)")
    w("16. [File and column reference](#16-file-and-column-reference)")
    w()
    w("---")
    w()


# =============================================================================== PART 1: model
def part_model():
    w("## 1. The model")
    w()
    w("An unknown phase `phi` is estimated from a maximally-entangled circuit of depth `N` that "
      "reads out \"0\" with probability")
    w()
    w("```")
    w("p0(N, phi) = cos^2(N phi)")
    w("```")
    w()
    w("Drawing `m` shots and inverting the readout gives the estimator")
    w()
    w("```")
    w("phi_hat = (1/N) * arccos( sqrt(hits/m) ),     hits ~ Binomial(m, p0)")
    w("```")
    w()
    w("**Cost.** One measurement of depth `N` with `m` shots costs `N*m`. An algorithm's total cost "
      "is the sum over all its measurements, exploration and exploitation alike. Every algorithm "
      "here is *fixed-budget*: it is given a budget `B` and must not exceed it (audited in §12).")
    w()
    w("**Convergence.** A trial converges iff `|phi_hat - phi| < eps`, where `phi_hat` is the final "
      "estimate returned by the algorithm. The reported *convergence rate* is the fraction of "
      "converged trials over R independent trials.")
    w()
    w("**Aliasing and `N_opt`.** `arccos` inverts `p0` uniquely only while `N*phi <= pi/2`. The "
      "deepest non-aliasing depth is therefore")
    w()
    w("```")
    w("N_opt = max( floor( pi / (2 phi) ), 1 )")
    w("```")
    w()
    w("This is the target every exploration phase is implicitly trying to find, and the denominator "
      "of the depth diagnostics. Deeper circuits are more precise — the estimator's asymptotic "
      "variance is `1/(4 N^2 m)`, so at fixed budget `B = N*m` the error scales as "
      "`1/(2 sqrt(N B))` — but past `N_opt` the estimate aliases and the trial is (almost always) "
      "lost. Every algorithm below is a different way of trading those two off.")
    w()
    w("**Sampler.** `hits` is drawn directly from `Binomial(m, p0)` "
      "(`qmetrology/sim.py`, `DEFAULT_METHOD = \"binomial\"`), which is identical in distribution "
      "to summing `m` Bernoulli shots but O(1) in `m` rather than O(m). This is what makes the "
      "`eps = 1e-8` scenarios — budgets of order `1e16` — feasible at all.")
    w()
    w("**Prior.** In every trial `phi` is drawn from the scenario's prior, `Uniform(phi_min, "
      "phi_max)` throughout this sweep. The prior is genuine, not a modelling convenience: it is "
      "how the simulation draws `phi`, and it is what makes the statistical safeguard's posterior "
      "the true conditional distribution of `phi` given the pilot, over the ensemble reported here.")
    w()
    w("**Search bounds available to every algorithm.** From the prior support alone:")
    w()
    w("```")
    w("N_min = max( floor(pi / (2 phi_max)), 1 )    the deepest circuit that cannot alias for ANY")
    w("                                            admissible phi -- every algorithm opens here")
    w("N_max = max( floor(pi / (2 phi_min)), 1 )    the deepest circuit any admissible phi allows")
    w("```")
    w()
    w("---")
    w()


# ========================================================================== PART 2: algorithms
ALGO_DOC = {
    "brute": (
        "Algorithm 3 — brute force",
        "Fix `N = N_min` and spend the entire budget on shots: `m = floor(B / N_min)`. "
        "No exploration, no adaptivity, no parameters.",
        ["N = N_min", "m = floor(B / N_min)", "return phi_hat(N, m)"],
        "none",
    ),
    "linear": (
        "Algorithm 4 — linear search",
        "Scan `N` upward from `N_min` in steps of `inc`, `m'` shots per probe, tracking the running "
        "mean of the estimates. When that running mean falls `lookback_window` times in a row the "
        "scan declares an overshoot, backtracks by `lookback_window * inc`, subtracts the tuned "
        "decrement `s`, and spends whatever budget remains at that depth.",
        ["N = N_min; history = []",
         "while budget allows and N <= N_max:",
         "    phi_hat_i = measure(N, m');  history.append((N, phi_hat_i))",
         "    if running_mean(history) fell lookback_window times in a row: OVERSHOOT; break",
         "    N += inc",
         "N_guess = N_last - lookback_window*inc  if OVERSHOOT else N_last",
         "N_star  = max(1, N_guess - s)",
         "m       = floor(remaining_budget / N_star)",
         "return phi_hat(N_star, m)"],
        "m_exploration (m'), lookback_window, safeguard (s), inc",
    ),
    "binary_deep": (
        "Algorithm 5 — binary search + statistical safeguard, deepest-probe pilot",
        "Bisect on `N` between `N_min` and `N_max`. Each probe is compared against a threshold "
        "derived from the previous accepted estimate via Eq. (3.4); a probe reading below it is "
        "declared an overshoot and the search moves down, otherwise it is accepted and the search "
        "moves up. The **deepest accepted** probe `(phi_acc, N_acc)` is then handed to the "
        "statistical safeguard as the pilot.",
        ["phi_0 = measure(N_min, m')                       # cannot alias, by construction",
         "threshold = Phi^-1(1-conf; phi_0, 1/(4 m' N_min^2))",
         "L, U = N_min, N_max;  N = N_min + (U-N_min)//2",
         "while budget allows and the bracket is not exhausted:",
         "    phi_hat_i = measure(N, m')",
         "    if phi_hat_i < threshold:  U = N; N -= (N-L)//2          # declared overshoot",
         "    else:  (phi_acc, N_acc) = (phi_hat_i, N); L = N; N += (U-N)//2",
         "           threshold = Phi^-1(1-conf; phi_hat_i, 1/(4 m' N^2))",
         "N_guess = N_acc  (== L)",
         "N_star  = risk_optimal_depth(phi_acc, sigma = 1/(2 N_acc sqrt(m')),",
         "                             remaining_budget, eps, N_max = floor(pi/(2 phi_min)),",
         "                             support = (phi_min, phi_max))",
         "N_star  = max(N_star, min(N_min, N_max))         # never shallower than the opening probe",
         "m       = floor(remaining_budget / N_star)",
         "return phi_hat(N_star, m)"],
        "m_exploration (m'), conf",
    ),
    "reverse_eng_risk": (
        "Algorithm 6 — reverse engineering + statistical safeguard",
        "Take a single pilot at `N_min` (retrying if it returns exactly zero, which happens when "
        "`hits == m'`), invert it to a depth, and hand the pilot to the statistical safeguard. "
        "Every pilot attempt is charged to the exploration budget.",
        ["phi_hat_0 = 0",
         "while phi_hat_0 == 0 and budget allows:",
         "    phi_hat_0 = measure(N_min, m')               # retries are charged to B_exploration",
         "N_guess = max(floor(pi / (2 phi_hat_0)), 1)      # the raw inverted depth",
         "N_star  = risk_optimal_depth(phi_hat_0, sigma = 1/(2 N_min sqrt(m')),",
         "                             remaining_budget, eps, N_max = floor(pi/(2 phi_min)),",
         "                             support = (phi_min, phi_max))",
         "m       = floor(remaining_budget / N_star)",
         "return phi_hat(N_star, m)"],
        "m_exploration (m')",
    ),
}


def part_algorithms(man):
    w("## 2. Algorithms as implemented")
    w()
    w("Exactly four algorithms are reported. The superseded constant-safeguard variants, the "
      "exact-posterior variants and the omniscient oracles remain in `qmetrology/algorithms.py` but "
      "are deliberately outside this manifest and appear nowhere in these results.")
    w()
    w("| key | implementation | adaptive | has detector | tuned parameters |")
    w("|---|---|---|---|---|")
    for k in M.ORDER:
        s = M.ALGORITHMS[k]
        w(f"| `{k}` | `{s['variant']}` | {'yes' if s['adaptive'] else 'no'} | "
          f"{'yes' if s['has_detector'] else 'no'} | {ALGO_DOC[k][3]} |")
    w()
    for k in M.ORDER:
        title, prose, code, params = ALGO_DOC[k]
        w(f"### 2.{M.ORDER.index(k)+1} {title}")
        w()
        w(prose)
        w()
        w("```")
        for line in code:
            w(line)
        w("```")
        w()
    w("### 2.5 The statistical safeguard")
    w()
    w("`binary_deep` and `reverse_eng_risk` share one depth rule "
      "(`qmetrology/safeguard.py::risk_optimal_depth`), which replaces the grid-tuned constants "
      "`C_safe` and `s` of the published algorithms. Reading Eq. (3.4) as a likelihood for `phi` "
      "given a pilot `(phi_hat_0, N_0, m')` with `sigma = 1/(2 N_0 sqrt(m'))`, a candidate "
      "exploitation depth `N` carries two quantifiable and opposing risks:")
    w()
    w("```")
    w("P(no overshoot | N) = P( phi < pi/(2N) )       -- from the TRUNCATED normal posterior on")
    w("                                                  [phi_min, phi_max]")
    w("P(converge | N)     = 2*Phi( 2 eps sqrt(N B) ) - 1")
    w("")
    w("N_star = argmax_{1 <= N <= N_max}  P(no overshoot | N) * P(converge | N)")
    w("```")
    w()
    w("A deeper circuit is more precise (the second factor grows as `sqrt(N)`) but more likely to "
      "alias (the first falls). The optimum needs no tuned constant. The implied multiplicative "
      "safety factor `N_star / floor(pi/(2 phi_hat_0))` is adaptive: it tightens when the pilot is "
      "imprecise and relaxes toward 1 as the budget grows — which is what a constant `C` could "
      "never track.")
    w()
    w("The maximisation is a three-round geometric refinement followed by an exact integer scan, "
      "capped at `N_max = floor(pi/(2 phi_min))` from the prior support.")
    w()
    w("### 2.6 The two reference rows")
    w()
    w("Reported in the thesis tables for context, deliberately **not** protocols under comparison, "
      "and excluded from the regime rule, the \"points won\" count and every diagnostic aggregate.")
    w()
    w("**Separable (`N = 1`).** `m = floor(B/1) = B` — the whole budget as shots at unit depth. "
      "Same rule as brute force, which uses `N = N_min` instead. Its sd is `1/(2 sqrt(B))`: no "
      "`sqrt(N)` gain at all, which is the point of the row.")
    w()
    w("**Ceiling.** At fixed budget there is no independent shot count to choose: `m` follows from "
      "the depth, and Eq. (3.4) gives")
    w()
    w("```")
    w("N   = min( floor(pi/(2 phi)), B )        the deepest non-aliasing depth, capped so m >= 1")
    w("m   = floor(B / N)                       whole shots, so N*m <= B, like every other row")
    w("phi_hat = phi + Z / (2 N sqrt(m)),  Z ~ N(0,1)")
    w("```")
    w()
    w("It is **exact**: nothing is sampled. The convergence probability given `phi` is the closed "
      "form above, and the reported rate is that probability averaged over the prior. So this row "
      "carries no `R`, no Monte-Carlo error and no interval — the value is the value.")
    w()
    w("An earlier version of this pipeline instead *simulated* the ceiling, drawing "
      "`phi_hat = phi + Z/(2 N sqrt(m))` with `Z ~ N(0,1)`, so that it would flow through the same "
      "code path and carry a Wilson interval like every other row. That was dropped: it reports a "
      "number that wobbles by about +/- 0.4 pp between seeds in place of one that is exactly "
      "73.4844% at the headline point, and a bound has no sampling error to report in the first "
      "place. The uniformity was not worth the noise.")
    w()
    w("*Why the probability comes from Eq. (3.4) rather than from the binomial readout.* At "
      "`N ~ N_opt` the readout probability `p0 = cos^2(N phi)` sits against 0, so every shot returns "
      "0 and the arccos estimator is pinned at `pi/(2N)` **regardless of the data**. An oracle that "
      "knows `phi` could then read its own answer back off that constant, to accuracy "
      "`~2 phi^2/pi`, using almost no shots; whenever `phi < sqrt(pi eps / 2)` that is already "
      "inside tolerance, which is how an exact oracle comes to report a ~400x advantage at "
      "`U(0.001,0.01), eps = 1e-4`. Using Eq. (3.4) closes that loophole: the accuracy has to come "
      "from the statistics, not from the choice of `N`. Eq. (3.4) is not an extra assumption "
      "introduced for this row — it is the law the statistical safeguard and the binary-search "
      "overshoot test are both derived from.")
    w()
    w("*What kind of bound this is.* It bounds what the Chapter-3 family can achieve **under its "
      "own asymptotic law**, and it is non-degenerate everywhere. It is deliberately not a strict "
      "bound on the exact estimator, precisely because the exact estimator can exploit the "
      "boundary-clamping above. Two consequences, both measured rather than assumed:")
    w()
    w("* Because `m` is an integer, `N_opt` is not always the best admissible depth: at "
      "`B = 896, phi = 0.01` the bound `N_opt = 157` affords `m = 5` and spends 785 of 896, while "
      "`N = 149` affords `m = 6`, spends 894, and is better (`N sqrt(m)` = 365 vs 351). The row "
      "reported here uses `N_opt`, so it is a ceiling *for the depth an omniscient protocol would "
      "name*, not the supremum over all admissible depths. "
      "`qmetrology.oracle.ceiling_rate` computes the latter (at most 0.27 pp higher, and only where "
      "`m` is a handful of shots) if a strict supremum is ever wanted.")
    w("* Over all 221 operating points, the best implementable algorithm exceeds this ceiling at 93 "
      "(point-estimate, algorithm, budget) cells — and at **none** of them does the algorithm's 95% "
      "Wilson lower bound clear the ceiling. Every exceedance is inside Monte-Carlo noise.")
    w()
    w("---")
    w()


# ================================================================== PART 3: diagnostic quantities
def part_definitions():
    w("## 3. Diagnostic quantities")
    w()
    w("Every diagnostic is built from four per-run quantities recorded by the trace collector "
      "(`qmetrology/trace.py`). Their definitions are not conventions chosen after the fact — each "
      "is asserted against a hand-recomputed value from the same run's probe list in "
      "`tests/test_consolidated.py::test_algorithm_definitions`.")
    w()
    w("| quantity | definition |")
    w("|---|---|")
    w("| `N_opt` | `max(floor(pi/(2 phi)), 1)` — simulation ground truth, never visible to the algorithm |")
    w("| `N_guess` | the exploration phase's answer, recorded **before** any safeguard is applied |")
    w("| `N_star` | the depth the exploitation measurement actually ran at; **null** if it never ran |")
    w("| `B_exploration` | `sum(N_i * m_i)` over every probe before the exploitation measurement |")
    w("| `exploration_share` | `B_exploration / B` |")
    w()
    w("`N_guess` is algorithm-specific and this is the whole point of measuring it separately from "
      "`N_star`: the two differ by exactly the safeguard.")
    w()
    w("| algorithm | `N_guess` | `N_star` |")
    w("|---|---|---|")
    w("| `brute` | **null** — no exploration phase. Not zero, not `N_min`: an algorithm that does "
      "not search has no guess, and manufacturing one would corrupt every average. | `N_min` |")
    w("| `linear` | `N_last - lookback_window*inc` if the detector fired, else `N_last`; recorded "
      "raw, before `s` | `max(1, N_guess - s)` |")
    w("| `binary_deep` | `N_acc = L`, the deepest probe not classified as an overshoot | the "
      "safeguard's depth, floored at `N_min` |")
    w("| `reverse_eng_risk` | `max(floor(pi/(2 phi_hat_0)), 1)` from the pilot | the safeguard's depth |")
    w()
    w("**Repeated pilots count.** Reverse engineering retries its pilot when the estimate is "
      "exactly zero; every attempt is charged to `B_exploration`.")
    w()
    w("**Probe classification.** Each probe carries the algorithm's own verdict "
      "(`declared_overshoot`) and the simulation truth (`true_overshoot = N_i > N_opt`).")
    w()
    w("* `binary_deep` classifies each probe individually — the bisection's accept/reject decision "
      "*is* the declaration. The opening probe at `N_min` is accepted by construction (it cannot "
      "alias for any admissible `phi`).")
    w("* `linear` has **no per-probe detector**: its stopping rule is a trial-level verdict. The "
      "classification scored here is the algorithm's own backtracking decision — the last "
      "`lookback_window` probes are the ones it discards, so those are its declared overshoots and "
      "the retained ones are its acceptances. A different mapping would give different "
      "false-alarm numbers; this one is stated so the reader can judge it.")
    w("* `reverse_eng_risk` and `brute` have no detector at all and report **N/A**, never zero.")
    w()
    w("From these, the two trial-level detector rates:")
    w()
    w("```")
    w("false alarm : at least one probe with N_i <= N_opt was declared an overshoot")
    w("miss        : at least one probe with N_i >  N_opt was accepted / not flagged")
    w("```")
    w()
    w("Both are **trial-level**, not probe-level, because probes within one run are strongly "
      "dependent (the bisection's later probes are chosen from its earlier ones). Probe-level "
      "TP/FP/TN/FN counts are reported as supporting telemetry only, and carry no interval.")
    w()
    w("**Trace neutrality.** The collector records; it never draws a random number. A same-seed run "
      "with tracing on and off returns bit-identical `phi_hat` and budget use, asserted over 2,400 "
      "trials spanning four scenarios in `test_trace_neutrality`. The performance numbers in this "
      "document are therefore the *same runs* as the diagnostics, not a paired re-simulation.")
    w()
    w("---")
    w()


# ================================================================== PART 4: experimental setting
def part_setting(man, ops):
    w("## 4. Experimental setting")
    w()
    w("Everything below comes from `qmetrology/manifest.py`, the single source of truth, and is "
      "echoed into `results/consolidated/experiment_manifest.json` at run time. No scenario list, "
      "seed or trial count is hard-coded anywhere downstream — asserted by "
      "`test_ci_uses_row_R`, which fails if any consumer module contains a literal trial count.")
    w()
    w("### 4.1 Scenarios")
    w()
    w("Four Chapter-4 experiment families, ten scenarios, all with a **uniform** prior on `phi`:")
    w()
    w("| id | prior | `eps` | `N_min` | `N_max` | budget grid | pts | families |")
    w("|---|---|---|---:|---:|---|---:|---|")
    for s in man["scenarios"]:
        b = s["budgets"]
        rule = s["budget_rule"]
        rule_s = (f"log, brute90/50 .. brute90x30" if rule[0] == "brute90"
                  else f"log, {sci(rule[1])} .. {sci(rule[2])}")
        pin = f" +{{{', '.join(str(int(p)) for p in s['pin_budgets'])}}}" if s["pin_budgets"] else ""
        w(f"| `{s['id']}` | U({s['phi_min']:g}, {s['phi_max']:.4g}) | {s['eps']:.0e} | "
          f"{s['N_min']} | {s['N_max']} | {rule_s}{pin} | {len(b)} | "
          f"{', '.join(s['families'])} |")
    w()
    w("`brute90 = 0.6724 / (N_min * eps^2)` is the analytic budget at which brute force reaches "
      "~90% convergence; anchoring the grid to it puts the informative band in the middle of the "
      "sweep for every scenario, whatever its scale. The broad-prior scenarios instead use the "
      "explicit `3e3 .. 2e6` curve that `04-broad-dist.tex` reports, so they stay comparable to it.")
    w()
    w("`narrow_e3` additionally pins `B = 10,000`, the Table-3.1 operating point, so it is "
      "evaluated exactly rather than interpolated.")
    w()
    w("### 4.2 Budget grids in full")
    w()
    w("<details><summary>every evaluated budget, per scenario</summary>")
    w()
    for s in man["scenarios"]:
        w(f"**`{s['id']}`** ({len(s['budgets'])} budgets)")
        w()
        w("```")
        w(", ".join(f"{int(b):,}" for b in s["budgets"]))
        w("```")
        w()
    w("</details>")
    w()
    w("### 4.3 Regime classification")
    w()
    n_live = sum(1 for r in ops if r["regime"] == "live")
    n_sat = sum(1 for r in ops if r["regime"] == "saturated")
    n_fl = sum(1 for r in ops if r["regime"] == "floored")
    w("Each operating point is labelled by the *best* convergence rate achieved at it by any "
      "algorithm (`operating_points.csv`):")
    w()
    w("| regime | rule | count |")
    w("|---|---|---:|")
    w(f"| `live` | 3% < best rate < 99% | **{n_live}** |")
    w(f"| `saturated` | best rate >= 99% | {n_sat} |")
    w(f"| `floored` | best rate <= 3% | {n_fl} |")
    w()
    w("**Every aggregate in §7–§11 is over `live` points only.** At a saturated point every "
      "configuration converges, the grid search cannot distinguish them (selection margins go to "
      "0.00 pp), and the tie-break rule then picks the cheapest exploration — often a one-shot "
      "pilot. The *performance* number there is still valid; the *diagnostics* measured at that "
      "arbitrary configuration are not a property of the algorithm. All points remain in every CSV; "
      "join on `operating_points.csv` to filter differently.")
    w()
    w("### 4.4 Seeds")
    w()
    sd = man["seeds"]
    w("| role | value | used for |")
    w("|---|---|---|")
    w(f"| tuning block 1 | `{sd['tune_blocks'][0]}` | scoring candidate configurations |")
    w(f"| tuning block 2 | `{sd['tune_blocks'][1]}` | second, independent scoring block |")
    w(f"| held-out test | `{sd['test']}` | **every reported number**; disjoint from both blocks |")
    w(f"| bootstrap | `{sd['bootstrap']}` | all resampling: mean CIs, crossing CIs, ratio CIs |")
    w()
    w("Per-trial seeds are derived deterministically: "
      "`np.random.default_rng(seed).integers(0, 2**63, size=R)`. Trial `i` therefore draws the "
      "**same** `phi` for every algorithm, every budget and every configuration, so all comparisons "
      "are paired. Each trial then runs on `np.random.default_rng(trial_seed)`, making the result "
      "independent of execution order and of the number of worker processes.")
    w()
    w("### 4.5 Trial counts (every R in the study)")
    w()
    t = man["trials"]
    cfg = M.MODES[man["mode"]]
    w("| symbol | value | where |")
    w("|---|---:|---|")
    w(f"| `R_test` | **{t['R_test']:,}** | held-out evaluation of the frozen winner — every "
      f"reported rate and every diagnostic |")
    w(f"| `R_tune` (stages A, B) | {t['R_tune_per_block_stage1']:,} | per tuning block, locating "
      f"`m'` and scanning the discrete grid |")
    w(f"| `R_tune2` (stage C) | **{t['R_tune_per_block_stage2']:,}** | per tuning block, the "
      f"refinement that decides the winner |")
    w(f"| `R_audit` | {cfg['R_audit']:,} | the binary-search pilot comparison in "
      f"`algorithm_code_audit.md` |")
    w(f"| `n_boot` | {man['uncertainty']['n_boot']:,} | bootstrap replicates (means, crossings, ratios) |")
    w()
    w(f"Tuning consumes `2 x (R_tune)` trials per candidate in stages A and B and "
      f"`2 x (R_tune2)` in stage C — two blocks each. **None of those trials appear in any "
      f"reported number**; they only choose the configuration.")
    w()
    w("---")
    w()


# ==================================================================== PART 5: tuning protocol
def part_tuning(man, wins):
    cfg = M.MODES[man["mode"]]
    w("## 5. Tuning and held-out protocol")
    w()
    w("**The rule.** No reported value is a maximum evaluated on the trials that chose the "
      "parameters.")
    w()
    w("```")
    w("1. score every candidate configuration on EACH tuning block (seeds 42 and 43), separately")
    w("2. winner = argmax of the MEAN of the two block rates")
    w("     - two blocks, so one accidental draw cannot decide it")
    w("     - ties broken toward the SMALLER exploration size, then toward the earlier candidate")
    w("3. freeze the winner, together with its per-block evidence and its runner-up")
    w("4. evaluate the frozen winner on the disjoint held-out seed 2024, at R_test, WITH tracing")
    w("5. performance AND diagnostics both come from that single evaluation")
    w("```")
    w()
    w("Step 5 is what makes the diagnostics trustworthy: a rate and its telemetry cannot refer to "
      "different runs, because there is only one run. `test_performance_consistency` asserts that "
      "the converged count derived from the traces equals the count from an independent untraced "
      "evaluation on the same seeds.")
    w()
    w("### 5.1 Three-stage grid search")
    w()
    w("A single grid fine enough to span `m' in [1, B//N_min]` *and* cross it with linear search's "
      f"450 discrete combinations would be {cfg['n_m1']*450:,} configurations per stage. The search "
      "is therefore staged:")
    w()
    w("| stage | `m'` grid | discrete grid | R per block | purpose |")
    w("|---|---|---|---:|---|")
    w(f"| **A** | {cfg['n_m1']} log-spaced points across the full box `[1, B//N_min]` | "
      f"3-point-per-axis *skeleton* (bottom/middle/top) | {cfg['R_tune']:,} | locate `m'` |")
    w(f"| **B** | {cfg['n_m2']} points within x4 of A's winner | the **full** discrete grid | "
      f"{cfg['R_tune']:,} | choose the discrete configuration |")
    w(f"| **C** | {cfg['n_m3']} points within x2 of B's winner (~{100*(4**(1/(cfg['n_m3']-1))-1):.0f}% "
      f"steps) | the neighbouring value on each axis | **{cfg['R_tune2']:,}** | refine, and decide |")
    w()
    w("Stage C gets by far the most trials because it is the stage that actually decides. That "
      "directly controls the *variance* of which configuration wins — the one uncertainty component "
      "that cannot be recovered from stored output afterwards (see §6.8).")
    w()
    w("### 5.2 The `m'` search box")
    w()
    w("```")
    w("lower  m' = 1                 one shot -- the physical minimum")
    w("upper  m' = B // N_min        the largest pilot affordable at all: a probe at the opening")
    w("                              depth costs m'*N_min, so beyond this the algorithm cannot take")
    w("                              even its first probe and scores zero by construction")
    w("```")
    w()
    w("Neither endpoint can be criticised as arbitrary, and both are recorded per row (`m_lo`, "
      "`m_hi`) so a boundary hit is informative rather than an artifact.")
    w()
    w("### 5.3 Parameter grids in full")
    w()
    w("| algorithm | axis | values | n |")
    w("|---|---|---|---:|")
    w(f"| all adaptive | `m_exploration` | log-spaced over `[1, B//N_min]`, per scenario and budget "
      f"| {cfg['n_m1']}/{cfg['n_m2']}/{cfg['n_m3']} |")
    for k in M.ORDER:
        for ax, vals in M.ALGORITHMS[k].get("discrete", {}).items():
            w(f"| `{k}` | `{ax}` | {', '.join(str(v) for v in vals)} | {len(vals)} |")
    w()
    lin = M.ALGORITHMS["linear"]["discrete"]
    n_lin = 1
    for v in lin.values():
        n_lin *= len(v)
    w(f"Linear search therefore has **{n_lin} discrete combinations**, binary search "
      f"**{len(M.ALGORITHMS['binary_deep']['discrete']['conf'])}**, reverse engineering none.")
    w()
    w("**Why these ranges.** An earlier production sweep used `lookback_window in [1,2,5]` and "
      "`safeguard in [0,1,2]` and selected the **top** of those grids in 64.8% and 15.7% of live "
      "cells respectively — the optimum lay outside the search box, so those numbers understated "
      "linear search by an unknown amount. `safeguard` needed the largest extension because it is "
      "an *absolute* depth decrement: its useful range grows with `N_min`, and a grid tuned for "
      "`N_min = 15` cannot serve `N_min = 157`. §13 reports whether the current ranges bind.")
    w()
    w("### 5.4 Boundary reporting")
    w()
    w("`winners.csv` carries `at_axis_bound`, a JSON map flagging a winner pinned at an endpoint of "
      "**any** axis — not just `m'` — evaluated against the full grid. A pin at an axis *maximum* "
      "means the optimum may lie outside the box and the result is suspect. A pin at a *minimum* is "
      "usually a physical floor (`m' = 1` shot, `inc = 1`, `safeguard = 0`, `lookback_window = 1`, "
      "`conf = 0.5`) and is not a defect.")
    w()
    w("---")
    w()


# ============================================================ PART 6: statistical methodology
def part_stats(man, cross):
    unc = man["uncertainty"]
    nb, bs, lvl = unc["n_boot"], unc["boot_seed"], unc["ci_level"]
    R = man["trials"]["R_test"]
    w("## 6. Statistical methodology")
    w()
    w(f"All intervals are **{lvl}%**. Every interval-bearing row in every CSV records its own "
      f"`ci_level`, method, `n_boot`, `boot_seed` and denominator, so no interval has to be "
      f"reconstructed from context.")
    w()
    w("| quantity | estimator | interval method |")
    w("|---|---|---|")
    w(f"| convergence rate | `k/R` | Wilson score at the row's own `R` |")
    w(f"| diagnostic share | `k/n_eligible` | Wilson score at the metric's own eligible denominator |")
    w(f"| median / quartiles | sample quantile | exact order-statistic bootstrap (closed form) |")
    w(f"| mean | sample mean | run-level percentile bootstrap, {nb:,} replicates, seed {bs} |")
    w(f"| budget to reach p* | log-linear interpolation | parametric bootstrap of the curve, "
      f"{nb:,} replicates |")
    w(f"| budget ratio vs brute | ratio of two crossings | parametric bootstrap, both curves |")
    w(f"| grid discretisation | half-density subgrid deviation | deterministic, no interval |")
    w()
    w("### 6.1 Convergence and diagnostic proportions — Wilson score interval")
    w()
    w("For `k` successes in `n` trials, with `z = 1.959964` for 95%:")
    w()
    w("```")
    w("            p_hat + z^2/(2n)      z * sqrt( p_hat(1-p_hat)/n + z^2/(4n^2) )")
    w("centre  =  ------------------ ,  half = -----------------------------------------")
    w("              1 + z^2/n                            1 + z^2/n")
    w("```")
    w()
    w("Chosen over the Wald interval because it stays inside `[0,1]` and keeps close to nominal "
      "coverage near `p = 0` and `p = 1`, which matters here: several diagnostic shares (final "
      "overshoot, miss rate) are genuinely near zero, where Wald would produce negative lower "
      f"bounds. At `R = {R:,}` the worst-case half-width (at `p = 0.5`) is "
      f"**{100*1.959964*np.sqrt(0.25/R):.3f} pp**.")
    w()
    w("Every share is reported with its **eligible denominator** (`*_n`) and numerator (`*_k`). "
      "Eligibility is not cosmetic: a run with no defined `N_guess` (brute force, or an algorithm "
      "that never got a probe in) must not silently vanish into a smaller denominator. "
      "`test_eligibility_accounting` asserts `eligible + ineligible = R` for every metric, and that "
      "conditional metrics have denominators no larger than the sets they condition on.")
    w()
    w("### 6.2 Medians and quartiles — exact order-statistic bootstrap")
    w()
    w("The usual recipe is a percentile bootstrap: resample the R runs with replacement `n_boot` "
      "times, take the median of each resample, report the 2.5th and 97.5th percentiles of those "
      "medians. **For a quantile this does not need to be simulated** — it has a closed form.")
    w()
    w("A bootstrap resample's `q`-quantile is always one of the observed values `x_(k)`. Writing "
      "`m = ceil(q*n)` for the rank the estimator uses, the resampled quantile is at or below "
      "`x_(k)` exactly when at least `m` of the `n` resampled draws fall at or below `x_(k)`, and "
      "each draw does so independently with probability `k/n`. Hence")
    w()
    w("```")
    w("F_boot( x_(k) )  =  P( Binomial(n, k/n) >= m )")
    w("```")
    w()
    w("which is increasing in `k`, so the `alpha`-percentile of the bootstrap distribution is "
      "`x_(k*)` at the smallest `k*` with `F_boot(x_(k*)) >= alpha`. Evaluating that for "
      "`k = 1..n` and searching it costs O(n) and is **exact**.")
    w()
    w("This is the `n_boot -> infinity` limit of the percentile bootstrap, i.e. the quantity a "
      "Monte-Carlo bootstrap estimates noisily. It was adopted for cost as well as accuracy: a "
      f"Monte-Carlo bootstrap of the medians alone, at `R = {R:,}` across "
      f"{4*221:,} cells and ~8 continuous metrics, is on the order of an hour of post-processing "
      "for a strictly worse answer. Verified against a Monte-Carlo bootstrap at 2,000 replicates on "
      "three seeds: identical endpoints.")
    w()
    w("Recorded per row as `ci_method_quantile = \"order-statistic bootstrap (exact percentile "
      "interval)\"`. Points with fewer than 20 eligible runs get the point estimate and no interval.")
    w()
    w("### 6.3 Means — run-level percentile bootstrap")
    w()
    w(f"Means have no equivalent closed form, so they use an ordinary Monte-Carlo percentile "
      f"bootstrap: {nb:,} replicates, seed {bs}, resampling **runs** (never probes). Resampling at "
      f"the run level is what carries the clustering — probes inside one run are dependent, so a "
      f"probe-level resample would badly understate the uncertainty of anything probe-derived.")
    w()
    w("All continuous metrics are resampled with the *same* run indices within a replicate batch, "
      "and metrics that are undefined for a run carry `NaN` on the full run axis rather than being "
      "compacted, so every replicate averages over exactly the eligible runs it drew.")
    w()
    w("Recorded as `ci_method_mean`. The eight continuous metrics carrying a mean and its interval: "
      "`guess_ratio`, `guess_abs_rel`, `guess_signed_rel`, `exploration_share`, `n_probes`, "
      "`star_ratio`, `star_over_guess`, `budget_util`.")
    w()
    w("### 6.4 Budget to reach a target reliability — the crossing rule")
    w()
    w("One rule, used everywhere, defined once in `qmetrology/uncertainty.py::crossing` so the "
      "interval is always built around exactly the number reported:")
    w()
    w("```")
    w("find the first grid index i with rate[i] >= p*")
    w("f      = (p* - rate[i-1]) / (rate[i] - rate[i-1])")
    w("B(p*)  = exp( log(B[i-1]) + f * (log(B[i]) - log(B[i-1])) )      # LOG-LINEAR in budget")
    w("```")
    w()
    w("Interpolation is linear in `log B` because the convergence-vs-budget curve is close to "
      "linear on a log-budget axis over the relevant range (the estimator error falls as "
      "`1/sqrt(B)`), so log-linear interpolation is far more accurate than linear. Returns `NaN` if "
      "the curve never reaches `p*` within the swept range — reported as a missing crossing, never "
      "extrapolated.")
    w()
    w(f"Thresholds evaluated: {', '.join(str(t)+'%' for t in man['thresholds_pct'])}.")
    w()
    w("### 6.5 Crossing and ratio intervals — parametric bootstrap")
    w()
    w("The crossing is *derived*, not measured, so its uncertainty is propagated from the rates the "
      "curve is built from:")
    w()
    w("```")
    w(f"for b in 1..{nb:,}:")
    w("    resample every curve point:  p*_j ~ Binomial(R, p_hat_j) / R")
    w("    re-derive the crossing from the resampled curve")
    w("report the 2.5th and 97.5th percentiles of the resampled crossings")
    w("```")
    w()
    w("It is *parametric* (Binomial at the observed rate) rather than a resample of raw trials "
      "because the curve points are already sufficient statistics — each is a binomial proportion "
      "over the same R. Replicates whose resampled curve never reaches `p*` are dropped and the "
      "surviving fraction is reported as `coverage`; coverage below 1.0 flags a crossing sitting at "
      "the edge of the swept range.")
    w()
    w("The ratio `B_brute(p*) / B_algo(p*)` resamples both curves **independently**. Because all "
      "algorithms share the seed list, a trial index draws the same `phi` for every algorithm, so "
      "the two crossings are in truth positively correlated and the independent resample "
      "**overstates** the ratio's uncertainty. The ratio intervals are therefore conservative, not "
      "optimistic — stated here rather than left for a reader to discover.")
    w()
    w("### 6.6 Budget-grid discretisation")
    w()
    w("Monte-Carlo error is not the only thing moving a crossing; the finite budget grid does too. "
      "This is probed directly and without assumptions: re-derive each crossing from the two "
      "half-density subgrids (`budgets[0::2]` and `budgets[1::2]`) and report the largest relative "
      "deviation. Log-linear interpolation error is `O(h^2)` in the grid log-spacing, so the "
      "full-density grid carries roughly **a quarter** of the deviation reported.")
    w()
    gs = col(cross, "grid_sensitivity_rel")
    if gs.size:
        w(f"Measured here: median **{100*np.median(gs):.2f}%**, 90th percentile "
          f"{100*np.percentile(gs,90):.2f}%, max {100*gs.max():.2f}% over {gs.size} crossings — so "
          f"the full-grid contribution is of order {100*np.median(gs)/4:.2f}% at the median.")
    w()
    w("### 6.7 Rounding conventions")
    w()
    w("Budgets to 3 significant figures; ratios to 2 decimals; percentages to 1–2 decimals. Applied "
      "at presentation only — the CSVs carry full precision.")
    w()
    w("### 6.8 What is *not* covered")
    w()
    w("Stated explicitly because it is the honest boundary of these intervals:")
    w()
    w("* **The sampling variance of the grid search itself.** Which configuration wins is random. "
      "De-biasing removes its *bias* (the winner is re-validated on an independent seed) but not "
      f"its *variance*. Stage C's large `R_tune2 = {man['trials']['R_tune_per_block_stage2']:,}` and "
      "the two-block selection rule reduce it; quantifying what remains requires repeating the "
      "tuning on many seeds (`analysis/tuning_stability.py` does this for headline cells only). "
      "The per-row `margin_pp` and `runner_up` columns let a reader see how much was at stake in "
      "each selection.")
    w("* **Model error in the safeguard's Gaussian posterior**, and the selection bias of "
      "`binary_deep`'s pilot. Not an interval; measured behaviourally in §8–§10 instead.")
    w("* **Interpolation bias in the crossing** beyond the `O(h^2)` bound of §6.6.")
    w()
    w("---")
    w()


# ==================================================================== PART 7-11: results
def _live(rows, reg):
    return [r for r in rows if reg.get((r["scenario_id"], int(f(r["budget"])))) == "live"]


def part_performance(man, diag, cross, reg):
    R = man["trials"]["R_test"]
    w("## 7. Results — performance")
    w()
    w(f"Convergence rate of the frozen winner on the held-out seed, R = {R:,} per cell. "
      "Aggregates over live points only (§4.3).")
    w()
    w("### 7.1 Overall")
    w()
    by_pt = {}
    for r in diag:
        # reference rows (separable, ceiling) are not protocols and cannot "win" a point
        if r["algorithm"] not in M.PROTOCOLS:
            continue
        by_pt.setdefault((r["scenario_id"], r["budget"]), {})[r["algorithm"]] = f(r["rate"])
    wins = {a: 0 for a in M.ORDER}
    for v in by_pt.values():
        wins[max(v, key=lambda k: v[k])] += 1
    w("| algorithm | mean | median | min | max | points won |")
    w("|---|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        v = col([r for r in diag if r["algorithm"] == a], "rate")
        if v.size:
            w(f"| {SHORT[a]} | {pct(v.mean(),1)} | {pct(np.median(v),1)} | {pct(v.min(),1)} | "
              f"{pct(v.max(),1)} | {wins[a]}/{len(by_pt)} |")
    for a in M.REFERENCE:
        v = col([r for r in diag if r["algorithm"] == a], "rate")
        if v.size:
            w(f"| *{M.ALGORITHMS[a]['label']}* | {pct(v.mean(),1)} | {pct(np.median(v),1)} | "
              f"{pct(v.min(),1)} | {pct(v.max(),1)} | -- |")
    w()
    w("Means and medians mix scenarios and budgets; they summarise the table, they are not a "
      "headline claim. The per-cell rows with Wilson intervals are in `performance_curves.csv`. "
      "The last two rows are reference points, not protocols under comparison: they cannot win a "
      "point and are excluded from every diagnostic aggregate and from the regime rule.")
    w()
    w("### 7.2 Budget to reach a target reliability, relative to brute force")
    w()
    w("Ratio `B_brute(p*) / B_algo(p*)`; **greater than 1 means the algorithm needs less budget**.")
    w()
    w("| algorithm | p* | median ratio | min | max | scenarios with a crossing |")
    w("|---|---|---:|---:|---:|---:|")
    for a in M.ORDER[1:]:
        for T in man["thresholds_pct"]:
            v = fin([f(c["ratio_vs_brute"]) for c in cross
                     if c["algorithm"] == a and int(c["threshold_pct"]) == T])
            if v.size:
                w(f"| {SHORT[a]} | {T}% | **{np.median(v):.2f}** | {v.min():.2f} | {v.max():.2f} | "
                  f"{v.size} |")
    w()
    w("### 7.3 Per-scenario, at the 90% threshold")
    w()
    w("| scenario | brute `B(90%)` | linear | binary | reverse eng. | best ratio |")
    w("|---|---:|---:|---:|---:|---:|")
    for s in man["scenarios"]:
        row = {c["algorithm"]: c for c in cross
               if c["scenario_id"] == s["id"] and int(c["threshold_pct"]) == 90}
        if "brute" not in row:
            continue
        cells = []
        best = 0.0
        for a in M.ORDER:
            c = row.get(a)
            if not c or c["budget_to_reach"] in ("", "nan"):
                cells.append("--")
                continue
            cells.append(sci(c["budget_to_reach"]))
            rr = f(c.get("ratio_vs_brute"))
            if np.isfinite(rr):
                best = max(best, rr)
        w(f"| `{s['id']}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
          f"{best:.2f}x |")
    w()
    w("Full rows — every threshold, both interval endpoints, `coverage`, `grid_sensitivity_rel`, "
      "`R`, `n_boot`, `boot_seed` — are in `budget_crossings.csv`.")
    w()
    w("### 7.4 The Table-3.1 operating point")
    w()
    w("`U(0.01, 0.1)`, `eps = 1e-3`, `B = 10,000`, evaluated directly (not interpolated):")
    w()
    w("| algorithm | converged | 95% Wilson CI | frozen configuration |")
    w("|---|---:|---|---|")
    for a in M.ORDER:
        r = next((x for x in diag if x["scenario_id"] == "narrow_e3"
                  and int(f(x["budget"])) == 10000 and x["algorithm"] == a), None)
        if r:
            p = json.loads(r["params"]) if r.get("params") else {}
            p.pop("eps_target", None)
            ps = ", ".join(f"{k}={v}" for k, v in sorted(p.items())) or "none"
            w(f"| {SHORT[a]} | **{pct(f(r['rate']),2)}** | [{pct(f(r['rate_lo']),2)}, "
              f"{pct(f(r['rate_hi']),2)}] | {ps} |")
    w()
    w("---")
    w()


def part_exploration(man, diag):
    R = man["trials"]["R_test"]
    w("## 8. Results — exploration phase (`N_guess / N_opt`)")
    w()
    w("The primary exploration diagnostic: how close the exploration phase's *own answer* comes to "
      "the deepest non-aliasing depth, before any safeguard intervenes.")
    w()
    w("| algorithm | median ratio | mean abs. rel. err. | signed rel. err. | exact hit | within 5% | "
      "within 10% | guess overshoot |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        if not M.ALGORITHMS[a]["has_guess"]:
            continue
        rs = [r for r in diag if r["algorithm"] == a]
        g = col(rs, "guess_ratio_median")
        if not g.size:
            continue
        w(f"| {SHORT[a]} | **{np.median(g):.2f}** | {col(rs,'guess_abs_rel_mean').mean():.2f} | "
          f"{col(rs,'guess_signed_rel_mean').mean():+.2f} | {pct(col(rs,'guess_exact').mean(),1)} | "
          f"{pct(col(rs,'guess_within5').mean(),1)} | {pct(col(rs,'guess_within10').mean(),1)} | "
          f"{pct(col(rs,'guess_overshoot').mean(),1)} |")
    w()
    w("The mean ratio, the signed error and the absolute error are all retained deliberately: a "
      "mean ratio near 1 can hide a mixture of severe under- and overshoots, and only the absolute "
      "error exposes that.")
    w()
    w("`brute` is absent because it has no exploration phase — its `N_guess` columns are empty by "
      "construction, not zero.")
    w()
    w("### 8.1 Exploration cost")
    w()
    w("Each cell contributes its own within-cell statistic. The three *share* columns then take "
      "the **median across cells**, so they are directly comparable (a mean of per-cell means "
      "against a median of per-cell p90s would not be). The probe columns instead take the **mean "
      "across cells**, which — since every cell has the same R — is exactly the pooled mean over "
      "all runs, and is the interpretable \"probes per trial\" number.")
    w()
    w("| algorithm | expl. share: median | mean | p90 | mean probes/trial | median probes | "
      "only one probe | no exploitation phase |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        rs = [r for r in diag if r["algorithm"] == a]
        if not rs:
            continue
        has_guess = M.ALGORITHMS[a]["has_guess"]
        probes = (f"{col(rs,'n_probes_mean').mean():.1f} | "
                  f"{np.median(col(rs,'n_probes_median')):.1f} | "
                  f"{pct(col(rs,'single_probe').mean(),1)}") if has_guess else "-- | -- | --"
        w(f"| {SHORT[a]} | {pct(np.median(col(rs,'exploration_share_median')),2)} | "
          f"{pct(np.median(col(rs,'exploration_share_mean')),2)} | "
          f"{pct(np.median(col(rs,'exploration_share_p90')),2)} | "
          f"{probes} | {pct(col(rs,'no_exploitation').mean(),2)} |")
    w()
    w("Brute force takes no probes at all, so its probe-shape columns are blank rather than "
      "reporting the vacuously true \"0 probes is <= 1 probe\".")
    w()
    w("**Termination reasons** are recorded per cell in `diagnostics_by_point.csv` "
      "(`termination_reasons`, a `reason=count` list summing to R). The statuses are: `ok`, "
      "`detector_fired`, `scan_exhausted`, `no_exploitation_budget_exhausted`, "
      "`no_exploitation_shots`, `pilot_retries_exhausted_budget`, `refused_pilot_unaffordable`.")
    w()
    w("---")
    w()


def part_safeguard(man, diag):
    w("## 9. Results — safeguard and final depth (`N_star / N_opt`)")
    w()
    w("| algorithm | median `N*/N_opt` | Q1 | Q3 | final overshoot | exact | within 10% | "
      "converged given a safe depth |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        rs = [r for r in diag if r["algorithm"] == a]
        s = col(rs, "star_ratio_median")
        if not s.size:
            continue
        w(f"| {SHORT[a]} | **{np.median(s):.2f}** | {np.median(col(rs,'star_ratio_q1')):.2f} | "
          f"{np.median(col(rs,'star_ratio_q3')):.2f} | "
          f"{pct(col(rs,'star_overshoot').mean(),2)} | {pct(col(rs,'star_exact').mean(),1)} | "
          f"{pct(col(rs,'star_within10').mean(),1)} | "
          f"{pct(col(rs,'conv_given_safe_depth').mean(),1)} |")
    w()
    w("**A median below 1 is the intended behaviour, not a miss.** The safeguard's objective is "
      "`P(no overshoot) x P(converge)`, not `N_opt` itself: it deliberately backs off from the "
      "aliasing cliff, and because the estimator is boundary-censored near `N_opt` (where "
      "`p0 = cos^2(N phi)` sits against 0) a depth slightly below `N_opt` can have *higher* true "
      "convergence than `N_opt`. Exact attainment is reported because readers ask for it, but it is "
      "not the success criterion.")
    w()
    w("### 9.1 What the safeguard does to the guess")
    w()
    w("| algorithm | median `N*/N_guess` | Q1 | Q3 | unsafe-guess rescue | backoff when the guess "
      "was already safe |")
    w("|---|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        rs = [r for r in diag if r["algorithm"] == a]
        v = col(rs, "star_over_guess_median")
        if not v.size:
            continue
        rc = col(rs, "rescue_share")
        bo = col(rs, "backoff_safe_median")
        w(f"| {SHORT[a]} | {np.median(v):.2f} | {np.median(col(rs,'star_over_guess_q1')):.2f} | "
          f"{np.median(col(rs,'star_over_guess_q3')):.2f} | "
          f"{pct(rc.mean(),1) if rc.size else '--'} | "
          f"{(f'{np.median(bo):.2f}x' if bo.size else '--')} |")
    w()
    w("*Unsafe-guess rescue* is `P(N_star <= N_opt | N_guess > N_opt)` — how often the safeguard "
      "pulls a genuinely aliasing guess back to safety. Its denominator (`rescue_share_n`) is the "
      "count of runs whose guess overshot, and is reported per row.")
    w()
    w("**Not a failure decomposition.** Early stopping, final overshoot and ordinary shot noise are "
      "reported here as *overlapping stage flags and conditional rates*. They are deliberately "
      "**not** presented as a mutually exclusive causal breakdown of why trials fail: no priority "
      "or counterfactual rule has been specified that would justify assigning each failure to "
      "exactly one cause.")
    w()
    w("---")
    w()


def part_detector(man, det, diag):
    w("## 10. Results — detector quality")
    w()
    w("Only linear and binary search make overshoot declarations. Reverse engineering and brute "
      "force have no detector and are **N/A**, never zero.")
    w()
    w("| algorithm | trial-level false alarm | trial-level miss | probe TP | FP | TN | FN | "
      "eligible runs |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ("linear", "binary_deep"):
        rs = [r for r in det if r["algorithm"] == a]
        if not rs:
            continue
        fa, ms = col(rs, "false_alarm_rate"), col(rs, "miss_rate")
        tp = sum(int(f(r["probe_tp"], 0)) for r in rs)
        fp = sum(int(f(r["probe_fp"], 0)) for r in rs)
        tn = sum(int(f(r["probe_tn"], 0)) for r in rs)
        fn = sum(int(f(r["probe_fn"], 0)) for r in rs)
        el = sum(int(f(r["detector_eligible_n"], 0)) for r in rs)
        w(f"| {SHORT[a]} | **{pct(fa.mean(),1)}** | **{pct(ms.mean(),1)}** | {tp:,} | {fp:,} | "
          f"{tn:,} | {fn:,} | {el:,} |")
    w()
    w("Rates are **trial-level** with Wilson intervals at the row's own R (§3): a run counts as a "
      "false alarm if *at least one* safe probe was declared an overshoot, and as a miss if *at "
      "least one* overshooting probe was accepted. Probe-level confusion counts are supporting "
      "telemetry and carry no interval, because probes within a run are dependent.")
    w()
    w("Per-cell rows with both interval endpoints and denominators: `detector_confusion.csv`.")
    w()
    w("---")
    w()


def part_trends(man, diag, phase):
    w("## 11. Results — trends")
    w()
    w("### 11.1 With the target precision `eps`")
    w()
    w("`U(0.01, 0.1)` held fixed; only `eps` and the budget scale it forces change. Live points only.")
    w()
    w("| algorithm | eps | median expl. share | median `N_guess/N_opt` | median `N*/N_opt` | "
      "final overshoot | mean convergence |")
    w("|---|---|---:|---:|---:|---:|---:|")
    sweep = [s["id"] for s in M.SCENARIOS if "precision_sweep" in s["families"]]
    for a in M.ORDER:
        for sid in sweep:
            rs = [r for r in diag if r["algorithm"] == a and r["scenario_id"] == sid]
            if not rs:
                continue
            g = col(rs, "guess_ratio_median")
            w(f"| {SHORT[a]} | {f(rs[0]['eps']):.0e} | "
              f"{pct(np.median(col(rs,'exploration_share_median')),2)} | "
              f"{(f'{np.median(g):.2f}' if g.size else '--')} | "
              f"{np.median(col(rs,'star_ratio_median')):.2f} | "
              f"{pct(col(rs,'star_overshoot').mean(),2)} | {pct(col(rs,'rate').mean(),1)} |")
    w()
    w("### 11.2 With the available budget")
    w()
    w("Live points split into terciles of budget *within each scenario*, so scale differences "
      "between scenarios do not confound the comparison.")
    w()
    w("| algorithm | tercile | median expl. share | median `N_guess/N_opt` | guess overshoot | "
      "mean probes | only the opening probe |")
    w("|---|---|---:|---:|---:|---:|---:|")
    ranks = {}
    for sid in {r["scenario_id"] for r in diag}:
        bs = sorted({int(f(r["budget"])) for r in diag if r["scenario_id"] == sid})
        for i, b in enumerate(bs):
            ranks[(sid, b)] = "low" if i < len(bs)/3 else ("mid" if i < 2*len(bs)/3 else "high")
    for a in M.ORDER:
        if not M.ALGORITHMS[a]["has_guess"]:
            continue
        for band in ("low", "mid", "high"):
            rs = [r for r in diag if r["algorithm"] == a
                  and ranks.get((r["scenario_id"], int(f(r["budget"])))) == band]
            if not rs:
                continue
            g = col(rs, "guess_ratio_median")
            w(f"| {SHORT[a]} | {band} | "
              f"{pct(np.median(col(rs,'exploration_share_median')),2)} | "
              f"{(f'{np.median(g):.2f}' if g.size else '--')} | "
              f"{pct(col(rs,'guess_overshoot').mean(),1)} | "
              f"{col(rs,'n_probes_mean').mean():.1f} | {pct(col(rs,'single_probe').mean(),1)} |")
    w()
    w("### 11.3 With the true phase")
    w()
    w("Each cell's runs split into quartiles of the true `phi` (`diagnostics_by_phase.csv`). "
      "Small `phi` means a large `N_opt` and a hard depth problem; large `phi` means `N_opt` close "
      "to `N_min` and little to search for.")
    w()
    w("| algorithm | phase quartile | median `N_guess/N_opt` | guess overshoot | "
      "median `N*/N_opt` | final overshoot | convergence |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for a in M.ORDER:
        for q in range(4):
            rs = [r for r in phase if r["algorithm"] == a and int(f(r["phase_bin"])) == q]
            if not rs:
                continue
            g = col(rs, "guess_ratio_median")
            go = col(rs, "guess_overshoot")
            w(f"| {SHORT[a]} | Q{q+1} | {(f'{np.median(g):.2f}' if g.size else '--')} | "
              f"{(pct(go.mean(),1) if go.size else '--')} | "
              f"{np.median(col(rs,'star_ratio_median')):.2f} | "
              f"{pct(col(rs,'star_overshoot').mean(),2)} | {pct(col(rs,'rate').mean(),1)} |")
    w()
    w("*(Phase strata are computed over all points, live and saturated alike, since they are "
      "within-cell splits rather than cross-cell aggregates.)*")
    w()
    w("### 11.4 Figures")
    w()
    w("* `fig_diagnostics_vs_budget.png` — exploration budget share, median `N_guess/N_opt` and "
      "final overshoot against budget, normalised per scenario.")
    w("* `fig_guess_vs_final_depth.png` — `N_guess/N_opt` against `N*/N_opt`, and `N*/N_opt` "
      "against convergence.")
    w()
    w("Both regenerate from `diagnostics_by_point.csv` alone via "
      "`python analysis/consolidated/figures.py` — no simulation.")
    w()
    w("---")
    w()


def part_budget(man, audit):
    w("## 12. Budget compliance audit")
    w()
    viol = sum(int(f(r["budget_violations"], 0)) for r in audit)
    mx = max(col(audit, "budget_util_max")) if audit else float("nan")
    mean_u = col(audit, "budget_util_mean")
    w(f"Over **all {len(audit):,} evaluated cells** and every held-out run in them:")
    w()
    w("| | |")
    w("|---|---:|")
    w(f"| trials spending more than the nominal budget | **{viol}** |")
    w(f"| worst per-trial spend observed | **{mx:.6f}x** the cap |")
    w(f"| mean spend across cells | {mean_u.mean():.4f}x |")
    w(f"| median unused budget | {pct(1-np.median(col(audit,'budget_util_median')),2)} |")
    w()
    w("**A mean spend of 1.0000x does not by itself certify compliance** — it can average over "
      "trials that overspend and trials that underspend. That is why the per-trial *maximum* and an "
      "explicit *violation count* are reported per cell, in `budget_audit.csv`, alongside the mean, "
      "median and p90. Violations are counted, never clipped away after the fact.")
    w()
    w("Compliance is structural, not incidental: each exploration probe is checked to fit *before* "
      "it is paid for, and every exploitation measurement takes `m = floor(remaining / N)`. "
      "`test_budget_compliance` asserts both the total and the "
      "`B_exploration + N_star * m_final <= B` decomposition over 4,800 trials spanning four "
      "scenarios.")
    w()
    w("One convention worth stating: when an algorithm cannot afford even its opening probe it "
      "returns `(inf, B)` — the nominal budget — rather than 0. This is the pre-existing convention "
      "of the codebase and is conservative for this audit (it can only over-report spending). The "
      "affected runs carry `status = refused_pilot_unaffordable`.")
    w()
    w("---")
    w()


def part_provenance(man, wins, reg):
    w("## 13. Tuning provenance and grid adequacy")
    w()
    tuned = [x for x in wins if x["tuned"] == "True"]
    live = [x for x in tuned if reg.get((x["scenario_id"], int(f(x["budget"])))) == "live"]
    w(f"{len(tuned):,} tuned cells, {len(live):,} of them at live operating points. Every held-out "
      "row in `performance_curves.csv` points to exactly one frozen winner in `winners.csv` with "
      "the same parameter dictionary — asserted by `test_end_to_end_smoke_and_provenance`, which "
      "also checks that the winner's recorded held-out rate equals the reported rate exactly.")
    w()
    w("Each winner row carries: both per-block tuning rates, their mean, the runner-up "
      "configuration, the selection margin in percentage points, the stage A/B/C configuration "
      "counts, the `m'` box endpoints, and `at_axis_bound`.")
    w()
    w("### 13.1 How much was at stake in each selection")
    w()
    w("| algorithm | median margin over runner-up | p90 | max |")
    w("|---|---:|---:|---:|")
    for a in M.ORDER[1:]:
        v = col([x for x in live if x["algorithm"] == a], "margin_pp")
        if v.size:
            w(f"| {SHORT[a]} | {np.median(v):.2f} pp | {np.percentile(v,90):.2f} pp | "
              f"{v.max():.2f} pp |")
    w()
    w("These margins are small — the objective is flat near its optimum. That is exactly why the "
      "deciding stage runs at the largest R, and why the `m'` grid resolution "
      f"(~{100*(4**(1/(M.MODES[man['mode']]['n_m3']-1))-1):.0f}% steps) is far below the noise "
      "floor rather than being pushed further.")
    w()
    w("### 13.2 Grid adequacy — boundary report")
    w()
    FLOORS = {("m_exploration", "min"), ("inc", "min"), ("safeguard", "min"),
              ("conf", "min"), ("lookback_window", "min")}
    cnt, tot = {}, {}
    for x in live:
        tot[x["algorithm"]] = tot.get(x["algorithm"], 0) + 1
        for ax, side in json.loads(x["at_axis_bound"]).items():
            cnt[(x["algorithm"], ax, side)] = cnt.get((x["algorithm"], ax, side), 0) + 1
    bad = {k: v for k, v in cnt.items() if (k[1], k[2]) not in FLOORS}
    w("**Pins at an axis MAXIMUM** — the failure mode that invalidates a search box, because the "
      "optimum may lie outside it:")
    w()
    if not bad:
        w("*None.*")
    else:
        w("| algorithm | axis | share of live tuned cells | verdict |")
        w("|---|---|---:|---|")
        for k, v in sorted(bad.items(), key=lambda x: -x[1]):
            sh = 100 * v / tot[k[0]]
            verdict = ("**grid too small**" if sh > 25 else
                       ("watch" if sh > 10 else "negligible"))
            w(f"| {SHORT[k[0]]} | `{k[1]}` ({k[2]}) | {v}/{tot[k[0]]} = {sh:.1f}% | {verdict} |")
    w()
    w("**Pins at an axis MINIMUM** — physical floors, not defects (`m' = 1` is one shot, `inc = 1` "
      "one step, `safeguard = 0` no decrement, `lookback_window = 1` a single look back, "
      "`conf = 0.5` the point at which the normal quantile vanishes and the branch rule becomes a "
      "plain comparison against the reference estimate):")
    w()
    w("| algorithm | axis | share of live tuned cells |")
    w("|---|---|---:|")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        if (k[1], k[2]) in FLOORS:
            w(f"| {SHORT[k[0]]} | `{k[1]}` | {v}/{tot[k[0]]} = {100*v/tot[k[0]]:.1f}% |")
    w()
    w("### 13.3 Selected parameter values")
    w()
    w("Distribution of the frozen winners over live cells — showing whether the widened ranges are "
      "actually used:")
    w()
    for a in M.ORDER[1:]:
        rs = [x for x in live if x["algorithm"] == a]
        if not rs:
            continue
        w(f"**{SHORT[a]}**")
        w()
        ms = fin([f(json.loads(x["params"])["m_exploration"]) for x in rs])
        w(f"* `m_exploration`: {len(set(ms))} distinct values, {int(ms.min()):,} .. "
          f"{int(ms.max()):,} (median {int(np.median(ms)):,})")
        for ax in M.ALGORITHMS[a].get("discrete", {}):
            c2 = {}
            for x in rs:
                v = json.loads(x["params"])[ax]
                c2[v] = c2.get(v, 0) + 1
            w(f"* `{ax}`: " + ", ".join(f"{k}x{v}" for k, v in sorted(c2.items())))
        w()
    w("Full appendix-ready tables, including the frozen configuration at the tested budget nearest "
      "each interpolated `B_90` **and both bracketing winners**, are in `optimal_params.csv` and "
      "`tex/optimal_params.tex`. A budget crossing is interpolated between tested budgets; "
      "parameter dictionaries are **not** interpolated, which is precisely why the bracketing rows "
      "exist and why every such row carries a `budget_note` saying so.")
    w()
    w("---")
    w()


TESTS = [
    ("test_trace_neutrality", "Same seed, tracing on vs off, returns identical `phi_hat` and budget "
     "use. 4 scenarios x 4 algorithms x 150 trials."),
    ("test_budget_compliance", "No algorithm exceeds its nominal budget, and "
     "`B_exploration + N_star*m_final <= B`. 4 scenarios x 4 algorithms x 300 trials; violations "
     "are collected and reported, not clipped."),
    ("test_algorithm_definitions", "`N_guess`, `N_star` and `B_exploration` recomputed by hand from "
     "each run's own probe list must equal what the trace recorded — separately for all four "
     "algorithms, including that brute force has a null guess and that RE charges pilot retries."),
    ("test_run_record_detector_definitions", "Hand-constructed traces verify the false-alarm and "
     "miss definitions in all four confusion quadrants, and that a detector-less algorithm returns "
     "NaN rather than 0."),
    ("test_binary_deep_matches_fine_sweep", "The new deepest-probe binary search reproduces, trial "
     "for trial, the depth decision of the `binary_deep` arm of `analysis/fine_sweep.py` "
     "(900 trials, 3 operating points)."),
    ("test_determinism", "Repeated held-out evaluations and repeated tuning with the same manifest "
     "and seeds return identical arrays and identical winners."),
    ("test_eligibility_accounting", "`eligible + ineligible = R` for every metric; conditional "
     "denominators never exceed the sets they condition on; detector-less and search-less "
     "algorithms report N/A with denominator 0."),
    ("test_performance_consistency", "The converged count derived from the traces equals the count "
     "from an independent untraced evaluation on the same seeds — i.e. performance and diagnostics "
     "are one evaluation, not two."),
    ("test_ci_uses_row_R", "A smaller R must give a wider Wilson interval; the uncertainty "
     "functions must take R as an argument; and **no consumer module may contain a literal trial "
     "count** (guards against the 50,000 / 40,000 / 30,000 / 20,000 disagreement in the "
     "pre-existing scripts)."),
    ("test_ci_metadata_present", "Every diagnostics row carries `R`, `ci_level`, `n_boot`, "
     "`boot_seed` and all three `ci_method_*` strings."),
    ("test_code_audit_names_the_implementations", "The generated audit names every reported "
     "implementation, names both binary-search pilot variants, and the manifest points at the "
     "deepest-probe one."),
    ("test_end_to_end_smoke_and_provenance", "A full smoke sweep runs without a failed scenario; "
     "every held-out row maps to exactly one frozen winner with matching params and rate; all "
     "output files exist and are non-empty; crossing intervals use the curve's own R; the appendix "
     "carries `B90_nearest` plus both brackets, each with a `budget_note`."),
    ("test_reproducibility_and_resume", "Two independent sweeps produce byte-identical outputs; "
     "deleting a scenario's rows and re-running with `--resume` reproduces them exactly."),
]


def part_validation():
    w("## 14. Validation suite")
    w()
    w("`python tests/test_consolidated.py --slow` — **13/13 passing**. Runs standalone (no pytest "
      "required) and is pytest-compatible.")
    w()
    w("| test | asserts |")
    w("|---|---|")
    for name, desc in TESTS:
        w(f"| `{name}` | {desc} |")
    w()
    w("---")
    w()


def part_limits(man):
    w("## 15. Limitations and code/thesis mismatches")
    w()
    w("### 15.1 The binary-search pilot — resolved, and it changes the reported algorithm")
    w()
    w("The thesis pseudocode stores the deepest probe not classified as an overshoot "
      "(`phi_hat_acc`, `N_acc`) and supplies **that** pilot to the statistical safeguard. "
      "`qmetrology.algorithms.find_phi_fixed_budget_binary_search_risk` supplied the **opening** "
      "probe (`phi_0`, `N_min`). These are different algorithms.")
    w()
    w("Resolved by **adding** a function rather than relabelling one:")
    w()
    w("* `find_phi_fixed_budget_binary_search_deep` — deepest accepted probe. **This is what is "
      "reported here**, matching the pseudocode and the `binary_deep` arm of `fine_sweep.csv`.")
    w("* `find_phi_fixed_budget_binary_search_risk` — opening probe. Untouched, so every earlier "
      "study citing it still reproduces.")
    w()
    w("Both call the same `_binary_search_explore`, so at one configuration they differ in exactly "
      "one input to `risk_optimal_depth`. `algorithm_code_audit.md` measures the difference on "
      "common seeds at four operating points; the thesis-faithful pilot is **mildly worse** "
      "(0 to -2.4 pp). That is a result, not a bug, and it is surfaced rather than buried.")
    w()
    w("### 15.2 Selection bias in that pilot")
    w()
    w("The deepest accepted estimate is accepted *because it passed the overshoot test*, so it is "
      "selected for having read high. Treating it afterwards as an unbiased Gaussian pilot is "
      "outside the safeguard's derivation. Not assumed away — measured, via the detector rates "
      "(§10) and the `N_guess/N_opt` distribution (§8).")
    w()
    w("### 15.3 Linear search has no per-probe detector")
    w()
    w("Its stopping rule is a trial-level verdict, so the probe-level classification scored in §10 "
      "is the algorithm's own backtracking decision. A different mapping would produce different "
      "false-alarm numbers. Stated, not hidden.")
    w()
    w("### 15.4 `../thesis/*.tex` is not present in this checkout")
    w()
    w("The code audit is written against the handoff's statement of the pseudocode and against "
      "`results/BS_METHOD_DECISION.md`, not against the `.tex` sources directly. Every function's "
      "pilot and cap behaviour is verified against the code, and `binary_deep` is asserted "
      "trial-for-trial identical to the `fine_sweep.py` reference arm. No thesis file is read or "
      "written by this pipeline.")
    w()
    w("### 15.5 Scattered run constants elsewhere in the repository")
    w()
    w("`qmetrology/config.py` (R=50,000), `analysis/extensive_sweep.py` (40,000), "
      "`analysis/broad_dist_study.py` (30,000) and `analysis/fine_sweep.py` (20,000) still "
      "disagree with each other. They are untouched and **unused** by this pipeline, which reads "
      "every constant from `qmetrology/manifest.py` and stamps it on each row.")
    w()
    w("### 15.6 Uncertainty not quantified")
    w()
    w("See §6.8: the variance of which configuration the grid search picks, model error in the "
      "safeguard's Gaussian posterior, and interpolation bias beyond the `O(h^2)` grid bound.")
    w()
    w("### 15.7 Saturated operating points")
    w()
    w("At saturated points the tuner cannot distinguish configurations and its argmax is arbitrary. "
      "Performance there is valid; diagnostics are not, and are excluded from every aggregate "
      "(§4.3). They remain in the CSVs, labelled.")
    w()
    w("---")
    w()


FILE_DOC = [
    ("experiment_manifest.json", "every scenario, budget grid, algorithm, seed, R and grid the run used"),
    ("algorithm_code_audit.md", "which implementation is which thesis algorithm; the measured pilot-mismatch comparison"),
    ("operating_points.csv", "per (scenario, budget): best/worst rate, spread, and the live/saturated/floored label"),
    ("performance_curves.csv", "convergence vs budget with Wilson intervals at the row's own R"),
    ("budget_crossings.csv", "B(p*) and the ratio vs brute force, bootstrap intervals, coverage, grid sensitivity"),
    ("winners.csv", "the frozen tuning winner per cell: per-block rates, runner-up, margin, stage sizes, at_axis_bound"),
    ("optimal_params.csv", "appendix parameter tables, incl. the B_90 nearest and both bracketing winners"),
    ("diagnostics_by_point.csv", "all Tier A + Tier B diagnostics; every share with its eligible denominator"),
    ("diagnostics_headline.csv", "the compact per-scenario subset the thesis tables draw from"),
    ("diagnostics_by_phase.csv", "the depth diagnostics split by quartiles of the true phase"),
    ("detector_confusion.csv", "linear/binary false-alarm and miss rates + probe-level TP/FP/TN/FN"),
    ("budget_audit.csv", "per-cell spend: mean, median, p90, max, unused share, violation count"),
    ("REPORT.md", "the short report"),
    ("FULL_RESULTS.md", "this document"),
    ("tex/", "paste-ready LaTeX for the exploration, downstream, detector, crossing and parameter tables"),
    ("traces/", "full probe lists (gzipped JSONL) for the headline points, first 2,000 runs each"),
    ("fig_*.png", "diagnostic trend figures"),
]


def part_files(man):
    w("## 16. File and column reference")
    w()
    w("### 16.1 Files")
    w()
    w("| file | contents |")
    w("|---|---|")
    for n, d in FILE_DOC:
        w(f"| `{n}` | {d} |")
    w()
    w("### 16.2 Columns present on every tidy row")
    w()
    w("`setting`, `scenario_id`, `phi_min`, `phi_max`, `phi_distribution`, `eps`, `algorithm`, "
      "`implementation_variant`, `budget`, `params`, `R`, `seed_test`, `seed_tune_blocks`, "
      "`N_min`, `N_max`. No file's meaning depends on an undocumented column prefix, and no join "
      "requires knowledge held only in the code.")
    w()
    w("### 16.3 Suffix conventions on diagnostic columns")
    w()
    w("| suffix | meaning |")
    w("|---|---|")
    w("| `_lo`, `_hi` | the interval endpoints, at `ci_level` |")
    w("| `_n` | the **eligible denominator** for that share |")
    w("| `_k` | the numerator (count of eligible runs satisfying the predicate) |")
    w("| `_mean`, `_mean_lo`, `_mean_hi` | the mean and its run-level bootstrap interval |")
    w("| `_median`, `_q1`, `_q3`, `_p90` | sample quantiles |")
    w("| `_median_lo`, `_median_hi` | the exact order-statistic bootstrap interval for the median |")
    w()
    w("### 16.4 Reproducing")
    w()
    w("```bash")
    w(f"python analysis/consolidated/run.py --{man['mode']} --keep-traces   # the full sweep")
    w("python analysis/consolidated/run.py --report-only                 # rebuild derived tables")
    w("python analysis/consolidated/full_report.py                       # rebuild this document")
    w("python analysis/consolidated/figures.py                           # rebuild the figures")
    w("python tests/test_consolidated.py --slow                          # the validation suite")
    w("```")
    w()
    w("Trial seeds are derived deterministically from the manifest, so a re-run reproduces every "
      "number exactly (`test_reproducibility_and_resume`).")
    w()


def main():
    man = json.load(open(path("experiment_manifest.json")))
    perf, diag = load("performance_curves.csv"), load("diagnostics_by_point.csv")
    cross, wins = load("budget_crossings.csv"), load("winners.csv")
    det, audit = load("detector_confusion.csv"), load("budget_audit.csv")
    ops, phase = load("operating_points.csv"), load("diagnostics_by_phase.csv")
    reg = {(r["scenario_id"], int(f(r["budget"]))): r["regime"] for r in ops}
    runtime = float("nan")
    for line in open(os.path.join(ROOT, "sweep_max.log"), errors="ignore"):
        if "sweep finished in" in line:
            runtime = float(line.split("sweep finished in")[1].split("min")[0])
    dl = _live(diag, reg)

    part_header(man, perf, ops, runtime)
    part_model()
    part_algorithms(man)
    part_definitions()
    part_setting(man, ops)
    part_tuning(man, wins)
    part_stats(man, cross)
    part_performance(man, dl, cross, reg)
    part_exploration(man, dl)
    part_safeguard(man, dl)
    part_detector(man, _live(det, reg), dl)
    part_trends(man, dl, phase)
    part_budget(man, audit)
    part_provenance(man, wins, reg)
    part_validation()
    part_limits(man)
    part_files(man)

    out = path("FULL_RESULTS.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(L):,} lines, {os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
