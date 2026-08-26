# Adaptive quantum metrology under resource constraints — code

Reproducible code for the bachelor's thesis *"Evaluating adaptive quantum
metrology protocols for parameter search under resource constraints"*
([main-thesis.pdf](main-thesis.pdf)).

The model estimates an unknown phase φ from a maximally-entangled circuit that
reads out "0" with probability `cos²(Nφ)`; drawing `m` shots and inverting gives
`φ̂ = (1/N)·arccos(√(hits/m))`. A run *converges* when `|φ̂−φ| < ε`; its cost
(budget) is `C = N·m`. We compare a brute-force baseline against adaptive
strategies (linear search, binary search, reverse engineering).

**Everything uses the fixed-budget formulation:** each algorithm is given the same
budget and we measure convergence; the *budget needed for a target reliability* is
found by **sweeping the budget** to the crossing. (The original variable-budget
"run-until-confident" algorithms are archived in `legacy/` — a single exploitation
count over-spends on easy phases, so they under-perform; see
`legacy/analysis_old/re_variable_demo.py`.)

## Layout

```
qmetrology/          # the package — fixed-budget, fully seeded
  sim.py             #   circuit simulation + estimator (fast binomial sampler)
  algorithms.py      #   fixed-budget brute / separable / linear / binary(+anneal) / reverse-eng (+ RE-lite)
  safeguard.py       #   exploitation depth from the asymptotic law — replaces the tuned C_safe / s
                     #   (derivation + assumptions: results/SAFEGUARD_DERIVATION.md)
  oracle.py          #   the omniscient ceilings: exact N_opt, the aliasing bound, and the Eq.(3.4) ceiling
  uncertainty.py     #   Wilson + bootstrap error bars, and the curve-crossing they are built around
  experiments.py     #   parallel, explicitly-seeded Monte-Carlo evaluator (+ error_quantiles)
  config.py          #   seeds, settings, per-cell params, grids, published values
  tables.py          #   fixed-budget %-converged numbers (Tables 3.1, 3.3)
  latex.py           #   the LaTeX emitters (booktabs/makecell style used by the manuscript)
  reproduce.py       #   headless entry point -> results/RESULTS.md
analysis/
  extensive_sweep.py #   the budget sweep (de-biased) -> story_{curves,cube,winners}.csv   [--fine | --max]
  oracle_curves.py   #   add the oracle ceilings to a sweep already on disk (no re-tuning needed)
  error_curves.py    #   estimator-error distribution vs budget -> error_curves.csv
  uncertainty.py     #   error bars for every reported number -> story_cube_ci.csv, UNCERTAINTY.md
  tuning_stability.py#   the one component that needs re-simulation: grid-search selection variance
  render_story.py    #   all four figures (edit PAPER_RC to restyle for the paper)
  make_results.py    #   assemble results/RESULTS.md (Tables 3.1/3.2/3.3 + beyond-the-paper study)
  make_tex.py        #   every thesis table as paste-ready .tex -> results/tex/
  safeguard_study.py #   tuned-constant vs statistical safeguard, head to head -> results/SAFEGUARD.md
  re_share_sweep.py  #   exploration size as a budget share vs a shot count -> results/re_share*.csv
  pilot_share_report.py #  ... and its figure + summary -> fig_pilot_share.png, pilot_share_summary.csv
  exploration_study.py# linear search + the statistical safeguard (a negative result, see SAFEGUARD_DERIVATION.md 11)
  binary_pilot_study.py# which bisection probe feeds Eq. (3.8) -> results/binary_pilot.csv (12)
  binary_depth_study.py# should the depth just be the bisection's bound L -> binary_depth.csv (13)
results/             # RESULTS.md, all_numbers.csv, the CSVs, the figures, and tex/ (paste-ready tables)
reproduce.ipynb      # one notebook: estimator demo + results tables + figures + a grid you can tweak
notebooks/           # safeguard_playground.ipynb — self-contained sandbox for the new safeguard
                     # binary_pilot.ipynb        — which probe is the pilot, and should N = L (self-contained)
legacy/              # archived: variable-budget algorithms, conv95, old notebooks/scripts/results
```

## Consolidated results + algorithm diagnostics (start here)

One command regenerates every Chapter 4 performance number, its uncertainty, the exploration/
safeguard diagnostics and the appendix parameter tables — all from the *same* held-out Monte-Carlo
runs, so a reported value and its telemetry can never come from different trials.

```bash
python analysis/consolidated/run.py --quick          # ~2 min smoke run, identical code path
python analysis/consolidated/run.py --full           # production sweep (~1 h on 24 cores)
python analysis/consolidated/run.py --full --resume  # continue an interrupted sweep
python analysis/consolidated/run.py --full --keep-traces   # + full probe lists at headline points
python analysis/consolidated/run.py --report-only    # rebuild REPORT.md + derived tables, no sim
python analysis/consolidated/full_report.py         # rebuild FULL_RESULTS.md, no sim
python analysis/consolidated/thesis_tables.py      # thesis tables, SHORT (default)
python analysis/consolidated/thesis_tables.py --long   # ... per-scenario, for an appendix
python analysis/consolidated/thesis_figures.py     # the six Chapter-4 figures
python analysis/consolidated/error_curves.py       # |phi_hat-phi| quantiles (needed by fig_error)
python tests/test_consolidated.py --slow             # the validation suite
```

Two side studies answer specific review questions and write their own CSVs into
`results/consolidated/` without touching the sweep (see the `.md` next to each):

```bash
python analysis/consolidated/overshoot_criterion.py           # -> OVERSHOOT_CRITERION.md
python analysis/consolidated/linear_search_claims.py          # audits linear_search_revision_notes.md
python analysis/consolidated/linear_detector_study.py --curve # -> LINEAR_SEARCH.md
```

**Reported algorithms** (`qmetrology/manifest.py`, `ORDER`) — and only these:

| key | algorithm | tuned parameters |
|---|---|---|
| `brute` | brute force (Algorithm 3) | none |
| `linear` | linear search (Algorithm 4) | `m'`, `lookback_window`, `s`, `inc` |
| `binary_deep` | binary search + statistical safeguard, pilot = **deepest accepted probe** | `m'`, `conf` |
| `reverse_eng_risk` | reverse engineering + statistical safeguard | `m'` |

`binary_deep` (`find_phi_fixed_budget_binary_search_deep`) is new and it *replaces* what earlier
sweeps reported as binary search: `find_phi_fixed_budget_binary_search_risk` feeds the **opening**
probe to Eq. (3.8), the thesis pseudocode feeds the deepest probe the bisection did not flag. Both
functions now exist side by side and `results/consolidated/algorithm_code_audit.md` measures what
the difference costs on common seeds.

Everything the pipeline uses — scenarios, budget grids, tuning grids, seeds, `R_tune`, `R_test`,
bootstrap replicates — lives in `qmetrology/manifest.py`, and every output row stamps the values it
actually ran with. Parameters are tuned on two disjoint seed blocks (42, 43), frozen, and only then
evaluated on the held-out seed 2024, so no reported number is a maximum over its own trials.

Outputs, in `results/consolidated/`:

```
experiment_manifest.json     every scenario/algorithm/seed/R the run used
algorithm_code_audit.md      which function is which thesis algorithm + the pilot-mismatch measurement
performance_curves.csv       convergence vs budget, Wilson intervals at the row's own R
budget_crossings.csv         B(p*) and the ratio vs brute force, bootstrap intervals + grid sensitivity
winners.csv                  the frozen tuning winner, its per-block evidence, runner-up and margin
optimal_params.csv           appendix parameter tables (incl. the B_90 nearest/bracketing winners)
operating_points.csv         per (scenario, budget): best rate and the live/saturated/floored label
diagnostics_by_point.csv     Tier A + Tier B diagnostics, every share with its eligible denominator
diagnostics_headline.csv     the compact per-scenario subset the thesis tables draw from
diagnostics_by_phase.csv     the same depth diagnostics split by quantile bins of the true phase
detector_confusion.csv       linear/binary false-alarm and miss rates + probe-level TP/FP/TN/FN
budget_audit.csv             per-trial spend (mean/median/p90/max) and the violation count
FULL_RESULTS.md              the complete standalone record: model, algorithms, every
                             scenario/seed/R, full statistical methodology, all results,
                             validation and limitations  <- START HERE
REPORT.md                    the short write-up
tex/                         paste-ready LaTeX
tex/thesis/                  UPDATED versions of the Chapter-4 thesis tables (tab_summary_low_prec,
                             tab_summary_all, tab_scaling_with_prec, tab_broad_pi2/pi4,
                             tab_opt_param_first/second) + the three NEW compact diagnostics
                             tables (tab_diag_exploration/downstream/detector), each with a
                             _ci variant where intervals apply; all_thesis_tables.tex bundles them.
                             `results/tex/` is the OLD pre-consolidation set - see its STALE.md
fig_story / _small           convergence vs budget, headline scenario, with the 90% crossing
fig_pareto                   the budget/convergence frontier across all ten scenarios
fig_precision                budget ratio vs brute force at 90%, against epsilon
fig_error                    estimator-error distribution vs budget (median + IQR)
fig_broad                    convergence vs budget under the broad priors
fig_diagnostics_vs_budget    exploration cost and depth quality vs budget
fig_guess_vs_final_depth     what the safeguard does to the exploration's guess
error_curves.csv             |phi_hat - phi| quantiles behind fig_error
traces/                      full probe lists at the headline points (--keep-traces)
```

Every aggregate in `REPORT.md` is over the **live** operating points only — those where the best
algorithm converges between 3% and 99%. At a saturated point every configuration ties, so the
tuner's argmax, and every diagnostic measured at it, is Monte-Carlo noise rather than a property of
the algorithm. The per-point CSVs keep all points; join on `operating_points.csv` to filter.

`../thesis/*.tex` is never read or written; the LaTeX under `results/consolidated/tex/` is a
proposal to paste, not an edit.

The instrumentation is *neutral*: every reported algorithm takes an optional
`trace=qmetrology.trace.AlgorithmTrace()`, the collector draws no random numbers, and
`tests/test_consolidated.py::test_trace_neutrality` asserts that a same-seed run returns identical
`phi_hat` and budget use with tracing on and off.

## Reproduce

```bash
pip install numpy scipy tqdm pandas matplotlib
python analysis/extensive_sweep.py         # budget sweep (default ~13 min; --fine ~1 h; --max ~4.5 h)
python analysis/error_curves.py            # error-distribution curves (reads the sweep's winners)
python analysis/uncertainty.py             # error bars from the sweep, no re-simulation (~1 min)
python analysis/render_story.py            # fig_story / fig_error / fig_pareto / fig_precision
python analysis/safeguard_study.py         # tuned vs statistical safeguard (~15 min)
python -m qmetrology.reproduce             # assemble results/RESULTS.md (recomputes Tables 3.1/3.3, ~2-3 min)
python analysis/appendix_params.py         # optimal-parameter appendix -> optimal_params.{md,csv}
python analysis/make_tex.py                # every table as paste-ready LaTeX -> results/tex/

# exploration-size study (only needed if the sweep is re-run; ~2 h)
python analysis/re_share_sweep.py          # tuned m' vs a fixed budget share, over the whole sweep
python analysis/pilot_share_report.py      # -> results/fig_pilot_share.png + pilot_share_summary.csv
```

If a sweep is already on disk and you only want the omniscient ceilings added to it,
`python analysis/oracle_curves.py` evaluates them on the existing budget grid (~35 min) instead of
re-running the whole sweep — both oracles are parameter-free, so nothing has to be re-tuned.

### Reporting the numbers

`analysis/uncertainty.py` attaches a 95% interval to every reported quantity **without re-running any
simulation**: convergence rates are binomial proportions over R = 40,000 trials (Wilson interval,
≤ ±0.49 pp), and the derived budget-to-reach-p\* and its ratio get percentile intervals from a
parametric bootstrap of the convergence curve. It also reports how much of the crossing is moved by
the finite budget grid rather than by Monte-Carlo noise. The one component it cannot recover from
stored output is the variance of *which parameter configuration the grid search picks*;
`analysis/tuning_stability.py` measures that directly by repeating only the tuning step on several
seeds at the headline cells. See `results/UNCERTAINTY.md`.

Or open **[reproduce.ipynb](reproduce.ipynb)** — it demonstrates the estimator (the
`SimulateEntangledProtocol` algorithm), shows the reproduced tables and the four figures, and ends
with a grid-search cell you can tweak and re-run.

Everything is recomputed live with explicit seeds (`SEED_TUNE=42`, `SEED_TEST=2024`,
`SEED_UNBIASED=12345`); nothing is cached. Every adaptive number is **de-biased**:
parameters are grid-tuned on seed 42 and the winning config is re-validated on the
independent seed 2024 at **R = 40,000 trials**. The circuit readout uses a **binomial**
sampler (`hits ~ Binomial(m, cos²Nφ)`, identical in distribution to per-shot sampling,
~1000× faster for large `m`) — this is what makes the ε=10⁻⁸ / ~10¹⁴-budget sweeps feasible.

### Tune the grids yourself

`extensive_sweep.py` exposes the search via `--fine` / `--max` and the `grids()` /
`SCENARIOS` definitions at the top; `render_story.py` centralises plot style in `PAPER_RC`.
The convergence-vs-budget data behind every figure is in `results/story_curves.csv`.

## Findings — read before citing

Full write-up in **[results/RESULTS.md](results/RESULTS.md)**; the thesis-side edit list is in
**[results/THESIS_CHANGES.md](results/THESIS_CHANGES.md)**. All numbers below are de-biased
(grid-tuned on seed 42, validated on seed 2024 at R = 40,000).

**Binary search and reverse engineering no longer have a tuned safeguard.** The constants `C_safe`
and `s` are replaced by a depth derived from the same asymptotic law (Eq. 3.4) that already underpins
the binary-search overshoot criterion — see [results/SAFEGUARD_DERIVATION.md](results/SAFEGUARD_DERIVATION.md)
for the derivation and its assumptions, and [notebooks/safeguard_playground.ipynb](notebooks/safeguard_playground.ipynb)
to play with it. Wherever this README says "binary search" or "reverse engineering" it means that
version; the published constant-`C` numbers appear only as the delta.

1. **Table 3.1 (fixed budget 10,000):** brute 56.2%, linear 56.6%, binary 52.5%, reverse engineering
   **66.3%** — against the paper's stale 14.0% / 11.5%. The attainable ceiling is 73.5%.
2. **Table 3.2 (budget to reach 90%):** reverse engineering needs **1.47–1.86× less budget** than
   brute across the four columns, binary search 1.04–1.55×. The advantage grows with precision and
   **saturates at ×1.76 (reverse eng.) and ×1.80 (binary)** from ε ≤ 10⁻⁶, where the two draw level:
   binary is ahead at the 90/95% thresholds and behind at 50%, and the mean gap on the convergence
   curves is within one standard error. Reverse engineering's lead is a *low*-precision phenomenon.
3. **Table 3.3 (broad distributions):** reverse engineering now leads every column
   (22.5 / 31.7 / 42.0 / 52.1% against brute's 15.8 / 22.1 / 30.8 / 42.5%).
4. **Table 3.4 (φ ~ U(0.01, π/2)):** reverse engineering beats linear search at *every* budget, and
   binary search from 393,597 upward — the opposite of the published "linear dominates everywhere".
   Two separate causes: the published exploration grid was capped at `m' ≤ 3000` (an artefact, worth
   38% → 90%), and the statistical safeguard on top (90% → 100%).
5. **How good is this in absolute terms?** Granting perfect knowledge of the optimal depth
   `N = ⌊π/2φ⌋` but still requiring the estimate to be resolved by Eq. (3.4) gives a ceiling of
   **1.83–1.97×**. Reverse engineering realises **61% of it at ε=10⁻³ and ~90% at ε≤10⁻⁴**, and at
   broad priors sits within **0.1–2.9 pp** of it — so the remaining headroom is in the *low*-precision
   regime. This analytic ceiling is the only one reported: an oracle scored on the exact estimator
   converges off the deterministic readout at `Nφ ≈ π/2` rather than from the data, which inflates it.
6. **Reverse engineering's exploration size is quoted as a budget share, not a shot count.**
   ρ = 2% of the budget (floor 20 shots) costs **−0.06 pp** against grid-tuning `m'` at every budget
   separately, over all 546 live operating points of the 23-scenario sweep. The previously stated
   `m'=200` costs **−5.55 pp** (worst −99.4 pp: at `U(0.001,0.01)`, B=6,394 a 200-shot pilot costs more
   than the whole budget, so the trial is refused). This is a *reporting* change — the tuned rows are
   unchanged — see `results/fig_pilot_share.png` and `results/tex/tab_pilot_share.tex`.
7. **The safeguard is also more stable.** Removing the tuned constant cuts the tuning-selection
   variance ~3× for reverse engineering (±0.57% vs ±1.53% budget-equivalent over 5 tuning seeds), and
   its winning `m_exploration` is a median 5.3× smaller — the rule can act safely on a coarse pilot.

### Overnight (`--max`) configuration

`--max` is the wide/fine production configuration used for the final numbers:

| | `--full` | `--max` |
|---|---:|---:|
| budget points per scenario | 16 | 22 |
| operating points | 161 | 221 |
| `R_test` (held-out) | 40,000 | 50,000 |
| `R_tune` per block, stages A/B | 800 | 700 |
| `R_tune` per block, stage C (decides) | 1,500 | **6,000** |
| `m'` grid points, stages A/B/C | 16/9/9 | 24/13/15 |
| linear discrete combinations | 96 | **450** |
| `conf` values | 7 | 13 |

Tuning is three-stage, because a 450-combination discrete grid crossed with a 24-point `m'` grid
would be 10,800 configurations per stage:

* **A — locate `m'`**: the whole feasible box `[1, B // N_min]` against a 3-point-per-axis skeleton
  of the discrete grid.
* **B — search the discrete grid**: the *full* grid at `m'` within ×4 of A's winner.
* **C — refine**: `m'` within ×2 of B's winner (~10% steps) over neighbouring discrete values, at
  `R_tune2`. This is the stage that decides, so it gets the trials — it controls the *variance* of
  which configuration wins, the one uncertainty component that cannot be recovered from stored
  output afterwards.

`winners.csv` carries `at_axis_bound`, which flags a winner pinned at an endpoint of **any** axis,
not just `m'`. That flag exists because the first production sweep selected the top of the old
`lookback_window` grid `[1,2,5]` in 64.8% of live cells and the top of `safeguard` `[0,1,2]` in
15.7% — the optimum was outside the box and nothing reported it. The grids now run to
`lookback_window` 20, `safeguard` 24 and `inc` 8; `safeguard` needed the extra headroom because it
is an *absolute* depth decrement, so its useful range grows with `N_min`.

For unattended runs, `analysis/consolidated/supervise.sh` restarts the sweep with `--resume` if the
process dies, up to 12 times, and stops as soon as `REPORT.md` is written.
