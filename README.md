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
