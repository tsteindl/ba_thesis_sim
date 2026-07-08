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
  experiments.py     #   parallel, explicitly-seeded Monte-Carlo evaluator (+ error_quantiles)
  config.py          #   seeds, settings, per-cell params, grids, published values
  tables.py          #   fixed-budget %-converged numbers (Tables 3.1, 3.3)
  reproduce.py       #   headless entry point -> results/RESULTS.md
analysis/
  extensive_sweep.py #   the budget sweep (de-biased) -> story_{curves,cube,winners}.csv   [--fine | --max]
  error_curves.py    #   estimator-error distribution vs budget -> error_curves.csv
  render_story.py    #   all four figures (edit PAPER_RC to restyle for the paper)
  make_results.py    #   assemble results/RESULTS.md (Tables 3.1/3.2/3.3 + beyond-the-paper study)
results/             # RESULTS.md, all_numbers.csv, the CSVs, and the four figures
reproduce.ipynb      # one notebook: estimator demo + results tables + figures + a grid you can tweak
legacy/              # archived: variable-budget algorithms, conv95, old notebooks/scripts/results
```

## Reproduce

```bash
pip install numpy scipy tqdm pandas matplotlib
python analysis/extensive_sweep.py         # budget sweep (default ~13 min; --fine ~1 h; --max ~4.5 h)
python analysis/error_curves.py            # error-distribution curves (reads the sweep's winners)
python analysis/render_story.py            # fig_story / fig_error / fig_pareto / fig_precision
python -m qmetrology.reproduce             # assemble results/RESULTS.md (recomputes Tables 3.1/3.3, ~2-3 min)
```

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

Full write-up in **[results/RESULTS.md](results/RESULTS.md)**. In short:

1. **Table 3.1 (fixed budget 10,000):** narrow-range linear/binary are ~56%/48%, **not** the
   paper's stale 14.0%/11.5%. Reverse engineering ~60%, brute ~56%.
2. **Table 3.2 (budget to reach 90%):** reverse engineering needs **~1.2–1.6× less budget**
   than brute across the columns; binary/separable lose. The advantage **grows with precision
   and saturates at ~1.72×** (ε ≤ 10⁻⁵), and is range-insensitive under the uniform prior.
3. **Table 3.3 (broad distributions):** paper adaptive numbers were grid maxima (winner's curse);
   de-biased they drop, but linear and reverse-engineering still beat brute.
