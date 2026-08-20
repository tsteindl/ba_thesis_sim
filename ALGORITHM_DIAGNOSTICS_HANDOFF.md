# Consolidated results and algorithm diagnostics - implementation hand-off

Date: 2026-08-20

## Objective

Build one reproducible experiment pipeline that evaluates the algorithms used in the current thesis and produces, from the same held-out Monte Carlo runs:

1. the existing Chapter 4 performance results;
2. statistically correct uncertainty reporting;
3. diagnostics of the exploration phase and the final safeguard decision; and
4. the winning-parameter tables used in the appendix.

The diagnostics are not a separate experiment family. Instrument the experiments already needed for the Chapter 4 results, so that every performance value and its diagnostics refer to the same algorithm variant, configuration, scenario, budget, phase draws, and test run.

Do not edit any file under `../thesis/`. The repository may generate proposed Markdown, CSV, figures, and paste-ready LaTeX under `ba_thesis_sim/results/`, but thesis changes are recommendations for the user to apply separately.

## Sources of truth and preflight audit

Use the current thesis files as the source of truth for the algorithms and the experiment matrix:

- `../thesis/03-methodology.tex`
- `../thesis/04-results.tex`
- `../thesis/04-broad-dist.tex`
- `../thesis/91-appendix.tex`

Treat existing files in `results/` and older Markdown studies as evidence and implementation starting points, not as authoritative final values. Several studies concern different binary-search variants.

Before a production run, create a short machine-readable/text audit recording which implementation function corresponds to each thesis algorithm. There is already at least one material mismatch that must not be silently ignored:

- The current thesis binary-search pseudocode stores the deepest probe not classified as an overshoot (`phi_hat_acc`, `N_acc`) and supplies that pilot to the statistical safeguard.
- `qmetrology.algorithms.find_phi_fixed_budget_binary_search_risk` currently supplies the opening probe (`phi_0`, `N_min`) to `risk_optimal_depth`.

Implement or select an explicit function that matches the intended thesis algorithm. Do not overwrite or relabel an existing variant until same-seed comparisons and tests show what changed. If resolving the mismatch changes the reported algorithm rather than merely adding telemetry, surface it clearly in the generated report before treating the output as final.

Also consolidate currently scattered run constants. `qmetrology/config.py`, `analysis/extensive_sweep.py`, and `analysis/broad_dist_study.py` currently use different `R_TEST` values. Every output row must record its actual `R`, seeds, algorithm function/variant, and parameter configuration; uncertainty code must read these values rather than assume a global default.

## Minimal thesis-facing definitions the code must support

The thesis keeps the existing definition

```text
N_opt = floor(pi / (2 phi))
```

using the same lower bound of 1 as the simulation where necessary.

The primary exploration diagnostic is `N_guess / N_opt`. The final-depth diagnostic `N_star / N_opt` is secondary and measures the effect of the safeguard and final allocation.

Define `N_guess` consistently as follows:

### Linear search

`N_guess` is the depth returned by the search/stopping rule after its prescribed lookback backtracking, but before the separate safeguard decrement `s` is applied.

With the current implementation this is:

```text
N_guess = N_list[-1] - lookback_window * inc   if the stopping detector fired
          N_list[-1]                           otherwise
```

The actual exploitation depth after the safeguard is `N_star`.

### Binary search

`N_guess` is the exploration phase's deepest probed depth not classified as an overshoot:

```text
N_guess = N_acc = L
```

where `L` is the final lower/accepted bound in the current thesis pseudocode. Record the opening and deepest-accepted pilots separately because the codebase contains variants using either one in the statistical safeguard.

### Reverse Engineering

Keep the definition already present in the thesis:

```text
N_guess = floor(pi / (2 phi_hat_0))
```

Record this raw guess even if the statistical safeguard later chooses a different or support-capped `N_star`.

### Brute force and separable baselines

They have no exploration guess. Store `N_guess` as missing/N/A, `B_exploration = 0`, and record their actual final depth as `N_star`. Do not manufacture a guess for an algorithm that does not search.

### Exploration budget

`B_exploration` is the sum of `N_i * m_i` over every pilot/search probe before the final exploitation measurement. Include repeated pilot attempts (for example a retry after a zero Reverse Engineering estimate). Define

```text
exploration_share = B_exploration / nominal_budget.
```

Also retain total budget consumed and unused budget so equal-budget compliance can be audited.

## Experiment matrix

Do not invent a new headline experiment family. Build one canonical scenario manifest covering the experiments already used by Chapter 4:

1. Fixed-budget low-precision comparison: `phi ~ U(0.01, 0.1)`, `eps = 1e-3`, `B = 10,000`.
2. Budget-to-reliability comparisons, including the selected `eps = 1e-4` priors `U(0.01, 0.1)`, `U(0.001, 0.01)`, and `U(0.001, 0.1)`, plus the low-precision standard scenario.
3. Precision sweep for `U(0.01, 0.1)` over the epsilon values reported in the current results/appendix.
4. Broad-prior experiments used by `04-broad-dist.tex`, especially `U(0.01, pi/2)`, `eps = 1e-3`, over its reported budget curve; retain the narrower broad-prior cases needed by its comparison figure/table.

The manifest, not separate hard-coded lists in several scripts, should define scenario IDs, prior limits/distribution, epsilon, budgets or budget-grid rule, algorithms, tuning grids, tuning seeds, test seeds, `R_tune`, and `R_test`.

Collect diagnostics at every evaluated test point when this can be done in the same call. The main thesis will later select only a compact subset or aggregate summaries. Do not run a second diagnostic simulation when the same test evaluation can emit both performance and telemetry.

## Tuning and held-out evaluation

The final reported values must not be maxima evaluated on the same trials used for parameter selection.

Required procedure:

1. Evaluate each candidate configuration on an explicitly listed set of tuning seeds/trials.
2. Select the configuration using only the tuning results. Prefer multiple tuning seed blocks so selection is not determined by one accidental draw; record the aggregation rule.
3. Freeze the winner.
4. Evaluate the frozen winner on disjoint held-out test seeds/trials.
5. Compute both performance and diagnostics from that held-out evaluation.

The appendix must report the frozen configuration and the tested budget it belongs to. A budget crossing is interpolated between tested budgets, but parameter dictionaries are not interpolated. For a `B_90` parameter table, either report the winner at the nearest tested budget and say so, or report the two bracketing winners. Do not present a nearest-grid configuration as an exact optimum at an interpolated budget without qualification.

## Trace design

Instrument the algorithms without changing their stochastic behavior. A same-seed run with tracing disabled and enabled must return the same `phi_hat` and budget use.

A suggested internal trace model is:

```text
AlgorithmTrace
  algorithm / implementation_variant
  phi
  phi_hat_final
  converged
  nominal_budget
  budget_used_total
  budget_exploration
  N_opt
  N_guess          # null for non-search baselines
  N_star           # actual final exploitation depth, null if exploitation never ran
  m_final
  status / termination_reason
  opening_pilot_N, opening_pilot_phi_hat
  accepted_pilot_N, accepted_pilot_phi_hat
  probes[]

ProbeTrace
  stage_index
  N
  m
  phi_hat
  declared_overshoot   # true/false/null where no detector exists
  true_overshoot       # N > N_opt; simulation-only ground truth
  accepted             # where meaningful
```

Preserve the existing public `(phi_hat, budget_used)` behavior by default. Use an optional trace return or a trace collector passed into shared algorithm cores. Avoid maintaining separate copied implementations solely for diagnostics.

Do not store every full probe trace for the complete production sweep by default. At each operating point, keep the per-run scalar arrays in memory long enough to compute aggregates and bootstrap intervals, then write aggregate rows. Provide an optional `--keep-traces` mode for selected headline/debug points, using a compressed format such as `.npz` or JSONL gzip.

## Diagnostic definitions

All percentages must include their eligible denominator in the output. Runs with no defined `N_guess` or no exploitation phase must not silently disappear.

### Tier A - required core diagnostics

These are required in the consolidated output even if the thesis later reports only a subset.

#### Exploration depth quality

For adaptive algorithms and eligible runs:

- mean and median `N_guess / N_opt`;
- first and third quartiles of `N_guess / N_opt`;
- mean absolute relative distance `abs(N_guess - N_opt) / N_opt`;
- signed relative error `(N_guess - N_opt) / N_opt`;
- exact-hit share `P(N_guess == N_opt)`;
- within-5-percent and within-10-percent shares;
- guess overshoot share `P(N_guess > N_opt)`.

The mean ratio, signed error, and absolute error are all retained because a mean ratio alone can hide a mixture of severe under- and overshoots.

#### Exploration cost

- mean, median, first/third quartiles, and 90th percentile of `B_exploration / B`;
- mean and median number of exploration probes;
- share of runs with only the opening probe/no meaningful search step;
- share with no exploitation phase because exploration consumed the available budget;
- termination-reason counts.

#### Detector quality

For linear and binary search only, compare each algorithm decision with simulation truth `N_i > N_opt`:

- trial-level false-alarm share: at least one safe probe was declared an overshoot;
- trial-level miss share: at least one overshooting probe was accepted/not flagged;
- probe-level confusion counts (TP, FP, TN, FN) as supporting telemetry.

Use run-level bootstrap or run-level indicator intervals for the primary rates because probes within one run are dependent. Reverse Engineering and the baselines report these metrics as N/A.

#### Final depth and outcome

- mean/median/IQR of `N_star / N_opt`;
- final overshoot share `P(N_star > N_opt)`;
- convergence share conditional on a non-overshooting final depth;
- share of runs without a defined `N_star`.

### Tier B - compute if cheap, thesis inclusion optional

- exact/within-5-percent/within-10-percent rates for `N_star`;
- distribution of `N_star / N_guess`;
- unsafe-guess rescue share `P(N_star <= N_opt | N_guess > N_opt)`;
- backoff distribution for already-safe guesses;
- phase-stratified diagnostics by fixed phase quantiles or bins;
- budget-utilization and unused-budget distributions;
- accepted/rejected probe counts for binary search;
- diagnostic trends versus budget and epsilon.

Do not label early stopping, final overshoot, and ordinary shot-noise error as a mutually exclusive causal failure decomposition unless a defensible priority/counterfactual rule is specified. It is safe to report them as overlapping stage flags and conditional rates.

## Statistical reporting

Reuse and consolidate the sound parts of `qmetrology/uncertainty.py` and `analysis/uncertainty.py`.

Required outputs:

- convergence proportions: 95% Wilson intervals with the actual row-specific `R`;
- diagnostic proportions: 95% Wilson intervals for simple run-level indicators, or run-level percentile bootstrap intervals where derived from clustered probe data;
- continuous diagnostic summaries: median/IQR plus a run-level percentile bootstrap interval for the median; retain mean and bootstrap interval in the data even if not shown in the thesis;
- budget-to-threshold values: state and use one crossing rule (the current implementation uses log-linear interpolation);
- budget-to-threshold and ratio intervals: parametric bootstrap of the convergence curve;
- budget-grid sensitivity: preserve the existing grid-discretisation diagnostic;
- appropriate rounding: budgets to about three significant figures, ratios to two decimals unless the interval supports more, and percentages to at most one or two decimals.

Every CI output must identify the confidence level, method, number of bootstrap replicates, seed, and denominator/sample size.

## Outputs

Write new work under a stable directory such as `results/consolidated/` until it has passed validation. Suggested files:

```text
results/consolidated/experiment_manifest.json
results/consolidated/algorithm_code_audit.md
results/consolidated/winners.csv
results/consolidated/performance_curves.csv
results/consolidated/budget_crossings.csv
results/consolidated/diagnostics_by_point.csv
results/consolidated/detector_confusion.csv
results/consolidated/diagnostics_headline.csv
results/consolidated/optimal_params.csv
results/consolidated/budget_audit.csv
results/consolidated/REPORT.md
results/consolidated/tex/*.tex
```

Each CSV should use tidy rows with explicit `setting`, `phi_min`, `phi_max`, `phi_distribution`, `eps`, `algorithm`, `implementation_variant`, `budget`, seeds, `R`, and eligibility denominator columns. Avoid wide files whose meaning depends on undocumented column prefixes.

`REPORT.md` should distinguish:

1. performance findings;
2. exploration-phase findings centered on `N_guess / N_opt`;
3. safeguard/final-depth findings centered on `N_star / N_opt`;
4. statistical uncertainty and grid sensitivity;
5. code/thesis mismatches or unresolved limitations.

Generate appendix-ready parameter tables and candidate diagnostic tables, but do not modify `../thesis/*.tex`.

## Main-thesis reporting candidates

The production pipeline should make it easy to generate, without rerunning simulations:

1. A compact exploration table with algorithm, median `N_guess/N_opt` (IQR), exact/within-10-percent shares, guess overshoot share, median exploration-budget share, and probe count.
2. A compact downstream table or figure comparing `N_guess/N_opt` with `N_star/N_opt` and showing final overshoot/convergence.
3. A detector table for linear and binary search with false-alarm and miss rates.
4. Optional budget-trend figures showing how exploration share and depth quality change as the available budget grows.

The thesis will decide later which of these are important enough for the main text. The pipeline should compute them all once so selection does not require a new simulation.

## Tests and validation

At minimum add automated or executable checks for:

1. **Trace neutrality:** same seed and parameters produce identical `phi_hat` and budget use with tracing on/off.
2. **Budget compliance:** no fixed-budget algorithm spends more than its nominal budget; explicitly count/report violations rather than clipping them after the fact.
3. **Determinism:** repeated quick runs with the same manifest and seeds reproduce outputs.
4. **Definition checks:** hand-constructed traces verify each algorithm's `N_guess`, `N_star`, `B_exploration`, false-alarm, and miss definitions.
5. **Eligibility accounting:** every metric satisfies `eligible + ineligible = total test runs`.
6. **Performance consistency:** success counts derived from traces equal the performance curve count for the same evaluation.
7. **CI consistency:** intervals use the stored row-specific `R`; no hard-coded `40,000` or `50,000` mismatch.
8. **Winner provenance:** every reported held-out result points to exactly one frozen tuning winner and its tuning evidence.
9. **Thesis/code parity:** the generated code audit names the implementation and confirms the pilot/safeguard behavior for linear, binary, and Reverse Engineering.
10. **Quick/full modes and resume:** provide a small deterministic smoke run, a production mode, and resumable per-scenario/per-budget outputs.

## Completion criteria

The task is complete when:

- one documented command (plus an optional full/heavy flag) regenerates the consolidated outputs;
- all current Chapter 4 experiment families are represented by one manifest;
- the same held-out evaluations produce performance, diagnostics, and uncertainty outputs;
- `N_guess` is implemented and tested exactly as defined above;
- parameter tuning and held-out testing are separated and recorded;
- all proportions, crossings, and diagnostics have correct denominators and uncertainty metadata;
- appendix-ready winning-parameter tables are generated from the same winners used in the results;
- the binary-search thesis/code pilot mismatch is resolved explicitly or remains prominently flagged, never silently mixed;
- existing thesis `.tex` files remain unchanged.

