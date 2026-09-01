# Adaptive quantum metrology under resource constraints

Code and results for the bachelor's thesis *"Evaluating adaptive quantum metrology protocols for
parameter search under resource constraints"* (`main-thesis_newest.pdf`).

An unknown phase φ is estimated from a maximally-entangled circuit that reads out "0" with
probability `cos²(Nφ)`; drawing `m` shots and inverting gives `φ̂ = arccos(√(hits/m))/N`. A run
*converges* when `|φ̂−φ| < ε`; its cost is the budget `C = N·m`. Every algorithm is **fixed-budget**:
it is handed the same `C` and we measure convergence, and the budget needed for a target reliability
is found by sweeping `C` to the crossing. (A variable-budget "run until confident" formulation
over-spends on easy phases, which is why it is not used.)

The whole question is how to choose `N`. It must stay below `N_opt = ⌊π/2φ⌋` or the estimator
aliases, but φ is what we are trying to measure. Each adaptive algorithm spends part of its budget
discovering a usable `N`, then commits the rest to one measurement there.

## Layout

```
qmetrology/     the package: simulation, the algorithms, the safeguard, the manifest, the evaluator
analysis/       the pipeline that produces every reported number, plus the side studies
results/        everything the pipeline and the side studies write -- CSVs, figures, LaTeX
docs/           the hand-written derivations behind the generated results
thesis_code/    the appendix listings (Listings A.1-A.6) + a parity check against the package
tests/          the validation suite
notebooks/      a walkthrough of the model and a sandbox for the safeguard
```

## Reproduce

```bash
pip install numpy scipy pandas matplotlib

python analysis/run.py --quick       # ~2 min smoke run, identical code path
python analysis/run.py --max         # the production sweep (~5 h on 24 cores)
python analysis/run.py --max --resume --keep-traces   # continue, and keep probe-level traces
python analysis/run.py --max --report-only            # rebuild derived tables + REPORT.md, no sim

python analysis/full_report.py       # rebuild FULL_RESULTS.md
python analysis/thesis_tables.py     # the thesis tables -> results/tex/ (--long -> results/tex_long/)
python analysis/thesis_figures.py    # the Chapter-4 figures
python analysis/error_curves.py      # |φ̂−φ| quantiles behind fig_error
python analysis/variance_curves.py   # estimator variance vs budget

python tests/test_consolidated.py --slow   # the validation suite
```

`--keep-traces` writes the full probe list of the headline points to `results/traces/*.jsonl.gz`;
three of the density figures need them. For unattended runs `analysis/supervise.sh` restarts the
sweep with `--resume` if the process dies.

**Reported algorithms** (`qmetrology/manifest.py`, `ORDER`) — and only these:

| key | algorithm | tuned parameters |
|---|---|---|
| `brute` | brute force (Algorithm 3) | none |
| `linear` | linear search (Algorithm 4) | `m'`, `lookback_window`, `s`, `inc`, `mean_window` |
| `binary_deep` | binary search + statistical safeguard, pilot = deepest accepted probe | `m'`, `conf` |
| `reverse_eng_risk` | reverse engineering + statistical safeguard | `m'` |

`separable` (N=1) and `oracle_hl` (the analytic ceiling at `N_opt`) appear in the tables as
references, not as protocols under comparison.

## How a number gets made

One command regenerates every Chapter-4 performance number, its uncertainty, the exploration and
safeguard diagnostics and the appendix parameter tables — all from the **same** held-out runs, so a
reported value and its telemetry can never come from different trials. For each
(scenario, budget, algorithm) the pipeline

> tunes on the tuning blocks → freezes the winner → evaluates it **once** on the held-out seed with
> tracing on → emits the convergence rate, the diagnostics, the detector confusion and the budget
> audit from those same runs.

Parameters are tuned on two disjoint seed blocks (42, 43), frozen, and only then evaluated on the
held-out seed 2024, so no reported number is a maximum over its own trials. Everything the pipeline
uses — scenarios, budget grids, tuning grids, seeds, `R_tune`, `R_test`, bootstrap replicates —
lives in `qmetrology/manifest.py`, and every output row stamps the values it actually ran with.

The readout uses a **binomial** sampler (`hits ~ Binomial(m, cos²Nφ)`, identical in distribution to
per-shot sampling and ~1000× faster for large `m`), which is what makes the ε=10⁻⁸ sweeps feasible.

Tuning is three-stage, because the full discrete grid crossed with the `m'` grid would be ~10,800
configurations per stage: **A** locates `m'` against a coarse skeleton of the discrete grid; **B**
searches the full discrete grid near A's winner; **C** refines `m'` around B's winner at high `R`.
Stage C decides, so it gets the trials — it controls the variance of *which* configuration wins, the
one uncertainty component that cannot be recovered from stored output afterwards. `winners.csv`
carries `at_axis_bound`, flagging a winner pinned at the endpoint of any axis.

Every aggregate in `REPORT.md` is over the **live** operating points only — those where the best
algorithm converges between 3% and 99%. At a saturated point every configuration ties, so the
tuner's argmax and every diagnostic measured there is Monte-Carlo noise rather than a property of
the algorithm. The per-point CSVs keep all points; join on `operating_points.csv` to filter.

The instrumentation is *neutral*: every algorithm takes an optional `trace=qmetrology.trace.
AlgorithmTrace()`, the collector draws no random numbers, and `test_trace_neutrality` asserts that a
same-seed run returns identical `φ̂` and budget use with tracing on and off.

`../thesis/*.tex` is never read or written. The LaTeX under `results/tex/` is a proposal to paste.

## Results

Start with **[results/FULL_RESULTS.md](results/FULL_RESULTS.md)** — the complete standalone record:
model, algorithms, every scenario/seed/R, the statistical methodology, all results, validation and
limitations. **[results/REPORT.md](results/REPORT.md)** is the short version.

Key outputs in `results/`:

| file | what |
|---|---|
| `performance_curves.csv` | convergence vs budget, Wilson intervals at the row's own R |
| `budget_crossings.csv` | B(p\*) and the ratio vs brute force, bootstrap intervals + grid sensitivity |
| `winners.csv` | the frozen tuning winner, its per-block evidence, runner-up and margin |
| `optimal_params.csv` | the appendix parameter tables |
| `operating_points.csv` | per (scenario, budget): best rate and the live/saturated/floored label |
| `diagnostics_by_point.csv` | exploration and safeguard telemetry, every share with its denominator |
| `diagnostics_by_phase.csv` | the same, split by quantile bins of the true phase |
| `detector_confusion.csv` | linear/binary false-alarm and miss rates, probe-level TP/FP/TN/FN |
| `budget_audit.csv` | per-trial spend (mean/median/p90/max) and the violation count |
| `experiment_manifest.json` | every scenario/algorithm/seed/R the run used |
| `algorithm_code_audit.md` | which function is which thesis algorithm |
| `tex/`, `tex_long/` | paste-ready LaTeX tables (short and per-scenario) |

Headline: for φ ~ U(0.01, 0.1) all three adaptive algorithms beat brute force already at ε=10⁻³, and
the advantage grows with precision, plateauing near **1.84×** less budget for reverse engineering.
At a fixed budget of 10,000 (ε=10⁻³): brute 56.1%, linear 64.1%, binary 65.1%, reverse engineering
**66.3%**, against an attainable ceiling of 73.5%.

## Derivations

`docs/` holds the hand-written arguments the generated results rest on:

| document | subject |
|---|---|
| [SAFEGUARD_DERIVATION.md](docs/SAFEGUARD_DERIVATION.md) | the statistical safeguard, its assumptions and what each one buys |
| [OVERSHOOT_CRITERION.md](docs/OVERSHOOT_CRITERION.md) | the binary-search overshoot rule, its size and power |
| [BERRY_ESSEEN.md](docs/BERRY_ESSEEN.md) | the non-asymptotic justification of the normal quantile in Eq. (3.6) |
| [LINEAR_SEARCH.md](docs/LINEAR_SEARCH.md) | the linear-search detector bake-off and the `mean_window` axis |
| [POSTERIOR.md](docs/POSTERIOR.md) | the exact-posterior depth criterion |
| [BROAD_DIST.md](docs/BROAD_DIST.md) | broad priors, where the ranking reverses |

Their numbers are regenerated by:

```bash
python analysis/overshoot_criterion.py           # -> the overshoot_*.csv tables
python analysis/linear_detector_study.py --curve # -> the linear_detector_*.csv tables
python analysis/linear_search_claims.py          # re-derives Section 3.2.1's claims from results/
python analysis/audit.py                         # -> algorithm_code_audit.md
```

## Beyond the thesis

Three studies extend past what the thesis reports. They read the package and `results/` read-only
and write their own files into `results/`.

```bash
python analysis/ladder_study.py         # phase-unwrapping ladder      -> results/LADDER.md
python analysis/posterior_all_study.py  # exact posterior, all 3 algos -> results/posterior_all.csv
python analysis/posterior_explain.py    # the POSTERIOR.md figures
python analysis/binary_rescue_study.py  # can bisection inform the depth rule at all
python analysis/broad_dist_study.py     # broad priors                 -> results/broad_dist.csv
```

The **ladder** (`qmetrology/ladder.py`) is the one that changes the scaling class. Every thesis
algorithm inverts a single batch and is therefore capped at `N ≤ π/2φ`, so brute force and every
adaptive protocol alike cost `~1/ε²` and the advantage saturates — the plateau above is that
invertibility cap, not a physical limit. Unwrapping each measurement with the previous, coarser
estimate lifts the cap and recovers `~1/ε` scaling, so the budget ratio keeps growing with
precision instead of flattening. See [results/LADDER.md](results/LADDER.md).

The **exact posterior** is the Bayesian direction the thesis names in its Outlook: the likelihood
`p(D|φ)` is not limited by `Nφ < π/2`, so no expensively simulated probe has to be discarded.
