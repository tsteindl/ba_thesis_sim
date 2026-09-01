# Exact statistical safeguard and affected-algorithm rerun

## Objective

Replace the statistical safeguard's coarse-to-fine search by vectorized exhaustive enumeration of
the integer candidates

\[
N\in\{N_{\min},N_{\min}+1,\ldots,N_{\max}\}.
\]

Use the actual number of affordable exploitation shots, `m(N) = budget // N`, in the score. Then
retune and re-evaluate the two affected reported algorithms:

- `binary_deep`
- `reverse_eng_risk`

Do not rerun `brute` or `linear`; their implementations and results are unchanged. The broad-prior
scenarios have been removed from the thesis, so the production rerun should cover the eight
non-broad scenarios only.

The full rerun may start automatically after the implementation, unit tests, isolated smoke run,
and timing preflight below have passed. It must write to a new output directory and must not modify
`results/consolidated` while it is running.

## Non-negotiable behavior

1. The search starts at `N_min`, never at 1 unless `N_min == 1`.
2. The search ends at `N_max`, inclusive.
3. Every integer in this closed interval is evaluated. There is no geometric refinement, local
   interval, five-sigma cutoff, or hard cap.
4. The accuracy factor uses whole shots:

   ```text
   m(N)  = floor(B / N)
   p_acc = 2 Phi(2 eps N sqrt(m(N))) - 1     if m(N) >= 1
           0                                  otherwise
   ```

5. The validity factor remains the existing truncated-normal posterior probability from
   `_p_safe(..., support=(phi_min, phi_max))`.
6. The returned value is

   ```text
   argmax over N_min,...,N_max of p_safe(N) * p_acc(N).
   ```

   `numpy.argmax` gives the smallest `N` in an exact tie; retain and test this deterministic rule.
7. Invalid-input fallbacks return `N_min`, not 1. If the remaining budget is smaller than `N_min`,
   all candidates are unaffordable and the surrounding algorithm must take its existing
   no-exploitation path rather than inventing a shot.
8. Make `N_min` and `N_max` required arguments of `risk_optimal_depth`, preferably keyword-only, so
   no call site can silently revert to a lower bound of 1.

`N_min = floor(pi/(2 phi_max))` is the largest integer that stays on the first identifiable branch
for the complete prior support (`N_min * phi <= pi/2`; equality can occur only at an endpoint). It
is therefore a safe lower bound. Searching from `N_min` is a deliberate admissible-set constraint:
the safeguard should never choose a value shallower than the guaranteed-safe baseline. With whole
shots, floor effects mean this should be described as the exact maximizer **over the constrained
range**, not as a proof that no smaller integer could ever have a marginally larger numerical
score. For the current 177 active scenario/budget points, a direct integer audit nevertheless found
that `N_min` has at least as large an accuracy factor as every `N < N_min`; no excluded lower
candidate wins on the present grid.

## Core implementation

The vectorized core in `qmetrology/safeguard.py` should be equivalent to:

```python
candidates = np.arange(N_min, N_max + 1, dtype=np.int64)
m = int(budget) // candidates
p_acc = np.where(
    m >= 1,
    2.0 * ndtr(2.0 * eps * candidates.astype(float) * np.sqrt(m)) - 1.0,
    0.0,
)
score = _p_safe(candidates, phi_hat, sigma, support) * p_acc
return int(candidates[int(np.argmax(score))])
```

Avoid per-candidate Python loops. Candidate arrays may be cached by `(N_min, N_max)` if profiling
shows that allocation matters, but keep the public algorithm and the derivation simple.

Remove `_N_SEARCH`, `_N_ROUNDS`, `_HARD_CAP`, the geometric refinement loop, and the pilot-dependent
five-sigma search limit. Update `_score` so tests and diagnostics use the same integer-shot formula.
Update `effective_C` consistently.

## Files and call sites to audit

At minimum inspect and update:

- `qmetrology/safeguard.py`
- every `risk_optimal_depth(...)` call in `qmetrology/algorithms.py`, including retained legacy or
  audit variants, not only the two reported functions
- all tests and analysis scripts found by `rg -n "risk_optimal_depth|_score" .`
- `results/SAFEGUARD_DERIVATION.md`
- `analysis/consolidated/full_report.py`
- `qmetrology/manifest.py` metadata/docstrings that identify the safeguard implementation
- the appendix copies under `results/consolidated/thesis_code/`, together with their verification
  script

For Binary Search, pass the prior-derived `N_min` and the applicable upper cap. Once the subroutine
itself enforces `N >= N_min`, the later `max(N, N_min)` is redundant and may be removed. For Reverse
Engineering, pass the same prior-derived limits; it must no longer be able to select a value below
`N_min`.

Do not silently hand-edit only an appendix listing. The source implementation, appendix copy,
verification code, generated report description, and thesis pseudocode must ultimately describe the
same selection rule.

## Required tests

Add focused tests before any production run:

1. Compare the vectorized result with a deliberately simple scalar reference loop for randomized
   valid inputs, including truncated supports.
2. Assert `N_min <= result <= N_max` for every valid call and explicitly assert that a case with
   `N_min > 1` never returns 1.
3. Test the smallest-`N` tie rule.
4. Include budgets near affordability boundaries (`B < N_min`, `B == N_min`, and values at which
   `floor(B/N)` changes).
5. Include a regression case demonstrating that the score uses `floor(B/N)`, rather than the old
   smooth substitution `m = B/N`.
6. Test invalid inputs and collapsed truncated posteriors.
7. Verify that Binary Search and Reverse Engineering never exceed the nominal budget and that their
   recorded `N_star` stays in `[N_min, N_max]`.
8. Verify every live call site passes both bounds explicitly.
9. Run the existing consolidated smoke/test suite after the focused tests.

## Cost assessment and timing gate

The active thesis matrix has only two relevant interval sizes:

| Scenarios | Range | Candidates per safeguard call |
|---|---:|---:|
| `narrow_e3` through `narrow_e8` | 15 to 157 | 143 |
| `small_e4` | 157 to 1570 | 1,414 |
| `wide_e4` | 15 to 1570 | 1,556 |

There is no active thesis scenario with `N_max > 1570`. Tightening `epsilon` increases the budget
grid but does not enlarge the candidate interval.

A local vectorized timing check of the current score gave approximately 28 microseconds per call
for 15--157 and 78 microseconds per call for 157--1570. A hypothetical 15--15707 range took about
0.63 milliseconds per call. These are environment-specific measurements, not acceptance
thresholds, but they show that enumeration is not prohibitive for one call in any active scenario.

The important multiplier is the tuning procedure. In `--max` mode, one scenario/budget point makes
at most approximately:

- 872,800 safeguard calls for `binary_deep`
- 277,600 safeguard calls for `reverse_eng_risk`
- 1.15 million calls in total

The eight active scenarios contain 177 scenario/budget points, hence about 204 million calls before
duplicate-grid reductions. Enumeration can therefore add hours of aggregate CPU even though each
individual enumeration is cheap; multiprocessing should reduce wall time. This is still reasonable
for the active `N_max <= 1570` matrix, but it justifies a timing preflight rather than assuming that
“1,556 calculations” happens only once.

Before the max run, time at least 10,000 representative calls for 15--157 and 5,000 calls for both
157--1570 and 15--1570 using the new integer-shot score. Record the timings in the run log. If
15--1570 is unexpectedly above roughly 0.5 ms per call, profile vector allocation and `_p_safe`
before launching the full sweep. This is a diagnostic trigger, not a reason to reintroduce an
approximate search.

For future scenarios with intervals around 15--15707, enumeration remains simple but the repeated
tuning cost becomes material: at the measured rate, the safeguard score alone is about 12 CPU
minutes for a single max-mode operating point. Such a future scenario should receive its own timing
estimate, but it is not present in the thesis matrix.

## Safe affected-only rerun

The current `analysis/consolidated/run.py` always iterates over `M.ORDER`, writes all scenarios, and
uses scenario-level `--resume`. Do not point it at `results/consolidated` and do not try to obtain a
partial rerun merely by editing `M.ORDER`: that can overwrite or create provenance-inconsistent
files.

Add a tested selective mode (either CLI options in `run.py` or a small dedicated driver) with these
properties:

- accepts an explicit algorithm set and scenario set;
- records those filters in `experiment_manifest.json`;
- supports resume safely within the isolated partial output;
- writes all six raw tables for the selected cells;
- skips `add_baselines` and `finalize` until the partial rows have been merged into a complete
  staging dataset;
- does not classify a scenario as complete merely because an unrelated algorithm row exists;
- retains traces for the selected headline points when requested.

Use exactly these algorithms:

```text
binary_deep,reverse_eng_risk
```

Use exactly these scenarios:

```text
narrow_e3,narrow_e4,narrow_e5,narrow_e6,narrow_e7,narrow_e8,small_e4,wide_e4
```

The max matrix should produce 177 scenario/budget points and therefore 354 affected rows in each
one-row-per-cell raw table. `detector_confusion.csv` has only the 177 Binary Search rows; the phase
table has multiple phase-bin rows per cell.

Suggested isolated destination:

```text
results/consolidated_exact_safeguard_partial
```

After focused tests, the existing test suite, an isolated `--smoke` run, and the timing preflight
pass, start the `--max --resume --keep-traces` affected-only run automatically in the background.
On Windows, use `Start-Process` with `-WindowStyle Hidden`, redirect stdout and stderr to timestamped
log files inside the isolated output directory, and record the PID. Use the Python executable from
the environment in which the tests passed. Do not use `supervise.sh` unchanged: it contains a
machine-specific path.

## Merge and regeneration

Do not promote partial results directly. Build a second staging directory from the current complete
production data, then:

1. remove all broad-prior rows from the thesis staging dataset;
2. replace the non-broad `binary_deep` and `reverse_eng_risk` rows with the new partial rows;
3. retain the unchanged `brute`, `linear`, `separable`, and oracle rows;
4. use `(scenario_id, algorithm, budget)` as the cell key, adding `phase_bin` for phase-stratified
   rows;
5. assert uniqueness, the expected scenario/budget coverage, matching seeds/trial counts, and the
   new implementation metadata;
6. only then run `add_baselines`, `finalize`, `full_report.py`, error/variance curve generation,
   `thesis_tables.py`, and `thesis_figures.py` against the staging directory;
7. compare every changed thesis number and caption with the regenerated CSV source before any
   promotion.

Keep the old consolidated directory as a recoverable snapshot. Promotion should occur only after
the staged reports, tables, figures, budget audits, and appendix-code verification pass.

## Thesis follow-up after the rerun

The current thesis truthfully describes the implementation used for its present numbers: search
from 1, geometric refinement, and the smooth `m approximately B/N` score. After the new results are
accepted, update all of the following together:

- Theorem `thm:stat-safeguard`: both argmax expressions become
  `N' = N_min, ..., N_max`.
- The accuracy variance remains
  `sigma(N') = 1 / (2 N' sqrt(floor(B/N')))`, matching the new score exactly.
- The safeguard pseudocode loop starts at `N_min`, not 1.
- Remove the paragraph about geometric coarse-to-fine search and the smooth approximation.
- State that vectorized enumeration returns the exact maximizer of the discretized approximate
  score over the constrained interval. “Exact” modifies the numerical maximization, not the
  Gaussian/statistical model.
- Remove Binary Search's redundant post-safeguard floor.
- Regenerate the appendix Python listings, tables, figures, and all quoted results.

The rerun is scientifically necessary. Changing the safeguard changes both algorithms' outcomes
and can change the winning exploration parameter, so old tuned winners and old held-out results
cannot be reused.

## Completion checklist

- [ ] Exhaustive constrained enumeration implemented with whole shots.
- [ ] All call sites require and pass `N_min` and `N_max`.
- [ ] Focused correctness and budget tests pass.
- [ ] Existing suite passes.
- [ ] Isolated smoke run passes.
- [ ] Timing preflight is logged.
- [ ] Affected-only max run is started in an isolated directory with resumable logs.
- [ ] Partial row counts and metadata are verified.
- [ ] Complete active-scenario staging dataset is merged without stale affected rows.
- [ ] Reports, diagnostics, appendix code, tables, and figures are regenerated and checked.
- [ ] Only then are thesis claims and production artifacts promoted.
