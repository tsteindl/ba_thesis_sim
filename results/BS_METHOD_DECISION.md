# Binary search, the overshoot rule, and the statistical safeguard: decision memo

## Recommendation

The new experiments identify one coherent way to make bisection matter without introducing the full
exact posterior: **certified-only, precision-pooled bisection**.  Given the priority of retaining a
real binary-search contribution while accepting slight suboptimality, this is now the preferred
rescue candidate.  It changes only the exploration wrapper around the existing normal-law safeguard:

1. take the guaranteed-unaliased opening probe at `N_min`;
2. set `U = floor(pi/(2 phi_hat_0))` and propose the next midpoint of `[L,U]`;
3. before measuring it, require the existing truncated-normal model to assign at least 99.9%
   probability that the midpoint is unaliased;
4. if it fails, stop exploration; if it passes, measure it, perform the bisection comparison, and
   inverse-variance-pool it into the pilot;
5. give that pooled pilot and its precision to the existing statistical safeguard over the prior
   support.

The key word is *before*.  Certifying a probe before observing its outcome avoids reusing an estimate
selected because it happened to pass the overshoot test.  Moreover, an uncertified midpoint is not
measured at all: if its estimate cannot enter the safeguard, spending quantum resources on its branch
decision would recreate the original problem.

This does not establish global optimality and does not eliminate the approximate-normal model.  It
does make the dependence explicit and empirically testable.  The 99.9% threshold should be presented
as a declared per-probe risk tolerance, not as a performance-optimal constant.  The full exact
sequential posterior remains future work.

If changing the reported algorithm this late is judged too expensive, retain the original algorithm
and state the negative result plainly:

> Once a safe opening pilot and the statistical safeguard are available, reducing subsequent probes
> to binary safe/overshoot decisions does not add enough information to repay their cost.  The tuned
> optimum therefore approaches the one-pilot reverse-engineering protocol.  An exact sequential
> likelihood can recover the discarded information, but is left to future work.

Do not use the previously suggested `U`-cap compromise if the certified-only version is adopted; it
is weaker because the bisection still does not improve the pilot.

## Certified-only result

The final comparison fixed each method's configuration using an earlier tuning run, then used 50,000
fresh trials per point and common phase draws.  Intervals below are paired 95% Monte Carlo intervals.

| operating point | RE | certified BS | difference (pp) | mean bisection updates | runs with an update | contaminated run |
|---|---:|---:|---:|---:|---:|---:|
| headline low budget | 66.276% | 65.634% | -0.642 [-1.213, -0.071] | 0.476 | 22.41% | 0.004% |
| tight medium budget | 68.586% | 69.322% | +0.736 [+0.182, +1.290] | 3.015 | 89.95% | 0.004% |
| small phase | 95.462% | 95.124% | -0.338 [-0.495, -0.181] | 2.362 | 54.50% | 0.028% |
| wide prior | 74.788% | 75.036% | +0.248 [-0.259, +0.755] | 2.837 | 82.73% | 0.002% |

Thus it is not a new performance winner.  It is approximately performance-neutral across these four
stress points, with small gains and losses, while giving bisection a measurable and statistically
coherent role.  At the low-budget point, 77.6% of runs take no bisection step, which is the expected
resource-limited fallback rather than evidence that the higher-budget algorithm is fictitious.  At
the three other points it performs 2.36--3.01 certified updates on average.

The pooled pilot's mean absolute standardized error is 0.815--0.891 across the four points, close to
`E|Z| = 0.798` under the working Gaussian model.  Only 19 of 200,000 runs contained any certified
probe beyond the true `N_opt` (0.0095% overall).  These are empirical validation statements, not a
proof that the Gaussian approximation is exact.

Raw results: `certified_only_bs.csv` and `certified_only_bs_validation.csv`.  Reproduction:
`python analysis/certified_only_bs_study.py` and
`python analysis/certified_only_bs_validation.py --reps 50000` (using the project's Python 3
environment).

## Why the recent/deepest estimate is not the clean fix

The closest implementation of the proposed repair was tested:

1. take the safe opening estimate `phi_hat_0`;
2. set the initial upper bound to `N_guess=floor(pi/(2 phi_hat_0))` instead of the prior `N_max`;
3. bisect `[N_min,N_guess]` with the existing overshoot comparison;
4. feed the deepest accepted estimate to the existing statistical safeguard.

It genuinely runs (4.6--7.5 probes in the focused experiment) and ties RE at the tight-precision
headline points.  Across all six points, however, it averaged 1.62 pp below RE: -7.50 pp at the
low-budget headline point, -1.95 pp for the small-phase point, and differences within Monte-Carlo
noise at the two tight-precision and wide-prior points.  More importantly, the deepest accepted
estimate is selected *because it passed the test*.  Treating it afterward as an ordinary unbiased
Gaussian pilot contradicts the safeguard derivation.  A fresh confirmation probe removes that
selection issue but performed worse because it adds cost and does not remove alias ambiguity.

Raw results: `focused_bs_resolution.csv`.  Reproduction:
`python analysis/focused_bs_resolution.py`.

## What binary search is actually doing

The full existing diagnostics contain 69 operating points (23 scenarios, three live budgets each;
15,000 test trials per point).  At each point binary search is evaluated at its own tuned
configuration.

| diagnostic | result |
|---|---:|
| points with exactly one exploration probe (no bisection step) | 46.4% |
| among nontrivial points, mean probes | 10.58 |
| mean / median / maximum exploration-budget share | 8.89% / 3.79% / 31.16% |
| mean / median final depth `N*/N_opt` | 0.907 / 0.934 |
| mean final-depth overshoot rate | 0.50% |
| maximum final-depth overshoot rate | 5.97% |

The budget trend is present but not absolute: the one-probe share falls from 65.2% at the lowest live
budget in each scenario to 39.1% in the middle and 34.8% at the highest.  Mean exploration share falls
from 14.4% to 7.3% to 5.0%.  Thus the low-budget behavior is expected, but it still occurs in roughly
one third of the high-live-budget operating points.

The follow-up replay uses the saved winners, a fresh seed, and 5,000 trials per point:

| final-depth diagnostic, averaged equally over 69 points | rate |
|---|---:|
| exactly `N*=N_opt` | 20.79% |
| within 1% of `N_opt` | 26.97% |
| within 5% of `N_opt` | 55.54% |
| within 10% of `N_opt` | 72.02% |

Exact attainment is not the objective of the safeguard.  It deliberately backs off from the aliasing
cliff, and the transformed estimator is boundary-censored near `N_opt`; a depth slightly below it can
have higher true convergence.  Report exact attainment because readers will ask, but do not present it
as the primary success criterion.

Raw results: `binary_diagnostics.csv` and `binary_diagnostics_replay.csv`.  Reproduction:
`python analysis/binary_diagnostics.py` (tune + diagnose) and
`python analysis/binary_diagnostics_replay.py` (fresh replay of saved winners).

## How to discuss approximate normality

Separate two roles that currently get conflated.

1. **Binary branch rule.** In 62 of the 69 tuned diagnostic configurations, `conf=0.5`.  Since the
   normal quantile is then zero, the threshold is just the current reference estimate and the normal
   variance drops out.  This should be described as a sequential comparison heuristic, not as a 95%
   confidence test.  Its operating characteristic should be reported empirically.
2. **Statistical safeguard.** This does use the truncated-normal pilot posterior.  The finite-sample
   approximation is imperfect near the binomial boundary, but replacing the one-pilot RE calculation
   by the exact binomial posterior changes mean convergence by only +0.057 pp over the 166 stored
   operating points.  This isolates the normal approximation because RE has only one probe.  The
   corresponding binary-posterior gain (+0.664 pp on average) also recovers discarded history, so it
   must not be attributed to non-normality alone.

The theorem should therefore say "optimal for the stated surrogate model," not globally optimal.

## Numerical-verification subsection to add

The shortest convincing addition is one table/figure with:

1. empirical calibration of predicted overshoot probability (reliability bins, Brier score, and final
   overshoot rate);
2. exact-binomial versus normal-pilot ablation;
3. regret to the attainable oracle at 50%, 75%, 90%, and 99% reliability;
4. the process telemetry above: probes, bisection-step probability, exploration share, exact and
   near-`N_opt` rates, and final overshoot rate;
5. independent tuning/test seeds and Monte-Carlo intervals.

The stored oracle analysis already gives the strongest headline: at the 90% threshold the median
RE/statistical-safeguard budget is 1.05 times the attainable oracle budget, and 18 of 23 scenarios are
within 1.1 times.  This justifies keeping the approximation far better than a normality plot alone.

## Two issues to fix before final tables

1. The implementation currently caps the binary safeguard by prior support, whereas the thesis
   pseudocode sets `N_guess=L`.  These are different algorithms.  Choose the uncapped negative-result
   story or the `U`-cap compromise and make code, pseudocode, and tables agree.
2. The main sweep uses `m_exploration >= 20` for binary/RE but `>=3` for linear search.  Two scenarios
   are badly affected: lowering the floor moves the low-budget rate by about 47 pp.  Rerun those cells
   (or the sweep) before quoting worst-case or mean oracle gaps.  The median conclusions are not driven
   by those cells.
