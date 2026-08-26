# Adaptive quantum metrology under resource constraints — complete results

Consolidated performance, algorithm diagnostics and uncertainty for the four algorithms the thesis reports, all derived from one set of held-out Monte-Carlo runs.

| | |
|---|---|
| Generated | 2026-08-26 23:43 by `python analysis/consolidated/full_report.py` |
| Sweep command | `python analysis/consolidated/run.py --max --keep-traces` |
| Sweep mode | `max` |
| Sweep wall clock | 307 min (5.1 h), 24 cores |
| Repository commit | `49aec23` (working tree dirty) |
| Scenarios | 10 |
| Operating points | 221 (144 live, 77 saturated/floored) |
| Evaluated (point, algorithm) cells | 1326 |
| Held-out trials per cell | R = 50,000 |
| Total held-out trials | 66,300,000 |

> **Scope note.** This document is generated from `results/consolidated/*.csv` and nothing else. No thesis file (`../thesis/*.tex`) is read or written anywhere in this pipeline; the LaTeX in `results/consolidated/tex/` is a proposal to paste, never an edit.

**Contents**

1. [The model](#1-the-model)
2. [Algorithms as implemented](#2-algorithms-as-implemented)
3. [Diagnostic quantities](#3-diagnostic-quantities)
4. [Experimental setting](#4-experimental-setting)
5. [Tuning and held-out protocol](#5-tuning-and-held-out-protocol)
6. [Statistical methodology](#6-statistical-methodology)
7. [Results — performance](#7-results--performance)
8. [Results — exploration phase](#8-results--exploration-phase)
9. [Results — safeguard and final depth](#9-results--safeguard-and-final-depth)
10. [Results — detector quality](#10-results--detector-quality)
11. [Results — trends](#11-results--trends)
12. [Budget compliance audit](#12-budget-compliance-audit)
13. [Tuning provenance and grid adequacy](#13-tuning-provenance-and-grid-adequacy)
14. [Validation suite](#14-validation-suite)
15. [Limitations and code/thesis mismatches](#15-limitations-and-codethesis-mismatches)
16. [File and column reference](#16-file-and-column-reference)

---

## 1. The model

An unknown phase `phi` is estimated from a maximally-entangled circuit of depth `N` that reads out "0" with probability

```
p0(N, phi) = cos^2(N phi)
```

Drawing `m` shots and inverting the readout gives the estimator

```
phi_hat = (1/N) * arccos( sqrt(hits/m) ),     hits ~ Binomial(m, p0)
```

**Cost.** One measurement of depth `N` with `m` shots costs `N*m`. An algorithm's total cost is the sum over all its measurements, exploration and exploitation alike. Every algorithm here is *fixed-budget*: it is given a budget `B` and must not exceed it (audited in §12).

**Convergence.** A trial converges iff `|phi_hat - phi| < eps`, where `phi_hat` is the final estimate returned by the algorithm. The reported *convergence rate* is the fraction of converged trials over R independent trials.

**Aliasing and `N_opt`.** `arccos` inverts `p0` uniquely only while `N*phi <= pi/2`. The deepest non-aliasing depth is therefore

```
N_opt = max( floor( pi / (2 phi) ), 1 )
```

This is the target every exploration phase is implicitly trying to find, and the denominator of the depth diagnostics. Deeper circuits are more precise — the estimator's asymptotic variance is `1/(4 N^2 m)`, so at fixed budget `B = N*m` the error scales as `1/(2 sqrt(N B))` — but past `N_opt` the estimate aliases and the trial is (almost always) lost. Every algorithm below is a different way of trading those two off.

**Sampler.** `hits` is drawn directly from `Binomial(m, p0)` (`qmetrology/sim.py`, `DEFAULT_METHOD = "binomial"`), which is identical in distribution to summing `m` Bernoulli shots but O(1) in `m` rather than O(m). This is what makes the `eps = 1e-8` scenarios — budgets of order `1e16` — feasible at all.

**Prior.** In every trial `phi` is drawn from the scenario's prior, `Uniform(phi_min, phi_max)` throughout this sweep. The prior is genuine, not a modelling convenience: it is how the simulation draws `phi`, and it is what makes the statistical safeguard's posterior the true conditional distribution of `phi` given the pilot, over the ensemble reported here.

**Search bounds available to every algorithm.** From the prior support alone:

```
N_min = max( floor(pi / (2 phi_max)), 1 )    the deepest circuit that cannot alias for ANY
                                            admissible phi -- every algorithm opens here
N_max = max( floor(pi / (2 phi_min)), 1 )    the deepest circuit any admissible phi allows
```

---

## 2. Algorithms as implemented

Exactly four algorithms are reported. The superseded constant-safeguard variants, the exact-posterior variants and the omniscient oracles remain in `qmetrology/algorithms.py` but are deliberately outside this manifest and appear nowhere in these results.

| key | implementation | adaptive | has detector | tuned parameters |
|---|---|---|---|---|
| `brute` | `qmetrology.algorithms.find_phi_fixed_budget_brute_force` | no | no | none |
| `linear` | `qmetrology.algorithms.find_phi_fixed_budget_linear_search` | yes | yes | m_exploration (m'), lookback_window, safeguard (s), inc |
| `binary_deep` | `qmetrology.algorithms.find_phi_fixed_budget_binary_search_deep` | yes | yes | m_exploration (m'), conf |
| `reverse_eng_risk` | `qmetrology.algorithms.find_phi_fixed_budget_reverse_engineering_risk` | yes | no | m_exploration (m') |

### 2.1 Algorithm 3 — brute force

Fix `N = N_min` and spend the entire budget on shots: `m = floor(B / N_min)`. No exploration, no adaptivity, no parameters.

```
N = N_min
m = floor(B / N_min)
return phi_hat(N, m)
```

### 2.2 Algorithm 4 — linear search

Scan `N` upward from `N_min` in steps of `inc`, `m'` shots per probe, tracking the running mean of the estimates. When that running mean falls `lookback_window` times in a row the scan declares an overshoot, backtracks by `lookback_window * inc`, subtracts the tuned decrement `s`, and spends whatever budget remains at that depth.

```
N = N_min; history = []
while budget allows and N <= N_max:
    phi_hat_i = measure(N, m');  history.append((N, phi_hat_i))
    if running_mean(history) fell lookback_window times in a row: OVERSHOOT; break
    N += inc
N_guess = N_last - lookback_window*inc  if OVERSHOOT else N_last
N_star  = max(1, N_guess - s)
m       = floor(remaining_budget / N_star)
return phi_hat(N_star, m)
```

### 2.3 Algorithm 5 — binary search + statistical safeguard, deepest-probe pilot

Bisect on `N` between `N_min` and `N_max`. Each probe is compared against a threshold derived from the previous accepted estimate via Eq. (3.4); a probe reading below it is declared an overshoot and the search moves down, otherwise it is accepted and the search moves up. The **deepest accepted** probe `(phi_acc, N_acc)` is then handed to the statistical safeguard as the pilot.

```
phi_0 = measure(N_min, m')                       # cannot alias, by construction
threshold = Phi^-1(1-conf; phi_0, 1/(4 m' N_min^2))
L, U = N_min, N_max;  N = N_min + (U-N_min)//2
while budget allows and the bracket is not exhausted:
    phi_hat_i = measure(N, m')
    if phi_hat_i < threshold:  U = N; N -= (N-L)//2          # declared overshoot
    else:  (phi_acc, N_acc) = (phi_hat_i, N); L = N; N += (U-N)//2
           threshold = Phi^-1(1-conf; phi_hat_i, 1/(4 m' N^2))
N_guess = N_acc  (== L)
N_star  = risk_optimal_depth(phi_acc, sigma = 1/(2 N_acc sqrt(m')),
                             remaining_budget, eps, N_max = floor(pi/(2 phi_min)),
                             support = (phi_min, phi_max))
N_star  = max(N_star, min(N_min, N_max))         # never shallower than the opening probe
m       = floor(remaining_budget / N_star)
return phi_hat(N_star, m)
```

### 2.4 Algorithm 6 — reverse engineering + statistical safeguard

Take a single pilot at `N_min` (retrying if it returns exactly zero, which happens when `hits == m'`), invert it to a depth, and hand the pilot to the statistical safeguard. Every pilot attempt is charged to the exploration budget.

```
phi_hat_0 = 0
while phi_hat_0 == 0 and budget allows:
    phi_hat_0 = measure(N_min, m')               # retries are charged to B_exploration
N_guess = max(floor(pi / (2 phi_hat_0)), 1)      # the raw inverted depth
N_star  = risk_optimal_depth(phi_hat_0, sigma = 1/(2 N_min sqrt(m')),
                             remaining_budget, eps, N_max = floor(pi/(2 phi_min)),
                             support = (phi_min, phi_max))
m       = floor(remaining_budget / N_star)
return phi_hat(N_star, m)
```

### 2.5 The statistical safeguard

`binary_deep` and `reverse_eng_risk` share one depth rule (`qmetrology/safeguard.py::risk_optimal_depth`), which replaces the grid-tuned constants `C_safe` and `s` of the published algorithms. Reading Eq. (3.4) as a likelihood for `phi` given a pilot `(phi_hat_0, N_0, m')` with `sigma = 1/(2 N_0 sqrt(m'))`, a candidate exploitation depth `N` carries two quantifiable and opposing risks:

```
P(no overshoot | N) = P( phi < pi/(2N) )       -- from the TRUNCATED normal posterior on
                                                  [phi_min, phi_max]
P(converge | N)     = 2*Phi( 2 eps sqrt(N B) ) - 1

N_star = argmax_{1 <= N <= N_max}  P(no overshoot | N) * P(converge | N)
```

A deeper circuit is more precise (the second factor grows as `sqrt(N)`) but more likely to alias (the first falls). The optimum needs no tuned constant. The implied multiplicative safety factor `N_star / floor(pi/(2 phi_hat_0))` is adaptive: it tightens when the pilot is imprecise and relaxes toward 1 as the budget grows — which is what a constant `C` could never track.

The maximisation is a three-round geometric refinement followed by an exact integer scan, capped at `N_max = floor(pi/(2 phi_min))` from the prior support.

### 2.6 The two reference rows

Reported in the thesis tables for context, deliberately **not** protocols under comparison, and excluded from the regime rule, the "points won" count and every diagnostic aggregate.

**Separable (`N = 1`).** `m = floor(B/1) = B` — the whole budget as shots at unit depth. Same rule as brute force, which uses `N = N_min` instead. Its sd is `1/(2 sqrt(B))`: no `sqrt(N)` gain at all, which is the point of the row.

**Ceiling.** At fixed budget there is no independent shot count to choose: `m` follows from the depth, and Eq. (3.4) gives

```
N   = min( floor(pi/(2 phi)), B )        the deepest non-aliasing depth, capped so m >= 1
m   = floor(B / N)                       whole shots, so N*m <= B, like every other row
phi_hat = phi + Z / (2 N sqrt(m)),  Z ~ N(0,1)
```

It is **exact**: nothing is sampled. The convergence probability given `phi` is the closed form above, and the reported rate is that probability averaged over the prior. So this row carries no `R`, no Monte-Carlo error and no interval — the value is the value.

An earlier version of this pipeline instead *simulated* the ceiling, drawing `phi_hat = phi + Z/(2 N sqrt(m))` with `Z ~ N(0,1)`, so that it would flow through the same code path and carry a Wilson interval like every other row. That was dropped: it reports a number that wobbles by about +/- 0.4 pp between seeds in place of one that is exactly 73.4844% at the headline point, and a bound has no sampling error to report in the first place. The uniformity was not worth the noise.

*Why the probability comes from Eq. (3.4) rather than from the binomial readout.* At `N ~ N_opt` the readout probability `p0 = cos^2(N phi)` sits against 0, so every shot returns 0 and the arccos estimator is pinned at `pi/(2N)` **regardless of the data**. An oracle that knows `phi` could then read its own answer back off that constant, to accuracy `~2 phi^2/pi`, using almost no shots; whenever `phi < sqrt(pi eps / 2)` that is already inside tolerance, which is how an exact oracle comes to report a ~400x advantage at `U(0.001,0.01), eps = 1e-4`. Using Eq. (3.4) closes that loophole: the accuracy has to come from the statistics, not from the choice of `N`. Eq. (3.4) is not an extra assumption introduced for this row — it is the law the statistical safeguard and the binary-search overshoot test are both derived from.

*What kind of bound this is.* It bounds what the Chapter-3 family can achieve **under its own asymptotic law**, and it is non-degenerate everywhere. It is deliberately not a strict bound on the exact estimator, precisely because the exact estimator can exploit the boundary-clamping above. Two consequences, both measured rather than assumed:

* Because `m` is an integer, `N_opt` is not always the best admissible depth: at `B = 896, phi = 0.01` the bound `N_opt = 157` affords `m = 5` and spends 785 of 896, while `N = 149` affords `m = 6`, spends 894, and is better (`N sqrt(m)` = 365 vs 351). The row reported here uses `N_opt`, so it is a ceiling *for the depth an omniscient protocol would name*, not the supremum over all admissible depths. `qmetrology.oracle.ceiling_rate` computes the latter (at most 0.27 pp higher, and only where `m` is a handful of shots) if a strict supremum is ever wanted.
* Over all 221 operating points, the best implementable algorithm exceeds this ceiling at 93 (point-estimate, algorithm, budget) cells — and at **none** of them does the algorithm's 95% Wilson lower bound clear the ceiling. Every exceedance is inside Monte-Carlo noise.

---

## 3. Diagnostic quantities

Every diagnostic is built from four per-run quantities recorded by the trace collector (`qmetrology/trace.py`). Their definitions are not conventions chosen after the fact — each is asserted against a hand-recomputed value from the same run's probe list in `tests/test_consolidated.py::test_algorithm_definitions`.

| quantity | definition |
|---|---|
| `N_opt` | `max(floor(pi/(2 phi)), 1)` — simulation ground truth, never visible to the algorithm |
| `N_guess` | the exploration phase's answer, recorded **before** any safeguard is applied |
| `N_star` | the depth the exploitation measurement actually ran at; **null** if it never ran |
| `B_exploration` | `sum(N_i * m_i)` over every probe before the exploitation measurement |
| `exploration_share` | `B_exploration / B` |

`N_guess` is algorithm-specific and this is the whole point of measuring it separately from `N_star`: the two differ by exactly the safeguard.

| algorithm | `N_guess` | `N_star` |
|---|---|---|
| `brute` | **null** — no exploration phase. Not zero, not `N_min`: an algorithm that does not search has no guess, and manufacturing one would corrupt every average. | `N_min` |
| `linear` | `N_last - lookback_window*inc` if the detector fired, else `N_last`; recorded raw, before `s` | `max(1, N_guess - s)` |
| `binary_deep` | `N_acc = L`, the deepest probe not classified as an overshoot | the safeguard's depth, floored at `N_min` |
| `reverse_eng_risk` | `max(floor(pi/(2 phi_hat_0)), 1)` from the pilot | the safeguard's depth |

**Repeated pilots count.** Reverse engineering retries its pilot when the estimate is exactly zero; every attempt is charged to `B_exploration`.

**Probe classification.** Each probe carries the algorithm's own verdict (`declared_overshoot`) and the simulation truth (`true_overshoot = N_i > N_opt`).

* `binary_deep` classifies each probe individually — the bisection's accept/reject decision *is* the declaration. The opening probe at `N_min` is accepted by construction (it cannot alias for any admissible `phi`).
* `linear` has **no per-probe detector**: its stopping rule is a trial-level verdict. The classification scored here is the algorithm's own backtracking decision — the last `lookback_window` probes are the ones it discards, so those are its declared overshoots and the retained ones are its acceptances. A different mapping would give different false-alarm numbers; this one is stated so the reader can judge it.
* `reverse_eng_risk` and `brute` have no detector at all and report **N/A**, never zero.

From these, the two trial-level detector rates:

```
false alarm : at least one probe with N_i <= N_opt was declared an overshoot
miss        : at least one probe with N_i >  N_opt was accepted / not flagged
```

Both are **trial-level**, not probe-level, because probes within one run are strongly dependent (the bisection's later probes are chosen from its earlier ones). Probe-level TP/FP/TN/FN counts are reported as supporting telemetry only, and carry no interval.

**Trace neutrality.** The collector records; it never draws a random number. A same-seed run with tracing on and off returns bit-identical `phi_hat` and budget use, asserted over 2,400 trials spanning four scenarios in `test_trace_neutrality`. The performance numbers in this document are therefore the *same runs* as the diagnostics, not a paired re-simulation.

---

## 4. Experimental setting

Everything below comes from `qmetrology/manifest.py`, the single source of truth, and is echoed into `results/consolidated/experiment_manifest.json` at run time. No scenario list, seed or trial count is hard-coded anywhere downstream — asserted by `test_ci_uses_row_R`, which fails if any consumer module contains a literal trial count.

### 4.1 Scenarios

Four Chapter-4 experiment families, ten scenarios, all with a **uniform** prior on `phi`:

| id | prior | `eps` | `N_min` | `N_max` | budget grid | pts | families |
|---|---|---|---:|---:|---|---:|---|
| `narrow_e3` | U(0.01, 0.1) | 1e-03 | 15 | 157 | log, brute90/50 .. brute90x30 +{10000} | 23 | fixed_budget, budget_to_reliability, precision_sweep |
| `narrow_e4` | U(0.01, 0.1) | 1e-04 | 15 | 157 | log, brute90/50 .. brute90x30 | 22 | budget_to_reliability, precision_sweep |
| `narrow_e5` | U(0.01, 0.1) | 1e-05 | 15 | 157 | log, brute90/50 .. brute90x30 | 22 | precision_sweep |
| `narrow_e6` | U(0.01, 0.1) | 1e-06 | 15 | 157 | log, brute90/50 .. brute90x30 | 22 | precision_sweep |
| `narrow_e7` | U(0.01, 0.1) | 1e-07 | 15 | 157 | log, brute90/50 .. brute90x30 | 22 | precision_sweep |
| `narrow_e8` | U(0.01, 0.1) | 1e-08 | 15 | 157 | log, brute90/50 .. brute90x30 | 22 | precision_sweep |
| `small_e4` | U(0.001, 0.01) | 1e-04 | 157 | 1570 | log, brute90/50 .. brute90x30 | 22 | budget_to_reliability |
| `wide_e4` | U(0.001, 0.1) | 1e-04 | 15 | 1570 | log, brute90/50 .. brute90x30 | 22 | budget_to_reliability |
| `broad_pi4_e3` | U(0.01, 0.7854) | 1e-03 | 2 | 157 | log, 3,000 .. 2.00e+06 | 22 | broad_prior |
| `broad_pi2_e3` | U(0.01, 1.571) | 1e-03 | 1 | 157 | log, 3,000 .. 2.00e+06 | 22 | broad_prior |

`brute90 = 0.6724 / (N_min * eps^2)` is the analytic budget at which brute force reaches ~90% convergence; anchoring the grid to it puts the informative band in the middle of the sweep for every scenario, whatever its scale. The broad-prior scenarios instead use the explicit `3e3 .. 2e6` curve that `04-broad-dist.tex` reports, so they stay comparable to it.

`narrow_e3` additionally pins `B = 10,000`, the Table-3.1 operating point, so it is evaluated exactly rather than interpolated.

### 4.2 Budget grids in full

<details><summary>every evaluated budget, per scenario</summary>

**`narrow_e3`** (23 budgets)

```
896, 1,270, 1,799, 2,548, 3,610, 5,114, 7,244, 10,000, 10,262, 14,538, 20,594, 29,173, 41,326, 58,543, 82,931, 117,479, 166,419, 235,746, 333,954, 473,075, 670,151, 949,325, 1,344,800
```

**`narrow_e4`** (22 budgets)

```
89,653, 127,001, 179,908, 254,855, 361,024, 511,421, 724,471, 1,026,274, 1,453,804, 2,059,436, 2,917,365, 4,132,694, 5,854,310, 8,293,124, 11,747,910, 16,641,905, 23,574,663, 33,395,498, 47,307,541, 67,015,122, 94,932,574, 134,480,000
```

**`narrow_e5`** (22 budgets)

```
8,965,333, 12,700,150, 17,990,835, 25,485,535, 36,102,410, 51,142,110, 72,447,113, 102,627,447, 145,380,436, 205,943,651, 291,736,555, 413,269,439, 585,431,022, 829,312,426, 1,174,791,008, 1,664,190,563, 2,357,466,314, 3,339,549,897, 4,730,754,135, 6,701,512,293, 9,493,257,466, 13,447,999,999
```

**`narrow_e6`** (22 budgets)

```
896,533,333, 1,270,015,093, 1,799,083,511, 2,548,553,554, 3,610,241,091, 5,114,211,046, 7,244,711,353, 10,262,744,755, 14,538,043,655, 20,594,365,186, 29,173,655,514, 41,326,943,964, 58,543,102,238, 82,931,242,693, 117,479,100,897, 166,419,056,310, 235,746,631,456, 333,954,989,743, 473,075,413,571, 670,151,229,355, 949,325,746,642, 1,344,800,000,000
```

**`narrow_e7`** (22 budgets)

```
89,653,333,333, 127,001,509,327, 179,908,351,108, 254,855,355,420, 361,024,109,144, 511,421,104,605, 724,471,135,335, 1,026,274,475,590, 1,453,804,365,525, 2,059,436,518,680, 2,917,365,551,411, 4,132,694,396,433, 5,854,310,223,842, 8,293,124,269,377, 11,747,910,089,773, 16,641,905,631,032, 23,574,663,145,682, 33,395,498,974,351, 47,307,541,357,178, 67,015,122,935,579, 94,932,574,664,214, 134,480,000,000,000
```

**`narrow_e8`** (22 budgets)

```
8,965,333,333,333, 12,700,150,932,713, 17,990,835,110,839, 25,485,535,542,077, 36,102,410,914,495, 51,142,110,460,546, 72,447,113,533,589, 102,627,447,559,047, 145,380,436,552,546, 205,943,651,868,068, 291,736,555,141,147, 413,269,439,643,361, 585,431,022,384,237, 829,312,426,937,782, 1,174,791,008,977,376, 1,664,190,563,103,208, 2,357,466,314,568,226, 3,339,549,897,435,170, 4,730,754,135,717,893, 6,701,512,293,557,939, 9,493,257,466,421,480, 13,447,999,999,999,998
```

**`small_e4`** (22 budgets)

```
8,565, 12,133, 17,188, 24,349, 34,492, 48,861, 69,216, 98,051, 138,898, 196,761, 278,729, 394,843, 559,329, 792,336, 1,122,411, 1,589,990, 2,252,356, 3,190,652, 4,519,828, 6,402,718, 9,069,991, 12,848,407
```

**`wide_e4`** (22 budgets)

```
89,653, 127,001, 179,908, 254,855, 361,024, 511,421, 724,471, 1,026,274, 1,453,804, 2,059,436, 2,917,365, 4,132,694, 5,854,310, 8,293,124, 11,747,910, 16,641,905, 23,574,663, 33,395,498, 47,307,541, 67,015,122, 94,932,574, 134,480,000
```

**`broad_pi4_e3`** (22 budgets)

```
3,000, 4,088, 5,572, 7,595, 10,351, 14,108, 19,228, 26,207, 35,718, 48,681, 66,349, 90,429, 123,248, 167,979, 228,942, 312,031, 425,275, 579,619, 789,977, 1,076,679, 1,467,432, 2,000,000
```

**`broad_pi2_e3`** (22 budgets)

```
3,000, 4,088, 5,572, 7,595, 10,351, 14,108, 19,228, 26,207, 35,718, 48,681, 66,349, 90,429, 123,248, 167,979, 228,942, 312,031, 425,275, 579,619, 789,977, 1,076,679, 1,467,432, 2,000,000
```

</details>

### 4.3 Regime classification

Each operating point is labelled by the *best* convergence rate achieved at it by any algorithm (`operating_points.csv`):

| regime | rule | count |
|---|---|---:|
| `live` | 3% < best rate < 99% | **144** |
| `saturated` | best rate >= 99% | 77 |
| `floored` | best rate <= 3% | 0 |

**Every aggregate in §7–§11 is over `live` points only.** At a saturated point every configuration converges, the grid search cannot distinguish them (selection margins go to 0.00 pp), and the tie-break rule then picks the cheapest exploration — often a one-shot pilot. The *performance* number there is still valid; the *diagnostics* measured at that arbitrary configuration are not a property of the algorithm. All points remain in every CSV; join on `operating_points.csv` to filter differently.

### 4.4 Seeds

| role | value | used for |
|---|---|---|
| tuning block 1 | `42` | scoring candidate configurations |
| tuning block 2 | `43` | second, independent scoring block |
| held-out test | `2024` | **every reported number**; disjoint from both blocks |
| bootstrap | `12345` | all resampling: mean CIs, crossing CIs, ratio CIs |

Per-trial seeds are derived deterministically: `np.random.default_rng(seed).integers(0, 2**63, size=R)`. Trial `i` therefore draws the **same** `phi` for every algorithm, every budget and every configuration, so all comparisons are paired. Each trial then runs on `np.random.default_rng(trial_seed)`, making the result independent of execution order and of the number of worker processes.

### 4.5 Trial counts (every R in the study)

| symbol | value | where |
|---|---:|---|
| `R_test` | **50,000** | held-out evaluation of the frozen winner — every reported rate and every diagnostic |
| `R_tune` (stages A, B) | 700 | per tuning block, locating `m'` and scanning the discrete grid |
| `R_tune2` (stage C) | **6,000** | per tuning block, the refinement that decides the winner |
| `R_audit` | 50,000 | the binary-search pilot comparison in `algorithm_code_audit.md` |
| `n_boot` | 2,000 | bootstrap replicates (means, crossings, ratios) |

Tuning consumes `2 x (R_tune)` trials per candidate in stages A and B and `2 x (R_tune2)` in stage C — two blocks each. **None of those trials appear in any reported number**; they only choose the configuration.

---

## 5. Tuning and held-out protocol

**The rule.** No reported value is a maximum evaluated on the trials that chose the parameters.

```
1. score every candidate configuration on EACH tuning block (seeds 42 and 43), separately
2. winner = argmax of the MEAN of the two block rates
     - two blocks, so one accidental draw cannot decide it
     - ties broken toward the SMALLER exploration size, then toward the earlier candidate
3. freeze the winner, together with its per-block evidence and its runner-up
4. evaluate the frozen winner on the disjoint held-out seed 2024, at R_test, WITH tracing
5. performance AND diagnostics both come from that single evaluation
```

Step 5 is what makes the diagnostics trustworthy: a rate and its telemetry cannot refer to different runs, because there is only one run. `test_performance_consistency` asserts that the converged count derived from the traces equals the count from an independent untraced evaluation on the same seeds.

### 5.1 Three-stage grid search

A single grid fine enough to span `m' in [1, B//N_min]` *and* cross it with linear search's 450 discrete combinations would be 10,800 configurations per stage. The search is therefore staged:

| stage | `m'` grid | discrete grid | R per block | purpose |
|---|---|---|---:|---|
| **A** | 24 log-spaced points across the full box `[1, B//N_min]` | 3-point-per-axis *skeleton* (bottom/middle/top) | 700 | locate `m'` |
| **B** | 10 points within x4 of A's winner | the **full** discrete grid | 700 | choose the discrete configuration |
| **C** | 15 points within x2 of B's winner (~10% steps) | the neighbouring value on each axis | **6,000** | refine, and decide |

Stage C gets by far the most trials because it is the stage that actually decides. That directly controls the *variance* of which configuration wins — the one uncertainty component that cannot be recovered from stored output afterwards (see §6.8).

### 5.2 The `m'` search box

```
lower  m' = 1                 one shot -- the physical minimum
upper  m' = B // N_min        the largest pilot affordable at all: a probe at the opening
                              depth costs m'*N_min, so beyond this the algorithm cannot take
                              even its first probe and scores zero by construction
```

Neither endpoint can be criticised as arbitrary, and both are recorded per row (`m_lo`, `m_hi`) so a boundary hit is informative rather than an artifact.

### 5.3 Parameter grids in full

| algorithm | axis | values | n |
|---|---|---|---:|
| all adaptive | `m_exploration` | log-spaced over `[1, B//N_min]`, per scenario and budget | 24/10/15 |
| `linear` | `lookback_window` | 1, 2, 3, 4, 5, 6, 8, 12, 20 | 9 |
| `linear` | `safeguard` | 0, 1, 2, 3, 4, 6, 8, 12, 16, 24 | 10 |
| `linear` | `inc` | 1, 2, 3, 5, 8 | 5 |
| `linear` | `mean_window` | 0, 1, 2, 3, 4, 6, 8, 12 | 8 |
| `binary_deep` | `conf` | 0.5, 0.52, 0.55, 0.58, 0.62, 0.66, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99 | 13 |

Linear search therefore has **3600 discrete combinations**, binary search **13**, reverse engineering none.

**Why these ranges.** An earlier production sweep used `lookback_window in [1,2,5]` and `safeguard in [0,1,2]` and selected the **top** of those grids in 64.8% and 15.7% of live cells respectively — the optimum lay outside the search box, so those numbers understated linear search by an unknown amount. `safeguard` needed the largest extension because it is an *absolute* depth decrement: its useful range grows with `N_min`, and a grid tuned for `N_min = 15` cannot serve `N_min = 157`. §13 reports whether the current ranges bind.

### 5.4 Boundary reporting

`winners.csv` carries `at_axis_bound`, a JSON map flagging a winner pinned at an endpoint of **any** axis — not just `m'` — evaluated against the full grid. A pin at an axis *maximum* means the optimum may lie outside the box and the result is suspect. A pin at a *minimum* is usually a physical floor (`m' = 1` shot, `inc = 1`, `safeguard = 0`, `lookback_window = 1`, `conf = 0.5`) and is not a defect.

---

## 6. Statistical methodology

All intervals are **95%**. Every interval-bearing row in every CSV records its own `ci_level`, method, `n_boot`, `boot_seed` and denominator, so no interval has to be reconstructed from context.

| quantity | estimator | interval method |
|---|---|---|
| convergence rate | `k/R` | Wilson score at the row's own `R` |
| diagnostic share | `k/n_eligible` | Wilson score at the metric's own eligible denominator |
| median / quartiles | sample quantile | exact order-statistic bootstrap (closed form) |
| mean | sample mean | run-level percentile bootstrap, 2,000 replicates, seed 12345 |
| budget to reach p* | log-linear interpolation | parametric bootstrap of the curve, 2,000 replicates |
| budget ratio vs brute | ratio of two crossings | parametric bootstrap, both curves |
| grid discretisation | half-density subgrid deviation | deterministic, no interval |

### 6.1 Convergence and diagnostic proportions — Wilson score interval

For `k` successes in `n` trials, with `z = 1.959964` for 95%:

```
            p_hat + z^2/(2n)      z * sqrt( p_hat(1-p_hat)/n + z^2/(4n^2) )
centre  =  ------------------ ,  half = -----------------------------------------
              1 + z^2/n                            1 + z^2/n
```

Chosen over the Wald interval because it stays inside `[0,1]` and keeps close to nominal coverage near `p = 0` and `p = 1`, which matters here: several diagnostic shares (final overshoot, miss rate) are genuinely near zero, where Wald would produce negative lower bounds. At `R = 50,000` the worst-case half-width (at `p = 0.5`) is **0.438 pp**.

Every share is reported with its **eligible denominator** (`*_n`) and numerator (`*_k`). Eligibility is not cosmetic: a run with no defined `N_guess` (brute force, or an algorithm that never got a probe in) must not silently vanish into a smaller denominator. `test_eligibility_accounting` asserts `eligible + ineligible = R` for every metric, and that conditional metrics have denominators no larger than the sets they condition on.

### 6.2 Medians and quartiles — exact order-statistic bootstrap

The usual recipe is a percentile bootstrap: resample the R runs with replacement `n_boot` times, take the median of each resample, report the 2.5th and 97.5th percentiles of those medians. **For a quantile this does not need to be simulated** — it has a closed form.

A bootstrap resample's `q`-quantile is always one of the observed values `x_(k)`. Writing `m = ceil(q*n)` for the rank the estimator uses, the resampled quantile is at or below `x_(k)` exactly when at least `m` of the `n` resampled draws fall at or below `x_(k)`, and each draw does so independently with probability `k/n`. Hence

```
F_boot( x_(k) )  =  P( Binomial(n, k/n) >= m )
```

which is increasing in `k`, so the `alpha`-percentile of the bootstrap distribution is `x_(k*)` at the smallest `k*` with `F_boot(x_(k*)) >= alpha`. Evaluating that for `k = 1..n` and searching it costs O(n) and is **exact**.

This is the `n_boot -> infinity` limit of the percentile bootstrap, i.e. the quantity a Monte-Carlo bootstrap estimates noisily. It was adopted for cost as well as accuracy: a Monte-Carlo bootstrap of the medians alone, at `R = 50,000` across 884 cells and ~8 continuous metrics, is on the order of an hour of post-processing for a strictly worse answer. Verified against a Monte-Carlo bootstrap at 2,000 replicates on three seeds: identical endpoints.

Recorded per row as `ci_method_quantile = "order-statistic bootstrap (exact percentile interval)"`. Points with fewer than 20 eligible runs get the point estimate and no interval.

### 6.3 Means — run-level percentile bootstrap

Means have no equivalent closed form, so they use an ordinary Monte-Carlo percentile bootstrap: 2,000 replicates, seed 12345, resampling **runs** (never probes). Resampling at the run level is what carries the clustering — probes inside one run are dependent, so a probe-level resample would badly understate the uncertainty of anything probe-derived.

All continuous metrics are resampled with the *same* run indices within a replicate batch, and metrics that are undefined for a run carry `NaN` on the full run axis rather than being compacted, so every replicate averages over exactly the eligible runs it drew.

Recorded as `ci_method_mean`. The eight continuous metrics carrying a mean and its interval: `guess_ratio`, `guess_abs_rel`, `guess_signed_rel`, `exploration_share`, `n_probes`, `star_ratio`, `star_over_guess`, `budget_util`.

### 6.4 Budget to reach a target reliability — the crossing rule

One rule, used everywhere, defined once in `qmetrology/uncertainty.py::crossing` so the interval is always built around exactly the number reported:

```
find the first grid index i with rate[i] >= p*
f      = (p* - rate[i-1]) / (rate[i] - rate[i-1])
B(p*)  = exp( log(B[i-1]) + f * (log(B[i]) - log(B[i-1])) )      # LOG-LINEAR in budget
```

Interpolation is linear in `log B` because the convergence-vs-budget curve is close to linear on a log-budget axis over the relevant range (the estimator error falls as `1/sqrt(B)`), so log-linear interpolation is far more accurate than linear. Returns `NaN` if the curve never reaches `p*` within the swept range — reported as a missing crossing, never extrapolated.

Thresholds evaluated: 50%, 80%, 90%, 95%.

### 6.5 Crossing and ratio intervals — parametric bootstrap

The crossing is *derived*, not measured, so its uncertainty is propagated from the rates the curve is built from:

```
for b in 1..2,000:
    resample every curve point:  p*_j ~ Binomial(R, p_hat_j) / R
    re-derive the crossing from the resampled curve
report the 2.5th and 97.5th percentiles of the resampled crossings
```

It is *parametric* (Binomial at the observed rate) rather than a resample of raw trials because the curve points are already sufficient statistics — each is a binomial proportion over the same R. Replicates whose resampled curve never reaches `p*` are dropped and the surviving fraction is reported as `coverage`; coverage below 1.0 flags a crossing sitting at the edge of the swept range.

The ratio `B_brute(p*) / B_algo(p*)` resamples both curves **independently**. Because all algorithms share the seed list, a trial index draws the same `phi` for every algorithm, so the two crossings are in truth positively correlated and the independent resample **overstates** the ratio's uncertainty. The ratio intervals are therefore conservative, not optimistic — stated here rather than left for a reader to discover.

### 6.6 Budget-grid discretisation

Monte-Carlo error is not the only thing moving a crossing; the finite budget grid does too. This is probed directly and without assumptions: re-derive each crossing from the two half-density subgrids (`budgets[0::2]` and `budgets[1::2]`) and report the largest relative deviation. Log-linear interpolation error is `O(h^2)` in the grid log-spacing, so the full-density grid carries roughly **a quarter** of the deviation reported.

Measured here: median **2.59%**, 90th percentile 6.74%, max 13.14% over 197 crossings — so the full-grid contribution is of order 0.65% at the median.

### 6.7 Rounding conventions

Budgets to 3 significant figures; ratios to 2 decimals; percentages to 1–2 decimals. Applied at presentation only — the CSVs carry full precision.

### 6.8 What is *not* covered

Stated explicitly because it is the honest boundary of these intervals:

* **The sampling variance of the grid search itself.** Which configuration wins is random. De-biasing removes its *bias* (the winner is re-validated on an independent seed) but not its *variance*. Stage C's large `R_tune2 = 6,000` and the two-block selection rule reduce it; quantifying what remains requires repeating the tuning on many seeds (`analysis/tuning_stability.py` does this for headline cells only). The per-row `margin_pp` and `runner_up` columns let a reader see how much was at stake in each selection.
* **Model error in the safeguard's Gaussian posterior**, and the selection bias of `binary_deep`'s pilot. Not an interval; measured behaviourally in §8–§10 instead.
* **Interpolation bias in the crossing** beyond the `O(h^2)` bound of §6.6.

---

## 7. Results — performance

Convergence rate of the frozen winner on the held-out seed, R = 50,000 per cell. Aggregates over live points only (§4.3).

### 7.1 Overall

| algorithm | mean | median | min | max | points won |
|---|---:|---:|---:|---:|---:|
| Brute force | 51.3% | 49.0% | 8.8% | 97.4% | 0/144 |
| Linear search | 59.0% | 58.9% | 11.3% | 98.5% | 0/144 |
| Binary search | 61.2% | 61.8% | 12.1% | 98.7% | 63/144 |
| Reverse engineering | 61.6% | 63.1% | 11.9% | 98.9% | 81/144 |
| *Separable protocol (N=1)* | 22.0% | 15.7% | 0.0% | 96.2% | -- |

Means and medians mix scenarios and budgets; they summarise the table, they are not a headline claim. The per-cell rows with Wilson intervals are in `performance_curves.csv`. The last two rows are reference points, not protocols under comparison: they cannot win a point and are excluded from every diagnostic aggregate and from the regime rule.

### 7.2 Budget to reach a target reliability, relative to brute force

Ratio `B_brute(p*) / B_algo(p*)`; **greater than 1 means the algorithm needs less budget**.

| algorithm | p* | median ratio | min | max | scenarios with a crossing |
|---|---|---:|---:|---:|---:|
| Linear search | 50% | **1.76** | 1.26 | 2.07 | 10 |
| Linear search | 80% | **1.59** | 1.21 | 1.88 | 10 |
| Linear search | 90% | **1.49** | 1.14 | 1.67 | 10 |
| Linear search | 95% | **1.41** | 1.10 | 1.54 | 10 |
| Binary search | 50% | **2.08** | 1.41 | 2.38 | 10 |
| Binary search | 80% | **1.82** | 1.39 | 2.04 | 10 |
| Binary search | 90% | **1.68** | 1.32 | 1.84 | 10 |
| Binary search | 95% | **1.59** | 1.11 | 1.71 | 10 |
| Reverse engineering | 50% | **2.14** | 1.42 | 2.35 | 10 |
| Reverse engineering | 80% | **1.89** | 1.47 | 2.04 | 10 |
| Reverse engineering | 90% | **1.77** | 1.43 | 1.85 | 10 |
| Reverse engineering | 95% | **1.66** | 1.32 | 1.71 | 10 |

### 7.3 Per-scenario, at the 90% threshold

| scenario | brute `B(90%)` | linear | binary | reverse eng. | best ratio |
|---|---:|---:|---:|---:|---:|
| `narrow_e3` | 45,900 | 35,500 | 32,400 | 29,800 | 1.54x |
| `narrow_e4` | 4.53e+06 | 2.91e+06 | 2.83e+06 | 2.60e+06 | 1.74x |
| `narrow_e5` | 4.51e+08 | 2.76e+08 | 2.57e+08 | 2.51e+08 | 1.80x |
| `narrow_e6` | 4.51e+10 | 2.73e+10 | 2.54e+10 | 2.51e+10 | 1.80x |
| `narrow_e7` | 4.51e+12 | 2.70e+12 | 2.52e+12 | 2.49e+12 | 1.81x |
| `narrow_e8` | 4.51e+14 | 2.72e+14 | 2.55e+14 | 2.48e+14 | 1.82x |
| `small_e4` | 441,000 | 325,000 | 334,000 | 302,000 | 1.46x |
| `wide_e4` | 4.52e+06 | 3.19e+06 | 2.45e+06 | 2.45e+06 | 1.85x |
| `broad_pi4_e3` | 339,000 | 279,000 | 216,000 | 219,000 | 1.57x |
| `broad_pi2_e3` | 678,000 | 595,000 | 471,000 | 473,000 | 1.44x |

Full rows — every threshold, both interval endpoints, `coverage`, `grid_sensitivity_rel`, `R`, `n_boot`, `boot_seed` — are in `budget_crossings.csv`.

### 7.4 The Table-3.1 operating point

`U(0.01, 0.1)`, `eps = 1e-3`, `B = 10,000`, evaluated directly (not interpolated):

| algorithm | converged | 95% Wilson CI | frozen configuration |
|---|---:|---|---|
| Brute force | **56.09%** | [55.66%, 56.53%] | none |
| Linear search | **64.10%** | [63.68%, 64.52%] | inc=1, lookback_window=6, m_exploration=1, mean_window=4, safeguard=0 |
| Binary search | **65.07%** | [64.65%, 65.49%] | conf=0.5, m_exploration=114 |
| Reverse engineering | **66.21%** | [65.79%, 66.62%] | m_exploration=44 |

---

## 8. Results — exploration phase (`N_guess / N_opt`)

The primary exploration diagnostic: how close the exploration phase's *own answer* comes to the deepest non-aliasing depth, before any safeguard intervenes.

| algorithm | median ratio | mean abs. rel. err. | signed rel. err. | exact hit | within 5% | within 10% | guess overshoot |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear search | **1.00** | 0.16 | -0.11 | 37.3% | 50.9% | 61.1% | 6.6% |
| Binary search | **0.93** | 0.29 | -0.28 | 31.0% | 38.2% | 45.7% | 4.1% |
| Reverse engineering | **1.00** | 0.05 | +0.00 | 59.1% | 77.6% | 86.8% | 19.3% |

The mean ratio, the signed error and the absolute error are all retained deliberately: a mean ratio near 1 can hide a mixture of severe under- and overshoots, and only the absolute error exposes that.

`brute` is absent because it has no exploration phase — its `N_guess` columns are empty by construction, not zero.

### 8.1 Exploration cost

Each cell contributes its own within-cell statistic. The three *share* columns then take the **median across cells**, so they are directly comparable (a mean of per-cell means against a median of per-cell p90s would not be). The probe columns instead take the **mean across cells**, which — since every cell has the same R — is exactly the pooled mean over all runs, and is the interpretable "probes per trial" number.

| algorithm | expl. share: median | mean | p90 | mean probes/trial | median probes | only one probe | no exploitation phase |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brute force | 0.00% | 0.00% | 0.00% | -- | -- | -- | 0.00% |
| Linear search | 0.83% | 2.45% | 4.52% | 22.8 | 16.0 | 0.0% | 0.00% |
| Binary search | 2.70% | 2.83% | 3.09% | 4.4 | 1.0 | 53.5% | 0.00% |
| Reverse engineering | 1.33% | 1.37% | 1.33% | 1.0 | 1.0 | 98.6% | 0.04% |

Brute force takes no probes at all, so its probe-shape columns are blank rather than reporting the vacuously true "0 probes is <= 1 probe".

**Termination reasons** are recorded per cell in `diagnostics_by_point.csv` (`termination_reasons`, a `reason=count` list summing to R). The statuses are: `ok`, `detector_fired`, `scan_exhausted`, `no_exploitation_budget_exhausted`, `no_exploitation_shots`, `pilot_retries_exhausted_budget`, `refused_pilot_unaffordable`.

---

## 9. Results — safeguard and final depth (`N_star / N_opt`)

| algorithm | median `N*/N_opt` | Q1 | Q3 | final overshoot | exact | within 10% | converged given a safe depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brute force | **0.54** | 0.31 | 0.75 | 0.00% | 12.7% | 17.8% | 51.3% |
| Linear search | **0.95** | 0.72 | 0.97 | 1.72% | 22.8% | 54.2% | 59.7% |
| Binary search | **1.00** | 0.94 | 1.00 | 0.74% | 44.5% | 76.0% | 61.3% |
| Reverse engineering | **1.00** | 0.92 | 1.00 | 0.47% | 43.2% | 72.5% | 61.7% |

**A median below 1 is the intended behaviour, not a miss.** The safeguard's objective is `P(no overshoot) x P(converge)`, not `N_opt` itself: it deliberately backs off from the aliasing cliff, and because the estimator is boundary-censored near `N_opt` (where `p0 = cos^2(N phi)` sits against 0) a depth slightly below `N_opt` can have *higher* true convergence than `N_opt`. Exact attainment is reported because readers ask for it, but it is not the success criterion.

### 9.1 What the safeguard does to the guess

| algorithm | median `N*/N_guess` | Q1 | Q3 | unsafe-guess rescue | backoff when the guess was already safe |
|---|---:|---:|---:|---:|---:|
| Linear search | 0.98 | 0.95 | 1.00 | 44.6% | 1.00x |
| Binary search | 1.05 | 1.00 | 2.25 | 96.4% | 1.06x |
| Reverse engineering | 0.99 | 0.92 | 1.00 | 97.3% | 1.00x |

*Unsafe-guess rescue* is `P(N_star <= N_opt | N_guess > N_opt)` — how often the safeguard pulls a genuinely aliasing guess back to safety. Its denominator (`rescue_share_n`) is the count of runs whose guess overshot, and is reported per row.

**Not a failure decomposition.** Early stopping, final overshoot and ordinary shot noise are reported here as *overlapping stage flags and conditional rates*. They are deliberately **not** presented as a mutually exclusive causal breakdown of why trials fail: no priority or counterfactual rule has been specified that would justify assigning each failure to exactly one cause.

---

## 10. Results — detector quality

Only linear and binary search make overshoot declarations. Reverse engineering and brute force have no detector and are **N/A**, never zero.

| algorithm | trial-level false alarm | trial-level miss | probe TP | FP | TN | FN | eligible runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear search | **53.8%** | **6.6%** | 26,034,339 | 11,647,285 | 125,047,269 | 1,230,018 | 7,200,000 |
| Binary search | **22.8%** | **4.1%** | 14,367,412 | 4,664,659 | 12,323,125 | 335,727 | 7,200,000 |

Rates are **trial-level** with Wilson intervals at the row's own R (§3): a run counts as a false alarm if *at least one* safe probe was declared an overshoot, and as a miss if *at least one* overshooting probe was accepted. Probe-level confusion counts are supporting telemetry and carry no interval, because probes within a run are dependent.

Per-cell rows with both interval endpoints and denominators: `detector_confusion.csv`.

---

## 11. Results — trends

### 11.1 With the target precision `eps`

`U(0.01, 0.1)` held fixed; only `eps` and the budget scale it forces change. Live points only.

| algorithm | eps | median expl. share | median `N_guess/N_opt` | median `N*/N_opt` | final overshoot | mean convergence |
|---|---|---:|---:|---:|---:|---:|
| Brute force | 1e-03 | 0.00% | -- | 0.54 | 0.00% | 52.6% |
| Brute force | 1e-04 | 0.00% | -- | 0.54 | 0.00% | 52.6% |
| Brute force | 1e-05 | 0.00% | -- | 0.54 | 0.00% | 52.7% |
| Brute force | 1e-06 | 0.00% | -- | 0.54 | 0.00% | 52.7% |
| Brute force | 1e-07 | 0.00% | -- | 0.54 | 0.00% | 52.7% |
| Brute force | 1e-08 | 0.00% | -- | 0.54 | 0.00% | 52.7% |
| Linear search | 1e-03 | 2.68% | 0.72 | 0.71 | 2.17% | 57.8% |
| Linear search | 1e-04 | 2.88% | 1.00 | 0.92 | 2.36% | 60.9% |
| Linear search | 1e-05 | 1.18% | 1.00 | 0.95 | 0.79% | 63.0% |
| Linear search | 1e-06 | 0.34% | 1.00 | 0.95 | 0.45% | 63.7% |
| Linear search | 1e-07 | 0.06% | 1.00 | 1.00 | 0.49% | 64.2% |
| Linear search | 1e-08 | 0.00% | 1.00 | 0.99 | 0.41% | 64.1% |
| Binary search | 1e-03 | 15.56% | 0.54 | 0.87 | 1.56% | 58.7% |
| Binary search | 1e-04 | 5.44% | 0.97 | 0.92 | 1.71% | 61.4% |
| Binary search | 1e-05 | 1.57% | 0.98 | 0.97 | 0.27% | 64.2% |
| Binary search | 1e-06 | 0.04% | 1.00 | 1.00 | 0.19% | 64.9% |
| Binary search | 1e-07 | 0.00% | 1.00 | 1.00 | 0.09% | 65.1% |
| Binary search | 1e-08 | 0.00% | 1.00 | 1.00 | 0.13% | 65.0% |
| Reverse engineering | 1e-03 | 6.41% | 1.00 | 0.83 | 1.31% | 59.3% |
| Reverse engineering | 1e-04 | 2.75% | 1.00 | 0.94 | 0.36% | 63.7% |
| Reverse engineering | 1e-05 | 0.18% | 1.00 | 1.00 | 0.09% | 65.0% |
| Reverse engineering | 1e-06 | 0.00% | 1.00 | 1.00 | 0.05% | 65.1% |
| Reverse engineering | 1e-07 | 0.00% | 1.00 | 1.00 | 0.05% | 65.1% |
| Reverse engineering | 1e-08 | 0.00% | 1.00 | 1.00 | 0.05% | 65.1% |

### 11.2 With the available budget

Live points split into terciles of budget *within each scenario*, so scale differences between scenarios do not confound the comparison.

| algorithm | tercile | median expl. share | median `N_guess/N_opt` | guess overshoot | mean probes | only the opening probe |
|---|---|---:|---:|---:|---:|---:|
| Linear search | low | 0.54% | 1.00 | 9.2% | 20.3 | 0.0% |
| Linear search | mid | 0.87% | 1.00 | 7.1% | 22.7 | 0.0% |
| Linear search | high | 1.22% | 1.00 | 2.8% | 25.8 | 0.0% |
| Binary search | low | 5.58% | 0.97 | 6.2% | 4.4 | 52.8% |
| Binary search | mid | 2.69% | 0.94 | 3.0% | 4.1 | 57.4% |
| Binary search | high | 2.33% | 0.88 | 2.8% | 4.7 | 50.0% |
| Reverse engineering | low | 4.48% | 1.00 | 18.1% | 1.1 | 97.5% |
| Reverse engineering | mid | 1.43% | 1.00 | 20.1% | 1.0 | 99.1% |
| Reverse engineering | high | 0.13% | 1.00 | 19.9% | 1.0 | 99.5% |

### 11.3 With the true phase

Each cell's runs split into quartiles of the true `phi` (`diagnostics_by_phase.csv`). Small `phi` means a large `N_opt` and a hard depth problem; large `phi` means `N_opt` close to `N_min` and little to search for.

| algorithm | phase quartile | median `N_guess/N_opt` | guess overshoot | median `N*/N_opt` | final overshoot | convergence |
|---|---|---:|---:|---:|---:|---:|
| Brute force | Q1 | -- | -- | 0.21 | 0.00% | 67.9% |
| Brute force | Q2 | -- | -- | 0.43 | 0.00% | 68.3% |
| Brute force | Q3 | -- | -- | 0.65 | 0.00% | 68.1% |
| Brute force | Q4 | -- | -- | 0.88 | 0.00% | 68.3% |
| Linear search | Q1 | 0.65 | 5.3% | 0.61 | 1.17% | 80.8% |
| Linear search | Q2 | 0.83 | 4.8% | 0.78 | 0.84% | 74.4% |
| Linear search | Q3 | 0.95 | 5.0% | 0.89 | 1.35% | 70.0% |
| Linear search | Q4 | 0.94 | 3.0% | 0.89 | 1.19% | 67.7% |
| Binary search | Q1 | 0.69 | 11.3% | 0.83 | 1.07% | 83.5% |
| Binary search | Q2 | 0.76 | 8.9% | 0.95 | 0.55% | 75.9% |
| Binary search | Q3 | 0.91 | 6.5% | 0.96 | 0.26% | 71.0% |
| Binary search | Q4 | 1.00 | 6.8% | 1.00 | 0.10% | 68.1% |
| Reverse engineering | Q1 | 1.00 | 28.2% | 0.72 | 0.37% | 83.6% |
| Reverse engineering | Q2 | 1.00 | 19.3% | 0.90 | 0.44% | 76.2% |
| Reverse engineering | Q3 | 1.00 | 13.8% | 0.95 | 0.32% | 71.3% |
| Reverse engineering | Q4 | 1.00 | 8.7% | 0.95 | 0.11% | 68.5% |

*(Phase strata are computed over all points, live and saturated alike, since they are within-cell splits rather than cross-cell aggregates.)*

### 11.4 Figures

* `fig_diagnostics_vs_budget.png` — exploration budget share, median `N_guess/N_opt` and final overshoot against budget, normalised per scenario.
* `fig_guess_vs_final_depth.png` — `N_guess/N_opt` against `N*/N_opt`, and `N*/N_opt` against convergence.

Both regenerate from `diagnostics_by_point.csv` alone via `python analysis/consolidated/figures.py` — no simulation.

---

## 12. Budget compliance audit

Over **all 1,105 evaluated cells** and every held-out run in them:

| | |
|---|---:|
| trials spending more than the nominal budget | **0** |
| worst per-trial spend observed | **1.000000x** the cap |
| mean spend across cells | 0.9997x |
| median unused budget | 0.00% |

**A mean spend of 1.0000x does not by itself certify compliance** — it can average over trials that overspend and trials that underspend. That is why the per-trial *maximum* and an explicit *violation count* are reported per cell, in `budget_audit.csv`, alongside the mean, median and p90. Violations are counted, never clipped away after the fact.

Compliance is structural, not incidental: each exploration probe is checked to fit *before* it is paid for, and every exploitation measurement takes `m = floor(remaining / N)`. `test_budget_compliance` asserts both the total and the `B_exploration + N_star * m_final <= B` decomposition over 4,800 trials spanning four scenarios.

One convention worth stating: when an algorithm cannot afford even its opening probe it returns `(inf, B)` — the nominal budget — rather than 0. This is the pre-existing convention of the codebase and is conservative for this audit (it can only over-report spending). The affected runs carry `status = refused_pilot_unaffordable`.

---

## 13. Tuning provenance and grid adequacy

663 tuned cells, 432 of them at live operating points. Every held-out row in `performance_curves.csv` points to exactly one frozen winner in `winners.csv` with the same parameter dictionary — asserted by `test_end_to_end_smoke_and_provenance`, which also checks that the winner's recorded held-out rate equals the reported rate exactly.

Each winner row carries: both per-block tuning rates, their mean, the runner-up configuration, the selection margin in percentage points, the stage A/B/C configuration counts, the `m'` box endpoints, and `at_axis_bound`.

### 13.1 How much was at stake in each selection

| algorithm | median margin over runner-up | p90 | max |
|---|---:|---:|---:|
| Linear search | 0.08 pp | 0.24 pp | 0.56 pp |
| Binary search | 0.00 pp | 0.12 pp | 0.50 pp |
| Reverse engineering | 0.04 pp | 0.19 pp | 0.64 pp |

These margins are small — the objective is flat near its optimum. That is exactly why the deciding stage runs at the largest R, and why the `m'` grid resolution (~10% steps) is far below the noise floor rather than being pushed further.

### 13.2 Grid adequacy — boundary report

**Pins at an axis MAXIMUM** — the failure mode that invalidates a search box, because the optimum may lie outside it:

| algorithm | axis | share of live tuned cells | verdict |
|---|---|---:|---|
| Linear search | `mean_window` (min) | 12/144 = 8.3% | negligible |
| Linear search | `inc` (max) | 8/144 = 5.6% | negligible |
| Linear search | `mean_window` (max) | 1/144 = 0.7% | negligible |
| Binary search | `conf` (max) | 1/144 = 0.7% | negligible |

**Pins at an axis MINIMUM** — physical floors, not defects (`m' = 1` is one shot, `inc = 1` one step, `safeguard = 0` no decrement, `lookback_window = 1` a single look back, `conf = 0.5` the point at which the normal quantile vanishes and the branch rule becomes a plain comparison against the reference estimate):

| algorithm | axis | share of live tuned cells |
|---|---|---:|
| Linear search | `inc` | 117/144 = 81.2% |
| Binary search | `conf` | 82/144 = 56.9% |
| Linear search | `safeguard` | 62/144 = 43.1% |
| Linear search | `m_exploration` | 36/144 = 25.0% |
| Linear search | `lookback_window` | 6/144 = 4.2% |

### 13.3 Selected parameter values

Distribution of the frozen winners over live cells — showing whether the widened ranges are actually used:

**Linear search**

* `m_exploration`: 92 distinct values, 1 .. 43,212,199 (median 92)
* `lookback_window`: 1x6, 2x26, 3x17, 4x6, 5x17, 6x39, 8x23, 12x10
* `safeguard`: 0x62, 1x50, 2x11, 3x9, 4x5, 6x5, 8x2
* `inc`: 1x117, 2x11, 3x4, 5x4, 8x8
* `mean_window`: 0x12, 1x46, 2x25, 3x26, 4x23, 6x5, 8x6, 12x1

**Binary search**

* `m_exploration`: 142 distinct values, 9 .. 241,392 (median 1,361)
* `conf`: 0.5x82, 0.52x10, 0.62x1, 0.66x3, 0.7x7, 0.75x9, 0.8x7, 0.85x4, 0.9x13, 0.95x7, 0.99x1

**Reverse engineering**

* `m_exploration`: 139 distinct values, 4 .. 6,586,591 (median 769)

Full appendix-ready tables, including the frozen configuration at the tested budget nearest each interpolated `B_90` **and both bracketing winners**, are in `optimal_params.csv` and `tex/optimal_params.tex`. A budget crossing is interpolated between tested budgets; parameter dictionaries are **not** interpolated, which is precisely why the bracketing rows exist and why every such row carries a `budget_note` saying so.

---

## 14. Validation suite

`python tests/test_consolidated.py --slow` — **13/13 passing**. Runs standalone (no pytest required) and is pytest-compatible.

| test | asserts |
|---|---|
| `test_trace_neutrality` | Same seed, tracing on vs off, returns identical `phi_hat` and budget use. 4 scenarios x 4 algorithms x 150 trials. |
| `test_budget_compliance` | No algorithm exceeds its nominal budget, and `B_exploration + N_star*m_final <= B`. 4 scenarios x 4 algorithms x 300 trials; violations are collected and reported, not clipped. |
| `test_algorithm_definitions` | `N_guess`, `N_star` and `B_exploration` recomputed by hand from each run's own probe list must equal what the trace recorded — separately for all four algorithms, including that brute force has a null guess and that RE charges pilot retries. |
| `test_run_record_detector_definitions` | Hand-constructed traces verify the false-alarm and miss definitions in all four confusion quadrants, and that a detector-less algorithm returns NaN rather than 0. |
| `test_binary_deep_matches_fine_sweep` | The new deepest-probe binary search reproduces, trial for trial, the depth decision of the `binary_deep` arm of `analysis/fine_sweep.py` (900 trials, 3 operating points). |
| `test_determinism` | Repeated held-out evaluations and repeated tuning with the same manifest and seeds return identical arrays and identical winners. |
| `test_eligibility_accounting` | `eligible + ineligible = R` for every metric; conditional denominators never exceed the sets they condition on; detector-less and search-less algorithms report N/A with denominator 0. |
| `test_performance_consistency` | The converged count derived from the traces equals the count from an independent untraced evaluation on the same seeds — i.e. performance and diagnostics are one evaluation, not two. |
| `test_ci_uses_row_R` | A smaller R must give a wider Wilson interval; the uncertainty functions must take R as an argument; and **no consumer module may contain a literal trial count** (guards against the 50,000 / 40,000 / 30,000 / 20,000 disagreement in the pre-existing scripts). |
| `test_ci_metadata_present` | Every diagnostics row carries `R`, `ci_level`, `n_boot`, `boot_seed` and all three `ci_method_*` strings. |
| `test_code_audit_names_the_implementations` | The generated audit names every reported implementation, names both binary-search pilot variants, and the manifest points at the deepest-probe one. |
| `test_end_to_end_smoke_and_provenance` | A full smoke sweep runs without a failed scenario; every held-out row maps to exactly one frozen winner with matching params and rate; all output files exist and are non-empty; crossing intervals use the curve's own R; the appendix carries `B90_nearest` plus both brackets, each with a `budget_note`. |
| `test_reproducibility_and_resume` | Two independent sweeps produce byte-identical outputs; deleting a scenario's rows and re-running with `--resume` reproduces them exactly. |

---

## 15. Limitations and code/thesis mismatches

### 15.1 The binary-search pilot — resolved, and it changes the reported algorithm

The thesis pseudocode stores the deepest probe not classified as an overshoot (`phi_hat_acc`, `N_acc`) and supplies **that** pilot to the statistical safeguard. `qmetrology.algorithms.find_phi_fixed_budget_binary_search_risk` supplied the **opening** probe (`phi_0`, `N_min`). These are different algorithms.

Resolved by **adding** a function rather than relabelling one:

* `find_phi_fixed_budget_binary_search_deep` — deepest accepted probe. **This is what is reported here**, matching the pseudocode and the `binary_deep` arm of `fine_sweep.csv`.
* `find_phi_fixed_budget_binary_search_risk` — opening probe. Untouched, so every earlier study citing it still reproduces.

Both call the same `_binary_search_explore`, so at one configuration they differ in exactly one input to `risk_optimal_depth`. `algorithm_code_audit.md` measures the difference on common seeds at four operating points; the thesis-faithful pilot is **mildly worse** (0 to -2.4 pp). That is a result, not a bug, and it is surfaced rather than buried.

### 15.2 Selection bias in that pilot

The deepest accepted estimate is accepted *because it passed the overshoot test*, so it is selected for having read high. Treating it afterwards as an unbiased Gaussian pilot is outside the safeguard's derivation. Not assumed away — measured, via the detector rates (§10) and the `N_guess/N_opt` distribution (§8).

### 15.3 Linear search has no per-probe detector

Its stopping rule is a trial-level verdict, so the probe-level classification scored in §10 is the algorithm's own backtracking decision. A different mapping would produce different false-alarm numbers. Stated, not hidden.

### 15.4 `../thesis/*.tex` is not present in this checkout

The code audit is written against the handoff's statement of the pseudocode and against `results/BS_METHOD_DECISION.md`, not against the `.tex` sources directly. Every function's pilot and cap behaviour is verified against the code, and `binary_deep` is asserted trial-for-trial identical to the `fine_sweep.py` reference arm. No thesis file is read or written by this pipeline.

### 15.5 Scattered run constants elsewhere in the repository

`qmetrology/config.py` (R=50,000), `analysis/extensive_sweep.py` (40,000), `analysis/broad_dist_study.py` (30,000) and `analysis/fine_sweep.py` (20,000) still disagree with each other. They are untouched and **unused** by this pipeline, which reads every constant from `qmetrology/manifest.py` and stamps it on each row.

### 15.6 Uncertainty not quantified

See §6.8: the variance of which configuration the grid search picks, model error in the safeguard's Gaussian posterior, and interpolation bias beyond the `O(h^2)` grid bound.

### 15.7 Saturated operating points

At saturated points the tuner cannot distinguish configurations and its argmax is arbitrary. Performance there is valid; diagnostics are not, and are excluded from every aggregate (§4.3). They remain in the CSVs, labelled.

---

## 16. File and column reference

### 16.1 Files

| file | contents |
|---|---|
| `experiment_manifest.json` | every scenario, budget grid, algorithm, seed, R and grid the run used |
| `algorithm_code_audit.md` | which implementation is which thesis algorithm; the measured pilot-mismatch comparison |
| `operating_points.csv` | per (scenario, budget): best/worst rate, spread, and the live/saturated/floored label |
| `performance_curves.csv` | convergence vs budget with Wilson intervals at the row's own R |
| `budget_crossings.csv` | B(p*) and the ratio vs brute force, bootstrap intervals, coverage, grid sensitivity |
| `winners.csv` | the frozen tuning winner per cell: per-block rates, runner-up, margin, stage sizes, at_axis_bound |
| `optimal_params.csv` | appendix parameter tables, incl. the B_90 nearest and both bracketing winners |
| `diagnostics_by_point.csv` | all Tier A + Tier B diagnostics; every share with its eligible denominator |
| `diagnostics_headline.csv` | the compact per-scenario subset the thesis tables draw from |
| `diagnostics_by_phase.csv` | the depth diagnostics split by quartiles of the true phase |
| `detector_confusion.csv` | linear/binary false-alarm and miss rates + probe-level TP/FP/TN/FN |
| `budget_audit.csv` | per-cell spend: mean, median, p90, max, unused share, violation count |
| `REPORT.md` | the short report |
| `FULL_RESULTS.md` | this document |
| `tex/` | paste-ready LaTeX for the exploration, downstream, detector, crossing and parameter tables |
| `traces/` | full probe lists (gzipped JSONL) for the headline points, first 2,000 runs each |
| `fig_*.png` | diagnostic trend figures |

### 16.2 Columns present on every tidy row

`setting`, `scenario_id`, `phi_min`, `phi_max`, `phi_distribution`, `eps`, `algorithm`, `implementation_variant`, `budget`, `params`, `R`, `seed_test`, `seed_tune_blocks`, `N_min`, `N_max`. No file's meaning depends on an undocumented column prefix, and no join requires knowledge held only in the code.

### 16.3 Suffix conventions on diagnostic columns

| suffix | meaning |
|---|---|
| `_lo`, `_hi` | the interval endpoints, at `ci_level` |
| `_n` | the **eligible denominator** for that share |
| `_k` | the numerator (count of eligible runs satisfying the predicate) |
| `_mean`, `_mean_lo`, `_mean_hi` | the mean and its run-level bootstrap interval |
| `_median`, `_q1`, `_q3`, `_p90` | sample quantiles |
| `_median_lo`, `_median_hi` | the exact order-statistic bootstrap interval for the median |

### 16.4 Reproducing

```bash
python analysis/consolidated/run.py --max --keep-traces   # the full sweep
python analysis/consolidated/run.py --report-only                 # rebuild derived tables
python analysis/consolidated/full_report.py                       # rebuild this document
python analysis/consolidated/figures.py                           # rebuild the figures
python tests/test_consolidated.py --slow                          # the validation suite
```

Trial seeds are derived deterministically from the manifest, so a re-run reproduces every number exactly (`test_reproducibility_and_resume`).

