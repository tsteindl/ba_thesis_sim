# Results — rebuilt on a boundary-free grid

_`binary_deep`, `re_fixed` and `linear_fixed` are re-tuned over `m' ∈ [1, budget // N_min]` with two-stage refinement (`analysis/fine_sweep.py`, 23 scenarios × 14 budgets, tuned seed 42 at R=600/1500, validated seed 2024 at R=20,000). Brute force, separable and the ceiling `oracle_hl` carry over from `story_curves.csv` — none of them has a tuning parameter._

_Binary search uses the deepest **accepted** probe as the Eq. (3.8) pilot (`results/ALGORITHM_BINARY.md`); everything else is the shipped algorithm._

## Table C — budget ratio vs brute force, by threshold

_Ratio = `B_brute(p*) / B_algo(p*)`: how much **less** budget the algorithm needs to first reach convergence `p*`. >1 wins._

| Scenario | p* | Linear | Binary | Rev. eng. | Ceiling | winner |
|---|---:|---:|---:|---:|---:|---|
| U(0.0001,0.001), eps=1e-04 | 75% | _0.72×_ | — | — | 2.61× | Linear search |
| U(0.0001,0.001), eps=1e-04 | 90% | _0.79×_ | _0.98×_ | _0.81×_ | 2.23× | Binary search |
| U(0.0001,0.001), eps=1e-06 | 50% | **1.10×** | **1.65×** | **1.97×** | 2.19× | Reverse engineering |
| U(0.0001,0.001), eps=1e-06 | 75% | **1.11×** | **1.59×** | **1.84×** | 2.00× | Reverse engineering |
| U(0.0001,0.001), eps=1e-06 | 90% | **1.08×** | **1.51×** | **1.69×** | 1.80× | Reverse engineering |
| U(0.0001,0.01), eps=1e-04 | 50% | **1.06×** | **1.60×** | **1.48×** | 2.82× | Binary search |
| U(0.0001,0.01), eps=1e-04 | 75% | **1.06×** | **1.59×** | **1.51×** | 2.28× | Binary search |
| U(0.0001,0.01), eps=1e-04 | 90% | **1.08×** | **1.51×** | **1.46×** | 1.97× | Binary search |
| U(0.0001,0.01), eps=1e-06 | 50% | **1.58×** | **2.62×** | **2.50×** | 2.68× | Binary search |
| U(0.0001,0.01), eps=1e-06 | 75% | **1.52×** | **1.99×** | **2.15×** | 2.23× | Reverse engineering |
| U(0.0001,0.01), eps=1e-06 | 90% | **1.38×** | **1.82×** | **1.84×** | 1.91× | Reverse engineering |
| U(0.0001,0.1), eps=1e-04 | 50% | **1.52×** | **2.49×** | **2.44×** | 2.86× | Binary search |
| U(0.0001,0.1), eps=1e-04 | 75% | **1.42×** | **2.10×** | **2.14×** | 2.33× | Reverse engineering |
| U(0.0001,0.1), eps=1e-04 | 90% | **1.31×** | **1.87×** | **1.80×** | 1.99× | Binary search |
| U(0.0001,0.1), eps=1e-06 | 50% | **1.99×** | **2.82×** | **2.74×** | 2.84× | Binary search |
| U(0.0001,0.1), eps=1e-06 | 75% | **1.65×** | **2.29×** | **2.28×** | 2.32× | Binary search |
| U(0.0001,0.1), eps=1e-06 | 90% | **1.49×** | **1.96×** | **1.92×** | 1.97× | Binary search |
| U(0.001,0.01), eps=1e-03 | 90% | _0.84×_ | _0.98×_ | _0.81×_ | 2.22× | Binary search |
| U(0.001,0.01), eps=1e-04 | 50% | **1.02×** | **1.38×** | **1.46×** | 2.23× | Reverse engineering |
| U(0.001,0.01), eps=1e-04 | 75% | **1.03×** | **1.39×** | **1.48×** | 2.02× | Reverse engineering |
| U(0.001,0.01), eps=1e-04 | 90% | **1.07×** | **1.29×** | **1.40×** | 1.84× | Reverse engineering |
| U(0.001,0.01), eps=1e-05 | 50% | **1.31×** | **1.67×** | **1.99×** | 2.19× | Reverse engineering |
| U(0.001,0.01), eps=1e-05 | 75% | **1.30×** | **1.61×** | **1.85×** | 1.99× | Reverse engineering |
| U(0.001,0.01), eps=1e-05 | 90% | **1.24×** | **1.53×** | **1.67×** | 1.80× | Reverse engineering |
| U(0.001,0.01), eps=1e-06 | 50% | **1.46×** | **2.04×** | **2.18×** | 2.18× | Reverse engineering |
| U(0.001,0.01), eps=1e-06 | 75% | **1.49×** | **1.87×** | **1.93×** | 1.99× | Reverse engineering |
| U(0.001,0.01), eps=1e-06 | 90% | **1.46×** | **1.67×** | **1.71×** | 1.79× | Reverse engineering |
| U(0.001,0.01), eps=1e-07 | 50% | **1.62×** | **2.15×** | **2.14×** | 2.18× | Binary search |
| U(0.001,0.01), eps=1e-07 | 75% | **1.48×** | **1.95×** | **1.96×** | 1.99× | Reverse engineering |
| U(0.001,0.01), eps=1e-07 | 90% | **1.35×** | **1.75×** | **1.70×** | 1.79× | Binary search |
| U(0.001,0.01), eps=1e-08 | 50% | **1.61×** | **2.20×** | **2.03×** | 2.18× | Binary search |
| U(0.001,0.01), eps=1e-08 | 75% | **1.42×** | **1.97×** | **1.95×** | 1.99× | Binary search |
| U(0.001,0.01), eps=1e-08 | 90% | **1.45×** | **1.74×** | **1.71×** | 1.79× | Binary search |
| U(0.001,0.1), eps=1e-04 | 50% | **1.43×** | **2.37×** | **2.42×** | 2.78× | Reverse engineering |
| U(0.001,0.1), eps=1e-04 | 75% | **1.43×** | **2.06×** | **2.13×** | 2.30× | Reverse engineering |
| U(0.001,0.1), eps=1e-04 | 90% | **1.27×** | **1.79×** | **1.82×** | 1.97× | Reverse engineering |
| U(0.001,0.1), eps=1e-06 | 50% | **1.94×** | **2.68×** | **2.62×** | 2.77× | Binary search |
| U(0.001,0.1), eps=1e-06 | 75% | **1.74×** | **2.22×** | **2.20×** | 2.29× | Binary search |
| U(0.001,0.1), eps=1e-06 | 90% | **1.56×** | **1.90×** | **1.89×** | 1.96× | Binary search |
| U(0.005,0.05), eps=1e-04 | 50% | **1.38×** | **1.50×** | **1.84×** | 2.20× | Reverse engineering |
| U(0.005,0.05), eps=1e-04 | 75% | **1.39×** | **1.55×** | **1.79×** | 2.00× | Reverse engineering |
| U(0.005,0.05), eps=1e-04 | 90% | **1.32×** | **1.45×** | **1.66×** | 1.80× | Reverse engineering |
| U(0.005,0.05), eps=1e-06 | 50% | **1.74×** | **2.16×** | **2.13×** | 2.19× | Binary search |
| U(0.005,0.05), eps=1e-06 | 75% | **1.56×** | **1.96×** | **1.97×** | 2.00× | Reverse engineering |
| U(0.005,0.05), eps=1e-06 | 90% | **1.46×** | **1.75×** | **1.72×** | 1.79× | Binary search |
| U(0.01,0.1), eps=1e-03 | 50% | **1.03×** | **1.45×** | **1.51×** | 2.27× | Reverse engineering |
| U(0.01,0.1), eps=1e-03 | 75% | **1.03×** | **1.43×** | **1.54×** | 2.06× | Reverse engineering |
| U(0.01,0.1), eps=1e-03 | 90% | **1.05×** | **1.40×** | **1.49×** | 1.87× | Reverse engineering |
| U(0.01,0.1), eps=1e-04 | 50% | **1.52×** | **1.71×** | **2.03×** | 2.25× | Reverse engineering |
| U(0.01,0.1), eps=1e-04 | 75% | **1.53×** | **1.73×** | **1.89×** | 2.04× | Reverse engineering |
| U(0.01,0.1), eps=1e-04 | 90% | **1.43×** | **1.59×** | **1.73×** | 1.84× | Reverse engineering |
| U(0.01,0.1), eps=1e-05 | 50% | **1.74×** | **2.12×** | **2.14×** | 2.24× | Reverse engineering |
| U(0.01,0.1), eps=1e-05 | 75% | **1.67×** | **1.96×** | **1.99×** | 2.04× | Reverse engineering |
| U(0.01,0.1), eps=1e-05 | 90% | **1.44×** | **1.76×** | **1.72×** | 1.83× | Binary search |
| U(0.01,0.1), eps=1e-06 | 50% | **1.78×** | **2.24×** | **2.18×** | 2.24× | Binary search |
| U(0.01,0.1), eps=1e-06 | 75% | **1.65×** | **2.02×** | **1.99×** | 2.04× | Binary search |
| U(0.01,0.1), eps=1e-06 | 90% | **1.51×** | **1.79×** | **1.73×** | 1.83× | Binary search |
| U(0.01,0.1), eps=1e-07 | 50% | **1.82×** | **2.25×** | **2.19×** | 2.24× | Binary search |
| U(0.01,0.1), eps=1e-07 | 75% | **1.66×** | **2.02×** | **2.01×** | 2.04× | Binary search |
| U(0.01,0.1), eps=1e-07 | 90% | **1.52×** | **1.79×** | **1.76×** | 1.83× | Binary search |
| U(0.01,0.1), eps=1e-08 | 50% | **1.82×** | **2.26×** | **2.18×** | 2.24× | Binary search |
| U(0.01,0.1), eps=1e-08 | 75% | **1.67×** | **2.02×** | **2.00×** | 2.04× | Binary search |
| U(0.01,0.1), eps=1e-08 | 90% | **1.56×** | **1.80×** | **1.76×** | 1.83× | Binary search |
| U(1e-05,0.01), eps=1e-05 | 50% | **1.27×** | **2.32×** | **2.35×** | 2.77× | Reverse engineering |
| U(1e-05,0.01), eps=1e-05 | 75% | **1.24×** | **2.04×** | **2.03×** | 2.26× | Binary search |
| U(1e-05,0.01), eps=1e-05 | 90% | **1.20×** | **1.80×** | **1.80×** | 1.94× | Binary search |

### Pooled

| p* | Linear | Binary | Rev. eng. |
|---|---:|---:|---:|
| 50% | 1.51× (median 1.52×) | 2.08× (median 2.16×) | 2.12× (median 2.14×) |
| 75% | 1.40× (median 1.46×) | 1.87× (median 1.96×) | 1.94× (median 1.97×) |
| 90% | 1.30× (median 1.35×) | 1.63× (median 1.75×) | 1.63× (median 1.72×) |

## Head-to-head, paired over all 308 operating points

| comparison | mean | median | wins |
|---|---:|---:|---:|
| binary − reverse engineering | -0.32 pp | -0.06 pp | 40% |
| binary − linear | +5.10 pp | +4.52 pp | 100% |
| reverse engineering − linear | +5.42 pp | +4.94 pp | 99% |

Mean convergence: linear **56.55%**, binary **61.65%**, reverse engineering **61.97%**, brute force 50.67%, ceiling 64.48%.

Share of the ceiling reached: linear 87.7%, binary 95.6%, reverse engineering 96.1%.

## What moved against the old sweep

| algorithm | mean | median | max gain | max loss | points gaining >1 pp |
|---|---:|---:|---:|---:|---:|
| Linear search | +0.10 pp | +0.02 pp | +10.89 | -3.61 | 10% |
| Binary search (deepest-accepted pilot) | +0.88 pp | +0.14 pp | +54.67 | -3.47 | 10% |
| Reverse engineering | +0.64 pp | -0.04 pp | +47.73 | -2.04 | 6% |

## Grid boundary diagnostics

| algorithm | winner at `m'=1` | winner at `m' = budget/N_min` | (old sweep: at grid min) |
|---|---:|---:|---:|
| Linear search | 15.6% | 0.0% | 37.5% |
| Binary search (deepest-accepted pilot) | 0.0% | 0.0% | 23.5% |
| Reverse engineering | 1.3% | 0.0% | 32.1% |

_`m'=1` is the physical minimum (one shot), so a hit there is a real optimum, not a truncated search. The upper endpoint is never selected._
