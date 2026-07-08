# Table 3.2 - framing comparison (budget to reach **90%** convergence)

_Lower budget = better. Ratio = brute / algorithm (>1 means adaptive needs LESS budget). Fixed-budget numbers from `story_cube.csv` (de-biased); variable-budget numbers from direct searches._

## A) Original framing - variable-budget Algorithm 6 (scalar exploitation)

The paper's Table 3.2 approach. Best honest configs (direct search): **adaptivity ties or loses.**

| Algorithm | budget @90%, eps=1e-4, U(0.01,0.1) |
|---|---:|
| Brute force | 4,516,530 |
| Linear search | ~4,510,000  (~ties) |
| Binary search | > 6,000,000  (loses) |
| Reverse engineering | 6,310,396  (x0.72, **loses**) |

> The conv95 grid protocol mis-selects RE here (it reports 59.8M / 100% - a coarse-grid overshoot). The true best variable-Alg.6 config is 6.31M, which still loses to brute. A single exploitation count cannot give the hard (large-phi) trials more shots without over-spending on the easy ones.

## B) Variable-budget, per-trial precision-targeted RE (improved; budget still the OUTPUT)

Same reverse-engineering idea, but the exploitation shots adapt to the inferred depth: spend `m ~ 1/(N*eps)^2` shots (enough to hit precision eps). Budget is emergent, not prescribed.

| Algorithm | budget @90%, eps=1e-4, U(0.01,0.1) |
|---|---:|
| Brute force | ~4,520,000 |
| Reverse engineering (improved) | **3,307,012  (x1.37, wins)** |

> Quick proof-of-concept (seed 2024, R=15k, coarse grid over exploration shots + a confidence factor). It shows the win is achievable in a genuinely variable-budget algorithm; a proper de-biased implementation would replace Algorithm 6's fixed exploitation count with this rule.

## C) Fixed-budget - budget swept to the 90% crossing (the BEYOND table)

Give each algorithm a per-estimation budget B, sweep B, report where convergence crosses 90%.

| Algorithm | eps1e-3 U(0.01,0.1) | eps1e-4 U(0.01,0.1) | eps1e-4 U(0.001,0.01) | eps1e-4 U(0.001,0.1) |
|---|---:|---:|---:|---:|
| Brute force | 45,975 | 4,516,540 | 441,393 | 4,520,682 |
| Linear search | 44,458 (x1.03) | 3,049,824 (x1.48) | 402,795 (x1.10) | 3,448,509 (x1.31) |
| Binary search | 59,717 (x0.77) | 3,470,419 (x1.30) | 823,323 (x0.54) | 4,767,751 (x0.95) |
| Reverse engineering | 37,705 (x1.22) | 2,864,239 (x1.58) | 359,610 (x1.23) | 2,912,396 (x1.55) |

_N_min=157 for U(0.001,0.01); N_min=15 elsewhere. Reverse engineering wins every column._

## Takeaway

- The advantage is **real** and is NOT an artifact of fixing the budget.

- It is forfeited only by Algorithm 6's **scalar** exploitation count (framing A).

- Both a per-trial precision-targeted **variable** algorithm (B) and the fixed-budget sweep (C) recover it (~1.4-1.6x). Framing B keeps budget as the output, which matches the Table 3.2 intent.

