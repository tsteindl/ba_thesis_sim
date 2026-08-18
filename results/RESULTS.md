# Results — adaptive quantum metrology under resource constraints

_All comparisons use the fixed-budget formulation: every algorithm is given the same budget `C = N·m` and we measure convergence (`|φ̂−φ| < ε`). Adaptive numbers are grid-tuned on seed 42 and re-validated on seed 2024 at high R. Tables 3.1/3.3 are recomputed here; Table 3.2 and the precision study come from the budget sweep (`analysis/extensive_sweep.py`; raw data in `story_cube.csv`, `story_curves.csv`, `error_curves.csv`)._

## Table 3.1 — % converged at fixed budget 10,000  (ε = 10⁻³, φ ~ U(0.01, 0.1))

| Algorithm | paper | this work |
|---|---:|---:|
| Separable (N=1) | 15.9% | 15.7% |
| Brute force | 56.3% | 56.2% |
| Linear search | 14.0% | 56.5% |
| Binary search | 11.5% | 63.2% |
| Reverse Engineering | 61.3% | 66.3% |
| Attainable ceiling (Eq. 3.4 at N_opt) | — | 73.5% |
| _Reverse Eng. Lite (Alg. 7)_ | _29.6% @ ~2000_ | _29.3% (m'=150, m=40, budget ~3,876)_ |

_Narrow-range linear/binary are much higher than the paper's stale 14.0% / 11.5%. RE-Lite is the one variable-budget algorithm still reported (Algorithm 7)._

### Using the parameters as stated in the text (not re-tuned)

| Algorithm | stated parameters | stated | grid-tuned | cost of not tuning |
|---|---|---:|---:|---:|
| Linear search | `m_exploration`=10, `lookback_window`=5, `safeguard`=1, `inc`=1 | 43.2% | 56.5% | -13.4 pp |
| Binary search | `m_exploration`=100, `conf`=0.95 | 65.5% | 63.2% | +2.3 pp |
| Reverse Engineering | `pilot_share`=0.02 | 64.8% | 66.3% | -1.4 pp |

_Reverse engineering's exploration size is stated as a **share of the budget** (ρ = 2%, floor 20 shots) rather than a shot count, because a shot count can only be right at one budget. Over the 546 operating points of the full sweep the stated ρ costs −0.06 pp against tuning at every budget separately, while the previously stated m′ = 200 costs −5.55 pp (worst −99.4 pp) — see `results/tex/tab_pilot_share.tex` and `results/fig_pilot_share.png`. Budget 10,000 is the regime where ρ is least favourable (the pilot floor governs below B ≈ 400·N_min), so this row is close to the worst case._

## Table 3.2 — budget to reach 90% convergence  (fixed-budget, swept to the crossing)

_The budget is the **output**: we sweep the allocated per-estimation budget B and report the smallest B at which convergence reaches 90% (log-interpolated crossing of the convergence-vs-budget curve). Ratio = brute ÷ algorithm (>1 ⇒ adaptive needs less)._

| Algorithm | eps1e-3 U(0.01,0.1) | eps1e-4 U(0.01,0.1) | eps1e-4 U(0.001,0.01) | eps1e-4 U(0.001,0.1) |
|---|---:|---:|---:|---:|
| Brute force | 45,868 | 4,504,343 | 440,367 | 4,516,936 |
| Separable (N=1) | 683,132 (×0.07) | 67,162,261 (×0.07) | — | 67,265,215 (×0.07) |
| Linear search | 43,442 (×1.06) | 3,135,633 (×1.44) | 400,006 (×1.10) | 3,479,710 (×1.30) |
| Binary search | 32,228 (×1.42) | 2,899,348 (×1.55) | 345,488 (×1.27) | 2,440,996 (×1.85) |
| Reverse Engineering | 30,010 (×1.53) | 2,570,140 (×1.75) | 299,926 (×1.47) | 2,430,064 (×1.86) |
| Attainable ceiling (Eq. 3.4 at N_opt) | 24,498 (×1.87) | 2,449,849 (×1.84) | 239,952 (×1.83) | 2,288,717 (×1.97) |

> **Why fixed-budget and not the run-until-done formulation:** the paper's variable-budget Algorithm 6 uses a single exploitation count for all φ, so it over-spends on easy (small-φ) trials — its mean budget for 90% is *higher* than brute (≈6.3M vs 4.5M at ε=10⁻⁴). Committing the whole budget to the inferred depth (fixed-budget) is what realises the advantage; that is the formulation reported here and in Tables 3.1/3.3.

## Table 3.3 — broad distributions  (ε = 10⁻³, budget 10,000, % converged)

| Algorithm | $\mathcal{U}(0.01, \pi/2)$ | $\mathcal{U}(0.01, \pi/4)$ | $\mathcal{U}(0.01, \pi/8)$ | $\mathcal{U}(0.01, \pi/16)$ |
|---|---:|---:|---:|---:|
| Brute force | 15.8% (15.9%) | 22.1% (20.4%) | 30.8% (31.3%) | 42.5% (43.3%) |
| Linear search | 18.5% (20.8%) | 23.5% (24.4%) | 32.8% (34.3%) | 42.6% (46.6%) |
| Binary search | 22.7% (16.3%) | 31.9% (21.3%) | 42.1% (30.4%) | 52.1% (39.3%) |
| Reverse Engineering | 22.5% (17.4%) | 31.7% (22.3%) | 42.0% (39.7%) | 52.1% (51.8%) |

_(this work vs paper). Paper adaptive numbers were grid maxima (winner's curse); de-biased they drop, but linear and reverse-engineering still beat brute for broad distributions._

## Beyond the paper — the advantage grows with precision, then saturates

Sweeping precision at φ ~ U(0.01, 0.1), reverse engineering's budget ratio at 90%:

| ε | 10⁻3 | 10⁻4 | 10⁻5 | 10⁻6 | 10⁻7 | 10⁻8 |
|---|---:|---:|---:|---:|---:|---:|
| **budget ratio vs brute** | 1.53× | 1.75× | 1.75× | 1.80× | 1.80× | 1.80× |

The advantage climbs from ~1.2× (ε=10⁻³) and **saturates from ε ≤ 10⁻⁶** — at ~1.76× for reverse engineering, with binary search drawing level there (~1.80× at the 90% threshold, but behind at 50%; the mean gap on the underlying curves is within one standard error, so neither leads). A tighter tolerance rewards the larger circuit depth N that the adaptive search selects, until the depth is capped by the prior support rather than by the budget.

**Robust to the prior range.** Across dynamic ranges from U(0.01,0.1) down to U(10⁻⁵,0.01), the high-precision (ε≤10⁻⁵) advantage stays in a tight band (1.54–1.76×): it is **precision-driven, not range-driven** (under the uniform prior, a wide range down does not add much small-φ mass). At low precision (ε=10⁻³) in narrow ranges the adaptive edge disappears and brute is competitive.

### Figures

![convergence-vs-budget with budget-ratio arrows]( )

![estimator error + uncertainty vs budget, ε to 10⁻⁸](fig_error.png)

![budget–convergence Pareto frontier + dominance strip](fig_pareto.png)

![budget ratio at 90% vs precision](fig_precision.png)

## Mechanism & caveats

- **Why adaptive wins:** brute fixes N = N_min, so its error 1/(2·N_min·√m) is φ-independent; adaptive infers N ≈ π/2φ, giving a per-sample edge (φ_max/φ)² — large only when φ ≪ φ_max. A tighter ε demands a larger N, which is exactly what the adaptive search supplies, so the advantage grows with precision (until it saturates at the Heisenberg limit).
- **It is a budget ratio, not an asymptotic win:** both curves reach ~100% eventually, so any single fixed-budget %-margin vanishes at high budget — report the budget ratio or the whole curve (fig_story).
- **Reverse engineering must not be used at φ_max = π/2** under the uniform prior (N-inference fails when N_min = 1 and it caps ~42%); linear search is the robust default there.
- **Binary search is not featured**: it wins only at high precision and loses at ε = 10⁻³; kept in the tables for completeness.
- **Log-uniform aside:** a scale-invariant prior spanning several decades yields a larger ~10× advantage via the same mechanism; we keep the paper's uniform assumption throughout.
## Methodology & reproducibility

- **R** = 50,000 trials per point for Tables 3.1/3.3; the budget sweep (Table 3.2 and the figures) uses R = 40,000. Tuning uses R = 2,000 (seed 42); validation uses seed 2024.
- **Sampler:** binomial (`hits ~ Binomial(m, cos²Nφ)`), identical in distribution to per-shot sampling and much faster for large m, which is what makes the ε=10⁻⁸ / ~10¹⁴-budget sweeps feasible.
- **Seeds:** SEED_TUNE=42, SEED_TEST=2024, SEED_UNBIASED=12345. Re-running is deterministic.
- **Regenerate:** `python analysis/extensive_sweep.py` (budget sweep) → `python analysis/error_curves.py` → `python analysis/render_story.py` (figures) → `python analysis/make_results.py` (this file).
## LaTeX (paste-ready)

Every thesis table is emitted as its own `.tex` file by `python analysis/make_tex.py` into `results/tex/` (and collected in `results/TEX.md`), so there is exactly one place that knows the manuscript's table style. Variants ending in `_ci.tex` carry 95% confidence intervals; see `results/UNCERTAINTY.md` for what those intervals cover.

