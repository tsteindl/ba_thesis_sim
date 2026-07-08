# Results — adaptive quantum metrology under resource constraints

_All comparisons use the fixed-budget formulation: every algorithm is given the same budget `C = N·m` and we measure convergence (`|φ̂−φ| < ε`). Adaptive numbers are grid-tuned on seed 42 and re-validated on seed 2024 at high R. Tables 3.1/3.3 are recomputed here; Table 3.2 and the precision study come from the budget sweep (`analysis/extensive_sweep.py`; raw data in `story_cube.csv`, `story_curves.csv`, `error_curves.csv`)._

## Table 3.1 — % converged at fixed budget 10,000  (ε = 10⁻³, φ ~ U(0.01, 0.1))

| Algorithm | paper | this work |
|---|---:|---:|
| Separable (N=1) | 15.9% | 15.7% |
| Brute force | 56.3% | 56.2% |
| Linear search | 14.0% | 56.6% |
| Binary search | 11.5% | 47.5% |
| Reverse Engineering | 61.3% | 60.3% |
| _Reverse Eng. Lite (Alg. 7)_ | _29.6% @ ~2000_ | _29.3% (m'=150, m=40, budget ~3,876)_ |

_Narrow-range linear/binary are much higher than the paper's stale 14.0% / 11.5%. RE-Lite is the one variable-budget algorithm still reported (Algorithm 7)._

## Table 3.2 — budget to reach 90% convergence  (fixed-budget, swept to the crossing)

_The budget is the **output**: we sweep the allocated per-estimation budget B and report the smallest B at which convergence reaches 90% (log-interpolated crossing of the convergence-vs-budget curve). Ratio = brute ÷ algorithm (>1 ⇒ adaptive needs less)._

| Algorithm | eps1e-3 U(0.01,0.1) | eps1e-4 U(0.01,0.1) | eps1e-4 U(0.001,0.01) | eps1e-4 U(0.001,0.1) |
|---|---:|---:|---:|---:|
| Brute force | 45,868 | 4,504,343 | 440,367 | 4,516,936 |
| Separable (N=1) | 683,132 (×0.07) | 67,162,261 (×0.07) | — | 67,265,215 (×0.07) |
| Linear search | 43,218 (×1.06) | 3,135,633 (×1.44) | 416,087 (×1.06) | 3,451,483 (×1.31) |
| Binary search | 58,396 (×0.79) | 3,436,485 (×1.31) | 800,957 (×0.55) | 4,941,085 (×0.91) |
| Reverse Engineering | 37,197 (×1.23) | 2,890,389 (×1.56) | 362,887 (×1.21) | 2,911,267 (×1.55) |

> **Why fixed-budget and not the run-until-done formulation:** the paper's variable-budget Algorithm 6 uses a single exploitation count for all φ, so it over-spends on easy (small-φ) trials — its mean budget for 90% is *higher* than brute (≈6.3M vs 4.5M at ε=10⁻⁴). Committing the whole budget to the inferred depth (fixed-budget) is what realises the advantage; that is the formulation reported here and in Tables 3.1/3.3.

## Table 3.3 — broad distributions  (ε = 10⁻³, budget 10,000, % converged)

| Algorithm | $\mathcal{U}(0.01, \pi/2)$ | $\mathcal{U}(0.01, \pi/4)$ | $\mathcal{U}(0.01, \pi/8)$ | $\mathcal{U}(0.01, \pi/16)$ |
|---|---:|---:|---:|---:|
| Brute force | 15.8% (15.9%) | 22.1% (20.4%) | 30.8% (31.3%) | 42.5% (43.3%) |
| Linear search | 18.7% (20.8%) | 23.8% (24.4%) | 32.1% (34.3%) | 42.7% (46.6%) |
| Binary search | 13.1% (16.3%) | 18.4% (21.3%) | 26.0% (30.4%) | 37.0% (39.3%) |
| Reverse Engineering | 16.0% (17.4%) | 25.0% (22.3%) | 38.9% (39.7%) | 50.1% (51.8%) |

_(this work vs paper). Paper adaptive numbers were grid maxima (winner's curse); de-biased they drop, but linear and reverse-engineering still beat brute for broad distributions._

## Beyond the paper — the advantage grows with precision, then saturates

Sweeping precision at φ ~ U(0.01, 0.1), reverse engineering's budget ratio at 90%:

| ε | 10⁻3 | 10⁻4 | 10⁻5 | 10⁻6 | 10⁻7 | 10⁻8 |
|---|---:|---:|---:|---:|---:|---:|
| **budget ratio vs brute** | 1.23× | 1.56× | 1.72× | 1.72× | 1.72× | 1.72× |

The advantage climbs from ~1.2× (ε=10⁻³) and **plateaus at ~1.72×** from ε=10⁻⁵ down to 10⁻⁸ (Heisenberg-limited saturation). A tighter tolerance rewards the larger circuit depth N that the adaptive search selects.

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

**Table 3.1**
```latex
\begin{tabular}{lcc}
\toprule
Algorithm & paper & this work \\
\midrule
Separable ($N=1$) & 15.9\% & 15.7\% \\
Brute force & 56.3\% & 56.2\% \\
Linear search & 14.0\% & 56.6\% \\
Binary search & 11.5\% & 47.5\% \\
Reverse engineering & 61.3\% & 60.3\% \\
\bottomrule
\end{tabular}
```

**Table 3.2**
```latex
\begin{tabular}{lrrrr}
\toprule
Algorithm & $\varepsilon{=}10^{-3}$, $\mathcal{U}(0.01,0.1)$ & $\varepsilon{=}10^{-4}$, $\mathcal{U}(0.01,0.1)$ & $\varepsilon{=}10^{-4}$, $\mathcal{U}(0.001,0.01)$ & $\varepsilon{=}10^{-4}$, $\mathcal{U}(0.001,0.1)$ \\
\midrule
Brute force & 45{,}868 & 4{,}504{,}343 & 440{,}367 & 4{,}516{,}936 \\
Linear search & 43{,}218\,($\times$1.06) & 3{,}135{,}633\,($\times$1.44) & 416{,}087\,($\times$1.06) & 3{,}451{,}483\,($\times$1.31) \\
Binary search & 58{,}396\,($\times$0.79) & 3{,}436{,}485\,($\times$1.31) & 800{,}957\,($\times$0.55) & 4{,}941{,}085\,($\times$0.91) \\
Reverse engineering & 37{,}197\,($\times$1.23) & 2{,}890{,}389\,($\times$1.56) & 362{,}887\,($\times$1.21) & 2{,}911{,}267\,($\times$1.55) \\
\bottomrule
\end{tabular}
```

**Table 3.3**
```latex
\begin{tabular}{lrrrr}
\toprule
Algorithm & $\mathcal{U}(0.01, \pi/2)$ & $\mathcal{U}(0.01, \pi/4)$ & $\mathcal{U}(0.01, \pi/8)$ & $\mathcal{U}(0.01, \pi/16)$ \\
\midrule
Brute force & 15.8\% & 22.1\% & 30.8\% & 42.5\% \\
Linear search & 18.7\% & 23.8\% & 32.1\% & 42.7\% \\
Binary search & 13.1\% & 18.4\% & 26.0\% & 37.0\% \\
Reverse engineering & 16.0\% & 25.0\% & 38.9\% & 50.1\% \\
\bottomrule
\end{tabular}
```
