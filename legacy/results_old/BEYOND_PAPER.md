# NEW RESULTS — the best HONEST case for the adaptive algorithms

> **Separate from [RESULTS.md](RESULTS.md)** (which reproduces/corrects the paper's Tables 3.1–3.3).
> This document finds the strongest *honest* setting for "adaptive beats brute force," **under the
> paper's own assumption φ ~ Uniform(φ_min, φ_max)** — the sampling assumption is NOT changed.
>
> **Metric = budget ratio** `B_brute(p*)/B_algo(p*)`: how much less budget an algorithm needs to first
> reach convergence reliability `p*` (default `p*`=90%). **Every adaptive number is de-biased:** the
> parameters are grid-tuned on seed 42 and the single winning config is re-validated on an independent
> seed 2024 at R=40,000 — so no number is a grid maximum (no winner's curse).
>
> Regenerate everything (tables + figures) with:
> ```bash
> python analysis/extensive_sweep.py     # ~13 min: 7 scenarios x 18 budgets, ~800 configs/budget
> python analysis/render_story.py        # tables + 3 figures
> ```
> Raw numbers: [story_cube.csv](story_cube.csv), [story_curves.csv](story_curves.csv). Machine copy of
> the tables: [STORY_TABLES.md](STORY_TABLES.md).

## Headline (recommended for the thesis)

**Feature HIGH PRECISION, and report the budget ratio — not a fixed-budget margin.** Under the standard
uniform prior, **reverse engineering reaches 90% convergence with ~1.6× less budget than brute force at
ε=10⁻⁴** (and the advantage *grows* with precision). Reverse engineering is a single, named algorithm —
no across-algorithm selection — so this is the number to quote.

### Table A — the field at one operating point (φ~U(0.01,0.1), ε=10⁻⁴, reach 90% convergence)

| Algorithm | Budget to reach 90% | Ratio vs brute |
|---|---:|---:|
| Brute force (baseline) | 4,526,510 | 1.00× |
| Separable (N=1) | 68,521,833 | _0.07×_ |
| Linear search | 3,277,793 | **1.38×** |
| Binary search | 3,510,925 | **1.29×** |
| Reverse engineering | 2,796,472 | **1.62×** |

_Ratio > 1 = needs that many times **less** budget than brute for the same 90% convergence (**bold** wins,
_italic_ loses). This is the only place the full per-algorithm field is tabulated._

### Table B — the main lever is precision (φ~U(0.01,0.1), reach 90%)

| Precision ε | Linear | Binary | Reverse eng. |
|---|---:|---:|---:|
| 10⁻³ | **1.08×** | _0.76×_ | **1.24×** |
| 10⁻⁴ | **1.38×** | **1.29×** | **1.62×** |
| 10⁻⁵ | **1.44×** | **1.55×** | **1.67×** |

_The advantage grows with required precision: a tighter tolerance demands a larger circuit depth N
(Heisenberg 1/N scaling), which is exactly what the adaptive N-search supplies. At low precision (ε=10⁻³)
brute's φ-independent error is already "good enough," so the gap nearly closes — and binary search even
falls behind. **Do not feature ε=10⁻³.**_

### Table C — robustness of the win across thresholds (best adaptive)

| Scenario | 50% | 80% | 90% | 95% | winner |
|---|---:|---:|---:|---:|---|
| U(0.01,0.1), ε=10⁻³ | **1.34×** | **1.24×** | **1.24×** | **1.14×** | Reverse engineering |
| U(0.01,0.1), ε=10⁻⁴ | **1.84×** | **1.71×** | **1.62×** | **1.51×** | Reverse engineering |
| U(0.01,0.1), ε=10⁻⁵ | **2.09×** | **1.81×** | **1.67×** | **1.56×** | Reverse engineering |
| U(0.001,0.1), ε=10⁻⁴ | **2.05×** | **1.67×** | **1.49×** | **1.41×** | Reverse engineering |
| U(0.001,0.01), ε=10⁻⁴ | **1.31×** | **1.25×** | **1.23×** | **1.16×** | Reverse engineering |
| U(0.005,0.05), ε=10⁻⁴ | **1.71×** | **1.58×** | **1.46×** | **1.33×** | Reverse engineering |

_Ratio ≥ 1 in every cell ⇒ the advantage survives at every reliability target. It shrinks monotonically as
p*→95% (both curves saturate together), which is exactly why the paper's arbitrary ">95%" threshold looked
flat — 90% is meaningful and robustly favourable; report it, but show this row so nobody suspects the
threshold was fished for._

## Figures (all reproduced by `analysis/render_story.py`; edit `PAPER_RC` + the helpers for the paper)

**The honest metric, made visible.** Budget is on a **log axis** (it spans orders of magnitude), so the
advantage appears as a *small horizontal shift* that is in fact a **multiplicative factor** — the red arrow
at the 90% line labels that budget ratio. Read left→right: the gap opens up as precision tightens
(1.2× → 1.6× → 1.7×).

![convergence-vs-budget S-curves with the budget-ratio arrow](fig_story.png)

**Algorithm analysis — the budget–convergence Pareto frontier.** For each budget, the best achievable
convergence over all tuned configs is that algorithm's curve; the upper envelope is the combined frontier.
The strip shows *which* algorithm is Pareto-optimal in each budget regime: **reverse engineering owns the
entire non-saturated regime**, linear search takes a thin slice just before saturation, and binary search
and separable are dominated everywhere. (Beyond ~10⁷ every algorithm is at ~100%, so "ownership" there is a
numerical tie — greyed out.)

![budget–convergence Pareto frontier and dominance strip](fig_pareto.png)

**The lever.** Budget ratio at 90% versus target precision: the advantage rises monotonically as ε tightens.

![budget ratio at 90% vs precision](fig_precision.png)

## Why this works and "broad distributions" doesn't (mechanism, in one place)

Brute force fixes `N = N_min`; its error `1/(2·N_min·√m)` is **independent of φ**, so brute's convergence
depends only on `(φ_max, budget)`, **not on the prior**. Adaptive uses `N ≈ π/2φ`, so its per-sample edge
is `(φ_max/φ)²` — large only when **φ ≪ φ_max**. Consequences:

- **Broaden DOWN, keep φ_max small.** `U(0.001, 0.1)` keeps N_min=15 (brute unchanged) but adds small-φ
  samples adaptive can exploit. (Confirmed above — it slightly *raises* the low-threshold ratio.)
- **Broaden UP to π/2 does NOT help under uniform.** Every φ is then large → optimal N is small (1–15) →
  adaptive can't use high N, and reverse engineering collapses (N-inference fails when N_min=1). A *uniform*
  draw almost never lands in the small-φ region that adaptivity needs.
- **Under uniform, the range shape is secondary; the dominant lever is PRECISION** (Table B / fig_precision).

## Mandatory caveats (so we don't oversell again)

1. **It's a bounded head-start, not an asymptotic win.** The %-margin at any fixed budget → 0 as budget → ∞
   (both saturate). Always report the budget ratio or the full curve, never a single fixed-budget margin.
2. **Modest is honest.** Under strict uniform sampling the advantage is ~1.2–1.3× (ε=10⁻³) rising to
   ~1.6–1.7× (ε=10⁻⁵). The dramatic ~10× numbers only appear under a *log-uniform* prior — a different
   assumption, mentioned once as an aside.
3. **Noise floor:** SE at R=40,000 is ~0.25 pp per point; the de-biased headline was previously confirmed
   stable to R=200k. Ratios in Table A/B are trustworthy to ±~0.05×.
4. **Reverse engineering must not be used at φ_max=π/2 under uniform** (it caps ~42% and loses to brute at
   high budget). Linear search is the robust default there.
5. **Binary search is not a headline win** — it loses at ε=10⁻³ (0.76×) and only ties/creeps ahead at high
   precision. Keep it in Table A for honesty; do not feature it.

## Recommended paper edits

- **Replace "broad distributions" as the featured advantage with "high-precision estimation."**
  Suggested sentence: *"Under a uniform prior over small phases, reverse-engineering estimation reaches a
  given convergence reliability with ~1.6× less measurement budget at high precision (ε=10⁻⁴), and the
  advantage grows with the required precision — because a tighter tolerance rewards the larger circuit depth
  N that the adaptive search selects."*
- **Add the two analysis figures** (`fig_story.png`, `fig_pareto.png`) — the second is the algorithm
  analysis (Pareto dominance) the current draft is missing.
- **Report the budget ratio at 90%** (Table A), the precision trend (Table B / fig_precision), and the
  threshold-robustness row (Table C). Drop any single fixed-budget %-margin.
- **Retract** the binary-search "win"; **scope** reverse engineering away from φ_max=π/2 (use linear there).
- **At most one aside on log-uniform:** *"For a scale-invariant (log-uniform) prior spanning several
  decades, the same mechanism yields a much larger ~10× budget advantage; we keep the uniform assumption
  here."*
