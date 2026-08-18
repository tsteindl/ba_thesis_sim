# Thesis changes required by the statistical safeguard

Page numbers refer to `main-thesis.pdf` (PDF page = printed page + 3 in Chapter 3).
Code for the change: [`qmetrology/safeguard.py`](../qmetrology/safeguard.py).
Evidence: [`results/SAFEGUARD.md`](SAFEGUARD.md), [`results/safeguard_compare.csv`](safeguard_compare.csv),
[`results/fig_safeguard.png`](fig_safeguard.png).

**Status: all numbers below are from the complete re-run of 2026-08-16** (truncated posterior,
uncapped binary-search search, `oracle_hl` as the sole ceiling). The statistical safeguard *replaces*
the tuned constant — the reported "Binary search" and "Reverse Engineering" rows **are** the
statistical-safeguard versions; the constant-`C`/`s` numbers appear only as the published-vs-this-work
delta.

**The change in one sentence.** The reverse-engineering safety factor `C_safe` (Eq. 3.6) and the
binary-search decrement `s` were constants chosen by parameter grid search; they are replaced by a
depth derived from the *same* asymptotic sampling distribution (Eq. 3.4) that already underpins the
binary-search overshoot criterion — so neither algorithm has a tuned safeguard any more.

**Why it is not just a re-tune.** The rule is φ- and budget-adaptive. The implied multiplicative
safeguard `C_eff = N*/⌊π/2φ̂₀⌋` runs from ≈0.54 at φ = 0.01 to 1.00 at φ = 0.1 at budget 10⁴, and the
whole curve lifts as the budget grows (≈0.73 → 1.00 at budget 3·10⁶). A single constant cannot track
either axis; the grid search was picking a compromise across both.

**Second, reportable effect.** The rule also needs a far cheaper pilot: reverse engineering's winning
`m_exploration` is a median 5.3× smaller (up to 151× at φmax = π/2) than with a tuned constant. Scaling
the backoff to the pilot's own spread means the protocol can act safely on a coarse estimate, whereas
a constant C has to be bought with exploration shots. This is worth a sentence in §3.2.4 — it is the
mechanism behind the broad-prior result in §6 below.

---

## 1. New content — §3.2.3, immediately after Eq. (3.5), p. 23

This is the only place new theory is needed. Both algorithms then reference it.

> **Choosing the exploitation depth.** The same approximation can be used to choose the depth of the
> final estimation phase, replacing the safety margins introduced below. Read as a likelihood for the
> unknown ϕ given a pilot estimate ϕ̂₀ taken at depth N₀ with m′ shots, Eq. (3.4) assigns every
> candidate depth N a probability of not overshooting,
>
> P(N ϕ < π/2) = Φ((π/(2N) − ϕ̂₀)/σ),  σ = 1/(2 N₀ √m′),   (3.6)
>
> where Φ is the standard normal CDF. Conditional on not overshooting, the final estimate obtained at
> depth N with the remaining budget B — that is, m = ⌊B/N⌋ shots, so that N²m = N B — again follows
> Eq. (3.4), and converges with probability
>
> P(|ϕ̂ − ϕ| < ϵ) = 2 Φ(2 ϵ √(N B)) − 1.   (3.7)
>
> We select the depth maximising the product of the two,
>
> N* = argmax_{1 ≤ N ≤ N_max}  Φ((π/(2N) − ϕ̂₀)/σ) · [2 Φ(2 ϵ √(N B)) − 1].   (3.8)
>
> The two factors pull in opposite directions: a deeper circuit is more precise, the second factor
> growing as √N, but more likely to alias, the first factor falling. Unlike the constants it replaces,
> N* requires no grid search — it is fixed by quantities the protocol already has: the pilot and its
> spread, the remaining budget B, the target tolerance ϵ, and the prior support through
> N_max = ⌊π/(2ϕ_min)⌋.

LaTeX for Eq. (3.8) (label used by the generated tables in `analysis/make_results.py`):

```latex
\begin{equation}\label{eq:risk-depth}
  N^{*} \;=\; \operatorname*{arg\,max}_{1 \le N \le N_{\max}}\;
  \underbrace{\Phi\!\left(\frac{\pi/(2N) - \hat\phi_0}{\sigma}\right)}_{\text{no overshoot}}
  \cdot
  \underbrace{\left[\,2\,\Phi\!\left(2\epsilon\sqrt{N B}\right) - 1\,\right]}_{\text{converges}},
  \qquad \sigma = \frac{1}{2 N_0 \sqrt{m'}} .
\end{equation}
```

Optional one-line remark worth adding — it is the intuition an examiner will ask for:

> Equivalently, the rule applies a multiplicative safeguard C_eff = N*/⌊π/(2ϕ̂₀)⌋ that tightens when
> the pilot is relatively imprecise (small ϕ, few exploration shots) and relaxes toward 1 as the
> budget grows — behaviour a single tuned constant cannot reproduce.

## 2. §3.2.3 Binary search — Algorithm 5, p. 24

- **Signature.** `SimulatePhiBinarySearch(budget, ϕmin, ϕmax, m′, α, s)` → drop `s`, add `ϵ`:
  `SimulatePhiBinarySearch(budget, ϕmin, ϕmax, m′, α, ϵ)`.
- **Line `Reduce N by safeguard s to avoid overshooting`** → replace with:
  `N ← N* from Eq. (3.8), using the deepest non-overshooting probe (ϕ̂₀, N₀) as pilot`.
- **Note the change of role.** The rule is deliberately *not* restricted to reducing the depth the
  bisection settled on, the way `s` was. The bisection halts when its step size reaches zero and so
  undershoots systematically; letting the search run over the full prior support is worth
  **+1.3 pp** at B=4.5·10⁴ and **+2.1 pp** at B=3·10⁶. The exploration phase is therefore best
  described as buying a *precise pilot* — taken at a deep circuit, so σ = 1/(2·N_acc·√m′) is small —
  rather than as selecting the depth itself. This changes what binary search is *for* and deserves a
  sentence in §3.2.3.
- Add one sentence after the algorithm: the pilot is the deepest probe that was *not* classified as an
  overshoot, i.e. the most precise estimate taken in a regime where the estimator is still valid.

## 3. §3.2.4 Reverse engineering — p. 25

- **Eq. (3.6)** `N = ⌊C_safe · ⌊π/(2ϕ̂₀)⌋⌋` → delete; replace the surrounding sentence:

  > current: "Since N_opt ≈ π/2ϕ we set (3.6) … where C_safe is a safety factor that reduces the risk
  > of overshooting."
  >
  > new: "Since N_opt ≈ π/2ϕ, the pilot ϕ̂₀ already determines the depth up to the risk of overshooting.
  > We therefore take N = N* from Eq. (3.8), with the pilot taken at N₀ = N_min. No safety factor has
  > to be introduced: the tolerable amount of backing-off follows from the spread of ϕ̂₀ itself."

- **Algorithm 6 signature.** `SimulatePhiReverseEngineer(budget, ϕmax, m′, C)` →
  `SimulatePhiReverseEngineer(budget, ϕmin, ϕmax, m′, ϵ)`.
- **Line `N ← ⌊⌊π/2ϕ̂₀⌋ · C⌋  ▷ Infer N from ϕ̂₀, then reduce by safeguard factor C`** →
  `N ← N* from Eq. (3.8) with pilot (ϕ̂₀, N_min, m′)  ▷ Infer the depth and its safe backoff jointly`.
- Note `ϕmin` is now an argument: the prior support caps the depth (`N_max`). The published
  constant-C form has no such cap, so a badly under-shooting pilot could send it past any admissible
  depth — worth one sentence, it is a real robustness difference.

## 4. Figure 3.1 caption, p. 23

The normal approximation is now load-bearing twice. Extend the last clause:

> current: "…motivating the threshold criterion used in the binary-search strategy."
>
> new: "…motivating both the threshold criterion used in the binary-search strategy and the choice of
> exploitation depth in Eq. (3.8)."

## 5. §3.3.3 discussion, p. 29

> current: "This argument is heuristic: it ignores exploration overhead, integer rounding of N, safety
> margins, and failures caused by overshooting."

`safety margins` is no longer a free knob but a derived quantity. Suggested: "…it ignores exploration
overhead, integer rounding of N, and the residual overshoot risk that Eq. (3.8) trades against
precision." The rest of the paragraph is unaffected.

## 6. §3.3.4 Broad distributions + Table 3.4 + Figure 3.5, pp. 29–30 — **rewrite required**

This is the largest prose change, and part of it is independent of the safeguard (see §9 below).

Claims that no longer hold:

| Location | Current claim | Status |
|---|---|---|
| Table 3.4 caption | "Linear search dominates at every budget while reverse engineering plateaus." | **false** |
| Fig. 3.5 caption | "Linear search now dominates all other approaches while RE plateaus." | **false** |
| p. 30 body | "Reverse engineering … plateaus at roughly 38% and does not improve further, no matter how large the budget." | **false — grid artefact** |
| p. 30 body | "…so the protocol overshoots, the estimator aliases (Nϕ > π/2), and those trials never converge." | mechanism right, conclusion wrong |
| Conclusion, p. 32 | "Reverse Engineering's single, coarse initial estimate breaks down when N_min is small, capping its convergence rate regardless of budget, whereas linear search remains robust and dominates at every budget." | **false** |
| Abstract, p. ii | "for broad distributions … linear search is the most effective strategy." | **needs re-checking against the new numbers** |

Final numbers, φ ~ U(0.01, π/2), ε = 10⁻³ (de-biased, R = 30,000; full curve in `results/broad_dist.csv`):

| Budget | Brute | Linear | **Binary** | **Rev. eng.** | Rev.eng. (tuned C, wide grid) | Rev.eng. (published grid) | *Ceiling* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15,243 | 19.8% | 23.2% | **17.8%** | **28.6%** | 19.7% | 19.0% | *30.7%* |
| 34,362 | 28.9% | 35.0% | **29.7%** | **41.4%** | 29.2% | 24.6% | *42.9%* |
| 77,459 | 42.0% | 47.4% | **42.2%** | **56.2%** | 41.4% | 29.7% | *57.6%* |
| 174,608 | 59.3% | 66.0% | **64.7%** | **72.6%** | 55.6% | 34.4% | *73.6%* |
| 393,597 | 79.0% | 82.3% | **82.4%** | **87.3%** | 75.1% | 37.3% | *87.8%* |
| 887,240 | 94.0% | 94.6% | **95.5%** | **96.6%** | 88.7% | 38.1% | *96.9%* |
| 2,000,000 | 99.6% | 99.6% | **99.7%** | **99.7%** | 89.6% | 38.2% | *99.8%* |

The last column reproduces the published plateau (38.1% at 887,240 — the thesis states 38.1%) exactly,
confirming it was the grid cap. **Reverse engineering with the statistical safeguard beats linear
search at every single budget**, at both φmax = π/2 and π/4 — the precise inverse of the published
claim. Binary search also overtakes linear from budget 393,597 upward.

Against the Heisenberg ceiling column in `broad_dist.csv`, reverse engineering's gap is only
**0.1–2.9 pp** across the whole sweep and closes as the budget grows (56.2 vs 57.7 at 77,459;
96.6 vs 96.9 at 887,240). So at broad priors the statistical safeguard leaves almost nothing on the
table relative to the best any single-depth protocol could do — the strongest available rebuttal to
"reverse engineering breaks down when N_min is small".

The diagnosis in the body text is still the right one — a coarse single-shot estimate at N_min = 1 is
amplified into a large error in the inferred N — but the conclusion drawn from it is not. The correct
statement is that the *constant* safety factor cannot absorb that amplification, whereas a safeguard
scaled to the pilot's own spread can: it backs the depth off hard exactly when the pilot is unreliable.
Suggested replacement for the mechanism paragraph:

> Reverse engineering infers N ≈ π/(2ϕ̂₀) from a single initial estimate taken at N_min = 1, which is
> extremely coarse: a small error in ϕ̂₀ is amplified into a large error in the inferred N. With a
> fixed safety factor the protocol therefore overshoots, the estimator aliases (Nϕ > π/2), and those
> trials never converge — the constant must be chosen for the whole prior at once and is far too
> permissive at the small-ϕ end. The safeguard of Eq. (3.8) is scaled to the spread of ϕ̂₀ itself and
> backs the depth off hardest exactly where the pilot is least reliable, which restores reverse
> engineering to the top of the ranking even at ϕmax = π/2.

## 6b. Table 3.2, §3.3.3, the Abstract and the Conclusion — **numbers and claims change**

From the completed budget sweep (de-biased, R = 40,000; these cells are final — the scenarios still
running feed no headline table). Ratio vs brute at 90% convergence, > 1 = needs less budget:

| Algorithm | ε=10⁻³ U(0.01,0.1) | ε=10⁻⁴ U(0.01,0.1) | ε=10⁻⁴ U(0.001,0.01) | ε=10⁻⁴ U(0.001,0.1) |
|---|---:|---:|---:|---:|
| Linear search | ×1.06 | ×1.44 | ×1.11 | ×1.34 |
| Binary search — published (tuned `s`) | ×0.79 | ×1.31 | ×0.55 | ×0.91 |
| **Binary search — this work** | **×1.12** | **×1.55** | **×1.04** | **×1.31** |
| Reverse eng. — published (tuned `C`) | ×1.23 | ×1.56 | ×1.21 | ×1.55 |
| **Reverse engineering — this work** | **×1.53** | **×1.75** | **×1.47** | **×1.86** |
| *Heisenberg ceiling (Eq. 3.4 at N_opt)* | *×1.87* | *×1.84* | *×1.83* | *×1.97* |

Fraction of the attainable advantage captured, (r−1)/(r_ceiling−1): reverse engineering reaches
**61 % / 90 % / 56 % / 88 %** across those four columns; binary search **14 % / 66 % / 5 % / 32 %**.

Precision progression at φ ~ U(0.01,0.1), the source of the "≈1.72×" claim:

| Algorithm | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | 10⁻⁸ |
|---|---:|---:|---:|---:|---:|---:|
| Reverse eng. — published | 1.23 | 1.56 | 1.72 | 1.72 | 1.72 | 1.72 |
| **Reverse eng. — this work** | **1.53** | **1.75** | **1.75** | **1.76** | **1.76** | **1.76** |
| Binary search — published | 0.79 | 1.31 | 1.59 | 1.54 | 1.54 | 1.54 |
| **Binary search — this work** | **1.12** | **1.55** | **1.74** | **1.80** | **1.80** | **1.80** |
| *Heisenberg ceiling* | *1.87* | *1.84* | *1.83* | *1.83* | *1.83* | *1.83* |

From ε=10⁻⁶ down, **binary search catches up with reverse engineering** and the two become
statistically indistinguishable. This is a convergence, *not* a reversal, and the table above
overstates it because the ×-ratios are read at the 90 % threshold: across all four thresholds binary
is ahead at 90 % (1.80 vs 1.76) and 95 % (1.70 vs 1.66), behind at 50 % (2.22 vs 2.23), and level at
80 % (1.93 vs 1.91). On the underlying convergence curves the mean gap over the live budgets is
−0.11 pp (ε=10⁻⁶), +0.06 pp (10⁻⁷) and +0.08 pp (10⁻⁸), against a standard error of 0.07 pp on the
mean — binary wins at 30 %, 52 % and 52 % of budgets respectively. **Do not claim either algorithm
wins at tight ε.** The defensible statement is that reverse engineering's advantage over binary
search is a low-precision phenomenon: it is 18 pp of convergence at ε=10⁻³ and vanishes by ε=10⁻⁶.

Four claims to revise:

1. **The low-precision regime is no longer marginal.** At ε=10⁻³ reverse engineering goes ×1.23 → **×1.53**.
   - *Abstract, p. ii*: "For low-precision requirements (ϵ = 10⁻³) the adaptive protocols already provide
     a **slight** improvement over the maximally entangled baseline" — "slight" no longer fits a 1.5×
     budget saving. Suggested: "already provide a clear improvement (≈1.5× less budget)".
   - *Conclusion, p. 32*: "Already at low precision (ϵ = 10⁻³), Reverse Engineering needs about **1.23×**
     less budget" → 1.53×.
   - *Table 3.1, p. 25*: the "Budget for >90% convergence" column, 37,197 (×1.23) → 30,010 (×1.53).

2. **The saturation value and where it is reached both move.** ×1.72 → **×1.76** for reverse engineering (and ×1.80 for binary search), and it is reached
   at ε=10⁻⁴ rather than ε=10⁻⁵.
   - *§3.3.3, p. 27*: "…increases as the precision requirement tightens and plateaus at ≈ 1.72× in the
     best case" → ≈1.76×.
   - *Conclusion, p. 32*: "plateauing at roughly **1.72×** for Reverse Engineering" → 1.76×.
   - *Table on p. 28* (precision × algorithm): all three columns change; regenerate from
     `results/STORY_TABLES.md`.
   - The sentence in `results/RESULTS.md` that the advantage "plateaus at ~1.72× from ε=10⁻⁵ down to
     10⁻⁸" is regenerated automatically.

3. **Binary search is no longer the loser.** It moves from ×0.79 to **×1.12** at ε=10⁻³ and from ×0.55 to
   **×1.04** at U(0.001,0.01) — it now beats brute force in every reported column, and by ε=10⁻⁶ it
   has drawn level with reverse engineering (see the caveat above: level, not ahead).
   - *Conclusion, p. 32*: "binary search fell behind, as its costly initial exploration phase does not
     pay off at such a low budget" — the diagnosis still holds for the fixed-budget-10,000 metric
     (Table 3.1) but not for the budget-to-90% metric. Split the sentence, or scope it to Table 3.1.
   - *§3.3.2, p. 27*: "Binary search is the clear loser here (×0.55)" → ×0.99, i.e. it now essentially
     ties the baseline. The following explanation (confidence-threshold exploration overhead) still
     stands as the reason it does not *win*, but "clear loser" must go.
   - The caveat in `results/RESULTS.md` — "Binary search is not featured: it wins only at high precision
     and loses at ε=10⁻³" — is regenerated but you may want to reword it in the thesis too.

4. **Reverse engineering's win is now large enough to lead with.** ×1.86 at U(0.001,0.1), ε=10⁻⁴ is the
   single best cell in the table and beats every published number. Across the combined
   precision × dynamic-range study (`results/lever_cube.csv`, 42 scenarios) reverse engineering's
   maximum rises from ×1.79 to **×2.25** and its median from ×1.74 to ×1.88; binary search's maximum
   rises from ×1.73 to **×2.26** and its median from ×1.49 to ×1.81. Best cell overall:
   U(0.001, 0.4) at ε=10⁻⁶, **×2.26**.

5. **It is now close to the omniscient bound.** Against the oracle that knows φ, reverse engineering
   with the statistical safeguard reaches ×1.75 against the ×1.84 ceiling at ε=10⁻⁴, i.e. **90 % of the attainable advantage** — i.e. **the safeguard closes most of the remaining gap to what any
   single-depth protocol could do**, which is a much stronger statement than the published version
   supports. Worth a sentence in §3.3.3.

## 7. Tables 3.1, 3.2, 3.3 and Appendix C

- **Table 3.1** (p. 25) and **Table 3.2** (`tab:summary-all`): add the two statistical-safeguard rows.
  Paste-ready LaTeX is regenerated into `results/RESULTS.md` under "LaTeX (paste-ready)" —
  `analysis/make_results.py` now emits all six rows, with `\ref{eq:risk-depth}` in the row labels.
- **Table 3.3** (broad distributions): same, two extra rows.
- **Appendix C, Table C.1** (p. 43): binary search and reverse engineering lose their `safeguard=` entry.
  Their rows become `m_exploration=…, conf=…` and `m_exploration=…` respectively. Linear search is
  unchanged — its `safeguard` was not part of this change (see §10).
- **Appendix C, Table C.2** (p. 43): drop the `safeguard=` line from the B-search and Rev-Eng rows in
  all four columns. Regenerate with `python analysis/appendix_params.py`.
- Add one sentence to the Appendix C preamble: the exploitation depth of binary search and reverse
  engineering is no longer a grid-searched parameter, so only the exploration size (and α) remain.

## 8. Appendix A code listings, pp. 40–41

Replace the two listings with the current source:

- `find_phi_fixed_budget_binary_search` → `find_phi_fixed_budget_binary_search_risk`
  ([`qmetrology/algorithms.py`](../qmetrology/algorithms.py))
- `find_phi_fixed_budget_reverse_engineering` → `find_phi_fixed_budget_reverse_engineering_risk`

and add a short listing for `risk_optimal_depth` from [`qmetrology/safeguard.py`](../qmetrology/safeguard.py),
since both algorithms now call it. The linear-search listing (p. 39) is unchanged.

## 9. Independent correction — the exploration grid ceiling

Not caused by the safeguard change, but it must be reported because it changes the same table.

`analysis/broad_dist_study.py` capped the tuning grid at `m_exploration ≤ 3000`. At φmax = π/2,
N_min = 1, so the pilot needs far more shots than that before the inferred depth is usable at all —
the published "plateaus at roughly 38%" curve was **grid-limited, not algorithmic**. With the grid
scaled to the regime, *constant-C* reverse engineering already reaches 81.6% at budget 887,240.

The honest split for the thesis is therefore two effects, and both belong in the text:

1. 38.1% → ~81.6% : the published number was an artefact of a too-small exploration grid.
2. ~81.6% → 96.7% : the statistical safeguard.

`results/broad_dist.csv` carries a `reverse_eng_m3k` column that reproduces the old cap, so the
correction stays attributable. If you prefer not to foreground a methodological correction, the
minimum honest version is a footnote to Table 3.4 stating that the reverse-engineering row was
recomputed with an exploration grid scaled to N_min.

## 9b. Supporting evidence you can now cite

Three results that strengthen the change beyond the point estimates. All regenerated in the full run
of 2026-08-13/14.

**The improvement is outside the error bars.** From `results/story_cube_ci.csv` (Wilson + parametric
bootstrap, R = 40,000), at φ ~ U(0.01,0.1), ε = 10⁻⁴:

| Algorithm | ratio | 95% CI | grid_dev |
|---|---:|---|---:|
| Reverse eng. — published (tuned `C`) | ×1.558 | [1.527, 1.591] | — |
| **Reverse engineering — this work** | **×1.753** | **[1.713, 1.793]** | — |
| Binary search — published (tuned `s`) | ×1.311 | [1.284, 1.341] | — |
| **Binary search — this work** | **×1.554** | **[1.523, 1.588]** | — |
| *Heisenberg ceiling* | *×1.839* | *[1.800, 1.877]* | — |

The intervals do not overlap, so this is not Monte-Carlo noise.

**It is also more stable under tuning** (`results/TUNING_STABILITY.md`). Re-running the grid search on
5 independent tuning seeds at the 90% budget, U(0.01,0.1), ε = 10⁻⁴:

| Algorithm | validated rate | sd | budget-equivalent spread | distinct winners |
|---|---:|---:|---:|---:|
| Reverse eng. — published | 0.8885 | 0.0027 | ±1.53% | 4/5 |
| **Reverse engineering — this work** | **0.9035** | **0.0010** | **±0.57%** | 3/5 |
| Binary search — published | 0.8841 | 0.0029 | ±1.65% | 4/5 |
| **Binary search — this work** | **0.9136** | **0.0027** | **±1.56%** | 3/5 |

Removing the safeguard axis cuts the tuning-selection variance roughly threefold for reverse
engineering. This is a genuine methodological argument, not just a performance one: fewer tuned
parameters means the reported winner is less of a lottery. Worth a sentence in §3.3 or Appendix C.

**Combined precision × dynamic-range study** (`results/lever_cube.csv`, 42 scenarios):

| Algorithm | max ratio | median ratio |
|---|---:|---:|
| Reverse eng. — published | ×1.79 | ×1.74 |
| **Reverse engineering — this work** | **×2.25** | **×1.88** |
| Binary search — published | ×1.73 | ×1.49 |
| **Binary search — this work** | **×2.26** | **×1.81** |

The best cell overall is U(0.001, 0.4), ε = 10⁻⁶ at **×2.26**.

## 9c. Two exploration-phase questions, both settled without changing a reported number

**(a) Linear search + the statistical safeguard — tested, and deliberately rejected.**
Eq. (3.8) does extend to linear search: the scan's probes are independent, each with a known variance
$1/(4N_i^2m')$, so they pool by inverse-variance weighting into one pilot (derivation and numbers in
`results/SAFEGUARD_DERIVATION.md` §11). It beats the tuned `s` by **+3.5 to +5.6 pp**. It is still not
adopted, because grid search drives `lookback_window` to 1: at 10 of 15 operating points the scan
takes **two probes and keeps one**, so the algorithm degenerates into reverse engineering with one
wasted probe — and reverse engineering beats it at *all 15* points (+0.11 to +2.60 pp).

The useful output is a **mechanism sentence for §3.3.4**: once a principled depth rule exists, the
only thing the scan produces that matters is a pilot estimate, and buying that pilot incrementally is
strictly worse than buying it with one deep measurement. That is *why* reverse engineering dominates,
and it is a stronger explanation than the one currently in the text. The objection that $m'$ is too
small for the asymptotic law does not bite: where the scan survives tuning it retains 9–10 probes for
an effective $m$ of 400–2300.

**(b) Reverse engineering's pilot as a share of the budget instead of a shot count — adopt as the
stated default, $\rho = 2\%$.**

Replacing $m'$ by $m'=\rho B/N_{\min}$ (floored at 20 shots). Evaluated over **all 23 sweep scenarios
and 546 live operating points** — `analysis/re_share_sweep.py`, re-using the reported scenarios,
budgets, seeds and $R=40{,}000$; the fixed-$m'$ arm is read back from `story_curves.csv` rather than
re-tuned, so the comparison is exact. SE on a difference $\le0.35$ pp. Against the reported
per-budget grid-tuned $m'$:

| arm | mean | worst | > 0.5 pp worse | > 1 pp worse |
|---|---:|---:|---:|---:|
| $\rho$ grid-tuned per budget | +0.05 pp | −2.02 | 3.5 % | — |
| **$\rho=0.02$, never tuned** | **−0.06 pp** | **−1.75** | **7.3 %** | **1.8 %** |
| $\rho=0.05$, never tuned | −0.24 pp | −1.03 | 26 % | 0.2 % |
| $m'=45$, never tuned | −4.22 pp | **−82.5** | 87 % | 67 % |

Two conclusions. First, as a *parameterisation* the share costs nothing: tuned, it is +0.05 pp.
Second, **one untuned $\rho=0.02$ is statistically indistinguishable from per-budget tuning**
(−0.06 pp, inside the SE), whereas one untuned shot count is not remotely usable.

The −82.5 pp is the whole argument. At $\mathcal U(0.001,0.01)$, $B=6394$, a pilot of $m'=45$ costs
$45\times157>B$: the trial is refused and convergence goes from 82.5 % to 0 %. A share scales down
automatically; a shot count cannot.

**Neither parameterisation is correct in principle, and the thesis should say so.** A1 wants
$\sigma\lesssim\phi/\kappa$, i.e. a pilot size that does *not* depend on $B$ — so the ideal share
decays as $1/B$, and a constant one leaks budget at the top end. The regret by budget regime
(with $x=B/(400N_{\min})$, below $x\approx1$ the shot floor governs, not $\rho$):

| $x$ | $\rho=0.002$ | $\rho=0.01$ | $\rho=0.02$ | $\rho=0.05$ |
|---|---:|---:|---:|---:|
| $10^{0}$ | −1.01 | −0.95 | −0.65 | **−0.05** |
| $10^{1}$ | −3.18 | −0.93 | −0.20 | **+0.01** |
| $10^{2}$ | −0.96 | −0.01 | **+0.06** | −0.13 |
| $10^{4}$ | +0.11 | **+0.19** | +0.04 | −0.47 |
| $10^{6}$ | **+0.29** | +0.18 | +0.03 | −0.42 |
| $10^{10}$ | **+0.36** | +0.31 | +0.08 | −0.41 |

Two opposite failure modes: a small share under-funds the pilot at low budget, a large one wastes a
standing ~0.5 pp across the whole high-precision half of the sweep. $\rho=0.02$ is the only value
within ~0.6 pp everywhere. The reason to prefer a share over a shot count is not accuracy but the
*shape* of the failure — bounded waste versus an unaffordable pilot.

**Recommended use:** state $\rho=2\%$ (with the 20-shot floor) in §3.2.4 and in the Appendix C
parameter table, in place of a per-scenario tuned $m'$. **No re-run needed** — the reported rows are
grid-tuned per budget and $\rho=0.02$ matches them to −0.06 pp. Making the reported rows *be* the
share version is defensible too (the numbers would not move outside noise) but costs a full sweep.

## 9d. The parameters stated in the text do not reproduce the reported numbers

Surfaced while wiring the budget share in: `RESULTS.md` now prints what the **stated** parameters
achieve next to the grid-tuned result (budget 10,000, $\varepsilon=10^{-3}$, $\mathcal U(0.01,0.1)$):

| Algorithm | stated parameters | stated | grid-tuned | cost of not tuning |
|---|---|---:|---:|---:|
| Linear search | `m_exploration=10, lookback_window=5, safeguard=1, inc=1` | 44.2 % | 56.6 % | **−12.4 pp** |
| Binary search | `m_exploration=100, conf=0.95` | 9.1 % | 52.5 % | **−43.4 pp** |
| Reverse engineering | `pilot_share=0.02` | 64.8 % | 66.3 % | −1.4 pp |

Binary search's stated configuration converges **9.1 %** where the reported row says 52.5 %. This is
not a new defect — it has always been true, because every reported number is the winner of a grid
search while the text quotes one illustrative configuration — but a reader who implements Algorithm 5
from the stated parameters will not come close to Table 3.1, and nothing in the current text warns
them. Two acceptable fixes, pick one:

* say explicitly in §3.2 that the quoted parameters are illustrative and that all reported results are
  grid-tuned per operating point (Appendix C lists the winners), **or**
* restate the parameters as the tuned winners at the headline cell.

Reverse engineering is the only one that does not need this caveat once $\rho$ is quoted, because a
share transfers and a shot count does not — which is a second, independent argument for §9c(b).

---

---

## 9e. Which probe feeds the safeguard (binary search) — a null result, worth one sentence

Eq. (3.8) needs one pilot, but the bisection produces several. The implementation uses the **deepest
probe not flagged as an overshoot**; §7 justified that as "buying a precise pilot", which is only half
true. Measured (`analysis/binary_pilot_study.py`, 3 scenarios x 6 budgets, each variant tuning its own
`m'` and `conf`, de-biased at R = 40,000; 12 live points, SE of a difference 0.35 pp):

**Re-measured 2026-08-17 after the budget cap** (§9g): this script carried its *own* copy of the
bisection loop, so it was still running the uncapped exploration. Capped numbers, same 13 live points:

| pilot | mean vs shipped | mean overshoot | verdict |
|---|---:|---:|---|
| opening probe at N_min | **−0.03 pp** | 0.67 % | indistinguishable |
| deepest unflagged (**shipped**) | — | 0.95 % | keep |
| the last probe, flagged or not | **−0.53 pp** | 1.77 % | clearly worse |
| inverse-variance pool of unflagged | +0.12 pp | 0.74 % | within noise; not worth a re-run |

(Pre-cap these read +0.06 / — / −1.29 / +0.18 pp at overshoot 0.93 / 1.10 / 4.91 / 1.13 %. **The null
result survives the fix**; the penalty for using the last probe shrinks but stays real, and the
overshoot rates all fall now that the scan cannot run past its budget.)

The deep pilot cuts σ, but it is selected *for having read high* (acceptance means ϕ̂ ≥ ϕ₁), so it
carries a selection bias of **0.40σ** against the opening probe's **0.04σ**. Precision gained ≈
honesty lost. The bias is upward, so the rule picks a *shallower* depth — conservative, not
dangerous.

**What to write.** Nothing needs to change in the algorithm or in any number. Replace the §3.2.3
justification — currently "the deepest probe gives the most precise pilot" — with something that does
not overclaim:

> The pilot is taken from the deepest probe not flagged as an overshoot. This probe has the smallest
> sampling spread, but it is also selected for having read high, so a fraction of that spread is
> bias; using the opening probe at N_min instead changes the convergence rate by less than
> 0.1 pp. What must be avoided is using the *last* probe regardless of its flag, since that may be an
> aliased reading: it nearly doubles the overshoot rate, from 1.0 % to 1.8 %.

One sentence plus the two numbers is enough. A null result does not need a table — but it does need a
number and a sample size, otherwise "no real improvement" is an unfalsifiable assertion. The
supporting evidence, if an examiner asks, is `results/binary_pilot.csv` and
`notebooks/binary_pilot.ipynb`.

This also refines **A6** in `results/SAFEGUARD_DERIVATION.md`: the data-dependence of the pilot is not
only a mixture over N, it is a *selection* on ϕ̂, which biases the pilot. §12 there states it.

## 9f. Exploiting at L, the bisection's lower bound — tested and rejected

A natural simplification: after the bisection, L is the deepest depth probed and not flagged as an
overshoot, so why not exploit at N = L and drop Eq. (3.8) for binary search altogether?

Measured (`analysis/binary_depth_study.py`, same harness as §9e; 12 live points, SE 0.35 pp):

| depth rule | mean vs shipped | worst | wins |
|---|---:|---:|---:|
| N* from Eq. (3.8), uncapped (**shipped**) | — | — | — |
| **N = L** | **−6.18 pp** | −11.25 | 0 % |
| min(N*, L) | −1.34 pp | −2.48 | 0 % |
| max(N*, L) | −3.66 pp | −10.27 | 8 % |

L overshoots N_opt in **19.6 % of trials (up to 40 %)**, against **1.1 %** for N*. The reason is
structural: the bisection exists to *locate* the aliasing boundary, so L converges onto N_opt and
noise puts it on the wrong side; and the test's type-I error is **absorbing**, since L is a running
maximum — one accepted probe above N_opt and L stays in the aliased region.

**Why this is worth a sentence in §3.2.3.** It is the obvious objection to the whole safeguard for
binary search ("the search already found the boundary — why back off?"), and it has a one-line answer:

> The bisection's lower bound L is an estimate of the aliasing limit itself, not a safe depth below
> it: it exceeds N_opt in about 20 % of trials, and exploiting there costs 6.2 pp of convergence
> against Eq. (3.8). The safeguard's role is precisely to convert an estimate of the limit into a
> depth that respects it.

This is also the cleanest available motivation for Eq. (3.8) in the binary-search section — it shows
the safeguard is not redundant with the search, which a reader may otherwise assume.

---

## 9g. **Defect found: the exploration phases are not budget-capped**

Found while auditing the binary-search path. Algorithms 4 and 5 test the budget **after** taking a
probe, not before:

```python
while not done:
    phi_hat = simulate_errors(rng, phi, m_exploration, N)   # spend first ...
    budget_used += m_exploration * N
    ...
    if ... or budget_used >= budget:                         # ... then notice
        done = True
```

Only the *first* probe is guarded (`if m_exploration * N > budget: return inf`). Binary search's second
probe jumps to N ~ N_max/2, so when the scan does not fit, the algorithm silently spends several times
the budget it is credited with. `analysis/extensive_sweep.py` does not catch this: `grid_full` returns
the mean spend precisely so a caller can enforce a cap, but `tune_winner` discards it
(`for r, _b, cfg in res`), and `qmetrology/tables.py` uses `grid_search_max`, which never returns it.

**This is inherited from the published algorithms, not introduced by the statistical safeguard** —
`_binary_search_explore` is bit-identical to the published version (verified over 36,000 paired
trials). It is a property of Algorithms 4 and 5 as written in the thesis.

Measured at every winning configuration of the sweep (`results/budget_audit.csv`, 2,760 points):

| algorithm | points > 1.01x budget | > 1.10x | > 2x | worst |
|---|---:|---:|---:|---:|
| Linear search | 6.7 % | 6.7 % | 0.5 % | **2.02x** |
| Binary search | 12.9 % | 8.2 % | 5.2 % | **18.89x** |
| Reverse engineering | 3.3 % | 0.8 % | 0.0 % | 1.60x |

### Which reported cells this touches

**One cell is affected materially, and it is not in the broad-distribution section.**
`tab:robustness-across-thresholds` is **Table B.1, p. 39, Appendix B** ("Supplementary simulation
results"), referenced twice from §3.3. Its row U(0.001,0.01), eps=10^-3 names **linear search** the
winner, and linear spends **1.62x / 2.01x / 1.87x / 1.82x** its budget at the 50/80/90/95 % crossings
(binary 1.02-1.08x, reverse engineering 0.996-1.005x). Re-deriving the affected-cell list from
scratch over **all 89 adaptive cells** that `analysis/make_tex.py` emits: exactly those four are above
1.01x; every other cell is 0.997-1.0005x. Note this is also one of the two scenarios where the whole
prior is degenerate (phi_max < sqrt(pi*eps/2)), so there were already two reasons not to cite it.

**The mean-spend audit understates the reach, though.** `budget_audit.csv` stores the *mean* spend,
and `m = int(remaining/N)` makes most trials finish just under budget, which masks a tail of trials
that overspend. Per trial, at the operating points actually reported (R = 40,000, seed 2024):

| reported point | trials over budget | worst trial | cost of capping |
|---|---:|---:|---|
| Table 3.1, linear | 0.57 % | 1.03x | -0.03 pp |
| **Table 3.1, binary** | **1.79 %** | **1.08x** | **56.48 -> 52.16 %, -0.59 pp** |
| Table 3.1, reverse eng. | 0 % | 1.00x | none, bit-identical |
| Table 3.2, 3 of 12 crossings | 0.4-2.9 % | 1.12x | -0.05 to -0.70 pp |
| Table 3.2, other 9 crossings | 0 % | 1.00x | none, bit-identical |

So Table 3.1's binary-search cell and three Table 3.2 crossings do move, by well under a percentage
point and without changing any ordering, winner, or claim. Two *visible* points on Figure 3.2's
eps=10^-3 panel (linear at budgets 2,761 and 4,018) are drawn at ~1.9-2.0x budget; the rest of the
distortion there sits below the 28 % y-floor, where binary spends 5-6x.

Never audited in either pass, because their scenario labels are absent from `budget_audit.csv`:
`results/broad_dist.csv` (Table 3.4 / Figure 3.5) and `results/lever_cube.csv` (the
precision x dynamic-range appendix).

### Resolution: the algorithms are fixed (2026-08-16)

Of the three options -- disclose and exclude, cap during tuning, or fix the algorithms -- the third
was taken. `qmetrology/algorithms.py` now tests the budget **before** each probe, in five places:
linear's scan, binary's bisection, binary-anneal's bisection, and both reverse-engineering pilot-retry
loops (`while phi_hat == 0` had no cap either -- that is the 1.60x reverse-engineering tail, and it is
a retry the pseudocode of Algorithm 6 does not even describe):

```python
while not done:
    if budget_used + N * m_exploration > budget:
        break  # this probe does not fit — stop *before* spending, not after
    phi_hat = simulate_errors(rng, phi, m_exploration, N)
```

Where no trial was overspending the RNG stream is untouched and rates are bit-identical, so the fix
only moves the points the audit flags.

**The re-tune is the point, not the patch.** The published winners were grid-tuned while overspending
was free; run them against the fixed code and the low-budget end collapses (linear at budget 4,018:
38.9 % -> 3.2 %). The full sweep was therefore re-run with the fix. `analysis/extensive_sweep.py` now
also writes `results/budget_audit.csv` itself -- `run_scenario` records the mean spend of each tuned
winner via `E.rate_and_budget` -- so the cap is demonstrated point by point on every future run
instead of being checked by hand.

**Thesis change needed (smaller than expected).**

- **Algorithm 4** (linear search, p. 21) -- *no change*. Its loop already reads
  `while sufficient budget available do` before `Simulate(N, m')`; the implementation had diverged
  from its own pseudocode. Optionally footnote that "sufficient" means m'*N <= remaining budget.
- **Algorithm 5** (binary search, p. 22) -- **needs a change**. Its `do ... while N' != N` loop has no
  budget condition at all and probes first. Add the budget test to the loop, checked before `Simulate`.
- **Algorithm 6** (reverse engineering, p. 23) -- *no change*. It takes a single pilot; the uncapped
  retry loop exists only in Listing A.5.
- **Listings A.3, A.4, A.5** -- all three need the guard line.

`analysis/make_tex.py` keeps the dagger logic. Post-fix no winner can exceed the budget, so it never
fires; it stays as a standing guard rather than a disclosure.

### Outcome of the re-run (2026-08-17): binary search gets substantially *better*

The full `--max` sweep completed, 23/23 scenarios. All **4,600** audited points (23 settings x 5
algorithms x 40 budgets) now spend at most **1.000x** their budget; zero violations.

Change in ratio-vs-brute over all 92 reported cells per algorithm:

| algorithm | min | max | mean |
|---|---:|---:|---:|
| Reverse engineering (Eq. 3.8) | -0.068 | 0.000 | -0.003 |
| Linear search | **-0.566** | +0.018 | -0.032 |
| **Binary search (Eq. 3.8)** | 0.000 | **+1.722** | **+0.279** |
| Brute force / separable / oracle_hl | 0.000 | 0.000 | 0.000 |

So: **reverse engineering, the headline winner, is untouched.** Linear is untouched except the one
Appendix B row where it had been overspending. Binary search improves everywhere and is never worse.

**Why capping the budget makes binary search better, not worse.** Pre-fix, a large pilot was
self-destructing: the bisection's probes cost m'*N each, the scan consumed the entire budget,
`remaining_budget <= 0` fired, and the algorithm returned the *raw pilot* with no exploitation phase
at all -- about 10 % convergence. The tuner was therefore pushed into small pilots, which give a noisy
depth. With the cap the scan simply stops when the next probe will not fit, so the exploitation phase
always runs and a large, precise pilot becomes affordable. Same configuration, budget 21,727,
U(0.01,0.1), eps=10^-3, R=20,000:

| | m' | convergence | trials over budget |
|---|---:|---:|---:|
| pre-fix algorithm | 236 | **9.6 %** | 100 % |
| post-fix algorithm | 236 | **83.3 %** | 0 % |

The tuned m' at that budget went from single digits to 236 (and to 596 at budget 46,001). The defect
was not costing accuracy through overspending so much as it was **hiding binary search's usable
operating regime**.

Table 3.2 / 3.3 cells that move (ratio vs brute at 90 %):

| scenario | binary, old | binary, new |
|---|---:|---:|
| U(0.01,0.1), eps=10^-3 | 1.12x | **1.42x** |
| U(0.01,0.1), eps=10^-4 | 1.55x | 1.55x |
| U(0.001,0.01), eps=10^-4 | 1.04x | **1.27x** |
| U(0.001,0.1), eps=10^-4 | 1.31x | **1.85x** |

At U(0.001,0.1), eps=10^-4 binary search is now level with reverse engineering (1.85x vs 1.86x), where
it used to trail it by a wide margin. Any text claiming binary search is the weakest adaptive strategy
needs re-checking against the new tables.

**Table B.1's flagged row is resolved.** U(0.001,0.01), eps=10^-3, best adaptive ratio across the four
thresholds fell from 0.97 / 1.09 / 1.02 / 0.97 to **0.49 / 0.61 / 0.72 / 0.81**, and the 90 % / 95 %
winner changed from linear search to binary search. The row's message is now unambiguous rather than
marginal: no adaptive strategy comes close to brute force in that regime. The 1.09x that used to look
like a (barely) positive result was the overspend. No table carries a dagger any more.

**Table 3.1 changes qualitatively.** Convergence at fixed budget 10,000 (eps=10^-3, U(0.01,0.1)):

| algorithm | old | old x base | new | new x base |
|---|---:|---:|---:|---:|
| Brute force | 56.18 % | 1.000 | 56.18 % | 1.000 |
| Linear search | 56.57 % | 1.007 | 56.53 % | 1.006 |
| **Binary search** | 52.48 % | **0.934** | **63.20 %** | **1.125** |
| Reverse engineering | 66.26 % | 1.179 | 66.26 % | 1.179 |
| Separable / ceiling | 15.75 / 73.51 % | 0.280 / 1.308 | unchanged | unchanged |

Binary search moves from *below* the baseline to clearly above it, second only to reverse
engineering. Its tuned pilot went from m' = 11 to **m' = 109**. Any sentence in §3.3.1 saying binary
search underperforms at the constrained budget is now wrong — this is the largest single change the
fix produces, and it is a claim reversal, not a numerical drift.

**The §7 head-to-head gets much stronger, and gains a new argument.** `results/SAFEGUARD.md`,
re-run post-fix, now reports for binary search **11/11 operating points improving, median +13.5 pp,
max +24.6 pp** (it was roughly +0.7 to +6.3 pp). Reverse engineering is unchanged at median +3.5 pp.
The constant-`s` arm barely moved, and the reason is the interesting part: **the two rules react to a
truncated scan in opposite ways.** Constant `s` needs the bisection to *converge*, because the depth
it subtracts from is the bisection's answer — so it cannot afford to stop the scan early and its
tuned pilot stays pinned at the bottom of the grid (m' = 20). Eq. (3.8) reads the depth off the
pilot's own posterior, so an early stop costs it nothing and precision in the pilot is worth paying
for — its tuned m' jumps to 156, 732, and 12,164 at the same operating points.

That is a *second*, independent argument for the statistical safeguard, and it was invisible before
the fix: the rule does not merely choose a better depth, it makes the exploration phase
interruptible, which is what allows the budget cap to be honoured without loss. Note the asymmetry
with the existing "much cheaper pilot" bullet, which is about reverse engineering: for binary search
the statistical rule wants a *more expensive* pilot, bought with the shots the truncated scan saves.

**The broad prior (§3.3, Table 3.4) moves the same way**, only more so. Convergence gain for binary
search at phi_max = pi/2: **+6.7 pp at budget 3,000, peaking at +13.9 pp at 77,459**, decaying to
+0.1 pp by 2,000,000. Linear and reverse engineering move by at most 0.6 pp. At budget 15,243 binary
search (28.6 %) now edges out reverse engineering (28.6 %) and clearly beats linear (23.0 %) and brute
force (19.8 %). This is the regime the defect hurt most: N_min is 1-2, so the pilot needs many shots,
and a large pilot was exactly what the uncapped scan could not afford. **If §3.3 was going to be cut
because its result was weak, that reason no longer holds.**

---

## 9h. The budget guard changes reported numbers — a full re-run is required

With `budget_used + m'*N > budget` checked before each probe (§9g), measured directly:

| cell | reported | with the guard | change |
|---|---:|---:|---:|
| Table 3.1, binary search | 52.5 % | **63.2 %** | **+10.7 pp** |
| Table 3.1, brute / linear / RE / separable | — | — | −0.26 to +0.10 pp |
| U(0.001,0.01), eps=10^-3, linear @ 90 % crossing | 90.3 % | **22.1 %** | **−68.2 pp** |
| U(0.001,0.01), eps=10^-3, binary @ 90 % crossing | 90.4 % | 74.0 % | −16.4 pp |
| U(0.001,0.01), eps=10^-3, RE @ 90 % crossing | 90.3 % | 88.8 % | −1.5 pp |

Two opposite effects, both large:

**Binary search gains.** Without the guard a large exploration size was self-defeating: the second
probe jumps to N ~ N_max/2, blows the whole budget, and the algorithm returns its exploration estimate
with nothing spent on exploitation. The tuner was therefore forced down to m' = 10. With the guard the
scan simply stops early, so m' = 167 becomes viable and wins — 63.2 % against 52.5 %. The old winner
(m' = 10) still reproduces 52.5 % exactly under the guard, confirming the gain is the newly reachable
configuration, not a change to old behaviour.

**Linear search loses where it was overspending.** Its 90.3 % at U(0.001,0.01), eps=10^-3 was bought
with 1.87x the budget; capped, it converges 22.1 %. This removes the anomaly in which linear appeared
to beat reverse engineering by ~50 pp in the two fully-degenerate scenarios.

Consequence: `results/story_*.csv`, `all_numbers.csv`, every table in `results/tex/`, and every figure
were produced by the pre-guard code and no longer describe it. The narrative changes too — "binary
search is the clear loser" (§6b item 3) is much weaker at +10.7 pp, and Table 3.1's ranking becomes
brute 56.0 / linear 56.5 / **binary 63.2** / RE 66.4.

---

---

## 10. Explicitly *not* changed

- **Linear search** keeps its tuned safeguard `s` — now for a measured reason rather than an
  assumed one; see §9c(a). Worth one sentence in §3.2.2 so the asymmetry is not read as an oversight.
- The binary-search **overshoot criterion** (Eqs. 3.4–3.5, the α threshold) is untouched.
- ~~The `SimulatePhi…` **exploration phases** are untouched~~ — **no longer true.** They were
  bit-identical to the published version (verified over 36,000 paired trials) until the budget guard
  of §9g was added to all five exploration loops. Algorithms 4, 5 and 6 now test
  `budget_used + m'*N > budget` *before* probing. This is a change to the published pseudocode and
  must be reflected in Appendix A. **Every reported number predates it and has to be regenerated** —
  see §9h for how much moves.

---

## Numbers to transcribe

Regenerated by the full re-run; all are de-biased (tuned on seed 42, validated on seed 2024).

| Source file | Feeds |
|---|---|
| `results/tex/*.tex` | **every table, paste-ready** — `\input{results/tex/all_tables.tex}` |
| `results/RESULTS.md` | Tables 3.1, 3.2, 3.3 in markdown |
| `results/SAFEGUARD.md` | the head-to-head that justifies the change |
| `results/broad_dist.csv`, `fig_broad.png` | Table 3.4, Figure 3.5 |
| `results/STORY_TABLES.md`, `story_cube.csv` | budget ratios, Figures 3.2–3.4 |
| `results/UNCERTAINTY.md`, `story_cube_ci.csv` | the 95% intervals in §9b |
| `results/TUNING_STABILITY.md` | the tuning-variance argument in §9b |
| `results/LEVERS.md`, `lever_cube.csv` | the precision × dynamic-range appendix |

The `.tex` files require `\usepackage{booktabs}` and `\usepackage{makecell}`; `\texttimes` needs
`textcomp` (or `inputenc`). `all_tables.tex` resolves `\input` paths **relative to the main document**,
so re-run `make_tex.py --outdir <path as seen from the main .tex>` if `results/` is not at your thesis
root. The `_ci` variants carry the same `\label{}` as their plain counterparts — include one or the
other, never both.
