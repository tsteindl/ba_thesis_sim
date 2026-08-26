# Proposed thesis changes: Linear Search

Everything below is a **proposal**. No `.tex` file has been edited.

Sources: the re-run sweep in `results/consolidated/` (`sweep_max_meanwindow.log`, 24.6 h,
2026-08-26), the previous sweep preserved verbatim in `results/consolidated_prewindow/`, the
detector study in `results/consolidated/LINEAR_SEARCH.md`, and the claim audit in
`results/consolidated/linear_search_claims.csv`.

**What changed in the algorithm.** The stopping rule now tests the mean of the **last $w$**
estimates instead of the mean of *all* estimates so far. The width $w$ is a tuned parameter, and
$w = 0$ means "all of them", i.e. the previously published cumulative mean. The old algorithm is
therefore a special case of the new one, which is why no result got worse by more than tuning noise.

---

## 1. The updated pseudocode

Four changes against your current version, marked `% CHANGED` below:

1. a new parameter $w$ and a **windowed** mean;
2. `\State $N^* \gets \max(1, N - s)$` — the implementation floors at 1 (Nina's item 6);
3. the backtracking comment no longer claims the backtrack is correct (Nina's item 5);
4. the budget test is stated explicitly: a probe that does not fit is **not taken**, so the scan
   never overspends.

Your current version already has `$N \gets N - l\cdot\textit{inc}$`, so Nina's item 4 is done.

```latex
\begin{algorithm}
    \caption{Estimate $\phi$ using Linear Search}
    \label{alg:linear-search}
    \begin{algorithmic}
        \Procedure{SimulatePhiLinearSearch}{\textit{budget}, $\phi_\text{min}, \phi_\text{max}, \textit{m}', \textit{l}, \textit{inc}, \textit{s}, \textit{w}$}
        \State $N_\text{max} \gets \frac {\pi} {2 \phi_\text{min}},
            \qquad N_\text{min} \gets \frac{\pi} {2\phi_\text{max}},
            \qquad N \gets N_\text{min}$
        \State $\bar \phi \gets 0,
            \qquad i \gets 0,
            \qquad B_\text{used} \gets 0$
        \Comment{Initialize window mean, counter variable}
        \State Initialize empty list $\hat\phi_\text{list}$

        \While {$B_\text{used} + m' N \leq \textit{budget}$}
        \Comment{A probe that does not fit is not taken}
        \State $\hat \phi \gets \Call{Simulate}{N, \textit{m}'},
            \qquad B_\text{used} \gets B_\text{used} + m' N$

        \State Append $\hat\phi$ to $\hat\phi_\text{list}$

        \State $\bar\phi' \gets$ mean of the last $\min(w, |\hat\phi_\text{list}|)$ entries of $\hat\phi_\text{list}$
        \Comment{$w = 0$: all entries} % CHANGED
        \If {$\bar \phi' < \bar \phi$}
        \State $i \gets i + 1$ \Comment{Update counter variable}
        \Else
        \State $i \gets 0$
        \EndIf

        \State $\bar \phi \gets \bar \phi'$
        \If {$i \geq l$} \Comment{The window mean has decreased $l$ times in a row}
        \State $N \gets N - l\cdot\textit{inc}$
        \Comment{Discard the $l$ probes that triggered the stopping rule} % CHANGED
        \State \textbf{break}
        \ElsIf {$N \geq N_\text{max}$ \textbf{or} $B_\text{used} \geq \textit{budget}$}
        \State \textbf{break}
        \Else
        \State $N \gets N + \textit{inc}$
        \Comment{Increment $N$}
        \EndIf
        \EndWhile

        \State
        $N^* \gets \max(1,\, N - s)$
        \Comment{Reduce $N$ by safeguard to avoid overshooting} % CHANGED

        \State $m \gets \lfloor (\textit{budget} - B_\text{used}) / N^*\rfloor$
        \Comment{Calculate shots such that remaining budget is exhausted}
        \State $\hat \phi \gets \Call{Simulate}{N^*, m}$

        \State \Return $\hat\phi$
        \EndProcedure
    \end{algorithmic}
\end{algorithm}
```

### Optional: the two edge cases

The implementation also handles two cases the pseudocode is silent about, and it handles them
*differently* from what the pseudocode above would do — so this is a behavioural gap, not just an
omission. Neither case arises at any reported operating point (both need a budget too small to
afford the scan), so add these only if you want the pseudocode to be literally faithful.

```latex
        \If {$m' N_\text{min} > \textit{budget}$} \Comment{Not even the first probe is affordable}
        \State \Return $\infty$
        \EndIf
```

placed immediately after the initialization — without it the pseudocode would spend the whole budget
at $N_\text{min}$ instead of refusing — and

```latex
        \If {$B_\text{used} \geq \textit{budget}$} \Comment{Scan consumed the whole budget}
        \State \Return $\hat\phi_\text{list}[\,\max(1, |\hat\phi_\text{list}| - l)\,]$
        \EndIf
```

placed immediately before the $N^*$ line: with no budget left there is nothing to exploit with, so
the implementation returns a retained exploration estimate rather than calling \Call{Simulate}{} with
$m = 0$.

### Symbol to add to the notation list

| symbol | meaning | code name |
|---|---|---|
| $w$ | width of the mean tested by the stopping rule; $w = 0$ means the cumulative mean of the whole scan | `mean_window` |

---

## 2. Chapter 3: the stopping rule

### 2.1 Replace the methodology paragraph

Nina's suggested replacement, extended with the window. Suggested text:

> Linear Search uses a heuristic stopping rule rather than a calibrated hypothesis test. In the
> noiseless model the estimator equals $\phi$ on the admissible branch $0 \leq N\phi \leq \pi/2$.
> Immediately after the first aliasing boundary, for $\pi/2 \leq N\phi \leq \pi$, it becomes
> $\hat\phi_N = \pi/N - \phi$ and therefore decreases with $N$. This local post-boundary trend
> motivates testing for consecutive decreases. Writing $\bar\phi^{(w)}_k$ for the mean of the last
> $w$ estimates, the scan stops after $l$ consecutive falls of $\bar\phi^{(w)}$. Shot noise can
> produce the same event before the aliasing boundary, so premature stops are possible and no formal
> detection guarantee is claimed. We therefore assess the rule through held-out false-alarm and miss
> rates, the distance between $N_{\mathrm{guess}}$ and $N_{\mathrm{opt}}$, exploration cost, and
> final convergence performance in Section 4.5 and Appendix B.

### 2.2 Add the identity (short, exact, and it reframes the criticism)

This is the single most useful addition. Suggested text:

> For $k \geq w$ two consecutive windows differ only in the element that enters and the element that
> leaves, so
> $$\bar\phi^{(w)}_k - \bar\phi^{(w)}_{k-1} = \frac{\hat\phi_k - \hat\phi_{k-w}}{w},$$
> and the sign test $\bar\phi^{(w)}_k < \bar\phi^{(w)}_{k-1}$ is exactly
> $\hat\phi_k < \hat\phi_{k-w}$. A windowed rule therefore performs no averaging: it compares a probe
> with the probe $w$ steps earlier, which is Equation~\eqref{eq:overshoot-threshold} with
> $\hat\phi_{\mathrm{acc}} = \hat\phi_{k-w}$ and $\alpha = 1/2$. The cumulative rule $w = 0$ is the
> same sign test against a reference whose effective lag grows with $k$. The two search strategies of
> this chapter therefore share one overshoot criterion and differ only in search order.

In LaTeX:

```latex
For $k \geq w$ two consecutive windows differ only in the element that enters and the
element that leaves, so
\begin{equation}
    \bar\phi^{(w)}_k - \bar\phi^{(w)}_{k-1} = \frac{\hat\phi_k - \hat\phi_{k-w}}{w},
    \label{eq:window-identity}
\end{equation}
and the sign test $\bar\phi^{(w)}_k < \bar\phi^{(w)}_{k-1}$ is exactly
$\hat\phi_k < \hat\phi_{k-w}$.
```

Verified in `tests/test_consolidated.py::test_lds_moving_window_is_a_lagged_comparison`.

### 2.3 Say that $w = 0$ recovers the published rule

One sentence, so a reader understands the parameter is a generalisation and not a replacement:

> The width $w$ is tuned alongside $m'$, $l$, $\mathrm{inc}$ and $s$; $w = 0$ recovers the cumulative
> mean, so the tuning grid contains the simpler rule as a special case. Over the informative
> operating points the grid search selects $w = 0$ at 12 of 144 points and a finite window at the
> remaining 132.

### 2.4 Do **not** keep the "dilution by $1/k$" explanation

Nina is right that dividing by $k$ cannot change a sign, so the update magnitude is not the issue.
The real difference between $w = 0$ and $w > 0$ is the **lag of the reference**, per §2.2.

---

## 3. Chapter 4: numbers to update

Only Linear Search rows change. Brute force, Binary Search and Reverse Engineering are
**bit-identical** to the previous sweep across all 155 protocol rows — the same seeds and the same
grids produce the same numbers, so nothing else in Chapter 4 moves.

### 3.1 Table 4.1 (fixed budget $C = 10{,}000$, $\epsilon = 10^{-3}$)

| Algorithm | % converged | Budget for $>90\%$ |
|---|---:|---:|
| Brute force baseline | 56.09% | 45,900 |
| **Linear search** | **57.34% → 64.10%** | **42,500 → 35,500** |
| Binary search | 65.07% | 32,400 |
| Reverse engineering | 66.21% | 29,800 |
| Separable ($N=1$) | 15.68% | 680,000 |
| Oracle ($N_\text{opt}$) | 73.48% | 24,435 |

The improvement over brute force goes from $+1.25$ pp to $+8.01$ pp. As a share of the attainable
brute-force-to-oracle gap: **7% → 46%**.

### 3.2 Budget ratios against brute force at 90% reliability

| Scenario | LS old | LS new | BS | RE |
|---|---:|---:|---:|---:|
| $\epsilon = 10^{-3}$ | 1.08× | **1.29×** | 1.42× | 1.54× |
| $\epsilon = 10^{-4}$ | 1.56× | 1.56× | 1.60× | 1.74× |
| $\epsilon = 10^{-5}$ | 1.57× | **1.64×** | 1.75× | 1.80× |
| $\epsilon = 10^{-6}$ | 1.63× | 1.65× | 1.78× | 1.80× |
| $\epsilon = 10^{-7}$ | 1.65× | 1.67× | 1.79× | 1.81× |
| $\epsilon = 10^{-8}$ | 1.62× | **1.66×** | 1.77× | 1.82× |
| $U(0.001, 0.01)$ | 1.17× | **1.36×** | 1.32× | 1.46× |
| $U(0.001, 0.1)$ | 1.30× | **1.42×** | 1.84× | 1.85× |
| $U(0.01, \pi/4)$ | 1.11× | **1.22×** | 1.57× | 1.55× |
| $U(0.01, \pi/2)$ | 1.14× | 1.14× | 1.44× | 1.43× |

### 3.3 One reported ranking flips — $U(0.001, 0.01)$

Linear Search now needs **325,000** [321,000–329,000] against Binary Search's **334,000**
[330,000–338,000]. The intervals are disjoint, so this is resolved rather than noise. **This is the
only cell in the thesis where the order of two algorithms changes.** Any sentence asserting that
Binary Search dominates Linear Search in every setting needs a carve-out for this prior.

Everywhere else the ordering RE > BS > LS > brute holds, and the points-won count is unchanged
(BS 63, RE 81, LS 0, brute 0 of 144 informative points).

### 3.4 Section 4.2 prose

Two separate problems.

**(a) A pre-existing contradiction, independent of this change.** The text says Binary Search
underperforms the baseline and that all adaptive methods except Binary Search beat the baseline at
90% reliability. Table 4.1 says Binary Search reaches 65.07% against 56.09% and needs 32,400 against
45,900. Replace with: all three adaptive methods improve on the baseline; Reverse Engineering
performs best, Binary Search is close behind, Linear Search provides the smallest improvement.

**(b) The margin changes.** "Smallest improvement" is still correct, but it is now $+8.01$ pp rather
than $+1.25$ pp at the headline point, and Linear Search beats Binary Search at 9 of 144 informative
points (previously 1). The mean gap to Binary Search narrows from $-3.59$ pp to $-2.15$ pp.

### 3.5 Appendix C — tuned parameters

The Linear Search parameter tables gain a $w$ column. The other selected values shift only mildly
over the informative points — median $m'$ from 108 to 92, median $l$ from 7 to 5 — but the
**exploration cost halves** (median budget share 1.78% → 0.83%, §4.1), because a windowed rule
reaches its verdict without having to drag a long history along. Selected values at the headline
points:

| Scenario, budget | $m'$ | $l$ | $s$ | $\mathrm{inc}$ | $w$ |
|---|---:|---:|---:|---:|---:|
| $\epsilon = 10^{-3}$, $B = 10{,}000$ | 1 | 6 | 0 | 1 | 4 |
| $\epsilon = 10^{-3}$, $B = 41{,}326$ | 1 | 4 | 1 | 1 | 3 |
| $\epsilon = 10^{-4}$, $B = 2{,}917{,}365$ | 176 | 8 | 1 | 1 | **0** |
| $U(0.001,0.01)$, $B = 394{,}843$ | 1 | 12 | 8 | 2 | 8 |

Distribution of $w$ over the 144 informative points: $w=0$: 12, $w=1$: 46, $w=2$: 25, $w=3$: 26,
$w=4$: 23, $w=6$: 5, $w=8$: 6, $w=12$: 1.

---

## 4. Section 4.5 and Appendix B — the diagnostics, and the claim that must change

### 4.1 The numbers

Aggregated over the 144 informative operating points:

| Diagnostic | old | new |
|---|---:|---:|
| trial-level false-alarm rate | 45.3% | **53.8%** |
| trial-level miss rate | 7.7% | **6.6%** |
| median $N_\text{guess}/N_\text{opt}$ | 1.00 | 1.00 |
| guesses within 10% of $N_\text{opt}$ | 68.6% | **61.1%** |
| median $N^*/N_\text{opt}$ | 0.95 | 0.95 |
| final overshoot $P(N^* > N_\text{opt})$ | 2.35% | **1.72%** |
| median exploration budget share | 1.78% | **0.83%** |

### 4.2 The interpretive claim that no longer holds

**The detector diagnostics get worse while the algorithm gets better.** False alarms rise 8.5 pp and
the within-10% share falls 7.5 pp, while convergence rises by up to 6.8 pp, final overshoot falls by
a quarter and exploration cost more than halves. The thesis therefore cannot present the
false-alarm rate or the within-10% share as measures of how good the search is. Concretely, the
reading "Linear Search is weak at $\epsilon = 10^{-3}$ because only 32.0% of its guesses land within
10% of $N_\text{opt}$" is untenable: the configuration that converges 6.8 pp better manages 15.4%.

The mechanism is that at tight budgets, never aliasing is worth more than sitting close to
$N_\text{opt}$; the winning rule deliberately stops short and spends the saved budget on shots at a
safe depth.

### 4.3 Suggested replacement for the paragraph after Tables 4.5/4.6

> The diagnostics show that Linear Search is a conservative but imperfect boundary-localization
> heuristic. Its trial-level false-alarm rate is 53.8%, so the stopping rule should not be
> interpreted as a reliable overshoot classifier. A false alarm does not, however, imply a large
> loss: aggregated over the informative operating points the median $N_{\mathrm{guess}}/N_{\mathrm{opt}}$
> is 1.00, and after the safeguard the median exploitation ratio is 0.95 with a final overshoot rate
> of 1.72%, at a median exploration cost of 0.83% of the budget. The false-alarm rate is moreover
> not a measure of fitness: among the stopping rules evaluated in Appendix~\ref{app:detector-study},
> the one maximizing convergence at the fixed-budget operating point also has the highest
> false-alarm rate, because it stops short of the aliasing boundary deliberately.

### 4.4 Why the false-alarm rate is high — a derivation to add

This answers Nina's objection with a mechanism instead of an apology, and it explains the regime
dependence. Suggested text:

> A high false-alarm rate is unavoidable for any rule based on consecutive decreases. On the safe
> branch the readout probability is $p_0 = \cos^2(N\phi)$, so a probe returns $K = 0$ with
> probability $\sin^{2m'}(N\phi) = \sin^{2m'}(r\pi/2)$, with $r = N\phi/(\pi/2)$. As $r \to 1^-$ that
> probability tends to one and the estimate ceases to be random: it becomes the single atom
> $\pi/(2N)$, which decreases with $N$. In that zone the rule fires with probability approaching one
> while every probe is still safe. The zone occupies the last
> $1 - \tfrac{2}{\pi}\arcsin\!\big(2^{-1/2m'}\big) \approx \tfrac{2}{\pi}\sqrt{\ln 2 / m'}$ of the
> depth range, i.e. 36% of it at $m' = 2$ but only 4% at $m' = 176$ — which is why the measured
> false-alarm rate falls from 77.7% at the fixed-budget operating point to 30.2% at
> $\epsilon = 10^{-4}$.

Numbers from `results/consolidated/linear_degenerate_zone.csv` (exact, nothing simulated).

Note this is the same degeneracy that makes the Binary Search criterion sharp, seen from *below* the
boundary rather than above it — so Section 2.7's fold, Section 3.2.2's criterion and this paragraph
are one phenomenon and should cross-reference each other.

### 4.5 The $2^{-l}$ argument should not be used

If the text (or a footnote) reasons that $l$ consecutive decreases have probability $\approx 2^{-l}$,
remove it. Measured at the fixed-budget point, the probability that the scan stops before
$N_\text{opt}$ is 78.2% at $l = 4$ and still **63.6%** at $l = 12$, where the independence argument
predicts 0.3%. The decreases are strongly dependent for the reason in §4.4. Use measured rates
(`results/consolidated/linear_detector_streaks.csv`).

### 4.6 Appendix B.2

| Setting | median $N_\text{guess}/N_\text{opt}$ | mean abs. rel. error | within 10% |
|---|---:|---:|---:|
| $\epsilon = 10^{-3}$ | 0.83 → **0.72** | 0.29 → 0.29 | 32.0% → **15.4%** |
| $\epsilon = 10^{-4}$ | 1.00 → 1.00 | 0.05 → **0.09** | 87.6% → **73.4%** |
| $\epsilon = 10^{-5}$ | 1.00 → 1.00 | 0.04 → 0.04 | 89.6% → 89.5% |

Regenerate from `results/consolidated/diagnostics_by_point.csv`. Per §4.2 the accompanying prose
must not read these as a fitness ranking.

---

## 5. Chapter 2 and cross-references

- **Equation (2.84)**, "converges to zero": still contradicts Eq. (2.85). Past the first aliased
  branch, 49.5% of steps in $N$ *increase* the estimate — it is a sawtooth, and only its envelope
  decays. Replace the global statement with the first-aliased-branch calculation
  ($\hat\phi_N = \pi/N - \phi$, $\mathrm{d}\hat\phi/\mathrm{d}N = -\pi/N^2$).
- **Equation (2.85)** should carry a label (e.g. `eq:folded-estimator`) so Chapter 3 can cite the
  fold directly.
- **Section 2.7's** "this property will be employed in Section 4" should point at Chapter 3, which
  is where both the Binary Search criterion and the Linear Search false-alarm analysis now use it.

---

## 6. Optional: a subsection or appendix on the detector study

If you want to show the alternative was tested rather than assumed, `tab_linear_detector.tex` and
`fig_linear_detector.png` are ready to `\input` / `\includegraphics`. They report six stopping
rules — cumulative mean, moving window, pre/post windows with a noise threshold, the
Equation~(3.6) criterion at lag $w$, a pooled-reference version, and a CUSUM — each tuned over the
same grids on the same tuning blocks and scored on the same held-out trials. The pre/post, CUSUM and
pooled rules are exactly the three "defensible replacements" a reviewer would ask for; all were
built and none beats the windowed rule where it matters.

```latex
\input{results/consolidated/tex/thesis/tab_linear_detector.tex}
```

---

## 7. Status of Nina's eight minimum changes

| # | Change | Status |
|---|---|---|
| 1 | "we deduce that we have overshot" → "the rule declares an overshoot" | **still to do** (prose) |
| 2 | Remove "reliably determine an overshoot with very few exploration shots" | **still to do** — and the rate is now 53.8%, so it is more clearly wrong |
| 3 | Replace "estimates tend toward zero" with the first-aliased-branch calculation | **still to do** — §5 above |
| 4 | Algorithm 4: $N \leftarrow N - l\,\mathrm{inc}$ | **already correct** in your current pseudocode |
| 5 | Comment: "Discard the $l$ probes that triggered the stopping rule" | **in the new pseudocode** (§1) |
| 6 | $N^* = \max\{1, N_\text{guess} - s\}$ | **in the new pseudocode** (§1) |
| 7 | Add an interpretation after Tables 4.5/4.6 | **draft in §4.3** |
| 8 | Distinguish the low-precision from the high-precision result | **still to do** — and the contrast is now smaller: 1.29× vs 1.56–1.67× rather than 1.08× vs 1.56× |

All 54 quantitative claims in `linear_search_revision_notes.md` were verified against the sweep they
were written about (`results/consolidated_prewindow/`); see `linear_search_claims.csv`. They are
sound, but the ones describing Linear Search's *performance* now describe the superseded version.

---

## 8. Regenerated artifacts to re-`\input`

All under `results/consolidated/`, all rebuilt from the new sweep:

| | |
|---|---|
| tables | `tex/thesis/*.tex` — 19 files, including `all_thesis_tables.tex` |
| figures | `fig_story`, `fig_story_small`, `fig_precision`, `fig_pareto`, `fig_algorithm_diagnostics`, `fig_error`, `fig_variance`, `fig_error_variance`, `fig_error_density`, `fig_signed_error_density`, `fig_phi_hat_density`, `fig_broad`, `fig_overshoot_criterion`, `fig_linear_detector` |
| reports | `REPORT.md`, `FULL_RESULTS.md` |

Two cautions:

- `results/consolidated_prewindow/` holds the sweep the current draft and Nina's notes describe.
  Keep it until the text matches the new numbers.
- `variance_curves.py` is **not** part of `run.py`. After any future re-run it must be run by hand
  before `thesis_figures.py`, or `fig_variance` and `fig_error_variance` will silently keep old data.
