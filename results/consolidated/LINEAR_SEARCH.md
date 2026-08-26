# Linear search: what to do with the section

> **Status: adopted.** Algorithm 4 now has a `mean_window` parameter and the production sweep was
> re-run with it (`sweep_max_meanwindow.log`, 24.6 h, 2026-08-26). Everything in
> `results/consolidated/` is post-change; the sweep the thesis draft and
> `linear_search_revision_notes.md` describe is preserved verbatim in
> [`../consolidated_prewindow/`](../consolidated_prewindow/). **No `.tex` file was edited — the
> prose changes in Sections 6 and 8 are still yours to make.**

Answers [`linear_search_revision_notes.md`](../../linear_search_revision_notes.md), and in
particular the question it leaves open: *would a moving-window mean be better?* It would, and it
now is: the width of the mean is a tuned parameter whose `w = 0` is the previously published
cumulative rule, so the grid contains the old algorithm and the tuner picks between them.

```bash
python analysis/consolidated/run.py --max --keep-traces         # the re-run that adopted it (24.6 h)
python analysis/consolidated/linear_search_claims.py            # audits the notes vs the OLD sweep
python analysis/consolidated/linear_detector_study.py --curve   # the detector bake-off (~2.5 h)
python analysis/consolidated/thesis_tables.py                   # -> tex/thesis/tab_linear_detector.tex
python analysis/consolidated/thesis_figures.py --only linear_detector
```

---

## 0. What the re-run actually produced

The full sweep (10 scenarios, 221 budgets, 144 informative operating points) with `mean_window`
in the grid. Brute force, binary search and reverse engineering are **bit-identical** to the
previous sweep across all 155 protocol rows — only linear search moved.

| | old | new |
|---|---:|---:|
| convergence at $B = 10{,}000$, $\epsilon = 10^{-3}$ | 57.34% | **64.10%** |
| budget for 90% there | 42,500 | **35,500** |
| mean change across the 144 live points | — | **+1.44 pp** |
| points improved by >0.5 pp / worsened by >0.5 pp | — | **81 / 6** (worst −0.92 pp) |
| points won (best of four) | 0 | 0 |
| beats binary search | 1/144 | **9/144** |
| mean gap to binary search | −3.59 pp | **−2.15 pp** |
| mean gap to brute force | +6.29 pp | **+7.72 pp** |

Budget for 90% convergence, as a ratio against brute force:

| scenario | LS old | LS new | BS | RE |
|---|---:|---:|---:|---:|
| $\epsilon = 10^{-3}$ | 1.08× | **1.29×** | 1.42× | 1.54× |
| $\epsilon = 10^{-4}$ | 1.56× | 1.56× | 1.60× | 1.74× |
| $\epsilon = 10^{-5}$ | 1.57× | 1.64× | 1.75× | 1.80× |
| $\epsilon = 10^{-6}$ | 1.63× | 1.65× | 1.78× | 1.80× |
| $\epsilon = 10^{-7}$ | 1.65× | 1.67× | 1.79× | 1.81× |
| $\epsilon = 10^{-8}$ | 1.62× | 1.66× | 1.77× | 1.82× |
| $U(0.001, 0.01)$ | 1.17× | **1.36×** | 1.32× | 1.46× |
| $U(0.001, 0.1)$ | 1.30× | **1.42×** | 1.84× | 1.85× |
| $U(0.01, \pi/4)$ | 1.11× | **1.22×** | 1.57× | 1.55× |
| $U(0.01, \pi/2)$ | 1.14× | 1.14× | 1.44× | 1.43× |

**One reported ranking flips.** On $U(0.001, 0.01)$ linear search now needs **325,000**
[321,000–329,000] against binary search's **334,000** [330,000–338,000] — non-overlapping
intervals, so it is resolved, not noise. That is the only cell in the thesis where the order of two
algorithms changes; everywhere else the ordering RE > BS > LS > brute is preserved.

**How often the old rule is still chosen.** Over the 144 live points the tuner selects
`mean_window = 0` — the published cumulative mean — at **12** points, and a finite window at the
other 132 (median $w = 2$; distribution 0:12, 1:46, 2:25, 3:26, 4:23, 6:5, 8:6, 12:1). The
fallback is what keeps $\epsilon = 10^{-4}$ from regressing.

**Aggregate diagnostics** over the same 144 points — note that the detector looks *worse* while the
algorithm is better, which is the point Section 4 derives:

| | old | new |
|---|---:|---:|
| trial-level false alarm | 45.3% | **53.8%** |
| trial-level miss | 7.7% | 6.6% |
| median $N_\text{guess}/N_\text{opt}$ | 1.00 | 1.00 |
| guesses within 10% of $N_\text{opt}$ | 68.6% | **61.1%** |
| final overshoot $P(N^* > N_\text{opt})$ | 2.35% | **1.72%** |
| median exploration budget share | 1.78% | **0.83%** |

Exploration got *cheaper* (share more than halved) and final overshoot fell by a quarter, while the
false-alarm rate rose by 8.5 pp and the within-10% share fell by 7.5 pp. Tables 4.5/4.6 and Appendix
B.2 therefore cannot be read as fitness measures — see Section 4.

---

## 1. Short answers

**Is every number in the notes right?** Yes. All 54 checkable statements were recomputed — from the
stored sweep, from the live source, or in closed form ([`linear_search_claims.csv`](linear_search_claims.csv)).
Three quantiles differ by one phase gate because they are re-drawn with a different probe stream;
nothing else differs at all. The notes are a sound basis for the revision.

**Would a moving window be better?** Yes, in exactly the regime the critique targets — and the
production sweep has now confirmed it end to end: **57.34% → 64.10%** at the fixed-budget headline
point (Section 0). The bake-off below predicted 64.64% from a better-tuned analytic replay; the real
tuner is noisier and landed 0.5 pp lower, as flagged in advance.

**Is it a new algorithm needing new theory?** No, and this is the part worth acting on. Once a
trailing window is full, "the moving-window mean fell" is *identically* "$\hat\phi_k <
\hat\phi_{k-w}$", because two adjacent windows differ only in the element that enters and the one
that leaves. The supervisor's suggestion therefore performs no averaging at all: it is the thesis's
**own** overshoot criterion, Eq. (3.6), evaluated against the probe $w$ steps back rather than the
one immediately before, at $\alpha = 0.5$. The tuner picks $\alpha = 0.5$ unprompted at three of the
six points, all of them the ones where the gain is large. The "defensible replacement detector" the
notes say would be needed is therefore already derived in Chapter 3.

**Is the 45.3% false-alarm rate evidence of a bad rule?** No — and this is the strongest available
defence of the section. The rule that maximises convergence at the headline point has a **93.9%**
false-alarm rate, *higher* than the published rule's 77.7%, while cutting the final overshoot rate
from 9.6% to 2.2%. Section 4 derives why: a fixed fraction of the safe depth range is a zone in
which the estimator is not random at all, so any decrease-based rule must fire there.

---

## 2. Every claim in the notes, recomputed

[`linear_search_claims.csv`](linear_search_claims.csv), 54 rows, 0 disagreements.

| Group | Result |
|---|---|
| aggregate diagnostics | 45.3% false alarm, 7.7% miss, median $N_\text{guess}/N_\text{opt} = 1.00$, 68.6% within 10%, median $N^*/N_\text{opt} = 0.95$, 2.35% final overshoot, 1.78% exploration share, 144 informative points — **all confirmed to two decimals** |
| Appendix B.2 | 0.83 / 0.29 / 32.0% at $\epsilon = 10^{-3}$; 1.00 / 0.05 / 87.6% at $10^{-4}$; 1.00 / 0.04 / 89.6% at $10^{-5}$ — **all confirmed** |
| tuned parameters | 22 of 144 informative points select $l = 1$; headline windows $l = 4$ / $3$ / $8$ — **confirmed**. Also **44 of 144 select $\mathrm{inc} > 1$**, which is why the Algorithm 4 correction `N ← N − l·inc` is not a marginal case |
| end-to-end | 57.34% vs 56.09% at $B = 10{,}000$; 42,500 vs 45,900 and 2.91M vs 4.53M at 90%; ratios 1.08× and 1.56× — **all confirmed** |
| Section 4.2 inconsistency | binary search really is 65.07% vs 56.09%, and 32,400 vs 45,900 — the prose **is** wrong, as the notes say |
| code vs pseudocode | the code really does backtrack by `l·inc`, really does apply `max(1, N_guess − s)`, really does test the cumulative mean, and really does define its overshoot declaration as "the probes I backtracked over" — **all four confirmed** |
| noiseless branches | $g = \phi$ on the admissible branch and $\pi/N - \phi$ on the first aliased one, both exact to $10^{-15}$; the slope really is $-\pi/N^2$. Past the first aliased branch **49.5% of steps increase $g$** — the current thesis sentence about tending to zero is refuted, exactly as the notes claim; it holds only for the envelope |

The false-alarm distance diagnostics the notes report from a one-off reproduction now come from a
script, at the full held-out $R = 50{,}000$ rather than the 2,000 stored traces:

| Point | Median gates below $N_\text{opt}$ | p90 | p95 | Within 10% | Trial-level false alarm |
|---|---:|---:|---:|---:|---:|
| $B = 10{,}000$, $\epsilon = 10^{-3}$ | 5 | 64 | 90 | 27.2% | **77.7%** |
| $B = 41{,}326$, $\epsilon = 10^{-3}$ | 11 | 59 | 82 | 17.4% | **52.1%** |
| $B = 2{,}917{,}365$, $\epsilon = 10^{-4}$ | 1 | 22 | 40 | 68.1% | **30.2%** |

The notes say the false-alarm rate at the fixed-budget headline point is "substantially higher" than
the 45.3% aggregate. It is **77.7%**, and that number belongs in the text.

---

## 3. The bake-off

[`analysis/consolidated/linear_detector_study.py`](../../analysis/consolidated/linear_detector_study.py).
Six candidate stopping rules, each replacing **only** the overshoot verdict of Algorithm 4 — the
scan, the backtracking and the safeguard are untouched — each tuned over the same
$(m', \mathrm{inc}, s)$ grids the manifest gives Algorithm 4, on the same two tuning blocks, and
scored on the same held-out seed. 2,920 rule/safeguard configurations at up to 105 exploration
settings per operating point, 13 operating points.

### Why the comparison is exact

The depths the scan can afford are fixed *before any datum is seen*: a probe costs $m' N$, so which
probes exist depends on the budget, not on the estimates. The scan can therefore be run once per
trial to its affordable end and every rule replayed on that recorded stream — each rule stopping
where it would have stopped, having seen exactly the probes it would have seen. This is not an
approximation, and `tests/test_consolidated.py::test_lds_replay_reproduces_algorithm_4` asserts it:
over 240 real trials the replay reproduces Algorithm 4's own $N_\text{guess}$, $N^*$, false-alarm
verdict and miss verdict **exactly**.

The exploitation phase is then evaluated analytically: once $N^*$ and $m$ are fixed,
$P(|\hat\phi - \phi| < \epsilon)$ is an exact binomial quantity — the expression
`qmetrology/oracle.py` already uses for the oracle, verified identical to machine precision.
Integrating out the final coin flip is what lets a 0.3 pp difference be resolved with the trials a
simulated comparison would need for 3 pp. Replaying the **published** configuration this way
reproduces the sweep's own held-out rates at all six points, within Monte-Carlo error
([`linear_detector_validation.csv`](linear_detector_validation.csv)); across six re-seedings the
analytic estimate has a standard deviation of 0.01 pp against the sweep's own 0.13 pp.

### The rules

| Rule | What it declares |
|---|---|
| `cumulative` | Algorithm 4 as published: $l$ consecutive falls of the cumulative mean |
| `window` | the supervisor's suggestion: $l$ consecutive falls of the mean of the last $w$ probes |
| `prepost` | last $w$ probes against the $w$ before them, with an Eq. (3.4) noise threshold $z$ |
| `threshold` | $\hat\phi_k < \hat\phi_{k-w} + z_\alpha/(2N_k\sqrt{m'})$, $l$ times running — Eq. (3.6) at lag $w$ |
| `pooled` | the same against the inverse-variance pooled mean of all previous probes, its own variance propagated |
| `cusum` | a CUSUM on the standardised drop below that pooled reference |

`prepost`, `pooled` and `cusum` are precisely the three replacements the notes name as what a
defensible detector would have to look like — "a comparison of pre- and post-windows with a
noise-dependent threshold, a CUSUM-style change detector, or a likelihood-based test". All three
were built, tuned and scored. None of them is the best.

### Results

Held-out convergence, $R = 50{,}000$ ([`linear_detector_bakeoff.csv`](linear_detector_bakeoff.csv)):

| Rule | $10^{-3}$, $B{=}10{,}000$ | $10^{-3}$, $B{=}41{,}326$ | $10^{-4}$, $B{=}2.92$M | $10^{-5}$, $B{=}292$M | $U(0.001,0.01)$ | $U(0.01,\pi/4)$ |
|---|---:|---:|---:|---:|---:|---:|
| cumulative, **as published** | 57.49 | 89.69 | 90.27 | 90.43 | 90.84 | 90.52 |
| cumulative, re-tuned here | 58.00 | 89.50 | 90.24 | 90.73 | 90.84 | 90.79 |
| moving window | **64.68** | **92.65** | 88.92 | 90.92 | **93.08** | 91.77 |
| pre/post windows | 62.28 | **92.91** | 89.63 | 91.34 | 91.89 | **93.15** |
| Eq. (3.6) at lag $w$ | 64.50 | 92.84 | 89.95 | 91.49 | 92.72 | 92.66 |
| Eq. (3.6) vs. pooled ref. | 61.31 | 92.33 | **90.95** | **91.60** | 91.54 | 91.79 |
| CUSUM | 59.77 | 91.27 | 90.16 | 91.59 | 90.77 | 91.76 |
| **gain over the better cumulative row** | **+6.68** | **+3.21** | **+0.69** | **+0.87** | **+2.24** | **+2.36** |

Two controls matter, and both hold:

- **It is not better tuning.** The published row and the re-tuned row differ by +0.5 pp at the
  headline, −0.2 pp at $B = 41{,}326$, and by less than 0.3 pp everywhere else. Better tuning of the
  *same* rule buys almost nothing; the rest is the rule.
- **It is not a different exploration size.** Pinning $m'$ and $\mathrm{inc}$ to the published
  values and letting only the stopping rule change
  ([`linear_detector_matched.csv`](linear_detector_matched.csv)):

  | Point | cumulative | best alternative at the same $m'$, inc | gain |
  |---|---:|---:|---:|
  | $10^{-3}$, $B = 10{,}000$ ($m'{=}2$) | 58.00 | 62.67 (Eq. 3.6, lag 2) | **+4.68** |
  | $10^{-3}$, $B = 41{,}326$ ($m'{=}15$) | 89.69 | 91.89 (Eq. 3.6, lag 4) | **+2.20** |
  | $10^{-4}$, $B = 2.92$M ($m'{=}176$) | 90.27 | 90.50 (pooled) | +0.23 |
  | $10^{-5}$, $B = 292$M ($m'{=}5565$) | 90.53 | 91.60 (pooled) | **+1.06** |
  | $U(0.001,0.01)$ ($m'{=}1$) | 90.84 | 91.65 (Eq. 3.6, lag 2) | +0.81 |
  | $U(0.01,\pi/4)$ ($m'{=}596$) | 90.79 | 92.70 (pre/post) | **+1.91** |

  Panel (c) of the figure shows the same thing across the whole $m'$ axis: the lagged rule is above
  the cumulative rule at *every* exploration size, not at a favoured one.

### The headline point, fully simulated

The analytic replay is what makes the fine comparisons possible, but the headline claim does not
depend on it. Running the real implementation
(`qmetrology.algorithms.find_phi_fixed_budget_linear_search_lagged`) on the same 50,000 held-out
trials:

| Configuration | Simulated convergence |
|---|---:|
| published Algorithm 4 ($m'{=}2$, $l{=}4$, $s{=}1$, $\mathrm{inc}{=}1$) | 57.34% ± 0.43 |
| Eq. (3.6) at lag 2, $\alpha = 0.5$ ($m'{=}1$, $l{=}4$, $s{=}0$, $\mathrm{inc}{=}1$) | **64.64% ± 0.42** |

57.34% is exactly the number in Table 4.1, as it must be — same algorithm, same seeds.

For context at that budget: brute force 56.09%, binary search 65.07%, reverse engineering 66.21%,
oracle 73.48%. Linear search currently captures **7%** of the gap between brute force and the
oracle; with the lagged rule it captures **49%**. The ordering of the three adaptive methods does
**not** change — linear search stays third — so Chapter 4's qualitative conclusion survives while
its "smallest improvement" becomes a real one.

### What it does to the 90% crossings

[`linear_detector_crossings.csv`](linear_detector_crossings.csv), interpolated with
`qmetrology.uncertainty.crossing`, the sweep's own rule, over the tested budgets bracketing the
crossing. Because the grid here is coarser than the sweep's, the honest reading is the *relative*
change between the two rows measured the same way:

| $\epsilon = 10^{-3}$ | budget for 90% | vs. brute force |
|---|---:|---:|
| published Algorithm 4 (this grid) | 42,297 | 1.085× |
| Eq. (3.6) at lag $w$ | **33,984** | **1.351×** |
| — for reference, the sweep's own values — | | |
| brute force | 45,900 | 1.000× |
| linear search, as published | 42,500 | 1.080× |
| binary search | 32,400 | 1.417× |
| reverse engineering | 29,800 | 1.540× |

A **19.7% reduction** in the budget linear search needs. Applied to the sweep's own 42,500 that is
about 34,100, i.e. a ratio near **1.34×** against brute force — closing roughly **80%** of the
distance between linear search and binary search. At $\epsilon = 10^{-4}$ the same comparison gives
2,871,632 → 2,745,767, a 4.4% reduction, moving the ratio from 1.577× to about 1.65×.

### Regime dependence

The gain is largest exactly where the notes locate the weakness — tight budget, low precision — and
smallest at $\epsilon = 10^{-4}$, where the published rule is already close to the ceiling. Section
4 derives why, and the reason is not a property of the cumulative mean but of the estimator.

---

## 4. The false-alarm argument is not $2^{-l}$ — it is far worse, and that is the point

The notes offer $1-(1-2^{-l})^{T-l+1}$ as "a crude illustration" and rightly say measured rates
should be used instead. Measured
([`linear_detector_streaks.csv`](linear_detector_streaks.csv)), the approximation is not merely
crude — it is wrong by orders of magnitude, in a way worth explaining:

| $l$ | measured $P(\text{stop before } N_\text{opt})$ | $1-(1-2^{-l})^{T-l+1}$ | the same with the measured $q$ |
|---:|---:|---:|---:|
| 1 | 95.2% | 100% | 100% |
| 4 | 78.2% | 72.3% | 35.8% |
| 8 | 66.4% | 6.0% | 0.8% |
| 12 | **63.6%** | 0.3% | 0.01% |

($B = 10{,}000$, $m' = 2$, $\mathrm{inc} = 1$; $T = 22.9$ safe comparisons per scan.) The
per-comparison fall probability is $q = 0.385$, not $\tfrac12$ — ties and the discreteness of the
estimator account for that. But even with the measured $q$, independence predicts a probability that
vanishes with $l$, while the measured probability **plateaus above 60%**.

### The mechanism

Consecutive falls are not near-independent, because the probes along a scan are not identically
distributed. On the safe branch the readout probability is $p_0 = \cos^2(N\phi)$, so a probe returns
$K = 0$ with probability $\sin^{2m'}(N\phi) = \sin^{2m'}(r\pi/2)$, where $r = N\phi/(\pi/2)$. As
$r \to 1^{-}$ that probability tends to one and the estimate stops being random at all: it becomes
the single atom $\pi/(2N)$, which **decreases with $N$**. In that zone *any* decrease-based rule
fires with probability approaching one while every probe is still safe — a false alarm by
definition, and an unavoidable one.

The zone's width is exact and scale-free
([`linear_degenerate_zone.csv`](linear_degenerate_zone.csv)): $P(K = 0) \ge \tfrac12$ for
$r \ge \tfrac{2}{\pi}\arcsin\!\big(2^{-1/2m'}\big)$, i.e. over the last

| $m'$ | 1 | 2 | 5 | 15 | 50 | 176 | 600 | 5565 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fraction of the depth range below $N_\text{opt}$ | 50.0% | 36.4% | 23.4% | 13.6% | 7.5% | 4.0% | 2.2% | 0.7% |

shrinking like $1/\sqrt{m'}$ — the small-angle form is
$1 - r \approx \tfrac{2}{\pi}\sqrt{\ln 2 / m'}$.

That one table explains the whole regime dependence:

- at the headline point, $m' = 2$: **36%** of the safe depth range is deterministic, the measured
  false-alarm rate is 77.7%, and the rule has real room to improve;
- at $B = 2{,}917{,}365$, $m' = 176$: **4%**, the false-alarm rate is 30.2%, the median false alarm
  stops **one** phase gate short, and no alternative rule finds much left to gain.

This is the same degeneracy [`OVERSHOOT_CRITERION.md`](OVERSHOOT_CRITERION.md) identifies as the
mechanism that makes the binary-search criterion *sharp*, seen from below the boundary instead of
above it. The two documents describe one phenomenon, and both should cite Eq. (2.85).

### What this does to the framing

The trial-level false-alarm rate is not a fitness measure for this algorithm, and the thesis should
stop implying that it is. At the headline point the bake-off's winner has the **highest**
false-alarm rate of any rule tested (93.9% against the published 77.7%) and the best convergence;
the rule with the *lowest* false-alarm rate (CUSUM, 19.9%) is nearly the worst. What matters is
where the stop lands and what survives the safeguard, and there the winner is better on every count:

| $B = 10{,}000$ | published | moving window |
|---|---:|---:|
| convergence | 57.49% | **64.68%** |
| trial-level false alarm | 77.7% | 93.9% |
| trial-level miss | 14.2% | **2.2%** |
| **final overshoot** $P(N^* > N_\text{opt})$ | 9.6% | **2.2%** |
| median exploration share | 4.4% | **3.0%** |

---

## 5. What switching would cost

The notes advise against replacing the detector because "all parameter tuning, convergence curves,
diagnostic tables, and held-out evaluations would need to be rerun". True — and worth pricing: the
production sweep is one command and, per [`../../sweep_max.log`](../../sweep_max.log), took
**307 minutes**. The implementation is already in place:
`find_phi_fixed_budget_linear_search_lagged` in `qmetrology/algorithms.py`, registered in
`FIXED_BUDGET` but deliberately **outside** `manifest.ORDER`, so nothing reported today is affected.
Promoting it needs a manifest entry with a tuning grid over
$(m', l, s, \mathrm{inc}, \text{lag}, \alpha)$ and one added key.

`tests/test_consolidated.py::test_lds_replay_reproduces_lagged_variant` asserts that this
sweep-ready implementation and the rule the bake-off scored are the same algorithm, so the measured
gain is the gain a re-run would deliver — with one caveat: the bake-off's tuner uses the exact
convergence probability as its objective and is less noisy than the production tuner. A re-run would
land slightly below these numbers for *every* arm, the published rule included.

What would change downstream: Table 4.1, the budget-to-reliability table, Appendix B.2, the
convergence figures, the diagnostics tables, the tuned-parameter appendix, and the prose in 4.2 and
4.5 describing linear search as the weakest adaptive method by a wide margin. The ordering of the
algorithms does not change.

---

## 6. Recommendation

Three coherent options, in increasing order of work.

**A — keep Algorithm 4, fix only the prose.** The notes' eight minimum changes, plus the additions
in Section 8 below. Defensible on its own, and it now comes with a *derivation* of the false-alarm
rate rather than an apology for it.

**B — keep Algorithm 4, and report the bake-off.** Add `tab_linear_detector` and
`fig_linear_detector` as a short subsection or appendix: "a moving-window rule was suggested; here
is what it actually is, what it is worth, and why we did not adopt it." This *answers* the critique
instead of deferring it. The notes' own standard is that a replacement "would have to specify and
validate a complete detector"; five complete detectors were specified and validated.

**C — adopt the lagged Eq. (3.6) rule and re-run.** About five hours of compute plus a day of
downstream edits, in exchange for linear search going from +1.25 pp to +7.3 pp over brute force at
the headline point, from 7% to 49% of the attainable gap, and from 1.08× to about 1.34× on the
$\epsilon = 10^{-3}$ budget ratio.

**My recommendation is B, and C if the schedule allows.** B is strictly better than A for the same
review effort, because it converts "we did not investigate the alternative" into "we investigated it
and here is the number". C is the scientifically strongest outcome and the least risky it will ever
be — the rule is implemented and tested, and it makes linear search and binary search share a single
overshoot criterion, differing only in search order, which is a better thesis.

The one thing I would **not** do is call a moving window future work. It is no longer untested, and
saying otherwise while [`linear_detector_bakeoff.csv`](linear_detector_bakeoff.csv) exists would be
the weakest available position.

---

## 7. The identity, for the thesis

Worth stating under any of the three options, because it is short, exact, and reframes the
criticism:

> Let $\bar\phi_k^{(w)}$ be the mean of the last $w$ probe estimates. For $k \ge w$ two consecutive
> windows differ only in the element that enters and the element that leaves, so
> $$\bar\phi^{(w)}_k - \bar\phi^{(w)}_{k-1} = \frac{\hat\phi_k - \hat\phi_{k-w}}{w},$$
> and the sign test $\bar\phi^{(w)}_k < \bar\phi^{(w)}_{k-1}$ is exactly
> $\hat\phi_k < \hat\phi_{k-w}$. A moving-window rule therefore performs no averaging: it compares a
> probe with the probe $w$ steps earlier, which is Equation (3.6) with
> $\hat\phi_{\mathrm{acc}} = \hat\phi_{k-w}$ and $\alpha = 1/2$. The cumulative-mean rule of
> Algorithm 4 is the same sign test against the running mean of the entire scan, i.e. against a
> reference whose effective lag grows with $k$.

Verified in `tests/test_consolidated.py::test_lds_moving_window_is_a_lagged_comparison`.

This also settles the "dilution by $1/k$" discussion. The notes are right that dividing by $k$
cannot change a sign. The real difference between the two rules is the **lag of the reference**: a
fixed lag $w$ gives every comparison the same lever arm across the aliasing boundary, whereas the
cumulative mean compares against an average whose effective lag keeps growing, so its comparisons
become steadily less local exactly as the scan approaches the boundary.

---

## 8. Prose fixes, consolidated

All eight of the notes' minimum changes stand and are confirmed against the code. Four additions:

1. **The false-alarm rate needs its mechanism, not just its value** (Section 4). Without it, Tables
   4.5/4.6 read as an admission; with it, they read as a measurement of a derived effect.
2. **Quote 77.7%, not only the 45.3% aggregate.** The notes are right that the headline point is
   worse; that is the number.
3. **`inc` matters more than the notes suggest**: 44 of 144 informative points select
   $\mathrm{inc} > 1$, so `N ← N − l·inc` is not an edge case.
4. **Section 2.7's forward reference** ("this property will be employed in Section 4") — already
   flagged in [`OVERSHOOT_CRITERION.md`](OVERSHOOT_CRITERION.md). The fold is now cited by both the
   binary-search criterion and the linear-search false-alarm analysis, so it should point at
   Chapter 3.

---

## 9. Artifacts

| File | What it is |
|---|---|
| [`linear_search_claims.csv`](linear_search_claims.csv) | every quantitative statement in the revision notes, recomputed, with a verdict |
| [`linear_detector_bakeoff.csv`](linear_detector_bakeoff.csv) | the tuned winner of every rule at every operating point, held out, with the full diagnostic set |
| [`linear_detector_matched.csv`](linear_detector_matched.csv) | the same with $m'$ and `inc` pinned to the published configuration |
| [`linear_detector_profile.csv`](linear_detector_profile.csv) | the best each rule reaches at every exploration size |
| [`linear_detector_grid.csv`](linear_detector_grid.csv) | the top of each rule's tuning landscape, so a losing rule can be seen to have been searched |
| [`linear_detector_streaks.csv`](linear_detector_streaks.csv) | measured false-alarm streak statistics against the $2^{-l}$ argument |
| [`linear_degenerate_zone.csv`](linear_degenerate_zone.csv) | the exact pre-boundary zone in which the estimator stops being random |
| [`linear_detector_crossings.csv`](linear_detector_crossings.csv) | the budget each rule needs for 90% convergence, on the sweep's own interpolation rule |
| [`linear_detector_validation.csv`](linear_detector_validation.csv) | the checks that the analytic replay reproduces the production sweep |
| [`tex/thesis/tab_linear_detector.tex`](tex/thesis/tab_linear_detector.tex) | ready-to-`\input` table |
| [`fig_linear_detector.png`](fig_linear_detector.png) / [`.pdf`](fig_linear_detector.pdf) | the four-panel summary |

New code: `analysis/consolidated/linear_detector_study.py`,
`analysis/consolidated/linear_search_claims.py`,
`qmetrology.algorithms.find_phi_fixed_budget_linear_search_lagged` (registered but deliberately
outside `manifest.ORDER`, so no reported number moves), `tab_linear_detector` in
`thesis_tables.py`, `fig_linear_detector` in `thesis_figures.py`, and four tests in
`tests/test_consolidated.py`. The full suite is 18/18 with `--slow`.
