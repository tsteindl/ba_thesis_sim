# Algorithm 5, updated: binary search with the deepest-accepted pilot

*Two forms below: a **readable** version whose math renders in the VS Code Markdown preview
(**Ctrl+Shift+V**), and the **LaTeX source** to paste into the thesis. `algorithmic` environments do not
render in Markdown, which is why the readable version is written out as a list.*

---

## 1. What changed, and why

| # | change | justification |
|---|---|---|
| 1 | The safeguard's pilot is the **deepest accepted probe** $(\hat\varphi_{\rm acc}, N_{\rm acc})$, not the opening probe $(\hat\varphi_0, N_0)$ | +0.21 pp where the bisection runs (median +0.18, wins 7/12); and it makes the bisection's output the input to Eq. (3.8), so binary search is no longer literally reverse engineering |
| 2 | $\sigma^2$ passed to the safeguard is $\dfrac{1}{4m'N_{\rm acc}^2}$, i.e. it tracks the pilot | required by 1; this is what makes the pilot $1.65\times$ sharper on average (up to $3.0\times$) |
| 3 | the safeguard searches $N \in [N_{\min}, N_{\max}]$ — upper end $N_{\max}$, **not** $L$; this resolves the `TODO: FIX` in the current draft | capping the search at $L$ costs **−2.76 pp** (50 points, R = 20,000). $L$ estimates the aliasing limit; it exceeds $N_{\rm opt}$ in ~20 % of trials, so it is not a safe ceiling. The search over $N$ is classical and free, so restricting it buys nothing and can only exclude the maximiser. The *lower* end may safely be raised to $N_{\min}$ — see §5. |
| 4 | The acceptance test is stated explicitly as **one-sided** at level $\alpha$ | it already is one-sided in the code; see §4 for why that is the correct choice and not merely the incumbent |
| 5 | $L$ and $U$ are kept, but only as **reported diagnostics** | $U$ was never used; $L$ is now reported (bracket quality) rather than consumed |

Everything else — the opening probe at $N_{\min}$, the arithmetic bisection step, the budget guard that
tests before spending, the termination condition, Eq. (3.4) and Eq. (3.8) themselves — is unchanged.

---

## 2. Readable form

**Procedure** `SimulatePhiBinarySearch(budget, `$\varphi_{\min}, \varphi_{\max}, m', \alpha, \varepsilon$`)`

1. $N_{\max} \gets \left\lfloor \dfrac{\pi}{2\varphi_{\min}} \right\rfloor$, $\quad N_{\min} \gets \left\lfloor \dfrac{\pi}{2\varphi_{\max}} \right\rfloor$, $\quad N \gets N_{\min}$
2. $\hat\varphi \gets \texttt{Simulate}(N, m')$, $\quad B_{\rm used} \gets m' N$
3. $\hat\varphi_{\rm acc} \gets \hat\varphi$, $\quad N_{\rm acc} \gets N$ — *the pilot: deepest probe not yet flagged*
4. $\varphi_1 \gets \hat\varphi + z_{\alpha}\,\sigma(N,m')$, where $\sigma(N,m') = \dfrac{1}{2N\sqrt{m'}}$ and $z_\alpha = \Phi^{-1}(\alpha)$ — *one-sided lower threshold*
5. $L \gets N_{\min}$, $\quad U \gets N_{\max}$
6. $N \gets N + \lfloor (U-N)/2 \rfloor$
7. **repeat**
   1. **if** $B_{\rm used} + m'N > \texttt{budget}$ **then break** — *test the budget before spending it*
   2. $\hat\varphi \gets \texttt{Simulate}(N, m')$, $\quad B_{\rm used} \gets B_{\rm used} + m'N$, $\quad N_{\rm old} \gets N$
   3. **if** $\hat\varphi < \varphi_1$ **then** — *flagged as an overshoot*
      - $N \gets N - \lfloor (N-L)/2 \rfloor$, $\quad U \gets N_{\rm old}$
   4. **else** — *accepted; this probe becomes the pilot*
      - $\hat\varphi_{\rm acc} \gets \hat\varphi$, $\quad N_{\rm acc} \gets N_{\rm old}$
      - $N \gets N + \lfloor (U-N)/2 \rfloor$, $\quad L \gets N_{\rm old}$
      - $\varphi_1 \gets \hat\varphi + z_\alpha\,\sigma(N, m')$ — *recomputed at the depth of the next probe*
   5. **until** $N_{\rm old} = N$ **or** $N < N_{\min}$ **or** $N > N_{\max}$ **or** $B_{\rm used} \ge \texttt{budget}$
8. $\sigma_{\rm acc}^2 \gets \dfrac{1}{4m'N_{\rm acc}^2}$
9. $B_{\rm rem} \gets \texttt{budget} - B_{\rm used}$
10. $N^\star \gets \texttt{StatisticalSafeguard}\big(B_{\rm rem},\ N_{\min},\ N_{\max},\ \varepsilon,\ \hat\varphi_{\rm acc},\ \sigma_{\rm acc}^2,\ \varphi_{\min},\ \varphi_{\max}\big)$
11. $m \gets \lfloor B_{\rm rem}/N^\star \rfloor$
12. $\hat\varphi \gets \texttt{Simulate}(N^\star, m)$
13. **return** $\hat\varphi$

where the safeguard is Eq. (3.8),

$$
N^{\star} \;=\; \arg\max_{N_{\min} \le N \le N_{\max}}\;
\underbrace{P\!\left(\varphi < \tfrac{\pi}{2N} \;\middle|\; \hat\varphi_{\rm acc}, \sigma_{\rm acc}\right)}_{\text{no aliasing}}
\;\cdot\;
\underbrace{\Big[\,2\,\Phi\!\big(2\varepsilon\sqrt{N B_{\rm rem}}\,\big) - 1\,\Big]}_{\text{sufficient precision}} .
$$

---

## 3. LaTeX source

```latex
\begin{algorithm}
    \caption{Simulate $\phi$ using Binary Search}
    \label{alg:binary-search}
    \begin{algorithmic}
        \Procedure{SimulatePhiBinarySearch}{\texttt{budget}, $\phi_\text{min}, \phi_\text{max}, \texttt{m}', \alpha, \epsilon$}
        \State $N_\text{max} \gets \left\lfloor\frac{\pi}{2 \phi_\text{min}}\right\rfloor,\ N_\text{min} \gets \left\lfloor\frac{\pi}{2 \phi_\text{max}}\right\rfloor,\ N \gets N_\text{min}$

        \State $\hat \phi \gets \Call{Simulate}{N, m'}$
        \State Update budget used

        \State $\hat\phi_\text{acc} \gets \hat\phi,\ N_\text{acc} \gets N$ \Comment{Pilot: deepest probe not classified as an overshoot}

        \State Calculate $\phi_1$ such that $\mathbb{P}\!\left(\hat\phi \leq \phi_1\right) = \alpha$, using $\hat\phi \sim \mathcal{N}\!\left(\hat\phi_\text{acc}, \tfrac{1}{4m'N^2}\right)$
        \Comment{One-sided: an aliased probe can only read low}
        \State

        \State $L \gets N_\text{min},\ U \gets N_\text{max}$ \Comment{Initialize lower and upper bound for $N$}
        \State $N \gets N + \lfloor (U-N)/2 \rfloor$ \Comment{Update $N$ to midpoint of the set $\{N,\ldots, U\}$}

        \Do
           \State \textbf{if} the next probe does not fit in the remaining budget \textbf{then break}
           \State $\hat \phi \gets \Call{Simulate}{N, \texttt{m}'}$
           \State Update budget used

           \State $N_\text{old} \gets N$ \Comment{Temporary $N$}

           \If {$\hat \phi < \phi_1$}
                \State $N \gets N - \lfloor (N - L)/2 \rfloor$ \Comment{Update $N$ to midpoint of the set $\{L,\ldots, N\}$}
                \State $U \gets N_\text{old}$ \Comment{Update upper bound for $N$}
           \Else
                \State $\hat\phi_\text{acc} \gets \hat\phi,\ N_\text{acc} \gets N_\text{old}$ \Comment{Sharper pilot: $\sigma \propto 1/N_\text{acc}$}
                \State $N \gets N + \lfloor (U - N) / 2 \rfloor$ \Comment{Update $N$ to midpoint of the set $\{N, \ldots, U\}$}
                \State $L \gets N_\text{old}$ \Comment{Update lower bound for $N$}
                \State
                \State Recompute $\phi_1$ using the updated value of $N$ and the corresponding variance
           \EndIf

        \doWhile {$N_\text{old} \neq N$}

        \State
        \State $\sigma_\text{acc}^2 \gets \frac{1}{4m'N_\text{acc}^2}$
        \State $B_\text{rem} \gets \texttt{budget} - B_\text{used}$
        \State $N^* \gets \Call{StatisticalSafeguard}{B_\text{rem}, N_\text{min}, N_\text{max}, \epsilon, \hat\phi_\text{acc}, \sigma_\text{acc}^2, \phi_\text{min}, \phi_\text{max}}$
        \Comment{$N_\text{max}$, not $L$: see text}

        \State Calculate shots $m$ such that remaining budget is exhausted
        \State $\hat \phi \gets \Call{Simulate}{N^*, \texttt{m}}$

        \State \Return $\hat\phi$

        \EndProcedure
    \end{algorithmic}
\end{algorithm}
```

---

## 4. Results

_All numbers in §4–§6 come from one run: `analysis/fine_sweep.py` (23 scenarios × 14 budgets = 308
operating points, tuned seed 42 at R=600/1500, validated seed 2024 at R=20,000) with diagnostics
re-measured at those winners by `analysis/binary_final_stats.py`. Brute force and the omniscient
ceiling `oracle_hl` carry over from `story_curves.csv` — neither has a tuning parameter._

### 4.1 Where binary search now stands

| | mean convergence | share of the ceiling |
|---|---:|---:|
| omniscient ceiling (`oracle_hl`) | 64.48 % | — |
| **reverse engineering** | **61.97 %** | **96.1 %** |
| **binary search, deepest-accepted pilot** | **61.65 %** | **95.6 %** |
| linear search | 56.55 % | 87.7 % |
| brute force | 50.67 % | 78.6 % |

Paired over all 308 points:

| comparison | mean | median | wins |
|---|---:|---:|---:|
| binary − reverse engineering | **−0.32 pp** | −0.06 pp | 40 % |
| binary − linear | +5.10 pp | +4.52 pp | 100 % |
| reverse engineering − linear | +5.42 pp | +4.94 pp | 99 % |

Budget ratio against brute force (how much *less* budget is needed to first reach `p*`), pooled:

| p* | Linear | **Binary** | Reverse engineering |
|---|---:|---:|---:|
| 50 % | 1.51× | **2.08×** | 2.12× |
| 75 % | 1.40× | **1.87×** | 1.94× |
| 90 % | 1.30× | **1.63×** | 1.63× |

Binary search and reverse engineering are **statistically tied**; both are clearly ahead of linear
search and brute force. Per-scenario ratios are in `results/RESULTS_UPDATED.md`.

### 4.2 The reporting statistics

| statistic | value |
|---|---|
| convergence rate | **61.65 %** |
| improvement over brute force | **+10.98 pp** mean, +11.17 pp median |
| share of the omniscient ceiling | **94.6 %** mean, 97.8 % median, 61.4 % worst |
| overshoot rate ($N^\star > N_{\rm opt}$) | **0.76 %** mean, 0.20 % median, 9.04 % worst |
| exploitation depth $N^\star/N_{\rm opt}$ | 0.922 mean, 0.952 median |
| mean distance from the aliasing limit | $\lvert N^\star - N_{\rm opt}\rvert / N_{\rm opt}$ = **8.1 %**; within 5 % of $N_{\rm opt}$ in 60.6 % of trials |
| exploration budget share | **5.7 %** mean, 3.0 % median, 24.7 % worst |
| probes per trial | 6.27 (of which 2.23 accepted) |
| pilot depth $N_{\rm acc}/N_{\min}$ | **1.69×** mean, 3.87× max |
| pilot itself aliased | 8.16 % of trials |
| bisection lower bound $L/N_{\rm opt}$ | 0.714 |

### 4.3 What changed against the published sweep

Two things move the numbers: the deepest-accepted pilot (§1) and a grid whose bounds cannot bind (§6).

| algorithm | mean | median | max gain | max loss | points gaining >1 pp |
|---|---:|---:|---:|---:|---:|
| binary search | **+0.88 pp** | +0.14 | **+54.67** | −3.47 | 10 % |
| reverse engineering | +0.64 pp | −0.04 | +47.73 | −2.04 | 6 % |
| linear search | +0.10 pp | +0.02 | +10.89 | −3.61 | 10 % |

Medians near zero with very large tails — the signature of a grid boundary that binds at a minority of
points and badly when it does. The two scenarios the published Table C reports as *losing* to brute
force were entirely that artifact; at $U(10^{-3},10^{-2})$, $\varepsilon=10^{-3}$, $B=3{,}643$:

| | published | now |
|---|---:|---:|
| brute force | 82.74 | 82.74 |
| linear search | 64.44 | 75.33 |
| reverse engineering | 28.08 | **75.78** |
| binary search | 28.08 | **82.19** |

$U(10^{-4},10^{-3})$, $\varepsilon=10^{-4}$ behaves identically. Adaptive methods still do not *beat*
brute force in that corner, but "no advantage" replaces "catastrophic failure".

**Broad priors are unaffected.** Re-running Table 3.4's scenarios ($\varphi_{\max}=\pi/4,\ \pi/2$, so
$N_{\min}=1$–2) on the same boundary-free grid moves binary by −0.16 pp, reverse engineering by
+0.00 pp and linear by −0.14 pp, with **zero boundary hits** (`analysis/broad_fine.py`,
`results/broad_fine.csv`). Table 3.4 stands as published.

---

## 5. How often does the bisection actually bisect?

This is the number a discussion section must state, because it is the first thing an examiner will ask.

> **At 174 of 308 operating points (56.5 %) the exploration genuinely bisects. At the other 134
> (43.5 %) it takes exactly one probe — which is reverse engineering.**

| | points | probes | pilot depth | exploration share | $N^\star/N_{\rm opt}$ | overshoot | rate | vs RE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| genuinely bisecting | 174 | **10.32** | **2.22×** | 2.6 % | 0.960 | 0.62 % | 62.60 % | **−0.76 pp** |
| single probe | 134 | 1.00 | 1.00× | 9.7 % | 0.872 | 0.93 % | 60.41 % | +0.24 pp |

Two honest consequences. Where the bisection runs it costs about **0.8 pp** against reverse
engineering — the exploration is not free and the deeper pilot does not repay it. Where it does not
run, binary search *is* reverse engineering and the +0.24 pp is tuning noise. The overall tie of
−0.32 pp is the average of those two regimes.

### 5.1 It is declined, not blocked

The obvious reading — "there was not enough budget for bisection steps" — is measurably wrong.
Sampling the degenerate points and asking whether *any* affordable $m'$ would yield ≥ 4 probes:

| | |
|---|---|
| bisection possible at some $m'$ | **33 of 34** |
| genuinely impossible | 1 of 34 |
| cost of forcing ≥ 4 probes | **−15.65 pp** mean, −11.10 pp median, −0.32 pp best case |
| budget rank of degenerate points within their scenario | mean 0.48; **44 % lie in the top half** |

If it were a budget floor, single-probe points would cluster at the bottom of each scenario's budget
range. They do not — several scenarios take one probe at *every* budget, including the largest tested.
The tuner can afford to bisect and declines because bisecting is worse.

### 5.2 What actually decides it: tolerance, and prior width

| $\varepsilon$ | bisects at |
|---|---:|
| $10^{-3}$ | 0 % of budgets |
| $10^{-4}$ | 21 % |
| $10^{-5}$ | 62 % |
| $10^{-6}$ | 72 % |
| $10^{-7}$, $10^{-8}$ | 100 % |

| prior width $\varphi_{\max}/\varphi_{\min}$ | bisects at |
|---|---:|
| 10 | 70 % of budgets |
| 100 | 32 % |
| $\ge 1000$ | **0 %** |

**Tolerance.** At loose $\varepsilon$ the target precision is reached well below the aliasing limit
($N^\star/N_{\rm opt} \approx 0.66$–$0.78$), so the limit never binds and there is nothing for a
bisection to locate. The transition sits between $10^{-4}$ and $10^{-5}$.

**Prior width, which overrides tolerance.** At $\varphi_{\max}/\varphi_{\min} \ge 1000$ the exploration
never bisects at *any* $\varepsilon$ or budget — including $U(10^{-4},10^{-1})$ at
$\varepsilon = 10^{-6}$, where the tolerance is tight. The cause is the arithmetic step: the first jump
goes to $(N_{\min}+N_{\max})/2$, roughly $500\,N_{\min}$ at that width, and affording it forces $m'$ so
small that the pilot is useless. One good probe wins. At width 100 the two effects compete and the
bisection appears only at the higher budgets.

*A one-line change would remove the second effect: stepping to the geometric mean of the bracket
instead of the arithmetic one restores the bisection at wide priors (1.00 → 8.07 probes at
$U(10^{-4},10^{-1})$). It does not change the conclusion — measured over 50 operating points it is
worth −0.45 pp — because the depth decision is already saturated (§4.1: 95.6 % of the ceiling).*

### 5.3 The pilot is aliased about a quarter of the time, and it does not matter

Where the bisection runs, the deepest accepted probe sits above $N_{\rm opt}$ in 26.8 % of trials
(8.16 % pooled over all points). That sounds fatal — an aliased estimate is the folded value, not a
noisy one — but the fold is marginal, and the acceptance test is what keeps it marginal.

| | aliased pilot | clean pilot |
|---|---:|---:|
| share of trials | 26.8 % | 73.2 % |
| $N_{\rm acc}/N_{\rm opt}$ | 1.023 (median 1.016, p95 1.062) | 0.908 |
| pilot bias $\hat\varphi_{\rm acc}/\varphi$ | **0.984** | 1.012 |
| chosen depth $N^\star/N_{\rm opt}$ | 0.984 | 0.967 |
| final overshoot | 2.78 % | 0.00 % |
| **convergence** | **55.07 %** | 51.19 % |

87.6 % of aliased pilots lie within 5 % of $N_{\rm opt}$, 99.9 % within 20 %, and **none beyond 2×**.
The mechanism: by §4 an aliased probe satisfies $\hat\varphi \le \pi/2N < \varphi$ — it can only read
*low* — and the acceptance test rejects low readings, so a badly folded probe never survives
acceptance. At $N_{\rm acc} = (1+\delta)N_{\rm opt}$ the folded value is $\varphi(1-\delta)/(1+\delta)$,
so $\delta \approx 0.02$ reproduces the measured 1.6 % bias.

Trials with an aliased pilot converge **+3.88 pp better**, because an aliased pilot is the *signature*
of a bracket that closed tightly onto $N_{\rm opt}$: those trials operate 1.7 % closer to the limit and
pay 2.78 % overshoot for it.

---

## 6. Grid bounds and how far the search was verified

The published sweep tunes $m'$ over `geomspace(20, m_hi)` for binary search and reverse engineering,
`geomspace(3, m_hi)` for linear, with `m_hi = clip(brute90, 200, 300000)`. In `story_winners.csv` the
winner sits **on** an endpoint at a large share of points, which means the optimum was outside the box:

| algorithm | at grid minimum | at grid maximum |
|---|---:|---:|
| linear search | 37.5 % | 1.3 % |
| reverse engineering | 32.1 % | 1.5 % |
| binary search | 23.5 % | 3.8 % |

The replacement bounds cannot bind for any reason except a genuine optimum:
$m' \in [\,1,\ \lfloor \texttt{budget}/N_{\min}\rfloor\,]$ — one shot at the bottom, and at the top the
largest pilot that can be afforded at all, since a probe at the opening depth costs $m'N_{\min}$. Resolution comes from a
two-stage search (coarse log sweep, then a refinement by ×4 either side of the winner at higher $R$).
Afterwards:

| algorithm | at $m'=1$ | at $m'=\lfloor B/N_{\min}\rfloor$ |
|---|---:|---:|
| binary search | **0.0 %** | 0.0 % |
| reverse engineering | 1.3 % | 0.0 % |
| linear search | 15.6 % | 0.0 % |

Linear's remaining hits are at the *physical* minimum of one shot, so they are genuine optima rather
than truncated searches; the upper endpoint is never selected.

**Was the resolution enough?** At sampled points the winner was re-tuned against a dense local grid —
25 values of $m'$ spanning ±a factor of 2.5, × 11 confidence levels, at 3× the tuning $R$ — and
validated on the held-out seed. The dense winner beat the two-stage winner by **+0.00 to +0.10 pp**
(one −0.49 pp outlier is test-seed noise). Nothing is being left on the table.

---

## 7. Why the test stays one-sided

The estimator is $\hat\varphi = \frac{1}{N}\arccos\sqrt{k/m}$, whose range is $[0, \pi/2N]$ — the sawtooth.
The relevant fact is not that it folds, but **which way** it folds:

$$
N > N_{\rm opt} \;\Longleftrightarrow\; \varphi > \frac{\pi}{2N}
\qquad\text{and}\qquad
\hat\varphi \le \frac{\pi}{2N} \ \ \text{always},
$$

so an aliased probe satisfies $\hat\varphi \le \pi/2N < \varphi$: **every tooth of the sawtooth lies below
the true $\varphi$.** Aliasing can only push the estimate *down*. The alternative hypothesis is therefore
strictly one-sided, and a two-sided test spends half its $\alpha$ on an upper rejection region where the
alternative has no mass.

Measured, over 167,264 probes at the tuned configurations: the two-sided **upper** region fires on
**3.70 %** of probes while the lower one fires on **65.01 %**. And in convergence:

| test | vs. shipped one-sided |
|---|---:|
| two-sided at the same $\alpha$ | **−0.08 pp** (SE 0.10) |
| one-sided with $\sigma = \sqrt{\sigma_{\rm prev}^2 + \sigma_{\rm new}^2}$ | −0.03 pp |

Both are null. Since performance does not decide it, the principle does: **keep the one-sided test.**
It is worth one sentence in the thesis that the two-sided variant was tried and is indistinguishable —
that converts an unexamined default into a tested choice.

---

## 8. Why the search may start at $N_{\min}$ but must not stop at $L$

**Lower end — free, and provably so.** With the prior support enforced (the truncated form of the
overshoot factor), for every $N \le N_{\min}$

$$
\frac{\pi}{2N} \;\ge\; \frac{\pi}{2N_{\min}} \;\ge\; \varphi_{\max}
\qquad\Longrightarrow\qquad
P\!\left(\varphi < \tfrac{\pi}{2N}\right) = 1 \ \text{exactly,}
$$

while the precision factor $2\Phi(2\varepsilon\sqrt{NB_{\rm rem}})-1$ is strictly increasing in $N$. The
objective is therefore non-decreasing on $[1, N_{\min}]$ and its maximiser is never strictly below
$N_{\min}$. Checked exhaustively at 4,000 random operating points:

| | count |
|---|---:|
| identical $N^\star$ from $[1,N_{\max}]$ and $[N_{\min},N_{\max}]$ | 3,060 |
| differ, but the objective is numerically identical at both (saturated plateau) | 940 |
| **differ with a genuinely better $N^\star$ below $N_{\min}$** | **0** |

Restricting is in fact slightly *better* than not: on the saturated plateau — where
$2\varepsilon\sqrt{NB_{\rm rem}} \gtrsim 8$ and the precision factor is $1.0$ in double precision across a
wide range of depths — `argmax` otherwise returns an arbitrary shallow depth, which distorts the
reported $N^\star/N_{\rm opt}$ without affecting convergence.

**Upper end — not free.** $L$ is an *estimate of the aliasing limit*, not a depth below it, and it
exceeds $N_{\rm opt}$ in about 20 % of trials. Capping the search there costs **−2.76 pp** on average
(50 operating points, R = 20,000; worst −14.97 pp). Since the search is classical, the cap saves
nothing and can only remove the maximiser.

*(Aside, same check: the bracketing search in `risk_optimal_depth` — three geometric refinement rounds
followed by an exact integer scan — returned the exhaustive-scan optimum in 4000/4000 cases, so its
lack of a unimodality guarantee is a theoretical caveat rather than a practical one.)*
