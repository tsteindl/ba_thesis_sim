# Overshoot criterion: the verdict, and exactly what to do

> Recommendations only — no `.tex` file was touched.
> Full derivation and numerical verification: [`BERRY_ESSEEN.md`](BERRY_ESSEEN.md).

---

## VERDICT: yes, the problem is solved

**Your criterion does not need Gaussianity to be justified, and it never did.** It is already an
exact statistical test — it was just never written as one. Nothing about the method, the threshold
formula, `conf`, the code, or any number in Chapter 4 has to change. What changes is roughly one
paragraph of Chapter 3 and one passage of Chapter 2.

The justification has three parts, and only the third involves an approximation:

1. **The test is exactly a one-sided binomial test.** `phi_hat = arccos(sqrt(K/m'))/N` is strictly
   *decreasing* in the count `K`, so `phi_hat_N < phi_1` is *identical* to `K > m' cos^2(N phi_1)`.
   Your rule is a cut on the raw count. Under the null "N is safe" the count is `Bin(m', cos^2(N phi))`,
   so the false-alarm rate is an exact binomial tail probability. No distributional assumption at all.
   *(Verified: 0 disagreements in 300,000 random draws.)*

2. **Its power needs no approximation either.** `arccos` returns a value in `[0, pi/2]`, so every
   estimate obeys `N phi_hat <= pi/2`, while overshooting means `N phi > pi/2`. Hence at any
   `N > N_opt` and for *every* outcome, `phi_hat <= pi/(2N) < phi` — an overshooting probe always
   reads below the truth. This is the finite-`N` content of your `phi_hat -> 0` limit.
   *(Verified: 200,000 overshooting draws, never positive.)*

3. **The Gaussian survives in one role only — placing the cut — and that use is bounded by a
   theorem.** Since the rule is a cut on `K`, the only thing that must be Gaussian is `K`, a sum of
   `m'` i.i.d. Bernoulli variables. Berry–Esseen bounds that error by
   `0.4748 (p^2+q^2)/sqrt(m' p q)`, **non-asymptotically, for every `m'`** — no "for `m` large
   enough". Measured: the achieved size stays within 0.04–0.08 of nominal in the regularity region.

   *Why `K` and not `phi_hat`?* Asymptotically the delta method does carry Gaussianity from `K` to
   `phi_hat` — that is your Lemma 2.6.3, and it is correct. But Berry–Esseen's value is that it is
   **finite-sample**, and finite-sample accuracy does not pass through a nonlinear map for free:
   `phi_hat`'s distance is measured against a different normal, and bounding it would need the
   curvature of `arccos(sqrt(p))`, which blows up at the aliasing boundary. Numerically the two
   distances agree to within 0.6 % inside the regularity region and diverge outside it (0.49 vs
   0.63 at `N = 0.98 N_opt`). Routing through `K` turns an empirical "close enough" into a bound.
   Full comparison in [`BERRY_ESSEEN.md`](BERRY_ESSEEN.md) Part 2c.

That is the answer to the criticism. You are no longer claiming `phi_hat` is Gaussian (which is
false near the boundary and was never provable); you are claiming `K` is approximately Gaussian,
which is a Bernoulli sum with a classical error bound.

**Should you replace the criterion with an exact binomial test and re-sweep? No.** The error budget
says it would fix the smaller of two errors: at `conf = 0.95` the normal-placed cut costs 0.015 of
miscalibration while the noisy reference costs 0.319 — 21× more — and an exact test does not touch
the reference. `conf` is grid-searched anyway, so exact calibration renames the knob without moving
the setting. Reasoning in [`BERRY_ESSEEN.md`](BERRY_ESSEEN.md) Part 6.

---

## WHAT TO DO — four required edits

### EDIT 1 (required) — Chapter 2, Section 2.7

**Replace** this passage (LaTeX source — shown as code so you can copy it):

```latex
Because $\arccos$ is bounded by $|\arccos(x)| \leq \pi$, the entangled estimator satisfies
\begin{align}
\lim_{N\to\infty} \hat\phi_\text{ent}^{(N)} = \lim_{N\to\infty} \frac 1 N  \arccos(\sqrt {p_0}) = 0,
\label{eq:est-conv}
\end{align}
implying that the estimator converges to zero. Once the optimal
choice of $N$ is exceeded, we can expect estimates to tend to zero.
```

**with** this:

```latex
Because $\sqrt{p_0} \in [0,1]$ and $\arccos$ maps $[0,1]$ onto $[0,\pi/2]$, every estimate obeys
\begin{align}
  0 \;\leq\; \hat\phi_{\text{ent}}^{(N)} \;=\; \frac{1}{N}\arccos\!\bigl(\sqrt{p_0(N)}\bigr)
    \;\leq\; \frac{\pi}{2N},
  \label{eq:est-bound}
\end{align}
where $p_0(N) = \cos^2(N\phi)$. The numerator is bounded while $1/N$ vanishes, so
\begin{align}
  \lim_{N\to\infty} \hat\phi_{\text{ent}}^{(N)} = 0,
  \label{eq:est-conv}
\end{align}
although the approach is not monotone: by \eqref{eq:folded-estimator} the estimator zig-zags within
the envelope $\pi/(2N)$. For what follows, \eqref{eq:est-bound} is the more useful statement,
because it holds at every finite $N$: as soon as $N\phi > \pi/2$, that is as soon as $N$ exceeds
$N_{\mathrm{opt}}$, it gives $\hat\phi_{\text{ent}}^{(N)} \leq \pi/(2N) < \phi$, so an overshooting
probe necessarily reports a phase below the true one.
```

**Three things changed and why:**

- `pi` became `pi/2`. Not cosmetic — the factor of two is what makes
  `phi_hat <= pi/(2N) < phi` close. With `pi` the detector argument cannot be completed at all.
- `p_0` is written as `p_0(N)`, and the limit is a squeeze. As written before, `arccos(sqrt(p_0))`
  was pulled outside the limit as though constant; it is not, and it has no limit — it oscillates
  forever. What makes your conclusion true is that it stays *bounded*.
- The last sentence is replaced. Your limit is correct, but "once `N_opt` is exceeded, we can expect
  estimates to tend to zero" attaches an `N -> infinity` conclusion to the event `N > N_opt`. At
  `N = 1.02 N_opt` the estimate is 2 % below `phi`, not near zero — and that is exactly where the
  rule decides. The inequality is what holds there.

**Also in this edit:** add `\label{eq:folded-estimator}` to Eq. (2.85) (the
`arccos|cos(N phi)|/N` equation), because the text above cites it. Consider moving Eq. (2.85) up to
sit just after this passage, where the non-monotonicity is first mentioned.

### EDIT 2 (required) — Chapter 3, Section 3.2.2

Add `\label{eq:overshoot-threshold}` to Equation (3.6). Edit 3 and the table caption both cite it.

### EDIT 3 (required) — Chapter 3, Section 3.2.2, immediately after Eq. (3.6)

**Insert** this paragraph (126 words) — copy into your `.tex`:

```latex
Although \eqref{eq:overshoot-threshold} is written through a normal quantile, the underlying test is
exact. Since $\hat\phi_N = \arccos(\sqrt{K/m'})/N$ is strictly decreasing in the count $K$, the event
$\hat\phi_N < \phi_1$ is identical to $K > m'\cos^2(N\phi_1)$, so the rule is a cut on the raw count
and its size under the null ``$N$ is safe'' is an exact binomial tail probability. The normal
approximation is therefore required only for $K$ itself, a sum of $m'$ i.i.d.\ Bernoulli variables,
for which the Berry--Esseen theorem bounds the error by
$C\,(p_0^2+(1-p_0)^2)/\sqrt{m'p_0(1-p_0)}$ with $C \leq 0.4748$, non-asymptotically and for every
$m'$. The rule's power requires no approximation at all: by \eqref{eq:est-bound},
$\hat\phi_N \leq \pi/(2N) < \phi$ at every $N > N_{\mathrm{opt}}$ and for every outcome, so an
overshooting probe always reads below the true phase.
```

You will need a citation for Berry–Esseen. Shevtsova (2011) for the constant `C <= 0.4748`; any
standard probability text (e.g. Durrett, *Probability: Theory and Examples*) for the theorem itself.

**Define `p_0` where you first use it**, since the paragraph relies on it: `p_0 = cos^2(N phi)` is
the true probability of reading out `0` at the candidate depth. It is unknown, which is fine and
worth one clause: the *test* never uses it (the cut `m' cos^2(N phi_1)` involves only `N`, `m'` and
the reference), and `p_0` enters only in stating how often the rule is wrong. The null
`H_0: N <= N_opt` is composite, so the rejection probability is a function of the operating point
and the test's size is its supremum over the safe range — which is what the table's interval
reports. See [`BERRY_ESSEEN.md`](BERRY_ESSEEN.md) Part 2b.

### EDIT 4 (required) — Chapter 3, Section 3.2.2, after that paragraph

`\input` the generated table:

```latex
\input{tables/tab_overshoot_operating}
```

Source: [`results/tex/tab_overshoot_operating.tex`](../results/tex/tab_overshoot_operating.tex).
It reports, per `m'`, the Berry–Esseen bound, the distance actually attained, and the largest gap
between the achieved size and the nominal level, for each `conf`. One sentence introducing it:

```latex
Table~\ref{tab:overshoot-operating} evaluates both the bound and the size the threshold actually
attains. The residual gap does not shrink with $m'$ because it reflects the discreteness of $K$
rather than a central-limit error.
```

---

## Recommended, but separable

### EDIT 5 — Chapter 2, Lemma 2.6.3

Add the hypothesis `0 < p_0 < 1` to the statement. The proof already needs it: its
`g'(p) = -1/(2N sqrt(p(1-p)))` is singular at the endpoints, and Theorem 2.6.2 requires `g'` finite
and non-zero. The proof itself needs no other change. Full wording in
`../binary_gaussianity/GAUSSIANITY_HANDOFF.md`.

This matters independently of the overshoot criterion, because Lemma 2.6.3 is what the *safeguard*
(Theorem 3.2.1) consumes.

### EDIT 6 — Chapter 2, Section 2.7

The sentence after Eq. (2.84) says the property "will be employed in Section 4". Overshoot detection
is defined in Chapter 3.

### EDIT 7 — Chapter 3, closing Section 3.2.2

Optional closing sentence. It is the strongest single argument that the rule is sound rather than
lucky, and it is currently nowhere in the thesis:

```latex
The rule needs a precise reference exactly when the decision is close, and binary search supplies
one automatically: the bracket narrows as the search proceeds, so the probes landing near
$N_{\mathrm{opt}}$ are the late ones, by which point $N_{\mathrm{acc}}$ is within a few percent of
$N$. For $\phi = 0.02$ the first probe compares against a reference $5.7$ times shallower but faces
a decision at $N = 1.10\,N_{\mathrm{opt}}$, whereas the probes at $1.006$ and $0.980$ arrive with
$N/N_{\mathrm{acc}} \approx 1.03$.
```

---

## Explicitly leave alone

- **The existing caveat sentence in 3.2.2** — *"Because the unknown mean is replaced by a noisy and
  adaptively selected estimate, and because the uncertainty of this reference estimate is not
  included, the procedure is a heuristic classification rule rather than an exact confidence
  test."* Keep it verbatim. It is still correct and now better supported: the test is exact *given*
  the threshold, and what is not exact is the unconditional level, because the threshold is
  estimated. It also already discloses the reference selection bias, so **nothing needs adding
  about that**, as you asked.
- **The criterion, the threshold formula, `conf`, and `qmetrology/algorithms.py`.**
- **Every tuned parameter and every number in Chapter 4, Appendix B and Appendix C.**
- **Lemma 2.6.3's proof, the delta method, Eqs. (3.4)–(3.6), Theorem 3.2.1, the safeguard.**

## Unrelated, found along the way

Section 4.2's prose still says binary search performs poorly and that "all adaptive strategies
except binary search beat the baseline", but Table 4.1 now has BS second best — 65.07 %, and 32,400
for >90 % convergence against the baseline's 45,900. Chapter 5 repeats the old figures
("60.3 % versus 56.2 %", BS "fell behind"). Nothing to do with this analysis, but worth fixing.

---

## Evidence behind each claim

| Claim | How it was checked |
|---|---|
| the rule is exactly a binomial cut | 0 disagreements over 300,000 random `(m', N, phi_1, K)`; asserted at runtime |
| an overshooting probe always reads low | 200,000 random overshooting draws, `max(phi_hat - phi) < 0`; asserted |
| Berry–Esseen bound is respected | true distance vs bound at every `m'`; asserted |
| the two-part error bound holds | 294 grid points in the regularity region, 0 violations; asserted |
| achieved size vs nominal | exact binomial, no simulation |

```
python analysis/overshoot_criterion.py     # all four assertions
python analysis/thesis_tables.py           # -> tab_overshoot_operating.tex
```

| File | Contents |
|---|---|
| [`BERRY_ESSEEN.md`](BERRY_ESSEEN.md) | full derivation, the closed-form threshold map, the re-sweep decision |
| [`overshoot_size.csv`](../results/overshoot_size.csv) | achieved vs nominal size, Berry–Esseen bound, true distance |
| [`overshoot_error_bound.csv`](../results/overshoot_error_bound.csv) | the two-part decomposition point by point |
| [`overshoot_power.csv`](../results/overshoot_power.csv) | exact `P(rule fires)` vs depth |
| [`overshoot_bracket_walk.csv`](../results/overshoot_bracket_walk.csv) | the probe sequences behind Edit 7 |
| [`results/tex/tab_overshoot_operating.tex`](../results/tex/tab_overshoot_operating.tex) | the table for Edit 4 |
| [`fig_overshoot_criterion.png`](../results/fig_overshoot_criterion.png) | optional figure; **recommended omitted** — the table carries the claim |
