# The statistical safeguard — derivation and its assumptions

Companion to [`qmetrology/safeguard.py`](../qmetrology/safeguard.py). Renders with KaTeX in the
VS Code markdown preview and on GitHub. This is the single, complete write-up: assumptions,
derivation, the truncation question, and every measurement behind them.

To *play* with the rule rather than read about it, open
[`notebooks/safeguard_playground.ipynb`](../notebooks/safeguard_playground.ipynb) — self-contained
(no imports from this repo), and it draws the optimisation landscape the rule maximises.

Everything below concerns one decision: after the exploration phase has produced a pilot estimate
$\hat\phi_0$, at what circuit depth $N$ should the remaining budget be spent? The published
algorithms answer with a grid-tuned constant ($C_{\text{safe}}$ for reverse engineering, $s$ for
binary search). This document derives the answer instead, and — the point of §3 — is explicit about
what has to be true for the derivation to mean anything.

---

## 0. The core assumptions, up front

| | Assumption | Status |
|---|---|---|
| **A1** | The estimator is asymptotically normal with $\operatorname{Var}(\hat\phi)=\dfrac{1}{4N^2m}$, **independent of $\phi$** | *Exact* to leading order; verified numerically in §2. This is Eq. (3.4) of the thesis. |
| **A2** | The pilot likelihood is **unimodal** over the prior support — no aliasing ambiguity | *Exact by construction* for reverse engineering ($N_0=N_{\min}$); *weaker* for binary search (§3.8) |
| **A3** | $\phi$ has a **uniform prior** on $[\phi_{\min},\phi_{\max}]$ | *Literally true* — the Monte Carlo draws it that way (§3.3). Not a modelling convenience. |
| **A4** | ~~The posterior truncation may be dropped~~ — **no longer assumed**: the exact truncated normal is used | *Retired.* Kept in §3.6–3.7 because the measurement matters: dropping it would cost $\le0.14$ pp, since the prior spans $W=\pi\sqrt{m'}(1-\phi_{\min}/\phi_{\max})\ge14\sigma$ |
| **A5** | Overshooting ($N\phi>\pi/2$) ⟹ the trial never converges | *Slightly conservative*; a thin band of marginal overshoots still converges (§8.2) |
| **A6** | In the convergence factor, $N$ is treated as fixed although it is chosen from the data | *Approximation*; the estimate itself is unbiased (fresh shots), only its marginal law is a mixture (§8.3) |
| **A7** | The chosen operating point stays where A1 holds, i.e. $m\cos^2(N\phi)\gtrsim 10$ | *Holds at moderate/high budget; violated in ~24 % of trials at $B=10^4$* (§9) |

**A3 is the strongest link in the chain, not the weakest** — see §3. The genuinely soft assumptions
are A5–A7. A4 has been *removed* by using the exact posterior (§3.6–3.7).

---

## 1. Setup

A circuit of depth $N$ measured $m$ times returns $\text{hits}\sim\text{Binomial}(m,p_0)$ with

$$p_0(\phi) \;=\; \cos^2(N\phi),$$

and the estimator inverts this:

$$\hat\phi \;=\; \frac{1}{N}\arccos\!\sqrt{\frac{\text{hits}}{m}} \;\in\;\Big[0,\ \frac{\pi}{2N}\Big].$$

The inversion is single-valued only while $N\phi\le\pi/2$. Beyond that the estimator *aliases*: it
returns a value in $[0,\pi/2N]$ that is unrelated to the true $\phi$, and the trial is lost. The
largest safe depth is

$$N_{\text{opt}}=\Big\lfloor \frac{\pi}{2\phi}\Big\rfloor,$$

which depends on the unknown $\phi$ — hence the need for a safeguard. The budget is $B=N\cdot m$ and a
trial *converges* when $|\hat\phi-\phi|<\varepsilon$.

---

## 2. A1 — the sampling law, and why its variance is $\phi$-free

With $\hat p=\text{hits}/m$ and $g(p)=\frac{1}{N}\arccos\sqrt p$, the delta method gives

$$g'(p)=-\frac{1}{2N\sqrt{p(1-p)}},\qquad
\operatorname{Var}(\hat p)=\frac{p_0(1-p_0)}{m},$$

$$\operatorname{Var}(\hat\phi)\;\approx\;g'(p_0)^2\operatorname{Var}(\hat p)
\;=\;\frac{1}{4N^2p_0(1-p_0)}\cdot\frac{p_0(1-p_0)}{m}
\;=\;\boxed{\frac{1}{4N^2m}}$$

The $p_0(1-p_0)$ factors cancel **exactly**. Equivalently, the Fisher information per shot is

$$I(\phi)=\frac{(p_0')^2}{p_0(1-p_0)}
=\frac{N^2\sin^2(2N\phi)}{\tfrac14\sin^2(2N\phi)}=4N^2,$$

constant in $\phi$, so the estimator saturates the Cramér–Rao bound uniformly. **This constancy is
what makes everything downstream clean**, and it is a genuine property of this particular estimator,
not a convenience.

Numerically (200,000 draws per row):

| $N$ | $m$ | $N\phi$ | $p_0$ | empirical sd | $1/(2N\sqrt m)$ | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 200 | 0.40 | 0.848 | 2.374e−3 | 2.357e−3 | 1.007 |
| 15 | 200 | 0.90 | 0.386 | 2.371e−3 | 2.357e−3 | 1.006 |
| 15 | 200 | 1.30 | 0.072 | 2.393e−3 | 2.357e−3 | 1.015 |
| 15 | 200 | **1.56** | 0.0001 | 7.21e−4 | 2.357e−3 | **0.306** |
| 100 | 500 | 0.90 | 0.386 | 2.236e−4 | 2.236e−4 | 1.000 |

Perfect in the bulk; it breaks at the boundary $N\phi\to\pi/2$ (last row) — see §8.1.

---

## 3. "But $\phi$ is fixed, not random" — the central question

You are right to push on this, and the resolution is worth stating precisely.

### 3.1 Two different Gaussian statements

They look identical, which is exactly why this is confusing:

$$\textbf{(S1)}\qquad \hat\phi \mid \phi \;\sim\; \mathcal{N}\!\left(\phi,\ \frac{1}{4N^2m}\right)$$

$$\textbf{(S2)}\qquad \phi \mid \hat\phi_0 \;\sim\; \mathcal{N}\!\left(\hat\phi_0,\ \frac{1}{4N_0^2m'}\right)$$

**(S1)** is the *sampling distribution*: $\phi$ is a fixed constant, and the randomness is the shot
noise across repetitions. This is what §2 derived and what asymptotic theory licenses. Uncontroversial.

**(S2)** is the *posterior*: it treats $\phi$ as the random quantity and $\hat\phi_0$ as fixed. This is
what the safeguard actually uses. **It is not the same statement as (S1)**, and writing
"$\phi\sim\mathcal N(\hat\phi_0,\tfrac{1}{4m'N^2})$" without justification is precisely the sleight of
hand you suspected.

### 3.2 The bridge is Bayes' theorem

$$p(\phi \mid \hat\phi_0) \;=\; \frac{p(\hat\phi_0 \mid \phi)\,\pi(\phi)}{\displaystyle\int p(\hat\phi_0 \mid \phi')\,\pi(\phi')\,\mathrm d\phi'}$$

Substituting (S1) as the likelihood, with $\sigma^2 = \frac{1}{4N_0^2m'}$:

$$p(\hat\phi_0\mid\phi)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(\hat\phi_0-\phi)^2}{2\sigma^2}\right)$$

### 3.3 Why the flip is *exact* here

Two properties of (S1) make the conversion lossless:

1. **The exponent depends only on the difference $(\hat\phi_0-\phi)$** — the Gaussian is symmetric in
   its two arguments.
2. **$\sigma$ does not depend on $\phi$** (that is A1, §2).

Because of (2) there is no $\phi$-dependent normalising factor left over. So *as a function of
$\phi$*, the likelihood is itself a Gaussian density centred at $\hat\phi_0$ with the same width. With
a flat prior $\pi(\phi)\propto\text{const}$ on $[\phi_{\min},\phi_{\max}]$:

$$p(\phi\mid\hat\phi_0)\;=\;\frac{\mathcal N(\phi;\ \hat\phi_0,\sigma^2)}{\displaystyle\int_{\phi_{\min}}^{\phi_{\max}}\!\mathcal N(\phi';\ \hat\phi_0,\sigma^2)\,\mathrm d\phi'}\;\cdot\;\mathbf 1\!\left[\phi\in[\phi_{\min},\phi_{\max}]\right]$$

— a **truncated** normal. This is what the code now uses. Dropping the truncation would give exactly
(S2); §3.6–3.7 record what that shortcut would have cost, because the measurement is worth reporting
even though the shortcut is no longer taken.

> Had the variance depended on $\phi$, the likelihood would have carried an extra $1/\sigma(\phi)$
> factor and the posterior would **not** have been Gaussian. The step is legitimate *because* the
> Fisher information is constant.

### 3.4 The prior really is uniform — this is the strong part

In most applications a flat prior is an assumption one has to defend. Here it is not: the experiment
**literally draws** $\phi\sim\mathcal U(\phi_{\min},\phi_{\max})$ in every Monte-Carlo trial
([`experiments.py`](../qmetrology/experiments.py), `_draw_phi`), and the thesis reports results
averaged over exactly that draw.

So $p(\phi\mid\hat\phi_0)$ is not a subjective belief. It is **the true conditional distribution of
$\phi$ given the pilot, over the ensemble of trials the thesis actually runs** — a frequentist
statement about a well-defined joint law $p(\phi,\text{data})$. No Bayesian commitment is required.

### 3.5 So what about "after the experiment starts, $\phi$ is fixed"?

Correct, and it does not break anything — because of what is being claimed.

Within a single run, $\phi$ is fixed and the event $\{N\phi<\pi/2\}$ is deterministic: it either
happens or it does not. The rule never claims otherwise. What it optimises is

$$\mathbb P\big(\text{converge}\big)\;=\;\mathbb E_{\phi\sim\mathcal U}\ \mathbb E_{\text{shots}\mid\phi}\big[\mathbf 1\{|\hat\phi-\phi|<\varepsilon\}\big],$$

the **expected convergence rate over the ensemble** — which is precisely the number the thesis
reports (a proportion over 40,000 trials, each with a freshly drawn $\phi$). The objective being
maximised is an unbiased estimate of the reported metric.

This is the ordinary logic of a decision rule, and the same logic as a confidence interval: a
particular 95 % interval either contains the fixed parameter or it does not; the "95 %" describes
coverage across repetitions. Here, likewise, the rule is the choice of $N$ that maximises the
long-run share of converged runs. It makes no probabilistic claim about your one fixed $\phi$.

**If** you instead cared about a guarantee for one specific $\phi$ — "for this particular phase, at
least 95 % of repeated measurements must converge" — the rule would need a different, minimax
formulation (maximise the *worst-case* over $\phi$, not the average). That is a real alternative, and
it would be more conservative. The average-case form is the right one here because the thesis's
metric is itself an average over the prior.

### 3.6 The truncation is real — and is now applied exactly

The posterior is a *truncated* normal; (S2) is not. An earlier version of this document claimed the
lower tail was "handled exactly by the $N_{\max}$ cap". **That was wrong**, and the code now uses the
exact form (`risk_optimal_depth(..., support=(phi_min, phi_max))`, on by default). The cap only forbids
$N$ so large that $\pi/(2N)<\phi_{\min}$; it does nothing about the distortion for admissible $N$,
where the truncated and untruncated CDFs genuinely differ:

$$\mathbb P_{\text{trunc}}\!\left(\phi<t\right)=\frac{\Phi\!\left(\frac{t-\hat\phi_0}{\sigma}\right)-\Phi\!\left(\frac{\phi_{\min}-\hat\phi_0}{\sigma}\right)}{\Phi\!\left(\frac{\phi_{\max}-\hat\phi_0}{\sigma}\right)-\Phi\!\left(\frac{\phi_{\min}-\hat\phi_0}{\sigma}\right)}$$

The two boundaries push in **opposite** directions:

* near $\phi_{\min}$, truncation removes the small-$\phi$ mass that made a deep circuit look safe, so
  it **lowers** $N^*$ (the untruncated form is *anti*-conservative there);
* near $\phi_{\max}$, knowing $\phi\le\phi_{\max}$ makes depth $N_{\min}$ provably safe, so it
  **raises** $N^*$ (the untruncated form is conservative there).

### 3.7 Why it is nevertheless negligible — the scale argument

Truncation only distorts the posterior within $\sim 2\sigma$ of a boundary, so what matters is the
width of the prior *measured in pilot standard deviations*. With the pilot at
$N_0=\lfloor\pi/(2\phi_{\max})\rfloor$, so that $\sigma\approx\phi_{\max}/(\pi\sqrt{m'})$:

$$W\;=\;\frac{\phi_{\max}-\phi_{\min}}{\sigma}\;=\;\pi\sqrt{m'}\left(1-\frac{\phi_{\min}}{\phi_{\max}}\right)$$

Remarkably, this depends only on the exploration size and the *ratio* of the prior limits — not on
their absolute scale, the budget, or $\varepsilon$. Verified against the exact $\sigma$:

| prior | $m'$ | $W$ predicted | $W$ actual |
|---|---:|---:|---:|
| U(0.01, 0.1) | 76 | 24.6 | 23.5 |
| U(0.001, 0.01) | 70 | 23.7 | 23.6 |
| U(0.01, π/2) | 16,074 | 395.8 | 395.8 |
| U(0.09, 0.1) | 100 | 3.1 | 3.0 |

Every prior in the thesis has $\phi_{\min}/\phi_{\max}\le 0.1$, and every selected $m'$ is $\ge 70$,
giving $W\ge 14$ — so at most ~$4/W\approx 15\,\%$ of trials sit in a boundary layer at all, and for
those the depth moves by $\le 5\,\%$ (worst case $\le 25\,\%$, at $m'=20$ on the broad prior, a
configuration the tuner never selects). For binary search the pilot sits at a *deeper* probe, so
$\sigma$ is smaller, $W$ larger, and the effect weaker still.

Measured end-to-end (de-biased, $R=40{,}000$, SE $\approx0.25$ pp), reproducible with
`python analysis/truncation_check.py`:

| operating point | plain normal | truncated | Δ |
|---|---:|---:|---:|
| U(0.01,0.1), ε=10⁻³, B=10⁴ | 66.22 % | 66.29 % | +0.07 pp |
| U(0.01,0.1), ε=10⁻³, B=4.5·10⁴ | 94.72 % | 94.72 % | +0.00 pp |
| U(0.001,0.01), ε=10⁻⁴, B=4·10⁵ | 93.83 % | 93.69 % | −0.14 pp |
| U(0.01,0.1), ε=10⁻⁴, B=3·10⁶ | 92.03 % | 92.03 % | +0.00 pp |
| U(0.01,π/2), ε=10⁻³, B=8.9·10⁵ | 96.72 % | 96.72 % | +0.00 pp |

**Verdict: the difference is immaterial, and the exact form is used anyway.** Every deviation is
inside the standard error and the two boundary effects partly cancel — so the untruncated shortcut
would have been defensible. The code nevertheless uses the truncated posterior, because it is the
correct one and costs nothing: one extra $\Phi$ evaluation per candidate depth. Pass
`support=None` to recover the untruncated form and reproduce the table above
(`python analysis/truncation_check.py`).

For the thesis, the honest one-liner is: *the posterior is a truncated normal; using the untruncated
form instead shifts convergence by at most 0.14 pp, well inside the 0.25 pp standard error.*

There is one regime where this would *not* hold: a prior with $\phi_{\min}/\phi_{\max}\to 1$ (say
U(0.09, 0.1)) drives $W\to 3$. But there the admissible depths collapse to
$N\in\{15,16,17\}$ — $N_{\max}/N_{\min}=\phi_{\max}/\phi_{\min}$ — so the argmax has almost nowhere
to move and the measured shift is again zero. Narrow priors distort the posterior most and constrain
the decision most, and the two effects offset.

### 3.8 Where A2 (unimodality) actually bites

The likelihood above is only single-moded if $\phi\mapsto\cos^2(N_0\phi)$ is injective on the prior
support. Since $\cos^2$ has period $\pi/N_0$ in $\phi$, this requires $N_0\phi\le\pi/2$ throughout.

* **Reverse engineering** takes the pilot at $N_0=N_{\min}=\lfloor\pi/(2\phi_{\max})\rfloor$, so
  $N_0\phi\le N_0\phi_{\max}\le\pi/2$ for every admissible $\phi$. **Exact by construction** — this is
  the real reason $N_{\min}$ is defined that way.
* **Binary search** takes the pilot at the deepest *non-overshooting* probe, which is at larger $N$.
  There, unimodality is **not** guaranteed a priori; what rules out the aliased branch is the
  bisection's own overshoot test. This is the weakest formal link in the binary-search variant and
  is worth stating in the thesis rather than glossing.

---

## 4. Factor 1 — the overshoot probability

Under §3, for a candidate depth $N$:

$$\mathbb P\big(\text{no overshoot}\mid\hat\phi_0\big)
=\mathbb P\!\left(\phi<\frac{\pi}{2N}\ \Big|\ \hat\phi_0\right)
=\frac{\Phi\!\left(\frac{\pi/(2N)-\hat\phi_0}{\sigma}\right)-\Phi\!\left(\frac{\phi_{\min}-\hat\phi_0}{\sigma}\right)}
       {\Phi\!\left(\frac{\phi_{\max}-\hat\phi_0}{\sigma}\right)-\Phi\!\left(\frac{\phi_{\min}-\hat\phi_0}{\sigma}\right)},
\qquad \sigma=\frac{1}{2N_0\sqrt{m'}}$$

Monotonically **decreasing** in $N$, and exactly $0$ once $\pi/(2N)\le\phi_{\min}$ — so the
$N_{\max}$ cap of §7 now falls out of the posterior instead of being imposed. Replacing this by the
plain $\Phi\!\left(\frac{\pi/(2N)-\hat\phi_0}{\sigma}\right)$ is the shortcut measured in §3.7.

## 5. Factor 2 — the convergence probability

Conditional on not overshooting, the final estimate at depth $N$ with $m=\lfloor B/N\rfloor$ shots
obeys (S1) again. The key algebraic step:

$$N^2 m \;=\; N^2\Big\lfloor\frac{B}{N}\Big\rfloor \;\approx\; N B
\qquad\Longrightarrow\qquad
\sigma_{\text{final}}=\frac{1}{2\sqrt{NB}}$$

so that

$$\mathbb P\big(|\hat\phi-\phi|<\varepsilon\big)
=2\Phi\!\left(\frac{\varepsilon}{\sigma_{\text{final}}}\right)-1
=2\Phi\!\left(2\varepsilon\sqrt{NB}\right)-1$$

Monotonically **increasing** in $N$ (as $\sqrt N$), and — crucially — **free of $\phi$**, again by A1.
That is what allows it to be evaluated without knowing $\phi$.

> Note the resource asymmetry: $N\cdot m=B$ is fixed, but $N^2m=NB$ grows with $N$. Depth is worth
> more than repetitions. That is the Heisenberg scaling, and it is why the safeguard is not simply
> "be as conservative as possible."

## 6. Why multiplying them is legitimate

The product is **not** an independence assumption. Writing $g(N)=2\Phi(2\varepsilon\sqrt{NB})-1$:

$$\mathbb P(\text{converge}\mid\hat\phi_0,N)
=\int p(\phi\mid\hat\phi_0)\Big[\mathbf 1\{N\phi<\tfrac\pi2\}\,g(N)+\mathbf 1\{N\phi\ge\tfrac\pi2\}\cdot 0\Big]\mathrm d\phi$$

$$=g(N)\int_{\phi<\pi/2N} p(\phi\mid\hat\phi_0)\,\mathrm d\phi
\;=\;g(N)\cdot\Phi\!\left(\frac{\pi/(2N)-\hat\phi_0}{\sigma}\right)$$

$g(N)$ comes out of the integral **exactly**, because it does not depend on $\phi$ (A1 again). The
only modelling input is A5, the zero in the second bracket.

## 7. The rule

$$\boxed{\;N^{*}=\operatorname*{arg\,max}_{1\le N\le N_{\max}}\;
\mathbb P\big(\phi<\tfrac{\pi}{2N}\mid\hat\phi_0\big)\cdot\Big[2\Phi\!\left(2\varepsilon\sqrt{NB}\right)-1\Big]\;}$$

with the first factor the truncated-normal tail of §4 (or, to within 0.14 pp, the plain
$\Phi\!\left(\frac{\pi/(2N)-\hat\phi_0}{\sigma}\right)$),

and $N_{\max}=\lfloor\pi/(2\phi_{\min})\rfloor$ from the prior support. No tuned constant appears
anywhere.

**The binary-search search is *not* capped at the bisection result.** That would mirror the published
"reduce $N$ by $s$" wording, but the bisection stops when its step size reaches zero and therefore
undershoots systematically; capping there throws away depths the pilot's own posterior can justify.
Measured (de-biased, R=20,000): capping at the bisected $N$ costs **1.3 pp** at $B=4.5\cdot10^4$,
**2.1 pp** at $B=3\cdot10^6$ and **0.8 pp** at U(0.001,0.01). Capping at the bisection's *upper
bound* — the shallowest depth actually flagged as an overshoot — is no better than capping at the
result. §13 measures the same effect more thoroughly (capping at $L$ costs 1.34 pp over 12 operating
points) and gives the reason: $N^{*}>L$ in 31.7 % of trials. So the exploration phase is best read as a way of buying a **precise pilot** (taken at a deep
circuit, so $\sigma=1/(2N_{\text{acc}}\sqrt{m'})$ is small), not as the depth selector itself.

The implied multiplicative safeguard $C_{\text{eff}}=N^*/\lfloor\pi/(2\hat\phi_0)\rfloor$ is what the
grid search was trying to approximate with one number:

| $\phi$ | $B=10^4$ | $B=4.5\cdot10^4$ | $B=3\cdot10^6$ |
|---:|---:|---:|---:|
| 0.01 | 0.541 | 0.599 | 0.726 |
| 0.03 | 0.788 | 0.846 | 0.904 |
| 0.05 | 0.871 | 0.903 | 0.935 |
| 0.10 | 0.933 | 1.000 | 1.000 |

Adaptive along **both** axes; a constant cannot be.

---

## 8. Where the assumptions break

### 8.1 A1 fails at the boundaries (the important one)

The normal approximation degrades when $m\,p_0$ is small, because $\text{hits}=0$ acquires a
non-negligible atom and the estimator is *censored* at $\hat\phi=\pi/(2N)$. KS statistic of the
standardised estimator:

| $N\phi$ | $m$ | $m\,p_0$ | KS | $\mathbb P(\text{hits}=0)$ |
|---:|---:|---:|---:|---:|
| 0.80 | 200 | 97.1 | 0.031 | 0.000 |
| 1.30 | 200 | 14.3 | 0.072 | 0.000 |
| 1.50 | 200 | 1.00 | 0.343 | 0.366 |
| 1.50 | 2000 | 10.0 | 0.087 | 0.000 |
| 1.56 | 2000 | 0.23 | 0.625 | 0.792 |

**Working rule: the law is trustworthy while $m\,p_0=m\cos^2(N\phi)\gtrsim 10$.** Note this alone
implies a ceiling on $C_{\text{eff}}$:

| $m$ | 50 | 200 | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|---:|---:|
| max $C_{\text{eff}}$ | 0.705 | 0.856 | 0.936 | 0.980 | 0.994 |

which is *strikingly close to the constants the grid search kept selecting* (0.8–0.85 at low budget,
0.95–0.975 at high budget). The tuned constant was, in effect, a crude estimate of this ceiling.

### 8.2 A5 is mildly conservative

An overshoot still converges if $\phi-\pi/(2N)<\varepsilon$. The band has width
$\approx\pi\varepsilon/(2\phi^2)$ in $N$:

| $\phi$ | $\varepsilon$ | $N_{\text{opt}}$ | band width | as % of $N_{\text{opt}}$ |
|---:|---:|---:|---:|---:|
| 0.05 | $10^{-3}$ | 31 | 0.64 | 2.1 % |
| 0.01 | $10^{-3}$ | 157 | 17.5 | **11.1 %** |
| 0.05 | $10^{-4}$ | 31 | 0.06 | 0.2 % |
| 0.05 | $10^{-6}$ | 31 | 0.00 | 0.0 % |

Negligible at tight $\varepsilon$; at small $\phi$ with loose $\varepsilon$ the rule gives away up to
~11 % of depth (≈5 % in $\sigma$). Always in the safe direction.

### 8.3 A6 — $N$ is data-dependent

$N^*$ is a function of $\hat\phi_0$, so the marginal law of the final estimate is a *mixture* over
$N$, not the single Gaussian factor 2 assumes. The estimate itself is not biased by this (the
exploitation shots are fresh — no data is reused), but the stated convergence probability is an
approximation to the mixture. Empirically this is small: the realised rates track the predicted ones
closely enough that the rule beats the tuned constant at 11/11 operating points.

---

## 9. Self-consistency: does the rule stay where its own assumptions hold? (A7)

The rule contains no explicit $m\,p_0$ constraint, so this must be checked rather than assumed. At
the depth actually chosen, over 3,000 simulated trials:

| Setting | $B$ | min $m p_0$ | median | % below 10 |
|---|---:|---:|---:|---:|
| U(0.01,0.1), $\varepsilon$=10⁻³ | 10,000 | 0.0 | 20.2 | **24.2 %** |
| U(0.01,0.1), $\varepsilon$=10⁻³ | 45,000 | 0.0 | 37.2 | 8.2 % |
| U(0.001,0.01), $\varepsilon$=10⁻⁴ | 400,000 | 0.2 | 118 | 1.6 % |
| U(0.01,0.1), $\varepsilon$=10⁻⁴ | 3,000,000 | 0.1 | 1,210 | 0.4 % |
| U(0.01,π/2), $\varepsilon$=10⁻³ | 890,000 | 0.3 | 34,047 | 0.2 % |

**Honest reading.** At moderate and high budget the rule sits comfortably inside its validity region —
the budget grows faster than the operating point approaches the boundary, since
$m\,p_0\approx B\phi\pi(1-C)^2/(2C)$ grows linearly in $B$. At the smallest budget ($10^4$) roughly a
quarter of trials land where the Gaussian law is *not* a good description, so there the rule is
partly operating on a model it has outrun.

It still wins there (+5.0 pp over the tuned constant at $B=10^4$), which says the failure is benign:
the trials where $m\,p_0$ is tiny are largely trials that were going to fail under any depth choice,
so mis-modelling them costs little. But it does mean **the low-budget result is empirically rather
than theoretically justified**, and the thesis should say so rather than claim the derivation covers
that regime.

---

## 10. Summary

The chain is:

1. $\operatorname{Var}(\hat\phi)=1/(4N^2m)$, exactly $\phi$-free — a real property of this estimator.
2. That constancy makes the likelihood→posterior flip exact, needing only a flat prior.
3. The flat prior is the experiment's actual generative model, so the posterior is a genuine
   frequentist conditional, not a belief.
4. $\phi$ being fixed within a run is not a problem, because the rule is a decision rule optimising
   the ensemble average — which is exactly the metric being reported.
5. The posterior is strictly a *truncated* normal, and the code uses that exact form. Dropping the
   truncation would have been worth $\le0.14$ pp, because the prior spans
   $W=\pi\sqrt{m'}(1-\phi_{\min}/\phi_{\max})\ge14$ pilot standard deviations for every prior
   used (§3.7).
6. The two factors multiply exactly, because the convergence factor is $\phi$-free.
7. The soft spots are A5 (mildly conservative), A6 (data-dependent $N$), and A7 (the low-budget
   regime, where ~24 % of trials sit outside the model's validity).

The comparison worth making: the thesis's *existing* binary-search criterion (Eq. 3.5) substitutes
$\hat\phi_0$ for the unknown mean and is described in the text as "a heuristic decision rule rather
than a rigorous confidence guarantee." The safeguard derived here makes the same substitution
*principled* — under A1 and A3 it is the exact posterior, not a proxy. **It rests on firmer ground
than the criterion already in the thesis.**

---

## 11. Why linear search does *not* get the same treatment

Linear search (Algorithm 4) is the one protocol still carrying a grid-tuned safeguard — the decrement
$s$. Extending Eq. (3.8) to it looks straightforward, and it is: the scan leaves a whole *history*
of estimates $\hat\phi_i$ at depths $N_i$, each obeying A1 with a known variance $1/(4N_i^2m')$, and
independent probes with known variances combine by inverse-variance weighting,

$$\hat\phi_{\text{pool}}=\frac{\sum_i N_i^2\,\hat\phi_i}{\sum_i N_i^2},\qquad
\sigma_{\text{pool}}=\frac{1}{2\sqrt{m'\sum_i N_i^2}},$$

with the last `lookback_window` probes dropped — the ones that triggered the overshoot verdict, for
which A2 fails. Feeding $(\hat\phi_{\text{pool}},\sigma_{\text{pool}})$ into Eq. (3.8) is then the
identical rule. Measured (`analysis/exploration_study.py`, grid-tuned on seed 42 and re-validated on
seed 2024 at $R=40{,}000$; mean over the budgets where the rate is between 2 % and 98 %):

| scenario | linear (tuned $s$) → statistical | using only the deepest probe |
|---|---:|---:|
| $\mathcal U(0.01,0.1)$, $\varepsilon=10^{-3}$ | **+5.02 pp** | +4.20 pp |
| $\mathcal U(0.01,0.1)$, $\varepsilon=10^{-4}$ | **+3.45 pp** | +3.73 pp |
| $\mathcal U(0.001,0.01)$, $\varepsilon=10^{-4}$ | **+5.62 pp** | +4.63 pp |

**It works, and it should still not be adopted.** The reason is visible in what the tuner selects.
Grid search drives `lookback_window` to 1, so at 10 of the 15 operating points the exploration takes
**exactly two probes and retains one**: the measured pooling factor $\sum_iN_i^2/N_{\text{last}}^2$ is
then $1.0$, i.e. no pooling occurs, and the median exploration spend is 0.1–6.2 % of the budget.
What remains is reverse engineering with one wasted probe — and reverse engineering beats it at
**every one of the 15 points**, by +0.11 to +2.60 pp.

So the gain is not the safeguard rescuing linear search; it is the safeguard making the linear scan
redundant. Once a principled depth rule exists, the only output of the scan that matters is a pilot
estimate, and buying that pilot with an incremental scan is strictly worse than buying it with a
single deep measurement. This is worth one sentence in §3.3.4 as the mechanism behind reverse
engineering's dominance, and it is the reason the reported "Linear search" row keeps its tuned $s$:
the alternative is not a better linear search, it is a worse reverse engineering.

A secondary observation, for the objection that $m'$ is too small here for A1 to hold at all: that is
not what fails. Where the scan *does* survive tuning (`lookback_window` $=5$), it retains 9–10 probes
and the pooling factor reaches 6.3–6.7, i.e. an effective $m$ of 400–2300 — comfortably inside A1.
The asymptotic law is fine; the scan is what has no value.

The depth chosen by the pooled variant also stays clear of the degenerate readout: the share of
trials with $m\,p_0<1$ is 0.02–7.7 %, below reverse engineering's 0.05–10.8 % at all 12 points
checked, so nothing new would have had to be disclosed had it been adopted.

---

## 12. Which probe should be the pilot? (binary search)

Eq. (3.8) needs one pilot $(\hat\phi_0,\sigma)$, but the bisection produces several estimates at
increasing depths. Four defensible choices:

| choice | $\sigma=1/(2N_0\sqrt{m'})$ | aliasing risk |
|---|---|---|
| `first` — the opening probe at $N_{\min}$ | largest | **none**: $N_{\min}\phi\le\pi/2$ for every admissible $\phi$, by A2 |
| `deep` — deepest probe not flagged as an overshoot | smaller by $N_{\min}/N_{\rm acc}$ | rests on the overshoot *test*, not on construction |
| `last` — the final probe, flagged or not | smaller | highest |
| `pool` — inverse-variance combination of the unflagged probes | smallest | as `deep` |

**`deep` is what `qmetrology/algorithms.py` implements**, and §7 justified it as "buying a precise
pilot". That justification is incomplete, and the measurement (`analysis/binary_pilot_study.py`;
3 scenarios $\times$ 6 budgets, each variant grid-tuning its own $m'$ **and** `conf`, de-biased on seed
2024 at $R=40{,}000$; 12 live points, SE of a difference 0.35 pp) shows why:

| variant | mean vs `deep` | median | worst | best | wins | mean overshoot |
|---|---:|---:|---:|---:|---:|---:|
| `first` | **+0.04 pp** | +0.06 | −0.80 | +0.66 | 50 % | 0.93 % |
| `deep` (shipped) | — | — | — | — | — | 1.10 % |
| `last` | **−1.34 pp** | −1.03 | −4.39 | +0.72 | 17 % | 4.91 % |
| `pool` | +0.24 pp | +0.10 | −0.11 | +1.17 | 75 % | 1.13 % |

**`first` and `deep` are indistinguishable** ($+0.04\pm0.10$ pp on the mean). The reason is not that the
extra precision is worthless — it is that it is paid for with a bias the model does not account for:

| variant | mean pilot bias | model $\sigma$ | bias / $\sigma$ |
|---|---:|---:|---:|
| `first` | $+3.3\cdot10^{-4}$ | $3.8\cdot10^{-3}$ | **0.09** |
| `deep` | $+2.4\cdot10^{-3}$ | $2.7\cdot10^{-3}$ | **0.87** |
| `pool` | $+2.0\cdot10^{-3}$ | $2.2\cdot10^{-3}$ | 0.90 |
| `last` | $-2.8\cdot10^{-4}$ | $3.1\cdot10^{-3}$ | −0.09 |

`deep` does cut $\sigma$ by about 30 %. But the probe it selects was selected *for having read high* —
acceptance means $\hat\phi\ge\phi_1$ — so it carries a **selection bias of nearly one full $\sigma$**, which
Eq. (3.4) knows nothing about. This is a genuine violation of A6 (it makes the pilot's marginal law
non-central, not merely a mixture), and it is the price of the smaller $\sigma$. The two effects cancel.

The bias is **upward**, so the rule believes $\phi$ is larger than it is and picks a *shallower* depth:
`deep` is conservative, not dangerous. That is why its overshoot rate (1.10 %) is no worse than
`first`'s (0.93 %) despite the deeper pilot.

**`last` is the one bad choice.** It may be a probe the test flagged, i.e. an aliased reading, biasing
the pilot *down* and pushing $N$ deeper — overshoot rises to 4.91 % and convergence falls 1.34 pp.
"Use the most recent estimate", read literally, means this.

A caveat that sharpens the picture: at a **fixed** `conf` the gap is larger — `first` beats `deep` by
$+0.75\pm0.27$ pp on paired trials. Letting each variant tune `conf` is what closes it; `deep` recovers
by choosing a laxer test (it tunes to $\alpha=0.5$ at 8 of 12 points), which admits probes earlier and
so reduces the selection bias. The two knobs are not independent.

**Verdict: keep `deep`, and justify it correctly.** It is not better because it is more precise; it is
*equivalent*, because the precision it gains is offset by a selection bias of comparable size.
Switching to `first` would be equally defensible and marginally simpler to describe (A2 would hold by
construction rather than through the test), but it is not an improvement and would cost a full re-run.
`pool` is the only variant that is ahead, by about two standard errors — not enough to justify a new
derivation.

Interactive version, self-contained: `notebooks/binary_pilot.ipynb`.

---

## 13. Should the exploitation depth just be $L$, the bisection's lower bound?

After the bisection, $L$ is the deepest depth probed and *not* flagged as an overshoot. It looks like
an empirically verified-safe depth, which suggests dropping Eq. (3.8) for binary search entirely and
exploiting at $N=L$. ($L$ is the same quantity as the `deep` pilot's depth of §12: the loop sets
`lb = temp_N` and `N_acc = temp_N` in the same branch, and $N$ never falls below `lb`.)

Measured (`analysis/binary_depth_study.py`; same harness as §12 — 3 scenarios $\times$ 6 budgets, each
rule grid-tuning its own $m'$ and `conf`, de-biased at $R=40{,}000$; 12 live points, SE 0.35 pp):

| depth rule | mean vs shipped | median | worst | wins |
|---|---:|---:|---:|---:|
| $N^{*}$ from Eq. (3.8), uncapped (**shipped**) | — | — | — | — |
| $N=L$ | **−6.18 pp** | −7.59 | −11.25 | 0 % |
| $\min(N^{*},L)$ | −1.34 pp | −1.38 | −2.48 | 0 % |
| $\max(N^{*},L)$ | −3.66 pp | −3.09 | −10.27 | 8 % |

**$L$ is not a safe depth, and it is not close.**

| | median depth / $N_{\rm opt}$ | $\mathbb P(\text{depth}>N_{\rm opt})$ |
|---|---:|---:|
| $L$ | 0.917 | **19.6 % (up to 40 %)** |
| $N^{*}$ | 0.827 | **1.1 % (up to 3.5 %)** |

The reason is structural, not marginal. The bisection's entire purpose is to *locate* the aliasing
boundary, so $L$ converges onto $N_{\rm opt}$ — and having converged onto it, sampling noise puts it on
the wrong side a large fraction of the time. The overshoot test is a hypothesis test with a type-I
error, and the error is **absorbing**: once a probe above $N_{\rm opt}$ is accepted, $L$ moves into the
aliased region and never comes back down, because $L$ is a running maximum. Eq. (3.8) instead backs
off to a median $0.83\,N_{\rm opt}$, trading a little precision for a $20\times$ lower aliasing rate.

So $L$ answers the wrong question. It estimates the quantity the protocol must stay *below*; it does
not say by how much, and "by nothing" is the one answer that is certainly wrong.

A related number, which also refines §7: $N^{*}>L$ in **31.7 %** of trials — the rule frequently wants a
depth the bisection never verified. That is why capping at $L$ costs 1.34 pp, consistent with the
1.3–2.1 pp measured in §7 for capping at the bisection's final $N$. Both caps discard depths the
pilot's own posterior can justify.

The trajectory plots in `notebooks/binary_pilot.ipynb` §3b show a single run doing this: an accepted
probe one step above $N_{\rm opt}$ lifts $L$ over the boundary permanently.

---

## 14. The exact posterior — dropping A1, A2 and A6 entirely

Everything above computes $\mathbb P(\phi<\pi/2N\mid\text{data})$ from **one** probe, by turning it into a
point estimate and invoking the asymptotic law. That route forces a classification: probes past the
aliasing point must be identified and discarded, because $\arccos$ cannot be inverted there. §12 and
§13 measured the damage that classification does. There is a route that avoids it.

### 14.1 The estimator loses a branch, not information

$\hat\phi=\frac1N\arccos\sqrt{k/m'}$ is a *bijection* from the count $k$, so the transformation itself
discards nothing. The loss is in the reading: $\arccos$ returns a principal value in $[0,\pi/2]$, so
$\hat\phi=\phi$ only while $N\phi\le\pi/2$. For $N\phi\in(\pi/2,\pi]$, $\cos^2$ folds and the estimator
returns $\pi/N-\phi$. The measurement was fully informative; it has merely been reflected onto a branch
the estimator cannot label.

### 14.2 The likelihood has no branches

For every probe, aliased or not,

$$k_i\sim\mathrm{Binomial}\bigl(m',\cos^2(N_i\phi)\bigr)$$

is a true statement about $\phi$. Under the uniform prior of A3 the posterior is therefore

$$\boxed{\;p(\phi\mid\mathcal D)\;\propto\;\prod_{i=1}^{K}\cos^{2k_i}(N_i\phi)\bigl(1-\cos^{2}(N_i\phi)\bigr)^{m'-k_i},\qquad\phi\in[\phi_{\min},\phi_{\max}]\;}$$

one-dimensional, evaluated on a grid (`qmetrology/posterior.py`). $\cos^2$ has period $\pi/N_i$, so a
deep probe alone yields a multimodal posterior — honest ambiguity, not a defect — resolved by the
opening probe, which is unimodal over the prior by construction.

**The old rule had to decide which probes were valid; this one never decides.** It carries every branch
as evidence and lets the probes resolve one another.

### 14.3 Why it is more precise

A probe at depth $N$ resolves $\phi$ to about $1/(2N\sqrt{m'})$ *on whichever branch it lies*. A
bisection places probes at $N\approx N_{\rm opt}$, and $N_{\rm opt}/N_{\min}=\phi_{\max}/\phi$. Measured,
the posterior's spread is **6–11$\times$ tighter** than the opening probe's $\sigma$.

### 14.4 What it costs in assumptions — less, not more

A1 (asymptotic normality), A2 (unimodality of the pilot likelihood) and A6 ($N$ treated as fixed though
data-dependent) are **all dropped**: the likelihood is exact, branches are kept, and no estimator is
formed. A3 (uniform prior) and A5 (overshoot $\Rightarrow$ failure) remain. Eq. (3.8) itself is
unchanged — only the first factor is computed differently.

### 14.5 Measured, over all 23 scenarios

`analysis/posterior_all_study.py`, 166 equal-cost operating points, each algorithm grid-tuned over its
own grid, de-biased at $R=30{,}000$. Paired within each algorithm (same exploration, both rules):

| algorithm | normal rule $\to$ exact posterior | helps at |
|---|---:|---:|
| **Linear search** | **+5.49 $\pm$ 0.19 pp** (worst +1.06, best +12.08) | **100 %** |
| Binary search | +0.66 $\pm$ 0.11 pp (worst −1.62, best +5.69) | 65 % |
| Reverse engineering | +0.06 $\pm$ 0.03 pp | 46 % |

The gain tracks how many *deep* probes an algorithm takes and throws away. Reverse engineering takes
one shallow probe, so there is nothing to recover and its normal approximation was already adequate.

### 14.6 The consequence: the exploration schedule stops mattering

Mean advantage over brute force at the same operating points:

| | vs brute | outright best at |
|---|---:|---:|
| Linear search + posterior | **+11.40 pp** | 26 % |
| Reverse engineering + posterior | +11.17 pp | 14 % |
| Binary search + posterior | +11.14 pp | 39 % |
| Reverse engineering (normal, as reported) | +11.12 pp | 14 % |
| Binary search (normal, from $\hat\phi_0$) | +10.47 pp | 11 % |
| Linear search (tuned $s$, as reported) | +5.91 pp | 0 % |

Binary search with the posterior is **statistically tied** with reverse engineering
(−0.04 $\pm$ 0.09 pp) and with linear search (−0.26 $\pm$ 0.19 pp). It wins outright most often (39 %)
but by no reliable margin on average.

So the honest conclusion is not "binary search wins". It is that **once the data are used properly,
how the exploration is scheduled stops mattering**: a linear scan, a bisection and a single shallow
probe all land within 0.3 pp. What separated the three algorithms in Chapter 3 was never the search
strategy — it was how much of the exploration each one threw away. Reverse engineering looked best
because, taking one probe, it had the least to throw away.

---

## 15. Can the overshoot criterion be repaired instead of replaced? (No)

§14's posterior works, but it replaces Chapter 3's estimator-plus-normal-law machinery wholesale. The
cheaper question is whether a small change to Eq. (3.5) recovers the benefit, leaving the thesis
structure intact.

**The candidate repair.** Eq. (3.5) flags a probe that reads *low*:
$\hat\phi<\phi_1=$ the $(1-\alpha)$-quantile of $\mathcal N(\hat\phi_{\rm prev},\sigma_{\rm prev}^2)$.
That is one-sided, and aliasing is not: past the aliasing point the estimator returns $\pi/N-\phi$, a
reflection that lands *above* the previous estimate as often as below. A bisection's second probe sits
at $N\approx N_{\max}/2$, far beyond $N_{\rm opt}$, so it is the probe most likely to alias, and a
one-sided test accepts it whenever the reflection happens to read high. Testing both tails is one line
and the same distributional argument.

**Measured** (`analysis/twosided_study.py`, 124 live equal-cost points, all 23 scenarios, each arm
grid-tuned, $R=30{,}000$), against reverse engineering:

| arm | mean vs RE | beats RE | overshoot |
|---|---:|---:|---:|
| deepest-accepted pilot, one-sided (**Chapter 3 as written**) | −12.31 pp | 1 % | 0.00 % |
| deepest-accepted pilot, **two-sided** | −12.29 pp | 1 % | 0.00 % |
| $N=L$ with the two-sided test | −8.89 pp | 1 % | 3.04 % |
| opening probe $\hat\phi_0$ as pilot | −0.41 pp | 42 % | 0.46 % |
| exact posterior (§14) | **+0.23 pp** | 72 % | 0.28 % |

Two-sided versus one-sided, paired on the same explorations: **+0.02 $\pm$ 0.02 pp.** It does nothing.

**Why the repair cannot work.** The diagnosis was wrong, and the arm's own overshoot rate says so:
**0.00 %**. The deep pilot never aliases — it *under*-shoots. With a small $\sigma$ the factor
$\mathbb P(\phi<\pi/2N)$ becomes a sharp step at $N=\pi/(2\hat\phi)$, so $N^{*}$ collapses onto the naive
depth; and because acceptance means $\hat\phi\ge\phi_1$, the accepted probe is selected for reading
high, so $\pi/(2\hat\phi)$ is systematically too small. A more reliable test does not remove that bias
— the bias is created by *conditioning on acceptance at all*, which is intrinsic to any rule that
classifies probes before using them.

There is also a hard limit on reliability: at $N=N_{\rm opt}$ the two branches coincide exactly
($\pi/N-\phi=\phi$ when $N=\pi/2\phi$), so no test can separate them there. That is harmless for the
*estimate* but fatal for any rule that needs the *classification*, which is why $N=L$ still overshoots
3 % of trials even with the stricter test.

**Conclusion.** The criterion is not the problem and cannot be made into the solution. The three
coherent positions are: keep the deepest-accepted pilot and accept that binary search trails reverse
engineering wherever the bisection actually runs (what Chapter 3 already reports); use $\hat\phi_0$ and
accept that the exploration becomes decorative; or use the exact posterior, which is the only option
where the exploration does real work *and* the algorithm is competitive.

---

## 16. Is knowing $N_{\rm opt}$ exactly actually worth anything? (Almost nothing)

The bisection locates $N_{\rm opt}$ to $\pm1$ in $O(\log N_{\max})$ probes. That looks like the most
valuable thing any exploration phase produces, and it invites the thought that the safeguard — which
deliberately backs *off* $N_{\rm opt}$ — is squandering it. It is not, and the reason is a property of
the objective rather than of the rule.

**Grant perfect knowledge and measure.** $N=N_{\rm opt}$ with $\phi$ known exactly, against the exact
posterior which has to infer everything (20,000 trials each):

| scenario | budget | $N=N_{\rm opt}$, $\phi$ known | exact posterior | ceiling |
|---|---:|---:|---:|---:|
| $\mathcal U(0.01,0.1)$, $10^{-3}$ | 22,413 | 86.94 % | 83.28 % | 88.72 % |
| $\mathcal U(0.01,0.1)$, $10^{-3}$ | 89,653 | 99.08 % | **99.19 %** | 99.55 % |
| $\mathcal U(0.01,0.1)$, $10^{-4}$ | 2.24 M | 88.13 % | 88.02 % | 88.72 % |
| $\mathcal U(0.01,0.1)$, $10^{-6}$ | 89.7 G | 99.50 % | **99.54 %** | 99.55 % |
| $\mathcal U(0.001,0.01)$, $10^{-5}$ | 21.4 M | 86.83 % | **87.26 %** | 88.32 % |
| $\mathcal U(0.001,0.01)$, $10^{-5}$ | 85.7 M | 98.97 % | **99.44 %** | 99.52 % |

Perfect knowledge of $N_{\rm opt}$ is worth about **+0.4 pp on average**, and at half these points the
inferring algorithm is already *ahead* of it.

**Why: the objective has a wide plateau below $N_{\rm opt}$ and a cliff above it.** For a single known
$\phi$ ($\phi=0.02$, $B=2.24\cdot10^6$, $\varepsilon=10^{-4}$, so $N_{\rm opt}=78$):

| $N$ | % of $N_{\rm opt}$ | $\mathbb P(\text{converge})$ | $m\,p_0$ |
|---:|---:|---:|---:|
| 46 | 59 % | 95.77 % | 17883 |
| 62 | 79 % | 98.19 % | 3814 |
| **76** | **97 %** | **99.16 %** | 76 |
| 77 | 99 % | 99.10 % | 27.6 |
| 78 | 100 % | 96.25 % | 3.3 |
| 79 | 101 % | **0.00 %** | 2.4 |

Every depth in $N\in[62,77]$ — a band **19 % of $N_{\rm opt}$ wide** — is within 1 % of the best. The
optimum sits at $0.97\,N_{\rm opt}$, and $N_{\rm opt}$ itself is 2.9 pp *worse* than the plateau, because
$m\,p_0$ has collapsed to 3.3 and the readout is no longer informative. One step further is zero.

So the objective rewards **not exceeding** $N_{\rm opt}$, not **locating** it. Resolving $N_{\rm opt}$ to
$\pm1$ is worth nothing over resolving it to $\pm10\%$; the whole value of the exploration is the
one-sided bound. That is exactly what Eq. (3.8) optimises, and it is why a safeguard that backs off is
not throwing information away — there is nothing left to gain.

(The first row of the earlier table is the exception that proves it: at $\phi=0.05$, $B=22{,}413$,
$N=N_{\rm opt}$ scores 96 % against a plateau of 89 %, but $m\,p_0=0.3$ there — it is converging off the
deterministic readout, i.e. the degeneracy of `qmetrology/oracle.py`, not off the data.)
