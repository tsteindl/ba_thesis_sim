# The Berry–Esseen route, in full

**Companion to [`OVERSHOOT_CRITERION.md`](OVERSHOOT_CRITERION.md)**, which holds the verdict and
the numbered edit list. This document is the detailed derivation, the numerical verification, and
the reasoning behind the decision not to replace the criterion. Recommendations only — no `.tex`
was touched.

> **Terminology.** This document analyzes the normal quantile used by the Binary Search
> *overshoot classifier*. The thesis's later *statistical safeguard* is a separate mechanism: it
> uses a pilot-based truncated-normal approximation to choose the exploitation value $N^*$. Part 7
> audits how the two arguments fit together.

---

## Part 0 — First, your updated passage

You now have (LaTeX source, shown as code so it can be copied):

```latex
Because $\arccos$ is bounded by $|\arccos(x)| \leq \pi$, the entangled estimator satisfies
\begin{align}
\lim_{N\to\infty} \hat\phi_\text{ent}^{(N)} = \lim_{N\to\infty} \frac 1 N \arccos(\sqrt {p_0}) = 0,
\label{eq:est-conv}
\end{align}
implying that the estimator converges to zero. Once the optimal
choice of $N$ is exceeded, we can expect estimates to tend to zero.
```

Splitting the limit from the interpretation is the right move. Three things I would still change.

**(a) `p_0` depends on `N`, and the way it is written hides that.** `p_0 = cos^2(N phi)` is a
function of `N`, so pulling `arccos(sqrt(p_0))` outside the limit as though it were a constant is
not quite legitimate — `arccos(sqrt(p_0(N)))` oscillates forever and has no limit. What makes the
statement true is that it stays *bounded* while `1/N` vanishes. The clean way to write it is a
squeeze:

```
0  <=  phi_hat_ent^(N)  =  (1/N) arccos(sqrt(p_0(N)))  <=  pi/(2N)  ->  0.
```

**(b) The bound should be `pi/2`, not `pi`.** Since `sqrt(p_0) in [0,1]` and `arccos` maps `[0,1]`
onto `[0, pi/2]`, the sharp statement is `arccos(sqrt(p_0)) <= pi/2`. It is not merely tidier: the
factor of two is what later lets you conclude `phi_hat <= pi/(2N) < phi` for an overshooting `N`
(Part 3). With `pi` that chain does not close, and the detector argument cannot be completed at all.

**(c) The last sentence still overreaches.** *"Once the optimal choice of `N` is exceeded, we can
expect estimates to tend to zero"* attaches an `N -> infinity` conclusion to the event
`N > N_opt`. Those are different regimes. At `N = 1.02 N_opt` the estimate is about 2 % below
`phi` — nowhere near zero — yet that is exactly where the rule has to decide. Numerically, at
`phi = 0.05` (`N_opt = 31`):

| `N` | 32 | 40 | 60 | 100 | 300 | 1000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phi_hat` | 0.0482 | 0.0285 | 0.0024 | 0.0128 | 0.0024 | 0.00027 | 0.00005 |

Non-monotone, as you said, and genuinely tending to zero — but only over orders of magnitude in
`N`, not just past `N_opt`. What is true immediately past `N_opt`, and is what the detector needs,
is the *inequality*, not the limit.

**Suggested replacement** — paste this into your `.tex`:

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
although the approach is not monotone: by \eqref{eq:folded-estimator} the estimator zig-zags
within the envelope $\pi/(2N)$. The bound \eqref{eq:est-bound} is the more useful statement for
what follows, because it holds at every finite $N$: as soon as $N\phi > \pi/2$, that is as soon as
$N$ exceeds $N_{\mathrm{opt}}$, it gives $\hat\phi_{\text{ent}}^{(N)} \leq \pi/(2N) < \phi$, so an
overshooting probe necessarily reports a phase below the true one.
```

This keeps your equation, keeps your conclusion, and adds the one line Chapter 3 needs.

---

## Part 1 — Why Berry–Esseen is even applicable

Your original difficulty was real. You wanted to show `phi_hat` is approximately Gaussian. But
`phi_hat = arccos(sqrt(K/m'))/N` is a *nonlinear* function of `K`, and the delta method's error
depends on the curvature of that function. Its derivative,

```
d/dp [ arccos(sqrt(p))/N ]  =  -1 / (2N sqrt(p(1-p)))
```

blows up as `p -> 0`, which is exactly the aliasing boundary. So no bound on the transformed
variable can be uniform, and that is why the attempts failed — not for want of effort, but because
the statement is false near the boundary.

**The escape is that the test never needs `phi_hat` to be Gaussian.** Because `phi_hat` is a
strictly *decreasing* function of `K`, the event the rule tests can be rewritten on the count scale
with no approximation at all:

```
phi_hat_N < phi_1
  <=>  arccos(sqrt(K/m')) < N phi_1
  <=>  sqrt(K/m')         > cos(N phi_1)        (arccos is decreasing)
  <=>  K                  > m' cos^2(N phi_1)   =: k*
```

valid for any `phi_1` with `0 <= N phi_1 <= pi/2`. Checked over **300,000 random `(m', N, phi_1, K)`:
0 disagreements**, asserted in the script.

So the rule is a cut on `K`, and `K ~ Bin(m', p_0)` is a **sum of `m'` i.i.d. Bernoulli variables**
— the exact object the Berry–Esseen theorem was written for. The nonlinearity has been moved out of
the probability statement and into the *location of the cut*, where it is a deterministic quantity
we can compute exactly (Part 4).

---

## Part 2 — The theorem, and what it gives

**Berry–Esseen.** Let `X_1, ..., X_n` be i.i.d. with mean `mu`, variance `sigma^2 > 0` and
`rho = E|X - mu|^3 < infinity`. Then for `S_n = sum X_i`,

```
sup_x | P( (S_n - n mu)/(sigma sqrt(n)) <= x ) - Phi(x) |  <=  C rho / (sigma^3 sqrt(n))
```

with `C <= 0.4748` (Shevtsova, 2011).

For `X ~ Bernoulli(p)`: `mu = p`, `sigma^2 = pq`, and

```
rho = E|X - p|^3 = q * p^3 + p * q^3 = pq(p^2 + q^2)
```

so

```
rho / (sigma^3 sqrt(n))  =  pq(p^2+q^2) / ( (pq)^{3/2} sqrt(n) )  =  (p^2 + q^2) / sqrt(n p q)
```

giving, for `K ~ Bin(m', p_0)`:

```
   sup_x | P( (K - m'p_0)/sqrt(m' p_0 q_0) <= x ) - Phi(x) |   <=   0.4748 (p_0^2 + q_0^2) / sqrt(m' p_0 q_0)
```

**This is non-asymptotic.** It holds for every `m'` and every `p_0`, with no "for `m` large enough".
That is precisely the property your earlier justification lacked, and it is why this route answers
the objection rather than restating it.

### Verified numerically

The bound is only useful if it is actually respected and not absurdly loose. Both checked:

| `m'` | `p_0` | `m' min(p,q)` | true distance | Berry–Esseen bound | holds |
|---:|---:|---:|---:|---:|:--:|
| 50 | 0.500 | 25.0 | 0.056 | 0.067 | yes |
| 50 | 0.200 | 10.0 | 0.084 | 0.114 | yes |
| 200 | 0.500 | 100.0 | 0.028 | 0.034 | yes |
| 200 | 0.200 | 40.0 | 0.042 | 0.057 | yes |
| 200 | 0.050 | 10.0 | 0.083 | 0.139 | yes |
| 800 | 0.500 | 400.0 | 0.014 | 0.017 | yes |
| 800 | 0.050 | 40.0 | 0.042 | 0.070 | yes |
| 800 | 0.0125 | 10.0 | 0.083 | 0.147 | yes |

Worst case over the whole regularity region `m' min(p_0, 1-p_0) >= 10`:

| `m'` | Berry–Esseen bound | true distance |
|---:|---:|---:|
| 50 | 0.114 | 0.084 |
| 200 | 0.139 | 0.083 |
| 800 | 0.147 | 0.083 |

The bound is a factor 1.4–1.8 loose, which is normal for Berry–Esseen. Two things worth noticing:

- **The true distance is 0.083 at every `m'`.** That is not coincidence — it is the ceiling imposed
  by the `m' min(p_0,1-p_0) >= 10` edge itself, and increasing `m'` just moves *where* that edge
  sits, not how bad it is there. It matches the independent Gaussianity audit (max KS 0.078–0.084)
  exactly, which is a useful cross-check that two different calculations agree.
- **The bound gets slightly worse as `m'` grows**, because a larger `m'` admits smaller `p_0` into
  the region. Restricted to fixed `p_0` the bound falls as `1/sqrt(m')`, as it must.

---

---

## Part 2b — What is `p_0`, and is it legitimate to use something unknowable?

**Definition, which the prose above should have stated outright:** `p_0 = cos^2(N phi)` is the true
probability that the circuit reads out `0` at the candidate depth `N`. It is the quantity the
estimator inverts, and — you are right — it is **unknown**, because `phi` is unknown.

That is not a problem, and it is worth being precise about why.

**The test never uses `p_0`.** The cut is

```
k* = m' cos^2(N phi_1),        phi_1 = phi_hat_acc + z_alpha / (2 N sqrt(m'))
```

which involves only `N`, `m'` and the reference estimate — all observable. You can run the rule
without knowing `p_0`, and indeed your code does exactly that. Nothing unknowable enters the
decision.

**`p_0` enters only when we ask how often the rule is wrong.** The statement is conditional:

> *if* `N` is safe and the true phase is `phi`, *then* `K ~ Bin(m', p_0)` with `p_0 = cos^2(N phi)`,
> and the probability of a false alarm is `1 - F_Bin(floor(k*); m', p_0)`.

This is the ordinary situation in hypothesis testing. The null hypothesis here is **composite** —
`H_0: N <= N_opt`, equivalently `N phi <= pi/2` — so it does not pin down a single distribution but
a family of them, indexed by where in the safe range you happen to be. The rejection probability is
therefore a *function* of the unknown parameter, not a single number, and the test's **size** is
that function's supremum over the null:

```
size = sup over safe (N, phi) of P(reject)
```

That is precisely what the table reports: the range of the rejection probability across safe depths
`N/N_opt`, whose upper end is the size. Reporting a range rather than one number is not a weakness
of the analysis; it is what a composite null requires.

**Contrast with the old justification.** Under the Gaussian story the unknown `p_0` was buried
inside a claim ("`phi_hat` is approximately normal with variance `1/(4m'N^2)`") that quietly holds
for some `p_0` and fails for others, with no way to see which. Writing the null explicitly in terms
of `p_0` makes the dependence visible and lets us compute where it is good — which is the whole
improvement.

---

## Part 2c — "If `K` is Gaussian, isn't `phi_hat` Gaussian too, by the delta method?"

Short answer: **asymptotically yes, and numerically yes inside the regularity region — but it is
not a theorem at finite `m'`, and that gap is exactly what Berry--Esseen was brought in to close.**

**Where you are right.** The delta method says that if `sqrt(m')(K/m' - p_0)` converges to a normal
and `g(p) = arccos(sqrt(p))/N` is differentiable at `p_0` with `g'(p_0)` finite and non-zero, then
`sqrt(m')(g(K/m') - g(p_0))` converges to a normal too. So Gaussianity does transfer — this is
exactly your Lemma 2.6.3, and it is correct on the interior.

**Where the transfer stops being free.** The delta method is a statement about limits. Berry--Esseen
is a statement about *every finite* `m'`, and that is the whole reason for invoking it. Finite-sample
accuracy does not pass through a nonlinear map automatically:

- Berry--Esseen bounds `sup_k |F_K(k) - Phi(k)|`, the distance from the count's law to *its* normal.
- `phi_hat`'s distance is measured against a *different* normal, the delta-method
  `N(phi, 1/(4m'N^2))`. Since `phi_hat = g(K/m')` with `g` monotone,
  `sup_y |F_{g}(y) - Phi_2(y)| = sup_k |F_K(k) - Phi_2(g(k))|`, and `Phi_2 o g` **is not a normal
  CDF**. The two suprema are different objects, so a bound on one is not automatically a bound on
  the other.
- Getting one for `phi_hat` directly would require controlling the curvature of `g` over the range
  where `K` has mass, and `g''` blows up as `p_0 -> 0` — the aliasing boundary. That is the same
  obstruction that defeated the earlier attempts, and it is real, not an artefact.

**What actually happens numerically.** The two distances turn out to be nearly identical inside the
regularity region, and to separate outside it:

| `m'` | `N/N_opt` | `m' min(p,1-p)` | KS of `K` | KS of `phi_hat` | Berry--Esseen bound |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.40 | 45 | 0.0327 | 0.0327 | 0.0387 |
| 200 | 0.50 | 100 | 0.0282 | 0.0282 | 0.0336 |
| 200 | 0.80 | 19 | 0.0607 | 0.0611 | 0.0945 |
| 800 | 0.88 | 28 | 0.0501 | 0.0503 | 0.0850 |

Across the region: **max ratio `KS(phi_hat)/KS(K) = 1.006`**, and the Berry--Esseen bound happens to
cover `phi_hat` at every point tested (14/14). Outside the region they come apart:

| `m'` | `N/N_opt` | `m' min(p,1-p)` | KS of `K` | KS of `phi_hat` |
|---:|---:|---:|---:|---:|
| 200 | 0.90 | 4.89 | 0.116 | 0.123 |
| 200 | 0.95 | 1.23 | 0.234 | 0.278 |
| 200 | 0.98 | 0.20 | 0.493 | **0.634** |

**So what should the thesis say?** Both framings are available, and they differ only in what kind of
claim you are making:

| Framing | The claim | Its status |
|---|---|---|
| via `K` (recommended) | Berry--Esseen bounds the error of the normal approximation to `K`, which is the only approximation the test uses | a **theorem**, non-asymptotic |
| via `phi_hat` | `phi_hat` is close to its delta-method normal, and numerically as close as `K` is to its own | **asymptotic theorem plus an empirical observation** |

The second is exactly the kind of "verified numerically, close enough" argument that drew the
criticism in the first place. The first is a bound. Since the test is a cut on `K` anyway, taking
the route through `K` costs one sentence and upgrades the claim from empirical to proved.

**You do not have to choose, though.** Both statements are true and they serve different consumers:

- **Lemma 2.6.3** (`phi_hat` asymptotically normal) stays exactly as it is. The **safeguard**
  (Theorem 3.2.1) consumes it, and there it is applied at safe `N`, where it holds.
- **Berry--Esseen on `K`** justifies the **overshoot threshold**, where a finite-sample statement is
  what is wanted.

Nothing you have written needs to be deleted. The new material sits alongside it.

---

## Part 3 — What the structural inequality proves, and what it does not

Berry–Esseen is about calibration under a safe probe. A different, exact fact gives the direction
in which an overshooting estimate moves. Since the estimator always lies in its principal range,

$$
0\leq \hat\phi_N
=\frac{1}{N}\arccos\!\sqrt{\frac{K}{m'}}
\leq \frac{\pi}{2N}.
$$

If $N$ has overshot, then $N\phi>\pi/2$, and therefore

$$
\hat\phi_N\leq\frac{\pi}{2N}<\phi.
$$

Thus, **every overshooting probe reports a value below the true phase**. This statement is exact for
every outcome $K$ and every shot count $m'$. It was checked over 200,000 random overshooting
configurations, with no violation.

However, this proves only the *direction* of the change. It does **not** prove that the implemented
rule detects every overshoot. The rule fires when

$$
\hat\phi_N<\phi_1,
\qquad
\phi_1=\hat\phi_{\mathrm{acc}}+z_\alpha\sigma_N,
\qquad
\sigma_N=\frac{1}{2N\sqrt{m'}},
$$

where $z_\alpha=\Phi^{-1}(\alpha)$ and $\alpha=1-\mathrm{conf}$. Even with a perfect reference
$\hat\phi_{\mathrm{acc}}=\phi$, the condition is

$$
\hat\phi_N<\phi+z_\alpha\sigma_N.
$$

For $\alpha<1/2$, we have $z_\alpha<0$, so the threshold lies *below* the true phase. A marginal
overshoot can therefore satisfy

$$
\phi+z_\alpha\sigma_N\leq\hat\phi_N<\phi
$$

and be missed. Only at $\alpha=1/2$, where $z_\alpha=0$, would a perfect reference make the exact
inequality $\hat\phi_N<\phi$ sufficient to detect every strict overshoot. A noisy, adaptively
selected reference introduces another source of misses and false alarms.

At the exact boundary $N\phi=\pi/2$, $p_0=0$, hence $K=0$ almost surely and
$\hat\phi_N=\pi/(2N)=\phi$. The estimator is not Gaussian there, but the rule remains perfectly
well-defined. With a perfect reference and $\alpha<1/2$, it does not fire. That is appropriate if
the boundary itself is counted as non-aliasing; the difficulty is distinguishing points just below
from points just above the boundary. Their single-probe binomial laws become arbitrarily similar,
so no threshold can simultaneously have negligible false alarms immediately below the boundary and
perfect power immediately above it.

---

## Part 4 — Where exactly the Gaussian threshold lands on the count scale

This part answers one precise question: if the phase threshold is constructed with a normal
quantile, where does that threshold land when expressed in units of the binomial count's standard
deviation?

### 4.1 Scope of the calculation

For the moment, make two idealizations:

1. the candidate is on the identifiable branch, $0<N\phi<\pi/2$; and
2. the accepted reference equals the true phase, $\hat\phi_{\mathrm{acc}}=\phi$.

The second assumption isolates the normal-quantile calibration from reference noise. Part 6 puts
the reference back. Define

$$
\theta:=N\phi,
\qquad
p_0:=\cos^2\theta,
\qquad
q_0:=1-p_0=\sin^2\theta.
$$

The count then satisfies

$$
K\sim\operatorname{Binomial}(m',p_0).
$$

The implemented lower phase threshold is

$$
\phi_1
=\phi+z_\alpha\sigma_N,
\qquad
z_\alpha:=\Phi^{-1}(\alpha),
\qquad
\sigma_N:=\frac{1}{2N\sqrt{m'}}.
$$

It is helpful to measure the phase displacement after multiplication by $N$:

$$
\delta
:=N(\phi_1-\phi)
=Nz_\alpha\sigma_N
=\frac{z_\alpha}{2\sqrt{m'}}.
$$

Therefore

$$
N\phi_1=\theta+\delta.
$$

For the usual settings $\alpha<1/2$, $z_\alpha<0$ and hence $\delta<0$: the lower phase threshold
lies below the true phase.

### 4.2 Convert the phase threshold into a count threshold

On $[0,\pi/2]$, the map

$$
k\longmapsto \frac{1}{N}\arccos\!\sqrt{\frac{k}{m'}}
$$

is strictly decreasing. Provided $0<N\phi_1<\pi/2$, the rejection event is therefore

$$
\hat\phi_N<\phi_1
\quad\Longleftrightarrow\quad
K>m'\cos^2(N\phi_1).
$$

Define the real-valued cutoff

$$
k^*:=m'\cos^2(N\phi_1)
=m'\cos^2(\theta+\delta).
$$

There is no rounding ambiguity in the probability statement: because $K$ is integer-valued,
$\{K>k^*\}=\{K>\lfloor k^*\rfloor\}$.

### 4.3 Measure the cutoff's displacement from the binomial mean

The binomial mean is $m'p_0=m'\cos^2\theta$. Hence

$$
\frac{k^*}{m'}-p_0
=\cos^2(\theta+\delta)-\cos^2\theta.
$$

Use

$$
\cos^2 x=\frac{1+\cos(2x)}{2}
$$

to obtain

$$
\cos^2(\theta+\delta)-\cos^2\theta
=\frac{\cos(2\theta+2\delta)-\cos(2\theta)}{2}.
$$

Now apply

$$
\cos A-\cos B
=-2\sin\!\left(\frac{A+B}{2}\right)
     \sin\!\left(\frac{A-B}{2}\right)
$$

with $A=2\theta+2\delta$ and $B=2\theta$. This gives the exact identity

$$
\boxed{
\frac{k^*}{m'}-p_0
=-\sin(2\theta+\delta)\sin\delta
}.
$$

No normal approximation or Taylor expansion has been used. Numerically, the identity agreed to
within $3.6\times10^{-16}$ over 200,000 random parameter triples.

### 4.4 Standardize the cutoff

Berry–Esseen concerns the standardized count

$$
Z:=\frac{K-m'p_0}{\sqrt{m'p_0q_0}}.
$$

The count cutoff $k^*$ corresponds to

$$
u
:=\frac{k^*-m'p_0}{\sqrt{m'p_0q_0}}.
$$

Substituting the exact displacement gives

$$
u
=-\frac{m'\sin(2\theta+\delta)\sin\delta}
        {\sqrt{m'p_0q_0}}.
$$

Since $0<\theta<\pi/2$,

$$
2\sqrt{p_0q_0}
=2\sin\theta\cos\theta
=\sin(2\theta).
$$

Therefore

$$
u
=-2\sqrt{m'}\sin\delta\,
  \frac{\sin(2\theta+\delta)}{\sin(2\theta)}.
$$

Finally, multiply and divide by $z_\alpha$:

$$
\boxed{
u=-z_\alpha A(m',z_\alpha)B(\theta,\delta)
}
$$

with

$$
A(m',z_\alpha)
:=
\begin{cases}
\dfrac{2\sqrt{m'}\sin\delta}{z_\alpha},&z_\alpha\neq0,\\[6pt]
1,&z_\alpha=0,
\end{cases}
$$

and

$$
B(\theta,\delta)
:=\frac{\sin(2\theta+\delta)}{\sin(2\theta)}.
$$

The value $A=1$ at $z_\alpha=0$ is the continuous extension of the quotient. This detail matters
because the thesis reports $\mathrm{conf}=0.5$, for which $\alpha=0.5$ and $z_\alpha=0$.

### 4.5 What would perfect placement look like?

The rejection event is $Z>u$. Under an exact standard normal law, its probability would be

$$
\mathbb P(Z>u)\approx1-\Phi(u)=\Phi(-u).
$$

The nominal lower-tail level is

$$
\alpha=\Phi(z_\alpha).
$$

Thus a perfectly placed cutoff has

$$
u=-z_\alpha,
$$

because then

$$
1-\Phi(u)=1-\Phi(-z_\alpha)=\Phi(z_\alpha)=\alpha.
$$

The two factors $A$ and $B$ measure the departure from this ideal:

- $A$ measures the error in replacing $\sin\delta$ by $\delta$. For fixed $z_\alpha$,
  $A\to1$ as $m'\to\infty$.
- $B$ measures the curvature of $\cos^2\theta$ at the operating point. For fixed interior
  $\theta$, $B\to1$ as $m'\to\infty$ because $\delta\to0$.
- The convergence is not uniform in $\theta$. The denominator $\sin(2\theta)$ tends to zero as
  $\theta\to0$ or $\theta\to\pi/2$. The latter is the aliasing boundary.

Consequently, the limit $u\to-z_\alpha$ is valid for a **fixed interior operating point**. It must
not be read as a uniform statement up to the boundary.

### 4.6 A numerical example

Take $m'=200$, $\alpha=0.05$, and $\theta=\pi/4$, so $p_0=q_0=1/2$. Then

$$
z_\alpha=-1.64485,
\qquad
\delta=-0.05815,
$$

$$
A=0.99944,
\qquad
B=0.99831,
\qquad
u=1.64115.
$$

The Gaussian upper tail at the actual cutoff is

$$
\Phi(-u)=0.05038,
$$

compared with the nominal $\alpha=0.05000$. At this interior point, the nonlinear threshold
placement contributes only about $3.8\times10^{-4}$ absolute probability error. The next part adds
the separate error from replacing the binomial count distribution by a normal distribution.

---

## Part 5 — The finite-sample error bound, step by step

### 5.1 Define the probability being bounded

Under the idealized perfect reference of Part 4, let

$$
R:=\{\hat\phi_N<\phi_1\}=\{K>k^*\}=\{Z>u\}
$$

be the event that a safe probe is classified as an overshoot. Its achieved false-alarm probability
at the fixed operating point $(m',p_0)$ is

$$
\beta(m',p_0):=\mathbb P_{p_0}(R)=\mathbb P_{p_0}(Z>u).
$$

The nominal probability is

$$
\alpha=\Phi(z_\alpha).
$$

Berry–Esseen gives, for $0<p_0<1$,

$$
\sup_{x\in\mathbb R}
\left|
\mathbb P_{p_0}(Z\leq x)-\Phi(x)
\right|
\leq
\varepsilon_{\mathrm{BE}}(m',p_0),
$$

where

$$
\varepsilon_{\mathrm{BE}}(m',p_0)
:=
C\,
\frac{p_0^2+q_0^2}{\sqrt{m'p_0q_0}}.
$$

Evaluate this inequality at the one cutoff the rule actually uses, $x=u$:

$$
\left|
\mathbb P_{p_0}(Z\leq u)-\Phi(u)
\right|
\leq\varepsilon_{\mathrm{BE}}(m',p_0).
$$

Taking complements does not change the absolute difference, so

$$
\left|
\mathbb P_{p_0}(Z>u)-\bigl(1-\Phi(u)\bigr)
\right|
\leq\varepsilon_{\mathrm{BE}}(m',p_0).
$$

Since $1-\Phi(u)=\Phi(-u)$, the triangle inequality yields

$$
\begin{aligned}
|\beta(m',p_0)-\alpha|
&\leq
|\beta(m',p_0)-\Phi(-u)|
+|\Phi(-u)-\Phi(z_\alpha)|\\
&\leq
\varepsilon_{\mathrm{BE}}(m',p_0)
+|\Phi(-u)-\Phi(z_\alpha)|.
\end{aligned}
$$

Therefore

$$
\boxed{
|\text{achieved false-alarm probability}-\text{nominal }\alpha|
\leq
\underbrace{|\Phi(-u)-\Phi(z_\alpha)|}_{\text{threshold-placement error}}
+
\underbrace{C\frac{p_0^2+q_0^2}{\sqrt{m'p_0q_0}}}_{\text{binomial-to-normal error}}
}.
$$

The first term is the finite-sample effect of putting a delta-method phase threshold onto the count
scale. The second is the Berry–Esseen bound for the count itself. Both terms are finite-sample and
computable once $(m',p_0,\alpha)$ is fixed.

The script checked this inequality at 294 grid points across
$m'\in\{50,200,800\}$, $\mathrm{conf}\in\{0.5,0.66,0.95\}$, and the stated regularity region,
with no violation.

### 5.2 Numerical size of the two terms

Using the same full regularity grid as the thesis table gives:

| $m'$ | maximum placement term | maximum Berry–Esseen term | maximum actual size error |
|---:|---:|---:|---:|
| 50 | 0.018 | 0.114 | 0.077 |
| 200 | 0.024 | 0.139 | 0.081 |
| 800 | 0.026 | 0.147 | 0.080 |

The placement term is smaller than the conservative Berry–Esseen term on this grid. The realized
size error is also smaller than the bound, as expected: Berry–Esseen controls the worst CDF error
over *all* cutoffs, whereas the criterion uses one cutoff.

These numbers should not be described merely as "small." For example, an absolute CDF error of
$0.08$ is modest on a $0$--$1$ scale but large relative to a nominal level of $0.05$. The scientific
statement is the quantitative one: in the declared region the rigorous worst-case bound is at most
$0.147$, the exactly enumerated Kolmogorov distance is at most about $0.084$, and the achieved size
at the criterion's own cutoff differs from nominal by at most $0.081$ in the reported grid.

### 5.3 Why Kolmogorov distance is the relevant metric

The left-hand side of the Berry–Esseen theorem,

$$
d_{\mathrm K}(F_Z,\Phi)
:=\sup_x|F_Z(x)-\Phi(x)|,
$$

is the Kolmogorov distance. It is not an unrelated normality diagnostic added after the fact. The
overshoot rule is a one-sided threshold event, so its probability is a CDF or tail probability at a
cutoff. A uniform CDF bound directly limits the absolute error of *every such threshold
probability*. That is exactly the error notion the rule needs.

The "true KS" column is not needed to make the theorem valid. It is useful because the
Berry–Esseen bound is conservative: exact enumeration of the binomial distribution shows how much
of the allowed error is actually attained. A histogram, Shapiro–Wilk test, or visual normality claim
would be less directly connected to the decision probability.

The implementation uses $C=0.4748$, which is a valid 2011 upper bound. A later result gives the
slightly sharper universal i.i.d. bound $C\leq0.4690$. Replacing $0.4748$ by $0.4690$ would reduce the
reported Berry–Esseen values by only about $1.2\%$ and would not change the interpretation.

### 5.4 When is $m'$ large enough?

There is no answer in terms of $m'$ alone. The bound also depends on the operating probability
$p_0$. For a desired absolute CDF error tolerance $\eta$ at a fixed $p_0$, a sufficient condition is

$$
m'
\geq
\frac{C^2(p_0^2+q_0^2)^2}{\eta^2p_0q_0}.
$$

Using $C=0.4748$:

| $p_0$ | $m'$ for bound $\leq0.10$ | $m'$ for bound $\leq0.05$ | $m'$ for bound $\leq0.01$ |
|---:|---:|---:|---:|
| 0.50 | 23 | 91 | 2,255 |
| 0.20 | 66 | 261 | 6,516 |
| 0.05 | 389 | 1,555 | 38,871 |

As $p_0\to0$ or $p_0\to1$, the required $m'$ diverges. Hence no finite $m'$ makes the bound small
uniformly all the way to either boundary.

For fixed $p_0$, the bound scales as $m'^{-1/2}$. It is better to call this the classical
square-root rate than a "fast" rate: halving the bound requires four times as many shots. Moreover,
the thesis's region

$$
m'\min(p_0,1-p_0)\geq10
$$

expands toward more extreme $p_0$ as $m'$ grows. At its moving edge, the expected rare outcome count
remains $10$, so the worst-case bound over the *whole region* need not decrease with $m'$.

The number $10$ is a conventional expected-count marker, not a theorem saying that normality
"starts" there. Berry–Esseen itself is valid for every $m'$ when $0<p_0<1$; the useful question is
whether its right-hand side is below a tolerance chosen for the decision problem.

### 5.5 What happens at the aliasing boundary?

At $N\phi=\pi/2$, $p_0=0$ and the standardized variable $Z$ is undefined because its variance is
zero. Berry–Esseen and the delta method do not apply there. Just inside the boundary, their bounds
become poor because $m'p_0$ is small.

This does not make the criterion undefined. The exact binomial law remains available, including the
degenerate boundary case. What fails is **normal calibration**, not the count-based definition of
the rule. Near the boundary, performance must therefore be justified by exact binomial
probabilities, a convolution that includes the noisy reference, or end-to-end simulation. The
Berry–Esseen table by itself does not establish near-boundary detector power.

### 5.6 Why the displayed worst case does not improve with $m'$

There are two interacting effects:

1. $K$ is discrete, so its CDF has jumps and the attainable rejection probabilities form a ladder.
2. As $m'$ increases, the condition $m'\min(p_0,q_0)\geq10$ admits operating points closer to the
   boundary, where the rare expected count is still only $10$.

At a fixed interior $p_0$, both the CDF jumps and the Berry–Esseen bound shrink with $m'$. The nearly
constant worst-case values in the table arise because the domain over which the maximum is taken is
changing with $m'$, not because the central limit theorem has stopped working.

---

## Part 6 — Should you replace the criterion with an exact binomial test?

There is an important qualification to the phrase "exact binomial test." The true count law is

$$
K\mid\phi\sim\operatorname{Binomial}\!\left(m',\cos^2(N\phi)\right),
$$

but $\phi$ is unknown. Replacing the normal quantile by a binomial quantile computed from

$$
p_{\mathrm{ref}}:=\cos^2(N\hat\phi_{\mathrm{acc}})
$$

would be exact only **conditional on treating the reference as the truth**. It would remove the
count-normal and phase-to-count placement approximations, but it would not remove:

- noise in $\hat\phi_{\mathrm{acc}}$;
- adaptive selection of the deepest accepted reference;
- the possibility that the accepted reference has itself overshot; or
- the fundamental near-boundary similarity between the safe and aliased branches.

Thus, a plug-in binomial quantile would not make the implemented adaptive procedure an exact
level-$\alpha$ test under the composite null "the candidate is safe."

The numerical error budget illustrates the point. At $m'=200$ and the representative operating
points used by the script:

| nominal $\alpha$ | normal placement of the cut | late-reference effect | first-reference effect |
|---:|---:|---:|---:|
| 0.50 | 0.061 | 0.001 | 0.022 |
| 0.34 | 0.047 | 0.042 | 0.105 |
| 0.05 | 0.015 | 0.063 | 0.319 |

These are illustrative pointwise differences, not uniform guarantees. They show that correcting the
normal cutoff can leave an error of comparable or larger size from the estimated reference.

There is also a practical reason not to change the current thesis algorithm now: $\mathrm{conf}$ is
tuned against end-to-end convergence rather than reported as a calibrated coverage guarantee. A
different cutoff changes the algorithm and would require re-tuning and re-evaluating all reported
results.

The defensible conclusion is therefore:

> Keep the present rule for this thesis, describe its cutoff calibration quantitatively, and retain
> the caveat that the adaptive plug-in rule is not an exact confidence test. If exact calibration of
> the full history becomes the objective, use the exact likelihood or posterior for all probe counts
> rather than merely replacing one normal quantile by a plug-in binomial quantile.

---

## Part 7 — What the current thesis contains, and what is still missing

The current thesis already incorporates most of the material that is necessary for the
Berry–Esseen justification of the binary-search overshoot threshold:

1. Lemma 2.6.3 restricts the delta-method normal approximation to the interior
   $0<N\phi<\pi/2$.
2. Section 3.2.2 states the exact monotone equivalence between the phase comparison and the count
   cutoff.
3. It gives the Bernoulli Berry–Esseen bound and defines $p_0=\cos^2(N\phi)$.
4. Equations (3.7)--(3.8) contain the closed-form threshold map derived in Part 4.
5. Table 3.1 separates the rigorous Berry–Esseen bound, the exactly enumerated Kolmogorov distance,
   and the achieved size error.
6. The prose explicitly says that the noisy, adaptively selected reference prevents the detector
   from being an exact confidence test.
7. The statistical-safeguard section separately labels its depth score as an asymptotic plug-in
   approximation and discloses the selection bias of the deepest accepted binary-search pilot.

The thesis does **not** need to reproduce every calculation or diagnostic in this note. It does,
however, still need six clarifications to make the logical scope unambiguous:

1. **Distinguish the two mechanisms.** Berry–Esseen justifies the normal quantile in the
   binary-search *overshoot classifier*. The later *statistical safeguard* is a different rule that
   uses a truncated-normal plug-in posterior to choose the exploitation value $N^*$.
2. **State the non-uniformity.** The limit in Equation (3.8) is for fixed
   $0<\theta<\pi/2$; it is not uniform as $\theta\to\pi/2$. Table 3.1 intentionally excludes the
   boundary through $m'\min(p_0,1-p_0)\geq10$.
3. **Handle $z_\alpha=0$.** Equation (3.8) contains a quotient by $z_\alpha$, but the thesis reports
   $\alpha=0.5$, for which $z_\alpha=0$. The quotient must be declared to have its continuous value
   $1$ at zero, as in Part 4.4.
4. **Do not turn directionality into a power guarantee.** The exact inequality
   $\hat\phi_N<\phi$ after overshooting does not imply
   $\hat\phi_N<\phi+z_\alpha\sigma_N$, and it says nothing about comparison with a noisy reference.
5. **State the threshold-range condition.** The equivalence
   $\hat\phi_N<\phi_1\Longleftrightarrow K>m'\cos^2(N\phi_1)$ uses the monotonicity of $\arccos$ on
   the principal branch and requires $0<N\phi_1<\pi/2$. If the adaptive threshold falls outside the
   estimator's range, the event is instead empty or automatic and should be handled directly.
6. **Do not equate the approximate safeguard score with the exact success probability.** In
   Theorem 3.2.2, the truncated-normal CDF and Gaussian accuracy factor define an approximate score.
   The second line of Equation (3.20) should therefore define the maximizer of that score, rather
   than be written as an exact equality to the maximizer of the unknown finite-sample probability.

For the statistical safeguard itself, the algebra is correct **under its working Gaussian model**:
the uniform prior and phase-independent asymptotic variance produce a truncated-normal plug-in
posterior, and the displayed objective is the product of approximate validity and accuracy factors.
It is not a finite-sample theorem about the implemented adaptive procedure. In particular, the
delta-method approximation used for the accuracy factor is not uniform at the branch boundary, and
the binary-search pilot is selected from the same exploration history. The thesis acknowledges the
second issue and repeatedly calls the score approximate; adding the first boundary qualification
would make the limitation complete.

---

## Part 8 — Reproducing all of it

```powershell
python analysis/overshoot_criterion.py
```

Asserts, and fails loudly if any breaks:

1. the binomial restatement is exact (0 disagreements in 300,000 draws);
2. an overshooting probe always reads below the truth (200,000 draws), which verifies direction but
   not perfect detector power;
3. the Berry–Esseen bound is respected by the true distance;
4. the two-part error bound holds at all 294 regularity-region grid points.

| File | Contents |
|---|---|
| [`overshoot_size.csv`](../results/overshoot_size.csv) | achieved vs nominal size, Berry–Esseen bound and true distance, per `(m', conf)` |
| [`overshoot_error_bound.csv`](../results/overshoot_error_bound.csv) | the two-part decomposition point by point, including the $A$ and $B$ factors of Part 4 |
| [`overshoot_power.csv`](../results/overshoot_power.csv) | exact $\mathbb P(\text{rule fires})$ versus depth, per $(m',\mathrm{conf},\mathrm{reference})$ |
| [`overshoot_bracket_walk.csv`](../results/overshoot_bracket_walk.csv) | the probe sequences used to assess reference quality as the bracket narrows |
| [`results/tex/tab_overshoot_operating.tex`](../results/tex/tab_overshoot_operating.tex) | the generated table |
