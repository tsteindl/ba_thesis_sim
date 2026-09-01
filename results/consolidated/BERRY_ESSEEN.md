# The Berry–Esseen route, in full

**Companion to [`OVERSHOOT_CRITERION.md`](OVERSHOOT_CRITERION.md)**, which holds the verdict and
the numbered edit list. This document is the detailed derivation, the numerical verification, and
the reasoning behind the decision not to replace the criterion. Recommendations only — no `.tex`
was touched.

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

## Part 3 — Power needs no approximation at all

Berry–Esseen is about the *size* of the test. The *power* side needs nothing:

```
sqrt(p_0) in [0,1]   =>   arccos(sqrt(p_0)) <= pi/2   =>   N phi_hat <= pi/2
```

and "N has overshot" means `N phi > pi/2`. Chaining,

```
phi_hat  <=  pi/(2N)  <  phi.
```

**An overshooting probe always reads below the truth** — every outcome `K`, every `N > N_opt`, every
shot count. Verified over 200,000 random overshooting configurations: `max(phi_hat - phi) < 0`,
never positive; asserted in the script.

So against a known `phi` the rule could never miss an overshoot. It misses only because it compares
against `phi_hat_acc`, which is itself an estimate. That is the whole failure mode, and it is a
statement about the reference, not about Gaussianity.

---

## Part 4 — Where exactly the Gaussian threshold lands (closed form)

Berry–Esseen bounds the normal approximation to `K`. But your threshold is not chosen on the `K`
scale — it is chosen on the `phi` scale, as `phi_1 = phi_hat_acc + z sigma_N` with
`sigma_N = 1/(2N sqrt(m'))`. To connect the two we need to know where that lands as a cut on `K`.
This is exact algebra, no approximation.

Write `theta = N phi` and `delta = N z sigma_N = z/(2 sqrt(m'))`. Then `N phi_1 = theta + delta`
(taking the reference at the true `phi` for now), and

```
k*/m'  =  cos^2(theta + delta).
```

Using `cos A - cos B = -2 sin((A+B)/2) sin((A-B)/2)` with `A = 2theta + 2delta`, `B = 2theta`, and
`cos^2 u = (1 + cos 2u)/2`:

```
k*/m' - p_0  =  [cos(2theta + 2delta) - cos(2theta)]/2  =  -sin(2 theta + delta) sin(delta)
```

**Exact.** Verified: `max |LHS - RHS| = 3.6e-16` over 200,000 random `(m', z, theta)`.

Now standardise. Since `p_0 = cos^2 theta` and `q_0 = sin^2 theta`, we have
`sin(2 theta) = 2 sin theta cos theta = 2 sqrt(p_0 q_0)`, so

```
u  :=  (k* - m' p_0)/sqrt(m' p_0 q_0)
    =  -m' sin(2theta+delta) sin(delta) / sqrt(m' p_0 q_0)
    =  -2 sqrt(m') sin(delta) * sin(2 theta + delta)/sin(2 theta)
    =  -z * A(m', z) * B(theta, delta)
```

with

```
A(m', z) = 2 sqrt(m') sin(delta)/z        ->  1   as m' grows          (the sin d ~ d error)
B(theta, delta) = sin(2theta+delta)/sin(2theta)  ->  1   as delta -> 0  (singular as sin 2theta -> 0)
```

Verified: `max |u + z A B| = 3.2e-13` over 200,000 draws.

**Read this off.** A perfectly placed cut would give `u = -z` exactly, hence size exactly `Phi(z)`.
The two correction factors say precisely how the delta method errs:

- `A` is the error from linearising `sin(delta) ~ delta`. It depends only on `m'` and `z`, never on
  where you are, and it is tiny: at `m' = 50, z = -1.645`, `A = 0.9978`.
- `B` is the error from linearising `cos^2` around `theta`. It is the one that matters, and it
  degrades as `sin(2 theta) -> 0`, i.e. as `theta -> 0` or `theta -> pi/2`. **`theta -> pi/2` is the
  aliasing boundary.** So the closed form independently rediscovers the boundary problem — and
  localises it in one factor.

---

## Part 5 — The complete error bound

Combining Parts 2 and 4:

```
| achieved size - nominal alpha |   <=   | Phi(-u) - Phi(z) |   +   C (p_0^2+q_0^2)/sqrt(m' p_0 q_0)
                                          ^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          threshold placement       normal vs binomial
                                          (closed form, Part 4)     (Berry-Esseen, Part 2)
```

Both terms are computable; neither is asymptotic. **Verified on 294 grid points across
`m' in {50,200,800}`, `conf in {0.5,0.66,0.95}` and safe depths inside the regularity region:
0 violations**, asserted in the script.

| `m'` | placement term | Berry–Esseen term | actual error |
|---:|---:|---:|---:|
| 50 | ≤ 0.017 | ≤ 0.110 | ≤ 0.073 |
| 200 | ≤ 0.023 | ≤ 0.133 | ≤ 0.073 |
| 800 | ≤ 0.013 | ≤ 0.104 | ≤ 0.033 |

**The placement term is negligible** — under 0.023 everywhere. The delta method puts the cut almost
exactly where it should go; essentially all of the (conservative) bound is Berry–Esseen. And the
error that actually materialises is smaller still, because Berry–Esseen is a worst case over all
`x` while the size only uses one.

### Why the residual does not vanish with more shots

The actual error stalls around 0.073 for `m' = 50` and `m' = 200`. That is **the discreteness of
`K`**, not a central-limit error: `K` takes integer values, so the attainable sizes form a finite
ladder and the achievable level cannot land exactly on `alpha`. Increasing `m'` refines the ladder
but simultaneously admits smaller `p_0` into the region, which coarsens it again. Worth one sentence
in the thesis, because a reader will otherwise ask why the numbers do not improve.

---

## Part 6 — Should you replace the criterion with an exact binomial test?

You could: choose `k*` directly as the exact binomial quantile so the size is `<= alpha` by
construction, dropping the normal entirely. **My recommendation is no**, for three reasons, in
increasing order of force.

**(1) It fixes the smaller of the two errors.** The size deviates from nominal for two independent
reasons: the normal-placed cut (what an exact test removes), and the reference `phi_hat_acc` being a
noisy estimate rather than `phi` (which it does not touch). At `m' = 200`:

| nominal `alpha` | normal approximation of the cut | reference = late probe | reference = first probe |
|---:|---:|---:|---:|
| 0.50 | 0.061 | 0.001 | 0.022 |
| 0.34 | 0.047 | 0.042 | 0.105 |
| 0.05 | 0.015 | 0.063 | 0.319 |

At the operating points that matter the reference term is comparable or much larger. At
`conf = 0.95` it is **21× larger**. An exact test would remove the middle column and leave the rest.

**(2) The nominal level is a tuning knob, not a quantity anyone acts on.** `conf` is chosen by grid
search against convergence rate. If the achieved size is 0.42 when the label says 0.50, the grid
search simply selects the label that lands where it wants. Making the label exact renames the knob;
it does not move the setting. This is different from a scientific context where `alpha = 0.05` is
reported as a guarantee — you never report it as one.

**(3) The cost is a full re-tune and re-sweep**, and every number in Chapter 4, Appendix B and
Appendix C changes. You would be spending that to remove a `<= 0.06` miscalibration in a parameter
that is tuned anyway, while the dominant term survives untouched.

**What I would do instead** — and this costs nothing — is say in one sentence that the exact
binomial size is available in closed form and that the normal quantile is used because it is
cheaper, with the discrepancy bounded as in Part 5. That converts an unexamined approximation into
a deliberate, quantified choice, which is exactly what the criticism asked for.

*If you later want the exact version anyway, it is a two-line change in
`_binary_search_explore`: replace `norm.ppf` with `scipy.stats.binom.ppf` on the count scale and
compare `K` directly. The surrounding algorithm, the safeguard and the bracket logic are untouched.
But do it as future work, not for this thesis.*

---

## Part 7 — Thesis edits

The numbered, copy-pasteable edit list lives in
[`OVERSHOOT_CRITERION.md`](OVERSHOOT_CRITERION.md) so there is one place to work from: four required
edits (Section 2.7 passage, label Eq. (3.6), the new paragraph, the table), three recommended ones
(Lemma 2.6.3's hypothesis, a stale cross-reference, the bracket-walk sentence), and an explicit list
of what to leave alone.

Nothing in that list changes the algorithm, any tuned parameter, or any number in Chapter 4.

---

## Part 8 — Reproducing all of it

```
python analysis/consolidated/overshoot_criterion.py
```

Asserts, and fails loudly if any breaks:

1. the binomial restatement is exact (0 disagreements in 300,000 draws);
2. an overshooting probe always reads below the truth (200,000 draws);
3. the Berry–Esseen bound is respected by the true distance;
4. the two-part error bound holds at all 294 regularity-region grid points.

| File | Contents |
|---|---|
| [`overshoot_size.csv`](overshoot_size.csv) | achieved vs nominal size, Berry–Esseen bound and true distance, per `(m', conf)` |
| [`overshoot_error_bound.csv`](overshoot_error_bound.csv) | the two-part decomposition point by point, incl. the `A` and `B` factors of Part 4 |
| [`overshoot_power.csv`](overshoot_power.csv) | exact `P(rule fires)` vs depth, per `(m', conf, reference)` |
| [`overshoot_bracket_walk.csv`](overshoot_bracket_walk.csv) | the probe sequences behind the Part 7 #9 sentence |
| [`tex/thesis/tab_overshoot_operating.tex`](tex/thesis/tab_overshoot_operating.tex) | the generated table |
