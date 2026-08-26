# Linear search: response to the supervisor comments

## Bottom line

Linear search can remain in the thesis, but it should not be presented as a reliable overshoot detector or as a method with a formal guarantee. The current evidence supports a narrower claim: it is a simple sequential boundary-localization heuristic whose end-to-end usefulness is regime-dependent. It is weak in the low-budget, low-precision setting and substantially more useful once the final-estimation budget dominates the exploration cost.

The main methodological problem is therefore the current framing, not the mere inclusion of the algorithm. In particular, the statement that the cumulative-mean rule can "reliably determine an overshoot" is contradicted by the reported false-alarm rates.

## What is and is not theoretically motivated

For noiseless data, the entangled estimator is

\[
g(N)=\frac{1}{N}\arccos\!\left(|\cos(N\phi)|\right).
\]

On the admissible branch, $0\leq N\phi\leq \pi/2$, one has $g(N)=\phi$. On the first aliased branch, $\pi/2\leq N\phi\leq\pi$,

\[
g(N)=\frac{\pi}{N}-\phi,
\qquad
\frac{\mathrm d g}{\mathrm dN}=-\frac{\pi}{N^2}<0.
\]

There is therefore a genuine local downward trend immediately after the first aliasing boundary. This supplies a mechanistic reason for looking for consecutive decreases. It does **not** imply that the estimator decreases for every $N>N_{\mathrm{opt}}$: beyond the first aliased branch it follows the sawtooth behavior already described around Equation (2.85).

## Assessment of the cumulative/running mean

The quantity used by the implementation is a **cumulative mean** (also commonly called a running mean), not a moving-window mean: every estimate collected since the start of the scan remains in the average. Nina was correct to insist on this distinction.

For the cumulative mean

\[
\bar\phi_k=\frac{1}{k}\sum_{j=1}^k\hat\phi_j,
\]

the update satisfies

\[
\bar\phi_k-\bar\phi_{k-1}
=\frac{\hat\phi_k-\bar\phi_{k-1}}{k}.
\]

Consequently, the factor $1/k$ shrinks the magnitude of the update but not its sign:

\[
\bar\phi_k<\bar\phi_{k-1}
\quad\Longleftrightarrow\quad
\hat\phi_k<\bar\phi_{k-1}.
\]

The supervisor's false-alarm concern is nevertheless correct. Before the boundary, shot noise alone can put a new estimate below the previous mean; with $l=1$, this is essentially a one-sided noisy comparison. Requiring $l$ consecutive decreases lowers the per-location false-trigger probability, but the scan performs many repeated comparisons, so the probability of at least one premature stop can still be large. This is why the rule must be evaluated empirically rather than defended as a calibrated statistical test.

This is not only an issue with an obsolete parameter choice: 22 of the 144 informative operating points in the current consolidated output select $l=1$. The headline configurations use larger windows ($l=4$ at $C=10{,}000$, $l=3$ near the low-precision 90% crossing, and $l=8$ near the $\epsilon=10^{-4}$ crossing), but the general method still includes the one-comparison case.

### Where Nina's criticism is correct

The cumulative-mean rule has no calibrated false-alarm probability. Before overshooting, the estimates fluctuate around approximately the same value. In an idealized symmetric-noise model, a fresh safe estimate has a probability close to one half of falling below the preceding cumulative mean. Thus, with $l=1$, one decrease really is close to a coin-flip decision.

For $l>1$, an informal independence approximation gives a probability of roughly $2^{-l}$ for $l$ consecutive decreases at one candidate location. The comparisons are not actually independent because they share the cumulative history, but the approximation shows the central problem: the scan offers many opportunities for such a streak. If a scan makes $T$ comparisons, a crude illustration of the probability of encountering at least one false streak is

\[
1-\left(1-2^{-l}\right)^{T-l+1}.
\]

This is not a theoretical error bound for the implemented algorithm, but it explains why a seemingly reasonable window such as $l=4$ can still yield many trial-level false alarms over a long scan. The measured false-alarm rates should be used instead of this approximation in the thesis.

The cumulative mean also gives all early observations permanent influence. It therefore does not estimate a *local* slope or compare two local regimes. Calling it a change-point detector would be too strong. Its actual decision is simply whether each of the last $l$ estimates lies below the cumulative mean that preceded it.

### Where the ``dilution by $1/k$'' explanation needs refinement

The cumulative-mean update is indeed smaller by a factor $1/k$, but the implementation tests only whether the update is negative. Since division by the positive number $k$ cannot change the sign, increasing $k$ does not mathematically make this particular sign test less sensitive. Equivalently, the algorithm could test $\hat\phi_k<\bar\phi_{k-1}$ directly and make exactly the same decisions.

The historical average still matters, but in a different way from simple signal attenuation. Before the boundary it remains near $\phi$, creating frequent noise-driven crossings. Immediately after the boundary, keeping that pre-boundary reference near $\phi$ can actually make the lower aliased estimates easier to flag. The problem is therefore not that the sign vanishes as $1/k$; it is that an unthresholded sign comparison is noisy and repeated many times.

### Would a moving-window mean be better?

A moving window is a plausible alternative because it focuses on recent probes and could respond more directly to a local trend. It is not automatically a statistically sound fix. A short window has higher variance, and applying the same ``mean decreased'' rule to it can still produce frequent false alarms on safe probes. A defensible replacement would have to specify and validate a complete detector, for example a comparison of pre- and post-windows with a noise-dependent threshold, a CUSUM-style change detector, or a likelihood-based test.

Changing to such a detector now would define a different Linear Search algorithm. All parameter tuning, convergence curves, diagnostic tables, and held-out evaluations would need to be rerun. The current cumulative-mean method is therefore best retained as an explicitly simple heuristic, while a moving-window or calibrated sequential detector can be named as future work.

### Practical verdict on the running mean

The running mean is not wholly unfounded: it is a simple way to look for the real downward trend on the first aliased branch, and its decisions have now been evaluated on held-out simulations. However, the data do not justify describing it as a reliable detector. The defensible claim is that it is a low-complexity trend heuristic whose premature-stop risk is measured explicitly and whose usefulness depends strongly on the resource regime.

## What the current held-out results say

The aggregate diagnostics in Tables 4.5 and 4.6 show:

- trial-level false-alarm rate: **45.3%**;
- trial-level miss rate: **7.7%**;
- median $N_{\mathrm{guess}}/N_{\mathrm{opt}}=1.00$, but only **68.6%** of guesses are within 10%;
- after the fixed decrement, median $N^*/N_{\mathrm{opt}}=0.95$ and final overshoot is **2.35%**;
- median exploration share is **1.78%**.

These values do not validate the rule as a reliable classifier. They show instead that many false alarms are conservative stops and that the final value of $N$ is often still useful.

The aggregate also hides a strong regime dependence. Appendix Table B.2 reports for Linear Search:

| Setting | Median $N_{\mathrm{guess}}/N_{\mathrm{opt}}$ | Mean absolute relative error | Within 10% |
|---|---:|---:|---:|
| $\epsilon=10^{-3},\ \phi\sim\mathcal U(0.01,0.1)$ | 0.83 | 0.29 | 32.0% |
| $\epsilon=10^{-4},\ \phi\sim\mathcal U(0.01,0.1)$ | 1.00 | 0.05 | 87.6% |
| $\epsilon=10^{-5},\ \phi\sim\mathcal U(0.01,0.1)$ | 1.00 | 0.04 | 89.6% |

A reproduction of the held-out runs gives the more direct distance diagnostic requested in the feedback:

- At $C=10{,}000$, a false-alarm run stops a median of 5 phase-gate uses below $N_{\mathrm{opt}}$, but the 90th and 95th percentiles are 64 and 91 below it. Only 26.7% of false-alarm runs land within 10% of $N_{\mathrm{opt}}$.
- Near the low-precision 90% crossing ($C=41{,}326$), the corresponding median is 10 below $N_{\mathrm{opt}}$, with 90th and 95th percentiles of 60 and 82. Only 17.6% of false-alarm runs land within 10%.
- Near the $\epsilon=10^{-4}$ 90% crossing ($C=2{,}917{,}365$), a false-alarm run stops a median of only 1 below $N_{\mathrm{opt}}$; 67.8% of false-alarm runs are within 10%.

Thus, the severe early-stop tail identified in the feedback is real at low precision. It becomes much less important in the high-precision regime, where more exploration shots are affordable and the final-estimation cost amortizes the search.

The end-to-end performance tells the same story:

- At $C=10{,}000$, Linear Search reaches 57.34% convergence versus 56.09% for the baseline. The improvement is statistically resolved at the reported Monte Carlo precision but practically small.
- At 90% reliability and $\epsilon=10^{-3}$, Linear Search needs about 42,500 budget versus 45,900 for the baseline, only a $1.08\times$ improvement.
- At 90% reliability and $\epsilon=10^{-4}$, it needs about 2.91 million versus 4.53 million for the baseline, approximately a $1.56\times$ improvement.

Linear Search is therefore not universally ineffective, but neither is it a generally reliable overshoot detector.

## Minimum changes needed in the thesis

1. Replace "we deduce that we have overshot" with "the rule declares an overshoot." A heuristic decision is not ground truth.
2. Remove "reliably determine an overshoot with very few exploration shots." The 45.3% aggregate false-alarm rate, and the substantially higher rate at the fixed-budget headline point, directly contradict it.
3. Replace the global claim that estimates "tend toward zero" whenever $N>N_{\mathrm{opt}}$ with the first-aliased-branch calculation above. The global noiseless behavior is a sawtooth.
4. In Algorithm 4, change $N\leftarrow N-l$ to $N\leftarrow N-l\,\mathrm{inc}$, matching the implementation and Appendix Listing A.3. The selected configurations sometimes have $\mathrm{inc}>1$.
5. Change the pseudocode comment "Backtrack to last $N$ before overshooting $N_{\mathrm{opt}}$" to "Discard the $l$ probes that triggered the stopping rule." The former is only true when the classification is correct.
6. State the safeguard as $N^*=\max\{1,N_{\mathrm{guess}}-s\}$, matching the implementation.
7. Add a short interpretation after Tables 4.5 and 4.6. At present the tables expose the detector weakness, but the prose does not explain it.
8. Explicitly distinguish the low-precision result from the high-precision result, preferably by citing Appendix Table B.2.

## Suggested replacement for the methodology paragraph

> Linear Search uses a heuristic stopping rule rather than a calibrated hypothesis test. In the noiseless model, the estimator equals $\phi$ on the admissible branch $0\leq N\phi\leq\pi/2$. Immediately after the first aliasing boundary, for $\pi/2\leq N\phi\leq\pi$, it becomes $\hat\phi_N=\pi/N-\phi$ and therefore decreases with $N$. This local post-boundary trend motivates testing for consecutive decreases. Writing $\bar\phi_k=k^{-1}\sum_{j=1}^k\hat\phi_j$, the condition $\bar\phi_k<\bar\phi_{k-1}$ is equivalent to $\hat\phi_k<\bar\phi_{k-1}$. Shot noise can produce the same event before the aliasing boundary, so premature stops are possible and no formal detection guarantee is claimed. We therefore assess the rule through held-out false-alarm and miss rates, the distance between $N_{\mathrm{guess}}$ and $N_{\mathrm{opt}}$, exploration cost, and final convergence performance in Section 4.5 and Appendix B.

## Suggested interpretation after the diagnostic tables

> The diagnostics show that Linear Search is a conservative but imperfect boundary-localization heuristic. Its trial-level false-alarm rate is 45.3%, so the cumulative-mean rule should not be interpreted as a reliable overshoot classifier. However, a false alarm does not necessarily imply a large loss in the selected phase-gate count: aggregated over the informative operating points, the median $N_{\mathrm{guess}}/N_{\mathrm{opt}}$ is 1.00 and 68.6% of guesses lie within 10% of $N_{\mathrm{opt}}$. After the fixed decrement, the median exploitation ratio is 0.95 and the final overshoot rate is 2.35%. Appendix Table B.2 reveals a pronounced regime dependence: at $\epsilon=10^{-3}$, only 32.0% of guesses lie within 10%, whereas this share rises to 87.6% at $\epsilon=10^{-4}$. Linear Search is therefore weak in the tight-budget setting but becomes an effective, low-overhead locator when the higher precision requirement makes additional exploration affordable.

## Separate inconsistency that should be corrected

The prose in Section 4.2 says that Binary Search underperforms the baseline and that all adaptive methods except Binary Search beat the baseline at 90% reliability. This contradicts the current Table 4.1, where Binary Search achieves 65.07% versus 56.09% at $C=10{,}000$ and needs 32,400 versus 45,900 budget at 90% reliability. The paragraph should instead say that all three adaptive methods improve on the baseline in this setting, Reverse Engineering performs best, Binary Search is close behind, and Linear Search provides the smallest improvement.

## Recommendation

Do not replace the detector at this stage unless a new algorithmic contribution is explicitly required. A moving-window or statistically calibrated change detector would require a new parameter search and a fresh held-out evaluation across every reported scenario. The safer and scientifically stronger revision is to keep Linear Search as the deliberately simple sequential heuristic, provide its local mathematical motivation, report its failure modes without euphemism, and make the regime-dependent conclusion explicit.
