# Finite-sample Gaussianity audit for binary search

## Bottom line

The simulations support a narrower claim than 'the exploration estimator is Gaussian': at the shot counts selected by the reported sweep, the raw one-shot estimator is close to its asymptotic normal law whenever both binomial outcomes have reasonable expected counts. The approximation still fails in a shrinking neighbourhood of every turning point, including the first aliasing boundary. Adaptive selection of the deepest accepted probe creates an additional distortion that increasing the shot count alone does not justify away.

This is therefore useful evidence for the local sampling model, but not a proof of the whole binary-search derivation. The algorithm should be defended by combining this finite-sample audit with the end-to-end detector and final-convergence diagnostics.

## What was tested

The full algorithm was replayed for 20,000 independent trials per operating point (seed family 20260824). The parameters were read from `results/consolidated/optimal_params.csv`.

The exact one-shot distribution was also enumerated. With `K ~ Bin(m, cos^2(N phi))`, there are only `m+1` possible estimates, so no Monte Carlo error is needed for the Kolmogorov--Smirnov (KS) distance or variance ratio. The comparison normal is centred at the folded phase `acos(|cos(N phi)|)/N`. Beyond the boundary it is not centred at the true phase.

A finite-sample regularity marker is `m min(p0, 1-p0) >= 10`. This is deliberately two-sided: near `p0=0` the estimator piles up at `K=0`, while near `p0=1` it piles up at `K=m`.

## Exact one-shot result at the selected shot counts

| exploration shots m | max KS in regular region | median KS | max relative SD error | max absolute bias / model SD |
|---:|---:|---:|---:|---:|
| 114 | 0.081 | 0.048 | 1.9% | 0.068 |
| 289 | 0.084 | 0.034 | 2.2% | 0.078 |
| 334 | 0.078 | 0.031 | 1.8% | 0.072 |
| 391 | 0.079 | 0.030 | 1.8% | 0.073 |
| 2635 | 0.083 | 0.013 | 2.1% | 0.081 |

The 0.10 KS line in the figure is a visual reference, not a formal acceptance threshold. The more interpretable variance result is that the asymptotic standard deviation is close to the exact standard deviation throughout the regular region. Close to `r=1`, the exact law becomes discrete and eventually degenerate, so no shot count makes the approximation uniform over the boundary itself.

## Critical implementation finding

At four of the five selected operating points the tuned exploration shot count makes the second bisection probe unaffordable. The algorithm therefore records exactly one probe in every replayed trial: the opening pilot at `N_min`. At those points neither the binary-search branch nor its overshoot criterion is exercised. The reported performance is produced by the opening pilot followed by the statistical safeguard.

| operating point | B | minimum B for opening + second probe | second probe fits? | opening-only trials |
|---|---:|---:|:---:|---:|
| Table 3.1 | 10,000 | 11,514 | no | 100.0% |
| narrow e3 near B90 | 29,173 | 29,189 | no | 100.0% |
| narrow e4 near B90 | 2,917,365 | 33,734 | yes | 0.0% |
| small e4 near B90 | 394,843 | 398,820 | no | 100.0% |
| wide e4 near B90 | 2,059,436 | 2,126,445 | no | 100.0% |

This is not a Gaussianity failure, but it changes what the headline settings can demonstrate. They support the opening-pilot safeguard, not the binary-search mechanism. Only the `narrow_e4` near-B90 point in this audit actually tests repeated bisection decisions.

## Connection to the best-performing parameters

| operating point | m | conf | replay conv. | mean probes | regular probes | guess overshoot | final overshoot | opening KS | opening mean/SD | selected-safe KS | selected-safe mean/SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Table 3.1 | 114 | 0.50 | 65.2% | 1.00 | 71.9% | 0.0% | 0.80% | 0.013 | +0.03/1.04 | 0.013 | +0.03/1.04 |
| narrow e3 near B90 | 289 | 0.50 | 88.3% | 1.00 | 88.8% | 0.0% | 0.14% | 0.006 | +0.00/1.01 | 0.006 | +0.00/1.01 |
| narrow e4 near B90 | 334 | 0.66 | 90.5% | 8.33 | 62.1% | 12.2% | 0.53% | 0.005 | +0.00/1.02 | 0.498 | +1.18/0.79 |
| small e4 near B90 | 391 | 0.50 | 92.9% | 1.00 | 88.5% | 0.0% | 0.07% | 0.014 | +0.04/1.01 | 0.014 | +0.04/1.01 |
| wide e4 near B90 | 2635 | 0.50 | 87.6% | 1.00 | 96.9% | 0.0% | 0.07% | 0.008 | -0.02/1.00 | 0.008 | -0.02/1.00 |

`opening` is the unselected pilot at the smallest number of phase gates and is safe by construction. `selected-safe` is the deepest accepted pilot, restricted to trials in which that number of phase gates was in fact non-aliasing. A standard normal would have mean 0, SD 1 and KS distance 0.

At 4 of the 5 selected operating points, `conf=0.5`. There the normal quantile in the branch comparison is exactly zero, so the variance formula does not affect the accept/reject branch. It remains relevant to the downstream statistical safeguard.

At the one active bisection point (`narrow_e4` near B90), the fresh, regular probe residuals are close to standard normal (mean +0.017, SD 1.004, KS 0.007). However, only 62.1% of probes satisfy the finite-sample regularity marker. More importantly, selecting the deepest accepted safe pilot shifts its standardised error to mean +1.18 with SD 0.79 and KS 0.498. The trial-level false-alarm and missed-overshoot rates are 57.6% and 12.2%, respectively. Thus the raw Gaussian approximation is good where its regularity condition holds, but the selected pilot is not an unbiased Gaussian observation.

## Recommended thesis claim

> Exact enumeration at the exploration shot counts selected by the parameter sweep shows that the transformed-binomial estimator is well approximated by the delta-method normal law away from the binomial endpoints. In the region with at least ten expected observations of each outcome, its exact standard deviation remains close to `1/(2 N sqrt(m))` and the exact CDF has a small KS distance from the corresponding normal CDF. The approximation is not uniform: it fails near the turning points, in particular at the aliasing boundary where the estimator becomes discrete or degenerate. We therefore treat the normal law as a local design approximation and validate the adaptive search separately by Monte Carlo.

Do not say that these plots prove Gaussianity at the boundary, or that they calibrate the deepest accepted pilot exactly. The latter is selected using the same noisy observations and must be described as an approximation checked by the operating diagnostics.

## Figures

- `fig_exact_normality_map.pdf`: exact KS distance and SD ratio over the first fold.
- `fig_exact_density_examples.pdf`: density-scaled exact PMFs with normal overlays.
- `fig_operating_pilot_cdfs.pdf`: empirical CDFs for the opening and deepest accepted pilots.

The density panels show binned probability mass because the estimator is discrete at finite `m`; strictly speaking it has a PMF, not a continuous PDF. The PNG versions are convenient for notebook-style inspection; the PDFs are vector graphics for a thesis draft.
