"""Tier A / Tier B diagnostics and their error bars, computed from one held-out evaluation.

Everything here consumes the per-run scalar arrays that `qmetrology.pipeline.heldout` returns, so a
performance number and its diagnostics can never come from different runs. Three rules the handoff
insists on and this module enforces:

  * every percentage carries its eligible denominator (`*_n` columns), and
    eligible + ineligible = R for every metric;
  * no algorithm gets a fabricated value for a quantity it does not define -- reverse engineering and
    brute force return NaN for the detector metrics, brute force for everything keyed on N_guess;
  * every interval records its level, method, replicate count, seed and sample size.

UNCERTAINTY METHODS
-------------------
proportions        Wilson score interval at the row's own R. Exact, no simulation.
quantiles          the run-level percentile bootstrap interval, evaluated ANALYTICALLY rather than by
                   resampling. For an order statistic the bootstrap CDF is exact and closed-form:
                       P(X*_(m) <= x_(k)) = P(Binomial(n, k/n) >= m),
                   so the alpha-percentile of the bootstrap distribution of the sample q-quantile is
                   x_(k) at the smallest k reaching alpha. This is the n_boot -> infinity limit of the
                   percentile bootstrap the handoff asks for, is free, and removes the replicate noise
                   a Monte-Carlo bootstrap would add. Recorded as method "order-statistic bootstrap
                   (exact percentile interval)".
means              ordinary Monte-Carlo run-level percentile bootstrap with `n_boot` replicates.
                   Runs are resampled, never probes: probes inside a run are dependent, so the
                   run-level resample is what carries the clustering.
"""
import numpy as np
from scipy.stats import binom

from .uncertainty import wilson, Z95

CI_LEVEL = 95
_QCACHE = {}

# Every continuous diagnostic that gets a mean + bootstrap interval. Fixed, so the output CSV has a
# stable header no matter which branches an operating point exercises.
CONT_METRICS = ["guess_ratio", "guess_abs_rel", "guess_signed_rel", "exploration_share",
                "n_probes", "star_ratio", "star_over_guess", "budget_util"]


# ------------------------------------------------------------------------------- proportions
def prop_ci(k, n):
    """(p, lo, hi) Wilson at the given denominator. NaN triple when the denominator is empty."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    lo, hi = wilson(p, n)
    return float(p), float(max(lo, 0.0)), float(min(hi, 1.0))


def share_ci(mask, eligible):
    """Share of `eligible` runs satisfying `mask`, with its Wilson interval and denominator."""
    n = int(eligible.sum())
    k = int((mask & eligible).sum())
    p, lo, hi = prop_ci(k, n)
    return p, lo, hi, n, k


# --------------------------------------------------------------------------------- quantiles
def _boot_cdf(n, m):
    """F_boot(x_(k)) = P(Bin(n, k/n) >= m) for k = 1..n -- the exact bootstrap CDF of X*_(m)."""
    key = (n, m)
    if key not in _QCACHE:
        k = np.arange(1, n + 1)
        _QCACHE[key] = binom.sf(m - 1, n, k / n)
        if len(_QCACHE) > 64:
            _QCACHE.pop(next(iter(_QCACHE)))
    return _QCACHE[key]


def quantile_ci(x, q, ci=CI_LEVEL):
    """(quantile, lo, hi) with the exact order-statistic bootstrap percentile interval."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"),) * 3
    xs = np.sort(x)
    if n < 20:   # too few runs for a meaningful interval; report the point value only
        return float(np.quantile(xs, q)), float("nan"), float("nan")
    m = max(1, min(n, int(np.ceil(q * n))))
    F = _boot_cdf(n, m)
    a = (1 - ci / 100) / 2
    klo = int(np.searchsorted(F, a, side="left"))
    khi = int(np.searchsorted(F, 1 - a, side="left"))
    klo = min(max(klo, 0), n - 1)
    khi = min(max(khi, 0), n - 1)
    return float(np.quantile(xs, q)), float(xs[klo]), float(xs[khi])


# ------------------------------------------------------------------------------------- means
def boot_means(cols, n_boot, seed, ci=CI_LEVEL, batch=100):
    """Run-level percentile bootstrap of the mean, for a dict {name: array}.

    All metrics are resampled with the SAME run indices inside a batch, which costs nothing and keeps
    the replicates comparable across metrics.
    """
    names = [k for k, v in cols.items() if np.isfinite(v).any()]
    if not names:
        return {k: (float("nan"),) * 3 for k in cols}
    mats, keep = {}, {}
    for k in names:
        v = np.asarray(cols[k], float)
        keep[k] = np.isfinite(v)
        mats[k] = v
    lens = {len(v) for v in mats.values()}
    assert len(lens) == 1, f"boot_means needs one run axis, got lengths {sorted(lens)}"
    n = lens.pop()
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in names}
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        for k in names:
            v = mats[k][idx]
            f = keep[k][idx]
            cnt = f.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                draws[k].append(np.where(cnt > 0, np.nansum(np.where(f, v, 0.0), axis=1)
                                         / np.maximum(cnt, 1), np.nan))
        done += b
    a = (100 - ci) / 2
    out = {}
    for k in cols:
        if k not in names:
            out[k] = (float("nan"),) * 3
            continue
        d = np.concatenate(draws[k])
        d = d[np.isfinite(d)]
        v = mats[k][keep[k]]
        out[k] = (float(v.mean()), float(np.percentile(d, a)), float(np.percentile(d, 100 - a)))
    return out


# -------------------------------------------------------------------------- the diagnostics
def _q(x, qq):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, qq)) if x.size else float("nan")


def point_diagnostics(A, statuses, spec, n_boot, boot_seed, ci=CI_LEVEL):
    """All Tier A + Tier B diagnostics for one operating point.

    `A` is the per-run array dict from `pipeline.heldout`; `spec` is the manifest entry for the
    algorithm (it decides which families of metric are defined at all).
    """
    R = len(A["converged"])
    out = {"R": R, "ci_level": ci, "n_boot": n_boot, "boot_seed": boot_seed,
           "ci_method_proportion": "wilson",
           "ci_method_quantile": "order-statistic bootstrap (exact percentile interval)",
           "ci_method_mean": f"run-level percentile bootstrap, {n_boot} replicates"}
    nan = float("nan")
    conv = A["converged"] > 0.5
    N_opt, N_guess, N_star = A["N_opt"], A["N_guess"], A["N_star"]
    B = A["nominal_budget"]

    # ---- performance, from the same runs -------------------------------------------------
    p, lo, hi = prop_ci(int(conv.sum()), R)
    out.update(rate=p, rate_lo=lo, rate_hi=hi, rate_n=R, rate_k=int(conv.sum()))

    # ---- eligibility bookkeeping ---------------------------------------------------------
    el_guess = np.isfinite(N_guess) & np.isfinite(N_opt) & spec["has_guess"]
    el_star = np.isfinite(N_star) & np.isfinite(N_opt)
    out["guess_eligible_n"] = int(el_guess.sum())
    out["guess_ineligible_n"] = R - int(el_guess.sum())
    out["star_eligible_n"] = int(el_star.sum())
    out["star_ineligible_n"] = R - int(el_star.sum())

    cont = {}

    # ---- Tier A: exploration depth quality -----------------------------------------------
    if el_guess.any():
        # kept at FULL length with NaN where ineligible: the mean bootstrap resamples RUNS, so every
        # metric must be indexed by the same run axis (the NaNs drop out inside each replicate).
        den = np.where(np.isfinite(N_opt) & (N_opt > 0), N_opt, np.nan)
        g = np.where(el_guess, N_guess / den, np.nan)
        sg = np.where(el_guess, (N_guess - den) / den, np.nan)
        cont["guess_ratio"] = g
        cont["guess_abs_rel"] = np.abs(sg)
        cont["guess_signed_rel"] = sg
        g = g[el_guess]
        for tag, qq in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
            v, l, h = quantile_ci(g, qq, ci)
            out[f"guess_ratio_{tag}"] = v
            if tag == "median":
                out["guess_ratio_median_lo"], out["guess_ratio_median_hi"] = l, h
        for name, mask in (("guess_exact", N_guess == N_opt),
                           ("guess_within5", np.abs(N_guess - N_opt) <= 0.05 * N_opt),
                           ("guess_within10", np.abs(N_guess - N_opt) <= 0.10 * N_opt),
                           ("guess_overshoot", N_guess > N_opt)):
            p, l, h, n, k = share_ci(mask, el_guess)
            out[name] = p; out[name + "_lo"] = l; out[name + "_hi"] = h
            out[name + "_n"] = n; out[name + "_k"] = k
    else:
        for c in ("guess_ratio_median", "guess_ratio_q1", "guess_ratio_q3",
                  "guess_ratio_median_lo", "guess_ratio_median_hi"):
            out[c] = nan
        for name in ("guess_exact", "guess_within5", "guess_within10", "guess_overshoot"):
            for sfx in ("", "_lo", "_hi"):
                out[name + sfx] = nan
            out[name + "_n"] = 0; out[name + "_k"] = 0

    # ---- Tier A: exploration cost --------------------------------------------------------
    share = np.where(B > 0, A["budget_exploration"] / np.maximum(B, 1e-30), nan)
    cont["exploration_share"] = share
    cont["n_probes"] = A["n_probes"]
    for tag, qq in (("median", 0.5), ("q1", 0.25), ("q3", 0.75), ("p90", 0.90)):
        v, l, h = quantile_ci(share, qq, ci)
        out[f"exploration_share_{tag}"] = v
        if tag == "median":
            out["exploration_share_median_lo"], out["exploration_share_median_hi"] = l, h
    out["n_probes_median"] = _q(A["n_probes"], 0.5)
    all_runs = np.ones(R, bool)
    for name, mask in (("single_probe", A["n_probes"] <= 1),
                       ("no_search_step", A["n_probes"] <= 1),
                       ("no_exploitation", ~np.isfinite(N_star))):
        p, l, h, n, k = share_ci(mask, all_runs)
        out[name] = p; out[name + "_lo"] = l; out[name + "_hi"] = h
        out[name + "_n"] = n; out[name + "_k"] = k
    # the specific reason "exploration consumed the available budget"
    st = np.array(statuses)
    starved = np.isin(st, ["no_exploitation_budget_exhausted", "no_exploitation_shots",
                           "refused_pilot_unaffordable"])
    p, l, h, n, k = share_ci(starved, all_runs)
    out["no_exploitation_budget"] = p; out["no_exploitation_budget_lo"] = l
    out["no_exploitation_budget_hi"] = h; out["no_exploitation_budget_n"] = n
    out["no_exploitation_budget_k"] = k
    vals, counts = np.unique(st, return_counts=True)
    out["termination_reasons"] = ";".join(f"{v}={c}" for v, c in zip(vals, counts))

    # ---- Tier A: detector quality --------------------------------------------------------
    if spec["has_detector"]:
        el_det = np.isfinite(A["false_alarm"])
        for name, arr in (("false_alarm", A["false_alarm"]), ("miss", A["miss"])):
            p, l, h, n, k = share_ci(arr > 0.5, el_det)
            out[name + "_rate"] = p; out[name + "_rate_lo"] = l; out[name + "_rate_hi"] = h
            out[name + "_rate_n"] = n; out[name + "_rate_k"] = k
        for f in ("tp", "fp", "tn", "fn"):
            out["probe_" + f] = int(np.nansum(A[f]))
        out["detector_eligible_n"] = int(el_det.sum())
    else:
        for name in ("false_alarm", "miss"):
            for sfx in ("_rate", "_rate_lo", "_rate_hi"):
                out[name + sfx] = nan
            out[name + "_rate_n"] = 0; out[name + "_rate_k"] = 0
        for f in ("tp", "fp", "tn", "fn"):
            out["probe_" + f] = 0
        out["detector_eligible_n"] = 0

    # ---- Tier A: final depth and outcome -------------------------------------------------
    if el_star.any():
        den = np.where(np.isfinite(N_opt) & (N_opt > 0), N_opt, np.nan)
        cont["star_ratio"] = np.where(el_star, N_star / den, np.nan)
        sr = (N_star / den)[el_star]
        for tag, qq in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
            v, l, h = quantile_ci(sr, qq, ci)
            out[f"star_ratio_{tag}"] = v
            if tag == "median":
                out["star_ratio_median_lo"], out["star_ratio_median_hi"] = l, h
        for name, mask in (("star_overshoot", N_star > N_opt),
                           ("star_exact", N_star == N_opt),
                           ("star_within5", np.abs(N_star - N_opt) <= 0.05 * N_opt),
                           ("star_within10", np.abs(N_star - N_opt) <= 0.10 * N_opt)):
            p, l, h, n, k = share_ci(mask, el_star)
            out[name] = p; out[name + "_lo"] = l; out[name + "_hi"] = h
            out[name + "_n"] = n; out[name + "_k"] = k
        safe = el_star & (N_star <= N_opt)
        p, l, h, n, k = share_ci(conv, safe)
        out["conv_given_safe_depth"] = p; out["conv_given_safe_depth_lo"] = l
        out["conv_given_safe_depth_hi"] = h; out["conv_given_safe_depth_n"] = n
        out["conv_given_safe_depth_k"] = k
    else:
        for c in ("star_ratio_median", "star_ratio_q1", "star_ratio_q3",
                  "star_ratio_median_lo", "star_ratio_median_hi",
                  "conv_given_safe_depth", "conv_given_safe_depth_lo",
                  "conv_given_safe_depth_hi"):
            out[c] = nan
        out["conv_given_safe_depth_n"] = 0; out["conv_given_safe_depth_k"] = 0
        for name in ("star_overshoot", "star_exact", "star_within5", "star_within10"):
            for sfx in ("", "_lo", "_hi"):
                out[name + sfx] = nan
            out[name + "_n"] = 0; out[name + "_k"] = 0

    # ---- Tier B --------------------------------------------------------------------------
    both = el_guess & el_star
    if both.any():
        cont["star_over_guess"] = np.where(both, N_star / np.maximum(N_guess, 1e-30), np.nan)
        ratio = (N_star / np.maximum(N_guess, 1e-30))[both]
        out["star_over_guess_median"] = _q(ratio, 0.5)
        out["star_over_guess_q1"] = _q(ratio, 0.25)
        out["star_over_guess_q3"] = _q(ratio, 0.75)
        unsafe = both & (N_guess > N_opt)
        p, l, h, n, k = share_ci(N_star <= N_opt, unsafe)
        out["rescue_share"] = p; out["rescue_share_lo"] = l; out["rescue_share_hi"] = h
        out["rescue_share_n"] = n; out["rescue_share_k"] = k
        safe_guess = both & (N_guess <= N_opt)
        bo = (N_star[safe_guess] / np.maximum(N_guess[safe_guess], 1e-30)) if safe_guess.any() else np.array([])
        out["backoff_safe_median"] = _q(bo, 0.5) if bo.size else nan
        out["backoff_safe_q1"] = _q(bo, 0.25) if bo.size else nan
        out["backoff_safe_q3"] = _q(bo, 0.75) if bo.size else nan
        out["backoff_safe_n"] = int(safe_guess.sum())
    else:
        for c in ("star_over_guess_median", "star_over_guess_q1", "star_over_guess_q3",
                  "rescue_share", "rescue_share_lo", "rescue_share_hi",
                  "backoff_safe_median", "backoff_safe_q1", "backoff_safe_q3"):
            out[c] = nan
        out["rescue_share_n"] = 0; out["rescue_share_k"] = 0; out["backoff_safe_n"] = 0

    util = np.where(B > 0, A["budget_used_total"] / np.maximum(B, 1e-30), nan)
    cont["budget_util"] = util
    out["budget_util_mean"] = float(np.nanmean(util))
    out["budget_util_median"] = _q(util, 0.5)
    out["budget_util_p90"] = _q(util, 0.90)
    out["budget_util_max"] = float(np.nanmax(util))
    out["budget_unused_mean"] = float(1.0 - np.nanmean(util))
    out["budget_violations"] = int(np.nansum(util > 1.0 + 1e-9))
    out["accepted_probes_mean"] = float(np.nanmean(A["n_accepted"]))

    # ---- means + their bootstrap intervals -----------------------------------------------
    bm = boot_means({k: cont[k] for k in CONT_METRICS if k in cont}, n_boot, boot_seed, ci)
    for k in CONT_METRICS:
        mu, l, h = bm.get(k, (nan, nan, nan))
        out[k + "_mean"] = mu
        out[k + "_mean_lo"] = l
        out[k + "_mean_hi"] = h
    return out


def phase_strata(A, spec, n_bins=4):
    """Tier B: the depth diagnostics split by fixed quantile bins of the TRUE phase."""
    phi = A["phi"]
    ok = np.isfinite(phi)
    if not ok.any():
        return []
    edges = np.quantile(phi[ok], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for i in range(n_bins):
        sel = ok & (phi >= edges[i]) & (phi < edges[i + 1])
        if not sel.any():
            continue
        Ng, Ns, No = A["N_guess"][sel], A["N_star"][sel], A["N_opt"][sel]
        eg = np.isfinite(Ng) & spec["has_guess"]
        es = np.isfinite(Ns)
        rows.append({
            "phase_bin": i, "phi_lo": float(np.min(phi[sel])), "phi_hi": float(np.max(phi[sel])),
            "n": int(sel.sum()),
            "rate": float(A["converged"][sel].mean()),
            "guess_ratio_median": _q(Ng[eg] / No[eg], 0.5) if eg.any() else float("nan"),
            "guess_overshoot": float((Ng[eg] > No[eg]).mean()) if eg.any() else float("nan"),
            "guess_n": int(eg.sum()),
            "star_ratio_median": _q(Ns[es] / No[es], 0.5) if es.any() else float("nan"),
            "star_overshoot": float((Ns[es] > No[es]).mean()) if es.any() else float("nan"),
            "star_n": int(es.sum()),
            "exploration_share_median": _q(A["budget_exploration"][sel]
                                           / np.maximum(A["nominal_budget"][sel], 1e-30), 0.5),
        })
    return rows
