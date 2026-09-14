"""Does Linear Search need a different stopping rule? (Section 4.5 / Algorithm 4)

The supervisor's objection is that the rule of Algorithm 4 -- stop after `l` consecutive falls of
the CUMULATIVE mean of the probe estimates -- is an uncalibrated repeated sign test, and that a
MOVING-WINDOW mean would be the natural repair. This module answers that empirically, by running
every candidate rule on the *same* probe stream and tuning each one under an identical protocol.

WHAT MAKES THE COMPARISON EXACT AND PAIRED
------------------------------------------
Algorithm 4's scan probes N = N_min, N_min + inc, ... and its cost is m' * N per probe, so the
sequence of depths it can afford is fixed *before* any datum is seen: which probes exist depends on
the budget, not on the estimates. The scan can therefore be run to its affordable end once per
trial, and every candidate rule replayed on that recorded stream -- each rule stops where it would
have stopped, having seen exactly the probes it would have seen. Nothing is approximated: a rule
that fires at probe j saw probes 0..j and spent m' * sum(N_0..N_j).

The exploitation phase is then evaluated *analytically* rather than simulated. Once the rule has
chosen N* = max(1, N_guess - s) and the remaining budget fixes m = floor(B_rem / N*), the
convergence probability P(|phi_hat - phi| < eps) is an exact binomial quantity (the same expression
qmetrology/oracle.py uses for the oracle). Averaging it over trials estimates the same convergence
rate the sweep measures, with the exploitation coin-flip integrated out -- so a 0.3 pp difference
between two rules is resolved with the trials a simulated comparison would need for 3 pp.
`--validate` checks this against the production sweep's own held-out rates.

THE RULES  (all are "declare an overshoot, backtrack, hand N_guess to the safeguard")
-------------------------------------------------------------------------------------
  cumulative  Algorithm 4 as published: `l` consecutive falls of the cumulative mean.
  window      the supervisor's suggestion: `l` consecutive falls of the mean of the last `w` probes.
              Once the window is full this contains no averaging at all: the trailing mean falls
              exactly when phi_hat_k < phi_hat_{k-w}, because the two windows differ only in their
              entering and leaving element. The moving window is therefore a LAGGED pairwise
              comparison, and w = 1 is the raw rule "the last `l` estimates each fell".
  prepost     mean of the last `w` probes against the mean of the `w` before them, declared only
              when the drop exceeds `z` standard deviations of that difference (Eq. 3.4 variances).
  threshold   the criterion Algorithm 5 already uses, Eq. (3.6), transplanted onto the scan and
              given the lag the identity above exposes:
              phi_hat_k < phi_hat_{k-lag} + z_alpha / (2 N_k sqrt(m')), `l` times in a row.
              At alpha = 0.5 the quantile vanishes and this IS the moving-window rule with w = lag,
              so the supervisor's suggestion and the thesis's own overshoot criterion are two
              members of one family -- which is why the whole (lag, alpha) plane is searched.
  pooled      the same test against the inverse-variance pooled mean of ALL previous probes, whose
              own uncertainty is propagated -- the likelihood-flavoured version.
  cusum       a CUSUM on the standardised drop below that pooled reference, declared at S >= h.

Every rule is tuned over its own grid crossed with the same (m', inc, s) axes the manifest gives
Algorithm 4, on the same two tuning blocks, and the winner is scored on the held-out seed. So a
rule that loses cannot be said to have lost because it was tuned less carefully.

    python analysis/linear_detector_study.py            # the six diagnostic points
    python analysis/linear_detector_study.py --curve    # + the 90%-crossing budgets
    python analysis/linear_detector_study.py --quick    # ~2 min smoke run
    python analysis/linear_detector_study.py --validate # only the sanity checks

Writes results/linear_detector_*.csv:

  bakeoff     the tuned winner of every rule at every operating point, held-out, with diagnostics
  matched     the same, with m' and inc pinned to the published configuration's values
  profile     the best each rule reaches at every exploration size
  grid        the top of each rule's tuning landscape
  streaks     the measured false-alarm streak statistics behind the "roughly 2^-l" argument
  zone        the exact width of the pre-boundary zone in which the estimator stops being random,
              which is what actually drives the false-alarm rate
  crossings   (--curve) the budget each rule needs for 90% convergence, and the ratio vs brute force
  validation  the two checks that the analytic replay reproduces the production sweep
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qmetrology import manifest as M
from qmetrology.experiments import _CTX, _draw_phi
from qmetrology.pipeline import seeds_for
from pipeline_io import Table, path

# The operating points. Three are the ones the revision notes argue about (the fixed-budget
# headline and the two 90%-reliability crossings); the other three widen the regime coverage to a
# tighter epsilon, a deeper prior and a broad prior, because the whole question is whether the
# rule's weakness is regime-dependent.
POINTS = [
    dict(sid="narrow_e3", budget=10_000,      tag="fixed-budget headline"),
    dict(sid="narrow_e3", budget=41_326,      tag="90% crossing, eps=1e-3"),
    dict(sid="narrow_e4", budget=2_917_365,   tag="90% crossing, eps=1e-4"),
    dict(sid="narrow_e5", budget=291_736_555, tag="90% crossing, eps=1e-5"),
    dict(sid="small_e4",  budget=394_843,     tag="deep prior, U(0.001,0.01)"),
    dict(sid="broad_pi4_e3", budget=312_031,  tag="broad prior, U(0.01,pi/4)"),
]

# Budgets swept to locate the 90%-reliability crossing under each rule (--curve). These are tested
# budgets of the production grid, chosen to bracket the crossing, so the interpolated budget is
# directly comparable with results/budget_crossings.csv.
CURVE_POINTS = [
    dict(sid="narrow_e3", budget=b, tag="crossing sweep, eps=1e-3")
    for b in (14_538, 20_594, 29_173, 58_543)
] + [
    dict(sid="narrow_e4", budget=b, tag="crossing sweep, eps=1e-4")
    for b in (1_453_804, 2_059_436, 4_132_694)
]

# Shared axes: exactly the manifest's grids for Algorithm 4, so no rule is searched over a wider
# safeguard or increment than the published one was.
INC_GRID = M.ALGORITHMS["linear"]["discrete"]["inc"]
S_GRID = M.ALGORITHMS["linear"]["discrete"]["safeguard"]
L_GRID = M.ALGORITHMS["linear"]["discrete"]["lookback_window"]

MODES = dict(
    full=dict(R_tune=10_000, R_test=50_000, n_m=20),
    quick=dict(R_tune=1_500, R_test=4_000, n_m=8),
)
SEED_STREAK = 90_210     # separate stream for the streak-statistics pass


# --------------------------------------------------------------------------- the scan and probes
def scan_depths(N_min, N_max, inc, m, budget):
    """The depths Algorithm 4 probes when its stopping rule never fires.

    Mirrors `qmetrology.algorithms._linear_search_explore` exactly, including the order of its two
    termination checks: a probe that does not fit is not taken, and the scan also stops once the
    depth reaches N_max or the budget is exactly exhausted. The sequence depends only on the budget,
    so it is the same for every trial -- which is what makes the replay below exact.
    """
    depths, used, N = [], 0, max(1, int(N_min))
    while True:
        if used + N * m > budget:
            break
        depths.append(N)
        used += N * m
        if N >= N_max or used >= budget:
            break
        N = min(N + inc, N_max)
    return np.asarray(depths, dtype=np.int64)


def probe_matrix(rng, phi, depths, m):
    """(R, K) matrix of exploration estimates: phi_hat = arccos(sqrt(hits/m')) / N, as in sim.py."""
    p0 = np.cos(np.outer(phi, depths)) ** 2
    hits = rng.binomial(m, p0)
    return np.arccos(np.sqrt(hits / m)) / depths


def conv_prob(N, phi, budget, eps):
    """Exact P(|phi_hat - phi| < eps) at depth N with floor(budget/N) shots, elementwise.

    The vectorised-over-everything twin of qmetrology.oracle.convergence_prob, which only vectorises
    over N; here the depth, the phase and the remaining budget all vary trial by trial.
    `--validate` asserts the two agree.
    """
    N = np.asarray(N, dtype=np.int64)
    m = np.asarray(budget, dtype=np.int64) // N
    lo_ang, hi_ang = N * (phi - eps), N * (phi + eps)
    feasible = (m >= 1) & (lo_ang < np.pi / 2)
    ms = np.where(feasible, m, 1)
    p0 = np.cos(N * phi) ** 2
    k_hi = np.where(lo_ang > 0, np.ceil(ms * np.cos(np.minimum(lo_ang, np.pi / 2)) ** 2) - 1, ms)
    k_lo = np.where(hi_ang < np.pi / 2,
                    np.floor(ms * np.cos(np.minimum(hi_ang, np.pi / 2)) ** 2), -1.0)
    p = binom.cdf(k_hi, ms, p0) - binom.cdf(k_lo, ms, p0)
    return np.where(feasible, np.clip(p, 0.0, 1.0), 0.0)


# ---------------------------------------------------------------------------------- the rules
def _first_true(S):
    """Index of the first True in each row, or -1. The scan stops at the first trigger."""
    return np.where(S.any(1), np.argmax(S, axis=1), -1)


def _streak(dec, l):
    """Rows x probes mask: `l` consecutive True in `dec`, counter reset by a single False.

    This is the counter of Algorithm 4 (`counter += 1` on a fall, `counter = 0` otherwise).
    """
    c = np.zeros(dec.shape[0], dtype=np.int64)
    out = np.empty(dec.shape, dtype=bool)
    for k in range(dec.shape[1]):
        c = np.where(dec[:, k], c + 1, 0)
        out[:, k] = c >= l
    return out


def _trailing_mean(P, w):
    """Mean of the last min(w, k+1) probes at each k. w = 0 means the cumulative mean."""
    R, K = P.shape
    cs = np.concatenate([np.zeros((R, 1)), np.cumsum(P, axis=1)], axis=1)
    k = np.arange(1, K + 1)
    lo = np.zeros(K, dtype=np.int64) if w <= 0 else np.maximum(0, k - w)
    return (cs[:, k] - cs[:, lo]) / (k - lo)


def _falls(mean):
    """`mean` fell relative to the previous probe. The first probe can never trigger, matching the
    `len(phi_hat_list) >= 2` guard in the implementation."""
    dec = np.zeros(mean.shape, dtype=bool)
    dec[:, 1:] = mean[:, 1:] < mean[:, :-1]
    return dec


def rule_cumulative(P, C, l):
    return _first_true(_streak(_falls(_trailing_mean(P, 0)), l)), l


def rule_window(P, C, w, l):
    return _first_true(_streak(_falls(_trailing_mean(P, w)), l)), l


def rule_prepost(P, C, w, z):
    """Post-window mean against pre-window mean, with an Eq. (3.4) noise threshold.

    sd(phi_hat_j) = 1/(2 N_j sqrt(m')), so a w-probe window mean has variance sum(sd_j^2)/w^2 and
    the two windows are independent. Fires when pre - post > z * sd(pre - post).
    """
    R, K = P.shape
    cs = np.concatenate([np.zeros((R, 1)), np.cumsum(P, axis=1)], axis=1)
    vs = np.concatenate([[0.0], np.cumsum(C["var"])])
    k = np.arange(K)
    hi, mid, lo = k + 1, k + 1 - w, np.maximum(0, k + 1 - 2 * w)
    ok = mid > lo
    mid_c = np.maximum(mid, lo)
    post = (cs[:, hi] - cs[:, mid_c]) / np.maximum(hi - mid_c, 1)
    pre = (cs[:, mid_c] - cs[:, lo]) / np.maximum(mid_c - lo, 1)
    sd = np.sqrt((vs[hi] - vs[mid_c]) / np.maximum(hi - mid_c, 1) ** 2
                 + (vs[mid_c] - vs[lo]) / np.maximum(mid_c - lo, 1) ** 2)
    fire = ok & ((pre - post) > z * sd)
    fire[:, 0] = False
    return _first_true(fire), w


def rule_threshold(P, C, alpha, l, lag=1):
    """Equation (3.6) against the probe `lag` steps back, `l` times in a row.

    With lag = 1 the reference is the deepest accepted estimate, so this is exactly Algorithm 5's
    criterion running in Algorithm 4's search order. With alpha = 0.5 the normal quantile is zero
    and the rule degenerates to the moving-window rule of width `lag`. The two knobs are therefore
    "how far back to compare" and "how much of a drop to insist on".
    """
    z = norm.ppf(alpha)
    dec = np.zeros(P.shape, dtype=bool)
    dec[:, lag:] = P[:, lag:] < P[:, :-lag] + z * C["sd"][lag:]
    return _first_true(_streak(dec, l)), l


def _pooled_reference(P, C):
    """Inverse-variance pooled mean of probes 0..k-1 and the sd of the pooled-vs-probe difference.

    Weights N_j^2 (Eq. 3.4 gives sd_j = 1/(2 N_j sqrt(m')), so w_j = 1/sd_j^2 = 4 m' N_j^2, and the
    constant cancels), and the reference's own variance 1/(4 m' sum N_j^2) is propagated -- which is
    exactly what Eq. (3.6) leaves out.
    """
    R, K = P.shape
    w = C["N"].astype(float) ** 2
    cw = np.concatenate([[0.0], np.cumsum(w)])[:K]              # sum of weights BEFORE probe k
    cwp = np.concatenate([np.zeros((R, 1)), np.cumsum(P * w, axis=1)], axis=1)[:, :K]
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = cwp / cw
        var_ref = 1.0 / (4.0 * C["m"] * cw)
    sd = np.sqrt(C["sd"] ** 2 + var_ref)
    return ref, sd


def rule_pooled(P, C, alpha, l):
    z = norm.ppf(alpha)
    ref, sd = _pooled_reference(P, C)
    dec = np.zeros(P.shape, dtype=bool)
    dec[:, 1:] = P[:, 1:] < ref[:, 1:] + z * sd[1:]
    return _first_true(_streak(dec, l)), l


def rule_cusum(P, C, k_drift, h):
    """CUSUM on the standardised drop below the pooled reference; backtrack = the excursion length.

    S_k = max(0, S_{k-1} + z_k - k_drift) with z_k = (ref_k - phi_hat_k)/sd_k, declared at S >= h.
    Unlike the streak rules this accumulates evidence, so several small drops can trigger it while a
    single large one need not.
    """
    R, K = P.shape
    ref, sd = _pooled_reference(P, C)
    with np.errstate(invalid="ignore"):
        z = (ref - P) / sd
    z[:, 0] = 0.0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.zeros(R)
    run = np.zeros(R, dtype=np.int64)
    fire = np.zeros((R, K), dtype=bool)
    back = np.zeros(R, dtype=np.int64)
    hit = np.zeros(R, dtype=bool)
    for k in range(K):
        S = np.maximum(0.0, S + z[:, k] - k_drift)
        run = np.where(S > 0, run + 1, 0)
        f = S >= h
        fire[:, k] = f
        new = f & ~hit
        back = np.where(new, np.maximum(run, 1), back)
        hit |= f
    return _first_true(fire), np.maximum(back, 1)


RULES = {
    "cumulative": dict(fn=rule_cumulative, grid=dict(l=L_GRID),
                       label="cumulative mean (Algorithm 4)"),
    "window": dict(fn=rule_window, grid=dict(w=[1, 2, 3, 4, 6, 8, 12], l=[1, 2, 3, 4, 6, 8, 12]),
                   label="moving-window mean"),
    "prepost": dict(fn=rule_prepost, grid=dict(w=[1, 2, 3, 4, 6, 8, 12],
                                               z=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
                    label="pre/post windows, noise threshold"),
    "threshold": dict(fn=rule_threshold, grid=dict(lag=[1, 2, 3, 4, 6, 8],
                                                   alpha=[0.5, 0.2, 0.05, 0.01],
                                                   l=[1, 2, 3, 4, 6, 8]),
                      label="Eq. (3.6) at lag $w$"),
    "pooled": dict(fn=rule_pooled, grid=dict(alpha=[0.5, 0.2, 0.05, 0.01],
                                             l=[1, 2, 3, 4, 6, 8]),
                   label="Eq. (3.6) vs. pooled reference"),
    "cusum": dict(fn=rule_cusum, grid=dict(k_drift=[0.0, 0.25, 0.5, 1.0],
                                           h=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
                  label="CUSUM on the pooled reference"),
}
RULE_ORDER = ["cumulative", "window", "prepost", "threshold", "pooled", "cusum"]


def rule_configs(rule):
    g = RULES[rule]["grid"]
    return [dict(zip(g, v)) for v in product(*g.values())]


# ------------------------------------------------------------------- replay -> outcome of a trial
def outcome(fire, back, C, phi, P, s, eps, budget):
    """Everything one (rule, s) choice produces on a recorded probe stream.

    Follows `find_phi_fixed_budget_linear_search` step for step: N_guess is the firing depth minus
    the backtrack (BEFORE the safeguard), N* = max(1, N_guess - s), and when the scan leaves no
    budget the algorithm returns the retained exploration estimate instead of running an
    exploitation phase.
    """
    depths, K = C["N"], len(C["N"])
    if K == 0:                                     # the opening probe never fit
        R = len(phi)
        z = np.zeros(R)
        return dict(conv=z, N_guess=np.zeros(R, np.int64), N_star=np.zeros(R, np.int64),
                    b_expl=z.copy(), n_probes=np.zeros(R, np.int64),
                    false_alarm=np.zeros(R, bool), miss=np.zeros(R, bool),
                    fired=np.zeros(R, bool))
    back = np.broadcast_to(np.asarray(back), fire.shape).astype(np.int64)
    fired = fire >= 0
    last = np.where(fired, fire, K - 1)
    inc = C["inc"]
    N_guess = depths[last] - np.where(fired, back * inc, 0)
    b_expl = C["cum_cost"][last]
    rem = budget - b_expl
    N_star = np.maximum(1, N_guess - s)

    conv = conv_prob(N_star, phi, np.maximum(rem, 0), eps)
    # no exploitation budget: the algorithm returns phi_hat_list[max(0, len - back)] as its answer
    dry = rem <= 0
    if dry.any():
        idx = np.clip(last + 1 - np.where(fired, back, 0), 0, K - 1)
        kept = P[np.arange(len(phi)), idx]
        conv = np.where(dry, (np.abs(kept - phi) < eps).astype(float), conv)

    n_opt = np.maximum((np.pi / (2.0 * phi)).astype(np.int64), 1)
    keep = np.maximum(0, last + 1 - np.where(fired, back, 0))    # probes NOT declared overshoots
    # depths increase along the scan, so the shallowest declared probe and the deepest retained one
    # decide both errors: a false alarm is a declared probe that was safe, a miss a retained one
    # that was not.
    shallowest_declared = depths[np.clip(keep, 0, K - 1)]
    deepest_retained = depths[np.clip(keep - 1, 0, K - 1)]
    false_alarm = fired & (shallowest_declared <= n_opt)
    miss = (keep >= 1) & (deepest_retained > n_opt)
    return dict(conv=conv, N_guess=N_guess, N_star=N_star, b_expl=b_expl.astype(float),
                n_probes=(last + 1).astype(np.int64), false_alarm=false_alarm, miss=miss,
                fired=fired)


def context(depths, m, inc):
    """The per-probe constants every rule needs: depth, Eq. (3.4) sd, its variance, running cost."""
    return dict(N=depths, m=int(m), inc=int(inc),
                sd=1.0 / (2.0 * depths * np.sqrt(m)),
                var=1.0 / (4.0 * depths.astype(float) ** 2 * m),
                cum_cost=np.cumsum(depths.astype(np.int64) * int(m)))


# ----------------------------------------------------------------------------------- the sweep
def phi_draws(seed, R, scen):
    """The same phases the production sweep draws, so this study is paired with it trial by trial."""
    return np.array([_draw_phi(np.random.default_rng(int(s)), scen["phi_min"], scen["phi_max"],
                               scen["phi_dist"]) for s in seeds_for(seed, R)])


def _score_one(task):
    """Score every (rule, params, s) at one (m', inc), on every requested phi block."""
    scen, budget, m, inc, blocks, chunk = task
    N_min, N_max = M.n_min_of(scen), M.n_max_of(scen)
    eps = scen["eps"]
    depths = scan_depths(N_min, N_max, inc, m, budget)
    C = context(depths, m, inc)
    out = {}
    for bi, (seed, phi_all) in enumerate(blocks):
        acc = {}
        for lo in range(0, len(phi_all), chunk):
            phi = phi_all[lo:lo + chunk]
            rng = np.random.default_rng([seed, m, inc, lo])
            P = (probe_matrix(rng, phi, depths, m) if len(depths)
                 else np.zeros((len(phi), 0)))
            for rule in RULE_ORDER:
                fn = RULES[rule]["fn"]
                for cfg in rule_configs(rule):
                    fire, back = ((np.full(len(phi), -1), 1) if len(depths) == 0
                                  else fn(P, C, **cfg))
                    for s in S_GRID:
                        o = outcome(fire, back, C, phi, P, s, eps, budget)
                        key = (rule, json.dumps(cfg, sort_keys=True), s)
                        a = acc.setdefault(key, [0.0, 0])
                        a[0] += float(o["conv"].sum())
                        a[1] += len(phi)
        for key, (tot, n) in acc.items():
            out.setdefault(key, []).append(tot / n)
    return m, inc, out


def sweep_point(pt, mode, n_jobs):
    """Tune every rule at one operating point, then evaluate the winners on the held-out seed."""
    scen = next(s for s in M.SCENARIOS if s["id"] == pt["sid"])
    budget, cfgm = int(pt["budget"]), MODES[mode]
    lo, hi = M.m_bounds(scen, budget)
    m_grid = set(int(x) for x in np.geomspace(lo, hi, cfgm["n_m"]))
    pub = published_config(pt["sid"], budget)
    if pub is not None:                 # so "same m', different rule" is an exact comparison
        m_grid.add(int(pub[0]["m_exploration"]))
    m_grid = sorted(m_grid)
    blocks = [(sd, phi_draws(sd, cfgm["R_tune"], scen)) for sd in M.SEED_TUNE_BLOCKS]

    tasks = [(scen, budget, m, inc, blocks, 4000) for m in m_grid for inc in INC_GRID]
    print(f"   tuning {len(tasks)} (m', inc) combinations x "
          f"{sum(len(rule_configs(r)) for r in RULE_ORDER) * len(S_GRID)} rule/safeguard "
          f"configurations", flush=True)
    scored = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_CTX) as ex:
        for m, inc, out in ex.map(_score_one, tasks, chunksize=1):
            for (rule, cfg, s), rates in out.items():
                scored[(rule, cfg, s, m, inc)] = rates
    return scen, budget, scored


def best_under(scored, rule, keep=lambda m, inc: True):
    """Best (mean block rate, config) for one rule over the (m', inc) combinations `keep` allows."""
    out = None
    for (r, cfg, s_, m, inc), rates in scored.items():
        if r != rule or not keep(m, inc):
            continue
        mean = float(np.mean(rates))
        if out is None or (mean, -m) > (out[0], -out[1]["m_exploration"]):
            out = (mean, dict(m_exploration=m, inc=inc, safeguard=s_, **json.loads(cfg)))
    return out


def winners_from(scored):
    """argmax of the MEAN block rate per rule, ties broken toward the smaller exploration size --
    the selection rule of qmetrology/pipeline.py, applied to every candidate rule identically."""
    best = {}
    for (rule, cfg, s, m, inc), rates in scored.items():
        mean = float(np.mean(rates))
        cur = best.get(rule)
        if cur is None or (mean, -m) > (cur[0], -cur[1]["m_exploration"]):
            best[rule] = (mean, dict(m_exploration=m, inc=inc, safeguard=s, **json.loads(cfg)),
                          rates)
    return best


def heldout(scen, budget, rule, cfg, R, chunk=4000):
    """Evaluate one frozen (rule, m', inc, s) on the held-out seed, keeping the per-trial arrays."""
    m, inc, s = cfg["m_exploration"], cfg["inc"], cfg["safeguard"]
    rp = {k: v for k, v in cfg.items() if k not in ("m_exploration", "inc", "safeguard")}
    depths = scan_depths(M.n_min_of(scen), M.n_max_of(scen), inc, m, budget)
    C = context(depths, m, inc)
    phi_all = phi_draws(M.SEED_TEST, R, scen)
    keys = ("conv", "N_guess", "N_star", "b_expl", "n_probes", "false_alarm", "miss", "fired")
    acc = {k: [] for k in keys}
    for lo in range(0, R, chunk):
        phi = phi_all[lo:lo + chunk]
        rng = np.random.default_rng([M.SEED_TEST, m, inc, lo])
        P = probe_matrix(rng, phi, depths, m) if len(depths) else np.zeros((len(phi), 0))
        fire, back = ((np.full(len(phi), -1), 1) if len(depths) == 0
                      else RULES[rule]["fn"](P, C, **rp))
        o = outcome(fire, back, C, phi, P, s, scen["eps"], budget)
        for k in keys:
            acc[k].append(o[k])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["phi"] = phi_all
    out["N_opt"] = np.maximum((np.pi / (2.0 * phi_all)).astype(np.int64), 1)
    return out


# ------------------------------------------------------------------------------ streak statistics
def streak_stats(scen, budget, m, inc, R, chunk=4000):
    """Measure what the "roughly 2^-l" argument in the revision notes actually is.

    For probes that are still SAFE (N <= N_opt), record how often the cumulative mean falls, and how
    often `l` such falls occur in a row anywhere in the safe part of the scan -- i.e. the probability
    that the rule stops the scan before the aliasing boundary is ever reached.
    """
    depths = scan_depths(M.n_min_of(scen), M.n_max_of(scen), inc, m, budget)
    if len(depths) < 2:
        return []
    phi_all = phi_draws(SEED_STREAK, R, scen)
    ls = [l for l in range(1, 13)]
    falls = ties = comparisons = 0
    safe_cmp = np.zeros(0)
    prem = {l: 0 for l in ls}
    prem_w4 = {l: 0 for l in ls}
    n = 0
    for lo in range(0, R, chunk):
        phi = phi_all[lo:lo + chunk]
        rng = np.random.default_rng([SEED_STREAK, m, inc, lo])
        P = probe_matrix(rng, phi, depths, m)
        n_opt = np.maximum((np.pi / (2.0 * phi)).astype(np.int64), 1)
        safe = depths[None, :] <= n_opt[:, None]           # probe is below the aliasing boundary
        cm = _trailing_mean(P, 0)
        d = cm[:, 1:] - cm[:, :-1]
        # a comparison is "safe" when both probes it involves are safe
        cmp_safe = safe[:, 1:] & safe[:, :-1]
        falls += int(((d < 0) & cmp_safe).sum())
        ties += int(((d == 0) & cmp_safe).sum())
        comparisons += int(cmp_safe.sum())
        safe_cmp = np.concatenate([safe_cmp, cmp_safe.sum(1)])
        for l in ls:
            fired = _first_true(_streak(_falls(cm), l))
            # premature: fired, and the shallowest declared probe was still safe
            keep = np.maximum(0, fired + 1 - l)
            prem[l] += int(((fired >= 0) & (depths[np.clip(keep, 0, len(depths) - 1)]
                                            <= n_opt)).sum())
            f4 = _first_true(_streak(_falls(_trailing_mean(P, 4)), l))
            k4 = np.maximum(0, f4 + 1 - l)
            prem_w4[l] += int(((f4 >= 0) & (depths[np.clip(k4, 0, len(depths) - 1)]
                                            <= n_opt)).sum())
        n += len(phi)
    q = falls / max(comparisons, 1)
    T = float(safe_cmp.mean())
    rows = []
    for l in ls:
        indep = 1.0 - (1.0 - q ** l) ** max(T - l + 1, 0.0)
        coin = 1.0 - (1.0 - 0.5 ** l) ** max(T - l + 1, 0.0)
        rows.append(dict(lookback_window=l,
                         p_fall_safe=round(q, 5), p_tie_safe=round(ties / max(comparisons, 1), 5),
                         safe_comparisons_mean=round(T, 3),
                         premature_stop_measured=round(prem[l] / n, 5),
                         premature_stop_indep_q=round(indep, 5),
                         premature_stop_coin_flip=round(coin, 5),
                         premature_stop_window4=round(prem_w4[l] / n, 5), R=n))
    return rows


# ---------------------------------------------------------------------------------- diagnostics
def diag_row(o, budget):
    """The Table 4.5/4.6 diagnostics, computed on the held-out arrays of one frozen rule."""
    g, no = o["N_guess"].astype(float), o["N_opt"].astype(float)
    ratio = np.where(no > 0, g / no, np.nan)
    star = np.where(no > 0, o["N_star"] / no, np.nan)
    fa = o["false_alarm"]
    short = (no - g)[fa]                       # phase gates the false alarm stopped short by
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    return dict(
        rate=float(o["conv"].mean()),
        rate_se=float(o["conv"].std(ddof=1) / np.sqrt(len(o["conv"]))),
        false_alarm_rate=float(fa.mean()), miss_rate=float(o["miss"].mean()),
        fired_rate=float(o["fired"].mean()),
        guess_ratio_median=float(np.nanmedian(ratio)),
        guess_within_10pct=float(np.nanmean(np.abs(ratio - 1.0) <= 0.10)),
        guess_mean_abs_rel_err=float(np.nanmean(np.abs(ratio - 1.0))),
        star_ratio_median=float(np.nanmedian(star)),
        star_overshoot=float(np.nanmean(o["N_star"] > o["N_opt"])),
        exploration_share_median=float(np.median(o["b_expl"] / budget)),
        n_probes_median=float(np.median(o["n_probes"])),
        fa_short_median=q(short, 50), fa_short_p90=q(short, 90), fa_short_p95=q(short, 95),
        fa_within_10pct=(float(np.mean(np.abs(ratio[fa] - 1.0) <= 0.10)) if fa.any()
                         else float("nan")),
        R=int(len(o["conv"])))


BAKEOFF_FIELDS = ["setting", "scenario_id", "eps", "budget", "point_tag", "rule", "rule_label",
                  "params", "tune_block_rates", "tune_mean", "rate", "rate_se",
                  "delta_vs_cumulative_pp", "delta_se_pp", "false_alarm_rate", "miss_rate",
                  "fired_rate", "guess_ratio_median", "guess_within_10pct",
                  "guess_mean_abs_rel_err", "star_ratio_median", "star_overshoot",
                  "exploration_share_median", "n_probes_median", "fa_short_median",
                  "fa_short_p90", "fa_short_p95", "fa_within_10pct", "R", "seed_test",
                  "seed_tune_blocks", "objective"]

GRID_FIELDS = ["scenario_id", "budget", "rule", "params", "tune_mean", "rank", "R_tune"]

# The best a rule can do at each exploration size, marginalised over inc, the safeguard and its own
# parameters. This separates "the rule is better" from "the rule prefers a different m'".
PROFILE_FIELDS = ["scenario_id", "budget", "rule", "m_exploration", "best_tune_mean",
                  "best_params", "R_tune"]

# The control the thesis actually needs: every rule forced onto the PUBLISHED exploration size and
# increment, so only the stopping rule and its safeguard differ.
MATCHED_FIELDS = ["setting", "scenario_id", "eps", "budget", "point_tag", "rule", "rule_label",
                  "m_exploration", "inc", "params", "tune_mean", "rate", "rate_se",
                  "delta_vs_cumulative_pp", "delta_se_pp", "false_alarm_rate", "miss_rate",
                  "guess_ratio_median", "guess_within_10pct", "star_ratio_median",
                  "star_overshoot", "exploration_share_median", "R"]

STREAK_FIELDS = ["setting", "scenario_id", "eps", "budget", "m_exploration", "inc",
                 "lookback_window", "p_fall_safe", "p_tie_safe", "safe_comparisons_mean",
                 "premature_stop_measured", "premature_stop_indep_q", "premature_stop_coin_flip",
                 "premature_stop_window4", "R", "seed"]

VAL_FIELDS = ["check", "scenario_id", "budget", "params", "expected", "obtained", "abs_diff",
              "note"]

CROSS_FIELDS = ["setting", "scenario_id", "eps", "rule", "rule_label", "threshold_pct",
                "budget_to_reach", "ratio_vs_brute", "published_linear_budget",
                "published_brute_budget", "budgets_used", "rates", "crossing_rule"]


# ----------------------------------------------------------------------------------- validation
def rule_of(cfg):
    """(rule name, replay config) for a configuration frozen by the sweep.

    Algorithm 4 gained a `mean_window` axis whose 0 is the cumulative mean, so a frozen config now
    names which rule it used and the replay has to follow it rather than assume the old one.
    """
    w = int(cfg.get("mean_window", 0) or 0)
    c = dict(m_exploration=cfg["m_exploration"], inc=cfg["inc"], safeguard=cfg["safeguard"],
             l=cfg["lookback_window"])
    if w > 0:
        return "window", dict(c, w=w)
    return "cumulative", c


def published_config(sid, budget):
    """The configuration the production sweep froze for Algorithm 4 at this operating point."""
    import csv
    p = path("winners.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["algorithm"] == "linear" and r["scenario_id"] == sid
                    and int(float(r["budget"])) == int(budget)):
                return json.loads(r["params"]), float(r["heldout_rate"])
    return None


def validate(mode):
    """Two checks, both of which must pass before any bake-off number is worth reading.

    1. `conv_prob` reproduces qmetrology.oracle.convergence_prob, which the oracle rows are built
       from -- so the analytic exploitation is the same quantity the thesis already reports.
    2. Replaying the PUBLISHED configuration through this module reproduces the sweep's own held-out
       convergence rate. That is the end-to-end check: it exercises the scan reconstruction, the
       backtracking, the safeguard and the analytic exploitation at once.
    """
    from qmetrology.oracle import convergence_prob
    rng = np.random.default_rng(7)
    rows = []
    worst = 0.0
    for _ in range(200):
        phi = float(rng.uniform(0.001, 0.5))
        N = int(rng.integers(1, 400))
        B = int(rng.integers(100, 10 ** 6))
        eps = float(10.0 ** rng.uniform(-6, -2))
        a = float(convergence_prob(N, phi, B, eps)[0])
        b = float(conv_prob(np.array([N]), np.array([phi]), np.array([B]), eps)[0])
        worst = max(worst, abs(a - b))
    rows.append(dict(check="conv_prob vs qmetrology.oracle.convergence_prob", expected=0.0,
                     obtained=round(worst, 12), abs_diff=round(worst, 12),
                     note="max |difference| over 200 random (N, phi, B, eps)"))
    print(f"   conv_prob agrees with the oracle to {worst:.2e}", flush=True)

    R = MODES[mode]["R_test"]
    for pt in POINTS:
        pub = published_config(pt["sid"], pt["budget"])
        if pub is None:
            continue
        cfg, published_rate = pub
        scen = next(s for s in M.SCENARIOS if s["id"] == pt["sid"])
        rule, c = rule_of(cfg)
        o = heldout(scen, int(pt["budget"]), rule, c, R)
        got = float(o["conv"].mean())
        se = float(o["conv"].std(ddof=1) / np.sqrt(R))
        rows.append(dict(check="published Algorithm 4 config, analytic replay vs swept held-out rate",
                         scenario_id=pt["sid"], budget=int(pt["budget"]),
                         params=json.dumps(cfg, sort_keys=True),
                         expected=round(published_rate, 5), obtained=round(got, 5),
                         abs_diff=round(abs(got - published_rate), 5),
                         note=f"analytic se {se:.5f}; the sweep's own rate carries a binomial se of "
                              f"about {np.sqrt(published_rate*(1-published_rate)/50000):.5f}"))
        print(f"   {pt['sid']:>13s} B={int(pt['budget']):>12,}  swept {published_rate:.4f}  "
              f"replay {got:.4f} (se {se:.4f})", flush=True)
    Table("linear_detector_validation.csv", VAL_FIELDS, reset=True).rows(rows)
    return rows


# ---------------------------------------------------------------------------------------- main
ZONE_FIELDS = ["m_exploration", "p_atom_target", "r_atom", "gates_below_N_opt_frac",
               "small_angle_approximation", "note"]


def degenerate_zone():
    """Why the false-alarm rate is high, computed exactly and without any tuned parameter.

    On the SAFE branch the readout probability is p0 = cos^2(N phi), so the probe returns K = 0 with
    probability sin^{2 m'}(N phi) = sin^{2 m'}(r pi/2). When that probability approaches one the
    estimate is no longer random: it is the single atom pi/(2N), which DECREASES with N. Every rule
    in this study fires on consecutive decreases, so in that zone it fires with probability
    approaching one -- while the probes are still safe, hence "false alarm".

    The zone is therefore not noise and not a defect of the cumulative mean. It is the same
    degeneracy analysis/overshoot_criterion.py identifies as the mechanism that makes
    the overshoot criterion sharp, seen from below the boundary. Its width shrinks like
    1/sqrt(m'): writing r = 1 - u,

        sin^{2m'}(r pi/2) = cos^{2m'}(u pi/2) ~ exp(-m' u^2 pi^2 / 4),

    so the atom dominates while u <~ (2/pi) sqrt(ln 2 / m'). That is the scale-free statement behind
    the measured false-alarm rates falling from 78% at m' = 2 to 30% at m' = 176.

    Written as results/linear_degenerate_zone.csv; nothing is simulated.
    """
    rows = []
    for m in (1, 2, 5, 15, 50, 176, 600, 5565):
        for target in (0.9, 0.5):
            # solve sin^{2m}(r pi/2) = target for r in (0, 1)
            r = 2.0 / np.pi * np.arcsin(target ** (1.0 / (2.0 * m)))
            approx = 1.0 - 2.0 / np.pi * np.sqrt(-np.log(target) / m)
            rows.append(dict(
                m_exploration=m, p_atom_target=target, r_atom=round(float(r), 6),
                gates_below_N_opt_frac=round(float(1.0 - r), 6),
                small_angle_approximation=round(float(approx), 6),
                note="P(K = 0) >= target for every probe with r >= r_atom, where the estimate is "
                     "the deterministic atom pi/(2N) and consecutive decreases are automatic"))
    Table("linear_degenerate_zone.csv", ZONE_FIELDS, reset=True).rows(rows)
    print("\n-- the pre-boundary zone where the estimate stops being random")
    print("   m'      r with P(K=0)=0.9    r with P(K=0)=0.5   (fraction of N_opt below which "
          "the scan is deterministic)")
    for m in sorted({r["m_exploration"] for r in rows}):
        a = [r for r in rows if r["m_exploration"] == m]
        r9 = next(x["r_atom"] for x in a if x["p_atom_target"] == 0.9)
        r5 = next(x["r_atom"] for x in a if x["p_atom_target"] == 0.5)
        print(f"   {m:>5d}   {r9:>10.4f}          {r5:>10.4f}        "
              f"{100*(1-r5):.1f}% of the depth range")
    return rows


def rule_crossings(threshold=0.90):
    """Interpolate the budget each rule needs for `threshold` convergence, from the swept points.

    Uses qmetrology.uncertainty.crossing -- the same log-linear rule the production sweep uses for
    results/budget_crossings.csv -- so the numbers can be put side by side.
    """
    import csv as _csv
    from qmetrology.uncertainty import crossing
    rows = list(_csv.DictReader(open(path("linear_detector_bakeoff.csv"), newline="")))
    pub = {}
    pb = path("budget_crossings.csv")
    if os.path.exists(pb):
        for r in _csv.DictReader(open(pb, newline="")):
            if int(float(r["threshold_pct"])) == int(100 * threshold):
                pub[(r["scenario_id"], r["algorithm"])] = float(r["budget_to_reach"])
    by = {}
    for r in rows:
        by.setdefault((r["scenario_id"], r["rule"]), []).append(
            (int(float(r["budget"])), float(r["rate"]), r["setting"], r["eps"], r["rule_label"]))
    out = []
    for (sid, rule), v in sorted(by.items()):
        v.sort()
        if len(v) < 3:
            continue
        b = [x[0] for x in v]
        rt = [x[1] for x in v]
        c = crossing(b, rt, threshold)
        brute = pub.get((sid, "brute"), float("nan"))
        out.append(dict(setting=v[0][2], scenario_id=sid, eps=v[0][3], rule=rule,
                        rule_label=v[0][4], threshold_pct=int(100 * threshold),
                        budget_to_reach=round(c, 1) if np.isfinite(c) else "",
                        ratio_vs_brute=round(brute / c, 3) if np.isfinite(c) else "",
                        published_linear_budget=pub.get((sid, "linear"), ""),
                        published_brute_budget=brute,
                        budgets_used=json.dumps(b), rates=json.dumps([round(x, 5) for x in rt]),
                        crossing_rule="qmetrology.uncertainty.crossing (log-linear)"))
    Table("linear_detector_crossings.csv", CROSS_FIELDS, reset=True).rows(out)
    print("\n-- budget for 90% convergence, by stopping rule")
    for r in out:
        if r["budget_to_reach"] == "":
            continue
        print(f"   {r['scenario_id']:>10s}  {r['rule']:<11s} {r['budget_to_reach']:>14,.0f}"
              f"   {r['ratio_vs_brute']}x brute   (published linear "
              f"{r['published_linear_budget']:,.0f})", flush=True)
    return out


def main():
    argv = sys.argv[1:]
    mode = "quick" if "--quick" in argv else "full"
    n_jobs = min(12, os.cpu_count() or 4)
    print(f"linear-search detector bake-off [{mode}]", flush=True)

    print("\n-- validation", flush=True)
    validate(mode)
    degenerate_zone()
    if "--validate" in argv or "--zone" in argv:
        return

    bake = Table("linear_detector_bakeoff.csv", BAKEOFF_FIELDS, reset=True)
    grid = Table("linear_detector_grid.csv", GRID_FIELDS, reset=True)
    prof = Table("linear_detector_profile.csv", PROFILE_FIELDS, reset=True)
    matched = Table("linear_detector_matched.csv", MATCHED_FIELDS, reset=True)
    streak = Table("linear_detector_streaks.csv", STREAK_FIELDS, reset=True)
    R = MODES[mode]["R_test"]

    points = list(POINTS)
    if "--curve" in argv:
        have = {(p["sid"], int(p["budget"])) for p in points}
        points += [p for p in CURVE_POINTS if (p["sid"], int(p["budget"])) not in have]
    for pt in points:
        scen = next(s for s in M.SCENARIOS if s["id"] == pt["sid"])
        print(f"\n== {scen['label']}  B={int(pt['budget']):,}  ({pt['tag']})", flush=True)
        scen, budget, scored = sweep_point(pt, mode, n_jobs)
        best = winners_from(scored)

        # the full tuning landscape, so a losing rule can be seen to have been searched properly
        rows = []
        for rule in RULE_ORDER:
            cand = [(float(np.mean(v)), dict(m_exploration=k[3], inc=k[4], safeguard=k[2],
                                             **json.loads(k[1])))
                    for k, v in scored.items() if k[0] == rule]
            cand.sort(key=lambda t: -t[0])
            for i, (mean, cfg) in enumerate(cand[:25]):
                rows.append(dict(scenario_id=scen["id"], budget=budget, rule=rule,
                                 params=json.dumps(cfg, sort_keys=True),
                                 tune_mean=round(mean, 5), rank=i + 1,
                                 R_tune=MODES[mode]["R_tune"]))
        grid.rows(rows)

        prof.rows([dict(scenario_id=scen["id"], budget=budget, rule=rule, m_exploration=m,
                        best_tune_mean=round(b[0], 5),
                        best_params=json.dumps(b[1], sort_keys=True),
                        R_tune=MODES[mode]["R_tune"])
                   for rule in RULE_ORDER
                   for m in sorted({k[3] for k in scored})
                   for b in [best_under(scored, rule, lambda mm, ii, _m=m: mm == _m)]
                   if b is not None])

        ref = None
        out_rows = []
        for rule in RULE_ORDER:
            mean, cfg, rates = best[rule]
            o = heldout(scen, budget, rule, cfg, R)
            d = diag_row(o, budget)
            if rule == "cumulative":
                ref = o["conv"]
            dif = o["conv"] - ref
            out_rows.append(dict(
                setting=scen["label"], scenario_id=scen["id"], eps=scen["eps"], budget=budget,
                point_tag=pt["tag"], rule=rule, rule_label=RULES[rule]["label"],
                params=json.dumps(cfg, sort_keys=True),
                tune_block_rates=json.dumps([round(r, 5) for r in rates]),
                tune_mean=round(mean, 5),
                delta_vs_cumulative_pp=round(100 * float(dif.mean()), 4),
                delta_se_pp=round(100 * float(dif.std(ddof=1) / np.sqrt(len(dif))), 4),
                seed_test=M.SEED_TEST,
                seed_tune_blocks=",".join(str(s) for s in M.SEED_TUNE_BLOCKS),
                objective="exact P(|phi_hat - phi| < eps) at the chosen (N*, m), averaged over "
                          "trials",
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}))
            print(f"   {rule:<11s} {100*d['rate']:6.2f}%  "
                  f"(d {100*float(dif.mean()):+5.2f} +- {100*float(dif.std(ddof=1)/np.sqrt(len(dif))):.2f} pp)"
                  f"  FA {100*d['false_alarm_rate']:5.1f}%  miss {100*d['miss_rate']:5.1f}%  "
                  f"{cfg}", flush=True)
        pub = published_config(pt["sid"], pt["budget"])
        if pub is not None:
            pc = pub[0]
            cfg = dict(m_exploration=pc["m_exploration"], inc=pc["inc"],
                       safeguard=pc["safeguard"], l=pc["lookback_window"])
            o = heldout(scen, budget, "cumulative", cfg, R)
            d = diag_row(o, budget)
            dif = o["conv"] - ref
            out_rows.append(dict(
                setting=scen["label"], scenario_id=scen["id"], eps=scen["eps"], budget=budget,
                point_tag=pt["tag"], rule="published",
                rule_label="cumulative mean, configuration frozen by the production sweep",
                params=json.dumps(cfg, sort_keys=True), tune_block_rates="", tune_mean="",
                delta_vs_cumulative_pp=round(100 * float(dif.mean()), 4),
                delta_se_pp=round(100 * float(dif.std(ddof=1) / np.sqrt(len(dif))), 4),
                seed_test=M.SEED_TEST,
                seed_tune_blocks=",".join(str(s) for s in M.SEED_TUNE_BLOCKS),
                objective="not tuned here: the configuration results/winners.csv "
                          "froze, replayed for reference",
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}))
            print(f"   {'published':<11s} {100*d['rate']:6.2f}%  "
                  f"(d {100*float(dif.mean()):+5.2f} pp)  FA {100*d['false_alarm_rate']:5.1f}%  "
                  f"{cfg}", flush=True)

            # matched control: same m' and inc as the published configuration, rule free
            mp, ip = pc["m_exploration"], pc["inc"]
            mref = None
            mrows = []
            for rule in RULE_ORDER:
                b = best_under(scored, rule, lambda m, i, _m=mp, _i=ip: m == _m and i == _i)
                if b is None:
                    continue
                mo = heldout(scen, budget, rule, b[1], R)
                md = diag_row(mo, budget)
                if rule == "cumulative":
                    mref = mo["conv"]
                mdif = mo["conv"] - mref
                mrows.append(dict(
                    setting=scen["label"], scenario_id=scen["id"], eps=scen["eps"],
                    budget=budget, point_tag=pt["tag"], rule=rule,
                    rule_label=RULES[rule]["label"], m_exploration=mp, inc=ip,
                    params=json.dumps(b[1], sort_keys=True), tune_mean=round(b[0], 5),
                    rate=round(md["rate"], 6), rate_se=round(md["rate_se"], 6),
                    delta_vs_cumulative_pp=round(100 * float(mdif.mean()), 4),
                    delta_se_pp=round(100 * float(mdif.std(ddof=1) / np.sqrt(len(mdif))), 4),
                    false_alarm_rate=round(md["false_alarm_rate"], 6),
                    miss_rate=round(md["miss_rate"], 6),
                    guess_ratio_median=round(md["guess_ratio_median"], 4),
                    guess_within_10pct=round(md["guess_within_10pct"], 4),
                    star_ratio_median=round(md["star_ratio_median"], 4),
                    star_overshoot=round(md["star_overshoot"], 6),
                    exploration_share_median=round(md["exploration_share_median"], 6),
                    R=md["R"]))
            matched.rows(mrows)
            print("   matched m'=%d inc=%d: " % (mp, ip)
                  + "  ".join(f"{r['rule'][:4]} {100*r['rate']:.2f}" for r in mrows), flush=True)

        bake.rows(out_rows)

        if pub is not None:
            cfg = pub[0]
            sr = streak_stats(scen, budget, cfg["m_exploration"], cfg["inc"],
                              MODES[mode]["R_tune"])
            streak.rows([dict(setting=scen["label"], scenario_id=scen["id"], eps=scen["eps"],
                              budget=budget, m_exploration=cfg["m_exploration"], inc=cfg["inc"],
                              seed=SEED_STREAK, **r) for r in sr])

    if "--curve" in argv:
        rule_crossings()
    print("\nwrote", path("linear_detector_bakeoff.csv"))


if __name__ == "__main__":
    main()
