"""The phase-search algorithms (Chapter 3).

Uniform signature: find_phi_*(rng, phi, phi_max, phi_min, **params) -> (phi_hat, budget_used).
The fixed-budget variants cap the total cost N*m at `budget`; the budget needed for a target
convergence is found by sweeping the budget (analysis/extensive_sweep.py).

The variable-budget ("run until confident") formulations live in legacy/variable_algorithms.py.
Only reverse-engineering-lite (Algorithm 7) is kept here in variable form, as Table 3.1 reports it.
"""
import numpy as np
from scipy.stats import norm

from .oracle import alias_depth, optimal_depth
from .posterior import depth_from_posterior, hits_from_estimate
from .safeguard import pilot_sd, risk_optimal_depth
from .sim import simulate_errors


# Omniscient benchmarks: the depth a protocol would pick if it already knew phi (qmetrology/oracle.py).
# Neither is an implementable algorithm — they bound what the whole family can achieve, so every
# adaptive result can be read as a fraction of the attainable gap rather than only against brute force.
def find_phi_fixed_budget_oracle(rng, phi, phi_max, phi_min, budget, eps_target):
    """Oracle: spend the whole budget at N* = argmax_N P(|phi_hat - phi| < eps), knowing phi.

    Upper bound for any protocol that commits its budget to a single depth — which is every algorithm
    in Chapter 3, the exploration phases existing only to identify that depth.
    """
    N = optimal_depth(phi, budget, eps_target)
    m = int(budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, m * N


def find_phi_fixed_budget_oracle_nopt(rng, phi, phi_max, phi_min, budget, eps_target=None,
                                      trace=None):
    """Brute force at the oracle-provided N_opt, using the real binomial estimator.

    This differs from ordinary brute force only in its number of phase gates: it is handed
    N_opt = floor(pi/(2 phi)) instead of using the prior-safe N_min.  It then takes the maximum
    number of whole shots allowed by the hard budget.  No Gaussian error is injected.
    """
    N = alias_depth(phi, budget)
    m = int(budget) // N
    phi_hat = simulate_errors(rng, phi, m, N)
    if trace is not None:
        trace.nominal_budget = float(budget)
        trace.set_exploration(0.0, N_guess=None, status="ok")
        trace.set_exploitation(N, m)
    return phi_hat, m * N


def find_phi_fixed_budget_oracle_alias(rng, phi, phi_max, phi_min, budget):
    """Backward-compatible name for the N_opt oracle."""
    return find_phi_fixed_budget_oracle_nopt(rng, phi, phi_max, phi_min, budget)


def find_phi_fixed_budget_ceiling(rng, phi, phi_max, phi_min, budget, eps_target=None, trace=None):
    """Backward-compatible name; the former artificial Gaussian runner has been removed."""
    return find_phi_fixed_budget_oracle_nopt(
        rng, phi, phi_max, phi_min, budget, eps_target=eps_target, trace=trace)


# Separable baseline (N = 1)
def find_phi_fixed_budget_separable(rng, phi, phi_max, phi_min, budget, trace=None):
    """Separable protocol: a single N=1 circuit measured `budget` times."""
    N = 1
    m = int(budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    if trace is not None:
        trace.nominal_budget = float(budget)
        trace.set_exploration(0.0, N_guess=None, status="ok")
        trace.set_exploitation(1, int(m))
    return phi_hat, m * N


# Brute force (Algorithm 3): fix N = N_min, spend all budget on m
def find_phi_fixed_budget_brute_force(rng, phi, phi_max, phi_min, budget, trace=None):
    N_min = max(np.pi // (2 * phi_max), 1)
    m = int(budget / N_min)
    phi_hat = simulate_errors(rng, phi, m, N_min)
    if trace is not None:
        # no exploration phase at all: N_guess stays null (handoff: do not manufacture a guess for
        # an algorithm that does not search), B_exploration = 0, and N_min is the final depth.
        trace.nominal_budget = float(budget)
        trace.set_exploration(0.0, N_guess=None, status="ok")
        trace.set_exploitation(int(N_min), int(m))
    return phi_hat, m * N_min


# Linear search (Algorithm 4): scan N upward, detect overshoot via lookback
def _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window, inc,
                           mean_window=0, trace=None):
    """Exploration scan of Algorithm 4 — probe N = N_min, N_min+inc, ... until the mean of the
    estimates falls `lookback_window` times in a row (the overshoot verdict).

    `mean_window` is the width of that mean: 0 (the default) averages every estimate collected so
    far, which is the cumulative mean of the published algorithm; w > 0 averages only the last w.
    The cumulative rule is the w -> infinity member of the same family, so the parameter is a strict
    generalisation and w = 0 reproduces the published scan exactly, RNG draw for RNG draw
    (tests/test_consolidated.py::test_linear_mean_window_default_is_cumulative).

    Why the width matters: for a full window the two consecutive means differ only in the element
    that enters and the one that leaves, so the test "the window mean fell" is identically
    phi_hat_k < phi_hat_{k-w}. A finite window therefore compares each probe with one a fixed depth
    behind it, while the cumulative mean compares against a reference whose effective lag keeps
    growing as the scan approaches the aliasing boundary. See results/consolidated/LINEAR_SEARCH.md.

    Returns (phi_hats, Ns, budget_used, overshot) or None if the first probe already exceeds the
    budget. Consumes the RNG in exactly the same order as the published algorithm, so the constant-`s`
    and statistical-safeguard variants stay paired trial-by-trial.

    `trace` records every probe. The scan's verdict is a TRIAL-level statement (the mean fell
    `lookback_window` times), so the per-probe classification is assigned afterwards by
    `_mark_linear_probes`: the probes the algorithm backtracks over are the ones it declares
    overshoots, the retained ones are the ones it accepts.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return None

    phi_hat_list, N_list = [], []
    counter = 0
    budget_used = 0
    r_mean = 0
    done = False
    while not done:
        if budget_used + N * m_exploration > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_list.append(N)
        if trace is not None:
            trace.probe(N, m_exploration, phi_hat)

        r_mean_new = np.mean(phi_hat_list[-mean_window:] if mean_window else phi_hat_list)
        if len(phi_hat_list) >= 2 and r_mean_new < r_mean:
            counter += 1
        else:
            counter = 0
        r_mean = r_mean_new

        if counter >= lookback_window:
            done = True
        elif N >= N_max or budget_used >= budget:
            done = True
        else:
            N += inc

    return phi_hat_list, N_list, budget_used, counter >= lookback_window


def _mark_linear_probes(trace, overshot, lookback_window):
    """Attach the scan's own classification to each probe.

    The stopping rule fires after `lookback_window` consecutive falls of the running mean and the
    algorithm then backtracks by exactly that many steps — i.e. it treats the last `lookback_window`
    probes as the aliased ones and keeps the rest. That is the only overshoot statement linear search
    makes, so it is the one scored against the truth `N_i > N_opt`. With no verdict, every probe is
    retained and therefore accepted.
    """
    if trace is None:
        return
    keep = max(0, len(trace.probes) - lookback_window) if overshot else len(trace.probes)
    for i, p in enumerate(trace.probes):
        p.declared_overshoot = bool(i >= keep)
        p.accepted = not p.declared_overshoot


def find_phi_fixed_budget_linear_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        lookback_window=5, safeguard=1, inc=1, mean_window=0,
                                        trace=None):
    """Algorithm 4. N_guess is the depth the scan returns after its lookback backtracking and BEFORE
    the safeguard decrement `s`; N_star = max(1, N_guess - s) is what the exploitation actually runs
    at (ALGORITHM_DIAGNOSTICS_HANDOFF.md, "Minimal thesis-facing definitions").

    `mean_window` is the width of the mean the stopping rule tests; 0 is the cumulative mean the
    algorithm was first published with, and is the default so that every earlier call site keeps its
    behaviour. It is a tuned parameter of the manifest grid."""
    if trace is not None:
        trace.nominal_budget = float(budget)
    out = _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget,
                                 lookback_window, inc, mean_window=mean_window, trace=trace)
    if out is None:
        if trace is not None:
            trace.set_exploration(0.0, status="refused_pilot_unaffordable")
        return np.inf, budget
    phi_hat_list, N_list, budget_used, overshot = out
    N = N_list[-1] - lookback_window * inc if overshot else N_list[-1]
    _mark_linear_probes(trace, overshot, lookback_window)
    if trace is not None:
        trace.set_exploration(budget_used, N_guess=int(N),
                              status="detector_fired" if overshot else "scan_exhausted")

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        if trace is not None:
            trace.status = "no_exploitation_budget_exhausted"
        return phi_hat_list[idx], budget_used

    N = max(1, N - safeguard)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    if trace is not None:
        trace.set_exploitation(N, m)
    return phi_hat, budget_used


def _linear_search_explore_lagged(rng, phi, phi_max, phi_min, m_exploration, budget,
                                  lookback_window, inc, lag, conf, trace=None):
    """Exploration scan of Algorithm 4 with the CUMULATIVE-MEAN stopping rule replaced by the
    overshoot criterion of Eq. (3.6), applied against the probe `lag` steps back.

    Declare a fall at probe k when

        phi_hat_k < phi_hat_{k-lag} + z_{1-conf} / (2 N_k sqrt(m')),

    and stop after `lookback_window` consecutive falls, exactly as the published scan does.

    Two special cases place this inside the existing thesis:
      conf = 0.5   the normal quantile vanishes and the rule is phi_hat_k < phi_hat_{k-lag}, which
                   (analysis/consolidated/linear_detector_study.py) is identically the
                   "moving window of width `lag`" rule, because a trailing window mean falls exactly
                   when its entering element is below its leaving one;
      lag = 1      the reference is the deepest probe not yet declared an overshoot, i.e. Algorithm
                   5's criterion evaluated in Algorithm 4's search order.

    The loop is duplicated from `_linear_search_explore` rather than factored out of it: the
    published scan is a reported result and is deliberately left untouched.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return None

    z = norm.ppf(1 - conf)
    phi_hat_list, N_list = [], []
    counter = 0
    budget_used = 0
    done = False
    while not done:
        if budget_used + N * m_exploration > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_list.append(N)
        if trace is not None:
            trace.probe(N, m_exploration, phi_hat)

        if len(phi_hat_list) > lag:
            threshold = phi_hat_list[-1 - lag] + z / (2 * N * np.sqrt(m_exploration))
            counter = counter + 1 if phi_hat < threshold else 0
        else:
            counter = 0

        if counter >= lookback_window:
            done = True
        elif N >= N_max or budget_used >= budget:
            done = True
        else:
            N += inc

    return phi_hat_list, N_list, budget_used, counter >= lookback_window


def find_phi_fixed_budget_linear_search_lagged(rng, phi, phi_max, phi_min, m_exploration, budget,
                                               lookback_window=4, safeguard=1, inc=1, lag=3,
                                               conf=0.5, trace=None):
    """NOT A REPORTED ALGORITHM (yet). Algorithm 4 with the lagged Eq. (3.6) stopping rule.

    Everything except the overshoot verdict is Algorithm 4: the same upward scan, the same
    backtracking by `lookback_window * inc`, the same safeguard N* = max(1, N_guess - s).

    It exists so that the alternative measured in
    results/consolidated/LINEAR_SEARCH.md can be swept end to end without editing the published
    algorithm. To report it, give it a manifest entry with a tuning grid over
    (m_exploration, lookback_window, safeguard, inc, lag, conf) and add its key to
    qmetrology.manifest.ORDER -- nothing else in the pipeline needs to change.
    """
    if trace is not None:
        trace.nominal_budget = float(budget)
    out = _linear_search_explore_lagged(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        lookback_window, inc, lag, conf, trace=trace)
    if out is None:
        if trace is not None:
            trace.set_exploration(0.0, status="refused_pilot_unaffordable")
        return np.inf, budget
    phi_hat_list, N_list, budget_used, overshot = out
    N = N_list[-1] - lookback_window * inc if overshot else N_list[-1]
    _mark_linear_probes(trace, overshot, lookback_window)
    if trace is not None:
        trace.set_exploration(budget_used, N_guess=int(N),
                              status="detector_fired" if overshot else "scan_exhausted")

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        if trace is not None:
            trace.status = "no_exploitation_budget_exhausted"
        return phi_hat_list[idx], budget_used

    N = max(1, N - safeguard)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    if trace is not None:
        trace.set_exploitation(N, m)
    return phi_hat, budget_used


def find_phi_fixed_budget_linear_search_risk(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, lookback_window=5, inc=1, pool=True):
    """DIAGNOSTIC, NOT A REPORTED ALGORITHM. Algorithm 4 with the tuned decrement `s` replaced by the
    risk-optimal depth (safeguard.py). Kept because the negative result is worth being able to redo.

    Linear search leaves a whole *history* of estimates rather than one pilot. Every probe obeys
    Eq. (3.4) with a known variance 1/(4 N_i^2 m'), and the probes are independent, so they combine
    by inverse-variance weighting (`pool=True`):

        phi_pilot = sum(N_i^2 phi_hat_i) / sum(N_i^2),   sigma = 1 / (2 sqrt(m' sum N_i^2)).

    `pool=False` uses only the deepest retained probe, i.e. literally the binary-search pilot. The
    probes that triggered the overshoot verdict are dropped -- the same `lookback_window` the
    published algorithm backs off by -- because those are the aliased ones and A2 fails for them.

    WHY IT IS NOT REPORTED (analysis/exploration_study.py, 15 operating points, de-biased):
    the rule does beat the tuned `s` by +3.5 to +5.6 pp on average, but it wins by *deleting the
    scan*. Grid search drives the configuration to lookback_window = 1, so at 10 of the 15 points the
    exploration takes exactly two probes and retains one -- the pooling factor sum(N_i^2)/N_last^2 is
    then 1.0, i.e. no pooling happens at all. What is left is reverse engineering with one wasted
    probe, and true reverse engineering beats it at every single point (+0.11 to +2.60 pp). The
    scan carries no information the depth rule needs, so there is no frontier point here.
    """
    out = _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget,
                                 lookback_window, inc)
    if out is None:
        return np.inf, budget
    phi_hat_list, N_list, budget_used, overshot = out

    keep = max(1, len(phi_hat_list) - lookback_window) if overshot else len(phi_hat_list)
    ph = np.asarray(phi_hat_list[:keep], dtype=float)
    Nk = np.asarray(N_list[:keep], dtype=float)
    finite = np.isfinite(ph)
    if not finite.any():
        return np.inf, budget_used
    ph, Nk = ph[finite], Nk[finite]

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        return phi_hat_list[idx], budget_used

    if pool:
        w = Nk ** 2
        phi_pilot = float((w * ph).sum() / w.sum())
        sigma = 1.0 / (2.0 * np.sqrt(m_exploration * w.sum()))
    else:
        phi_pilot, sigma = float(ph[-1]), pilot_sd(Nk[-1], m_exploration)

    N_max = max(int(np.pi // (2 * phi_min)), 1)
    N = risk_optimal_depth(phi_pilot, sigma, remaining_budget, eps_target,
                           N_max=N_max, support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


# Binary search (Algorithm 5): divide-and-conquer on N with a confidence bound
def _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf, trace=None):
    """Exploration phase of Algorithm 5 — bisection on N with the Eq.(3.4) overshoot test.

    Returns (phi_hat, N, budget_used, phi_acc, N_acc, phi_0, N_0, L, U, history) or None if the first
    probe already exceeds the budget.

      (phi_acc, N_acc)  the deepest probe *not* classified as an overshoot
      (phi_0, N_0)      the OPENING probe at N_min
      L                 the bisection's lower bound (== N_acc; kept explicit for the pseudocode)
      U                 the bisection's upper bound (the shallowest probe it rejected, else N_max)
      history           [(N_i, phi_hat_i)] for EVERY probe, including the flagged ones

    `trace` additionally records each probe with the bisection's own accept/reject decision, which is
    exactly the overshoot declaration scored in the detector diagnostics.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return None

    phi_hat = simulate_errors(rng, phi, m_exploration, N)
    budget_used = m_exploration * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
    phi_acc, N_acc = phi_hat, N
    phi_0, N_0 = phi_hat, N          # the opening probe -- the pilot Eq. (3.8) uses
    history = [(N, phi_hat)]         # EVERY probe, flagged or not (qmetrology/posterior.py)
    if trace is not None:
        # the opening probe at N_min cannot alias by construction, so the bisection never tests it
        trace.probe(N, m_exploration, phi_hat, declared_overshoot=False, accepted=True)
        trace.opening_pilot_N, trace.opening_pilot_phi_hat = int(N), float(phi_hat)

    lb, ub = N_min, N_max
    N += (ub - N) // 2

    done = False
    while not done:
        if budget_used + m_exploration * N > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += m_exploration * N
        history.append((N, phi_hat))
        temp_N = N
        rejected = phi_hat < phi_1
        if trace is not None:
            trace.probe(temp_N, m_exploration, phi_hat, declared_overshoot=bool(rejected),
                        accepted=not rejected)
        if rejected:
            N -= (N - lb) // 2
            ub = temp_N
        else:
            phi_acc, N_acc = phi_hat, temp_N
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        if temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True

    if trace is not None:
        trace.accepted_pilot_N, trace.accepted_pilot_phi_hat = int(N_acc), float(phi_acc)
    return phi_hat, N, budget_used, phi_acc, N_acc, phi_0, N_0, lb, ub, history


def find_phi_fixed_budget_binary_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        safeguard=1, conf=0.95):
    out = _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    phi_hat, N, budget_used, _pa, _Na, _p0, _N0, _L, _U, _h = out

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used

    N = max(N - safeguard, 1)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_risk(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, conf=0.95, depth_cap="support"):
    """Algorithm 5 with the tuned decrement `s` replaced by the risk-optimal depth (safeguard.py).

    PILOT: the OPENING probe (phi_0, N_min), not the deepest non-flagged one. Measured over 468
    equal-cost operating points of the full sweep (analysis/binary_pilot_sweep.py): +0.30 +/- 0.06 pp
    on the mean, driven by the wide-prior scenarios (+1.7 to +2.9 pp) and never materially worse in
    the narrow ones (-0.38 pp at worst). The opening probe cannot be aliased -- N_min*phi <= pi/2 for
    every admissible phi -- whereas the deepest *accepted* probe is selected for having read high and
    carries a 0.87 sigma bias that Eq. (3.4) does not model. Same pilot as reverse engineering.

    depth_cap selects the upper end of the Eq. (3.8) search ("N_guess" in the pseudocode):
        "support"  N_max = floor(pi/(2 phi_min)) from the prior -- the bisection only buys the pilot
        "L"        the bisection's lower bound (the deepest probe not flagged)
        "min"      min of the two
    See analysis/binary_depth_sweep.py for the measurement behind the default.
    """
    out = _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    phi_hat, _N_bisect, budget_used, _phi_acc, _N_acc, phi_0, N_0, L, _U, _hist = out

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used

    N_cap = {"support": max(int(np.pi // (2 * phi_min)), 1),
             "L": max(int(L), 1),
             "min": min(max(int(L), 1), max(int(np.pi // (2 * phi_min)), 1))}[depth_cap]
    N = risk_optimal_depth(phi_0, pilot_sd(N_0, m_exploration), remaining_budget,
                           eps_target, N_max=N_cap, support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_deep(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, conf=0.5, trace=None):
    """**The reported binary search**: Algorithm 5 with the statistical safeguard fed by the DEEPEST
    PROBE the bisection did not classify as an overshoot.

    This is the variant the thesis pseudocode describes, and the one measured in
    results/fine_sweep.csv (`binary_deep`) and results/binary_final_stats.csv. It differs from
    `find_phi_fixed_budget_binary_search_risk` — which is kept for the record — in one line: the
    pilot handed to Eq. (3.8) is `(phi_acc, N_acc)`, not the opening probe `(phi_0, N_min)`. Because
    N_acc >= N_min, the pilot sd 1/(2 N_acc sqrt(m')) is smaller by exactly N_acc/N_min, which is the
    only thing the bisection buys the depth rule.

      N_guess = N_acc = L,  the deepest accepted depth -- the exploration's answer, before safeguard.
      N_star  = the Eq. (3.8) risk-optimal depth over the prior support, floored at N_min.

    The floor `N >= min(N_min, N_max)` keeps the exploitation from running shallower than the
    opening probe, which no admissible phi would justify.

    CAVEAT, stated in results/BS_METHOD_DECISION.md and results/consolidated/algorithm_code_audit.md:
    the deepest accepted estimate is selected *because it passed the overshoot test*, so treating it
    afterwards as an unbiased Gaussian pilot is an approximation the safeguard derivation does not
    cover. The diagnostics measure the resulting bias directly (detector miss/false-alarm rates and
    the N_guess/N_opt distribution) instead of assuming it away.
    """
    if trace is not None:
        trace.nominal_budget = float(budget)
    out = _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf,
                                 trace=trace)
    if out is None:
        if trace is not None:
            trace.set_exploration(0.0, status="refused_pilot_unaffordable")
        return np.inf, budget
    _phi_last, _N_bisect, budget_used, phi_acc, N_acc, _phi_0, _N_0, L, _U, _hist = out

    N_guess = max(int(N_acc), 1)
    if trace is not None:
        trace.set_exploration(budget_used, N_guess=N_guess, status="ok")

    remaining_budget = budget - budget_used
    if remaining_budget <= 0 or not np.isfinite(phi_acc):
        if trace is not None:
            trace.status = ("no_exploitation_budget_exhausted" if remaining_budget <= 0
                            else "no_exploitation_pilot_nonfinite")
        return phi_acc, budget_used

    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_sup = max(int(np.pi // (2 * phi_min)), 1)
    N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m_exploration), remaining_budget, eps_target,
                           N_max=N_sup, support=(phi_min, phi_max))
    N = max(N, min(N_min, N_sup))
    m = int(remaining_budget / N)
    if m < 1:
        if trace is not None:
            trace.status = "no_exploitation_shots"
        return phi_acc, budget_used
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    if trace is not None:
        trace.set_exploitation(N, m)
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_anneal_m(rng, phi, phi_max, phi_min, m_exploration, budget, safeguard=1, conf=0.95, max_b_steps_sub=3, delta=5):
    """Binary-search variant that anneals m_exploration up as N grows (m *= (N/N_old)^2 on each step right)."""
    N_min = max(np.pi//(2*phi_max), 1)
    N_max = max(np.pi//(2*phi_min), 1)
    N = N_min
    max_b_steps = np.ceil(np.log2(N_max - N_min)) - max_b_steps_sub
    if m_exploration * N > budget:
        return np.inf, budget
    phi_hat = simulate_errors(rng, phi, m_exploration, N)
    budget_used = m_exploration * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
    lb = N_min
    ub = N_max
    N += (ub - N) // 2
    done = False
    b_steps = 0
    while not done:
        if budget_used + m_exploration * N > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += m_exploration * N
        step_left = (N - lb) // 2
        step_right = (ub - N) // 2
        if ub - lb <= delta:
            break
        N_old = N
        if phi_hat < phi_1:
            if step_left == 0:
                done = True
                break
            ub = N
            N -= step_left
        else:
            if step_right == 0:
                done = True
                break
            lb = N
            N += step_right
            m_exploration = max(int(m_exploration * (N / N_old)**2), 20)
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        b_steps += 1
        if b_steps >= max_b_steps or budget_used >= budget:
            done = True
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used
    N = max(N - safeguard, 1)
    m = int(remaining_budget/N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


# Reverse engineering (Algorithm 6) and its "lite" variant (Algorithm 7)
def find_phi_fixed_budget_reverse_engineering(rng, phi, phi_max, phi_min, m_exploration,
                                              budget, safeguard=0.9):
    N_min = max(np.pi // (2 * phi_max), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        if budget_used + m_exploration * N_min > budget:
            return np.inf, budget_used  # cannot afford another pilot retry
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    N = int(np.pi // (2 * phi_hat) * safeguard)
    remaining_budget = budget - budget_used
    if remaining_budget <= 0 or N <= 1:
        return phi_hat, budget_used
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += N * m
    return phi_hat, budget_used


def find_phi_fixed_budget_reverse_engineering_risk(rng, phi, phi_max, phi_min, m_exploration,
                                                   budget, eps_target, trace=None):
    """Algorithm 6 with the tuned safety factor C_safe (Eq. 3.6) replaced by the risk-optimal depth.

    N is capped by the prior support, N_max = floor(pi/(2 phi_min)) — the constant-C form has no such
    cap, so a badly under-shooting pilot could send it past any admissible depth.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    if trace is not None:
        trace.nominal_budget = float(budget)
    if m_exploration * N_min > budget:
        if trace is not None:
            trace.set_exploration(0.0, status="refused_pilot_unaffordable")
        return np.inf, budget
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        if budget_used + m_exploration * N_min > budget:
            if trace is not None:
                # every attempt is charged: repeated pilots are part of B_exploration
                trace.set_exploration(budget_used, status="pilot_retries_exhausted_budget")
            return np.inf, budget_used  # cannot afford another pilot retry
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
        if trace is not None:
            # reverse engineering has no overshoot detector: declared_overshoot stays None
            trace.probe(N_min, m_exploration, phi_hat)
    if trace is not None:
        trace.opening_pilot_N, trace.opening_pilot_phi_hat = int(N_min), float(phi_hat)
        trace.accepted_pilot_N, trace.accepted_pilot_phi_hat = int(N_min), float(phi_hat)
        # the thesis definition: the raw depth the pilot implies, before the safeguard
        trace.set_exploration(budget_used, N_guess=max(int(np.pi // (2 * phi_hat)), 1), status="ok")
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        if trace is not None:
            trace.status = "no_exploitation_budget_exhausted"
        return phi_hat, budget_used
    N = risk_optimal_depth(phi_hat, pilot_sd(N_min, m_exploration), remaining_budget,
                           eps_target, N_max=max(int(N_max), 1), support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    if m < 1:
        if trace is not None:
            trace.status = "no_exploitation_shots"
        return phi_hat, budget_used
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += N * m
    if trace is not None:
        trace.set_exploitation(N, m)
    return phi_hat, budget_used


def find_phi_fixed_budget_reverse_engineering_share(rng, phi, phi_max, phi_min, budget, eps_target,
                                                    pilot_share=0.02, m_floor=20, trace=None):
    """Algorithm 6 + statistical safeguard, with the pilot sized as a *share* of the budget.

    m' = pilot_share * budget / N_min, so the exploration phase costs `pilot_share` of the budget by
    construction instead of a fixed shot count. The point is not peak performance -- per-budget tuned,
    this and the shot-count form are within noise of each other (+0.05 pp over 546 operating points)
    -- but that ONE value transfers across scenarios and budgets, which no single shot count does.

    Measured over all 23 sweep scenarios (analysis/re_share_sweep.py, 546 live operating points,
    de-biased at R=40,000, SE on a difference 0.35 pp), against the per-budget grid-tuned m':

        pilot_share = 0.02, never tuned   mean -0.06 pp,  worst -1.75 pp,  1.8% of points > 1 pp
        m_exploration = 45, never tuned   mean -4.22 pp,  worst -82.5 pp,  67%  of points > 2 pp

    Neither parameterisation is right in principle: A1 wants sigma <~ phi/kappa, i.e. a shot count
    that does NOT depend on the budget, so the ideal *share* decays as 1/B and a constant one leaks
    budget at the top end (-0.4 pp at pilot_share=0.05). The reason to prefer a share anyway is the
    shape of the failure. Over-spending wastes a bounded fraction; a fixed shot count eventually
    cannot be afforded at all -- at U(0.001,0.01), B=6394 the pilot m'=45 costs 45*157 > B, the trial
    is refused, and convergence drops from 82.5% to 0%.

    m_floor is what actually governs below B ~ 400*N_min, where pilot_share*budget/N_min rounds to a
    handful of shots. It coincides with the bottom of the tuned grid there, so nothing is lost, but
    the rule is honestly "a budget share with a shot floor" -- two numbers, not one.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    m_exploration = max(int(m_floor), int(pilot_share * budget / N_min))
    return find_phi_fixed_budget_reverse_engineering_risk(
        rng, phi, phi_max, phi_min, m_exploration=m_exploration, budget=budget,
        eps_target=eps_target, trace=trace)


def find_phi_reverse_engineering_lite(rng, phi, phi_max, phi_min, m_exploration=100,
                                      m_exploitation=10):
    """Algorithm 7: infer N from one estimate, then exploit with a small m (no safeguard).
    Variable-budget; reported only in Table 3.1."""
    N_min = max(np.pi // (2 * phi_max), 1)
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    N = int(np.pi // (2 * phi_hat))
    phi_hat = simulate_errors(rng, phi, m_exploitation, N)
    budget_used += N * m_exploitation
    return phi_hat, budget_used



# ---------------------------------------------------------------------------------------------
# Exact-posterior variants. Instead of turning ONE probe into a point estimate and invoking the
# asymptotic law of Eq. (3.4), these feed the raw counts of EVERY probe into the exact likelihood
# (qmetrology/posterior.py). Nothing is discarded -- in particular the probes an overshoot test would
# reject, which are the sharpest evidence the exploration produces: k ~ 0 at depth N says phi ~ pi/2N.
def _history_to_counts(history, m):
    """[(N, phi_hat)] -> [(N, k)]. Exact: k = m cos^2(N phi_hat) inverts the estimator."""
    return [(int(N), hits_from_estimate(p, N, m)) for N, p in history if np.isfinite(p)]


def find_phi_fixed_budget_binary_search_post(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, conf=0.95):
    """Algorithm 5 with the depth chosen from the exact posterior over the whole probe history."""
    out = _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    phi_hat, _Nb, budget_used, _pa, _Na, _p0, _N0, _L, _U, history = out
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used
    counts = _history_to_counts(history, m_exploration)
    if not counts:
        return np.inf, budget_used
    N = depth_from_posterior(counts, m_exploration, phi_min, phi_max, remaining_budget, eps_target,
                             N_max=max(int(np.pi // (2 * phi_min)), 1))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


def find_phi_fixed_budget_reverse_engineering_post(rng, phi, phi_max, phi_min, m_exploration,
                                                   budget, eps_target):
    """Algorithm 6 with the depth chosen from the exact posterior of its single pilot probe.

    Reverse engineering has only ONE probe, so no information is being recovered from discarded
    ones -- but the exact posterior still differs from the normal approximation where it matters:
    the pilot sits at N_min, so for small phi the readout probability cos^2(N_min phi) is near 1, the
    binomial is hard against its boundary, and the Gaussian of Eq. (3.4) is a poor fit there.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        if budget_used + m_exploration * N_min > budget:
            break
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    remaining_budget = budget - budget_used
    if remaining_budget <= 0 or not np.isfinite(phi_hat):
        return phi_hat, budget_used
    counts = _history_to_counts([(N_min, phi_hat)], m_exploration)
    N = depth_from_posterior(counts, m_exploration, phi_min, phi_max, remaining_budget, eps_target,
                             N_max=max(int(np.pi // (2 * phi_min)), 1))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


def find_phi_fixed_budget_linear_search_post(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, lookback_window=5, inc=1):
    """Algorithm 4 with the depth chosen from the exact posterior over the whole scan.

    The scan visits many depths, so it produces the same kind of evidence a bisection does -- and,
    unlike the pooled-estimator version (find_phi_fixed_budget_linear_search_risk), the posterior can
    use the probes past the aliasing point instead of discarding them.
    """
    out = _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget,
                                 lookback_window, inc)
    if out is None:
        return np.inf, budget
    phi_hat_list, N_list, budget_used, _overshot = out
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        return phi_hat_list[idx], budget_used
    counts = _history_to_counts(list(zip(N_list, phi_hat_list)), m_exploration)
    if not counts:
        return np.inf, budget_used
    N = depth_from_posterior(counts, m_exploration, phi_min, phi_max, remaining_budget, eps_target,
                             N_max=max(int(np.pi // (2 * phi_min)), 1))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


# Registry used by config/tables. The *_risk entries carry the statistically derived safeguard
# (qmetrology/safeguard.py) instead of a grid-tuned constant; they are reported alongside the
# originals rather than replacing them.
FIXED_BUDGET = {
    "oracle": find_phi_fixed_budget_oracle,
    "oracle_nopt": find_phi_fixed_budget_oracle_nopt,
    "oracle_alias": find_phi_fixed_budget_oracle_alias,
    "separable": find_phi_fixed_budget_separable,
    "ceiling": find_phi_fixed_budget_ceiling,
    "brute": find_phi_fixed_budget_brute_force,
    "linear": find_phi_fixed_budget_linear_search,
    "binary": find_phi_fixed_budget_binary_search,
    "reverse_eng": find_phi_fixed_budget_reverse_engineering,
    "linear_risk": find_phi_fixed_budget_linear_search_risk,
    "binary_risk": find_phi_fixed_budget_binary_search_risk,
    "binary_deep": find_phi_fixed_budget_binary_search_deep,
    "reverse_eng_risk": find_phi_fixed_budget_reverse_engineering_risk,
    "reverse_eng_share": find_phi_fixed_budget_reverse_engineering_share,
    "binary_post": find_phi_fixed_budget_binary_search_post,
    "reverse_eng_post": find_phi_fixed_budget_reverse_engineering_post,
    "linear_post": find_phi_fixed_budget_linear_search_post,
    "linear_lagged": find_phi_fixed_budget_linear_search_lagged,
}

# linear_risk is deliberately absent: it is a diagnostic, not a reported variant (see its docstring)
RISK_OF = {"binary": "binary_risk", "reverse_eng": "reverse_eng_risk"}
