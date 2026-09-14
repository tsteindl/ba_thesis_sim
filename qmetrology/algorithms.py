"""The phase-search algorithms (Chapter 3).

Uniform signature: find_phi_*(rng, phi, phi_max, phi_min, **params) -> (phi_hat, budget_used).
Every algorithm here is fixed-budget: it caps the total cost N*m at `budget`, and the budget needed
for a target convergence is found by sweeping the budget (analysis/run.py). A single exploitation
count would over-spend on easy phases, so the variable-budget "run until confident" formulations are
not used.

qmetrology.manifest names the four reported by the thesis: brute_force, linear_search,
binary_search_deep and reverse_engineering_risk. The rest are the reference protocol (separable),
the audit arm the deepest-vs-opening-probe comparison needs (binary_search_risk), the tuned-constant
predecessors the broad-prior study reports next to them (binary_search, reverse_engineering), the
lagged-detector variant of the linear-search bake-off, and the three exact-posterior arms
.

The omniscient ceilings are analytic and live in qmetrology/oracle.py -- they are rates, not
runnable protocols.
"""
import numpy as np
from scipy.stats import norm

from .posterior import depth_from_posterior, hits_from_estimate
from .safeguard import pilot_sd, risk_optimal_depth
from .sim import simulate_errors


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
    estimates falls `lookback_window` times in a row (the overshoot verdict). The increment is
    clamped, N <- min(N + inc, N_max), so an inc > 1 cannot step the scan past the admissible
    interval and probe a depth the prior support already rules out.

    `mean_window` is the width of that mean: 0 (the default) averages every estimate collected so
    far, which is the cumulative mean of the published algorithm; w > 0 averages only the last w.
    The cumulative rule is the w -> infinity member of the same family, so the parameter is a strict
    generalisation and w = 0 reproduces the published scan exactly, RNG draw for RNG draw
    (tests/test_consolidated.py::test_linear_mean_window_default_is_cumulative).

    Why the width matters: for a full window the two consecutive means differ only in the element
    that enters and the one that leaves, so the test "the window mean fell" is identically
    phi_hat_k < phi_hat_{k-w}. A finite window therefore compares each probe with one a fixed depth
    behind it, while the cumulative mean compares against a reference whose effective lag keeps
    growing as the scan approaches the aliasing boundary.

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
            N = min(N + inc, N_max)

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
    at (see qmetrology/trace.py for the diagnostic definitions).

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
                   (analysis/linear_detector_study.py) is identically the
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
            N = min(N + inc, N_max)

    return phi_hat_list, N_list, budget_used, counter >= lookback_window


def find_phi_fixed_budget_linear_search_lagged(rng, phi, phi_max, phi_min, m_exploration, budget,
                                               lookback_window=4, safeguard=1, inc=1, lag=3,
                                               conf=0.5, trace=None):
    """NOT A REPORTED ALGORITHM (yet). Algorithm 4 with the lagged Eq. (3.6) stopping rule.

    Everything except the overshoot verdict is Algorithm 4: the same upward scan, the same
    backtracking by `lookback_window * inc`, the same safeguard N* = max(1, N_guess - s).

    It exists so that the alternative measured in
    the detector study can be swept end to end without editing the published
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

    PILOT: the OPENING probe (phi_0, N_min), not the deepest non-flagged one. It wins on the mean
    over the equal-cost operating points of the full sweep, driven by the wide-prior scenarios and
    never materially worse in the narrow ones. The opening probe cannot be aliased -- N_min*phi <=
    pi/2 for every admissible phi -- whereas the deepest *accepted* probe is selected for having
    read high and carries a bias that Eq. (3.4) does not model. Same pilot as reverse engineering.

    depth_cap selects the upper end of the Eq. (3.8) search ("N_guess" in the pseudocode):
        "support"  N_max = floor(pi/(2 phi_min)) from the prior -- the bisection only buys the pilot
        "L"        the bisection's lower bound (the deepest probe not flagged)
        "min"      min of the two
    The default is the one the sweep selected.
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
    # the "L"/"min" caps can land BELOW the prior's N_min (the bisection may settle at its own lower
    # bound), so the admissible set is anchored at whichever of the two is shallower -- never at 1.
    N_floor = min(max(int(np.pi // (2 * phi_max)), 1), N_cap)
    N = risk_optimal_depth(phi_0, pilot_sd(N_0, m_exploration), remaining_budget,
                           eps_target, N_min=N_floor, N_max=N_cap, support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
    return phi_hat, budget_used


def find_phi_fixed_budget_binary_search_deep(rng, phi, phi_max, phi_min, m_exploration, budget,
                                             eps_target, conf=0.5, trace=None):
    """**The reported binary search**: Algorithm 5 with the statistical safeguard fed by the DEEPEST
    PROBE the bisection did not classify as an overshoot.

    This is the variant the thesis pseudocode describes and the one the pipeline reports as
    `binary_deep`. It differs from
    `find_phi_fixed_budget_binary_search_risk` — which is kept for the record — in one line: the
    pilot handed to Eq. (3.8) is `(phi_acc, N_acc)`, not the opening probe `(phi_0, N_min)`. Because
    N_acc >= N_min, the pilot sd 1/(2 N_acc sqrt(m')) is smaller by exactly N_acc/N_min, which is the
    only thing the bisection buys the depth rule.

      N_guess = N_acc = L,  the deepest accepted depth -- the exploration's answer, before safeguard.
      N_star  = the Eq. (3.8) risk-optimal depth, the exact argmax over N_min..N_max.

    The safeguard enumerates the whole admissible interval, so N_star >= N_min holds by construction
    and no post-hoc floor is applied: the exploitation never runs shallower than the opening probe,
    a depth no admissible phi would justify.

    CAVEAT, also recorded in results/algorithm_code_audit.md:
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
    # no post-hoc floor: the search itself starts at N_min, so it cannot return anything shallower
    N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m_exploration), remaining_budget, eps_target,
                           N_min=min(N_min, N_sup), N_max=N_sup, support=(phi_min, phi_max))
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


def find_phi_fixed_budget_reverse_engineering(rng, phi, phi_max, phi_min, m_exploration,
                                              budget, safeguard=0.9):
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
    budget_used = m_exploration * N_min
    # phi_hat_0 == 0 (all m' shots read "0") inverts to an unbounded depth: report it at N_max.
    # Nonzero pilots keep the constant-C form's deliberate lack of an upper bound.
    N = int(N_max if phi_hat <= 0 else np.pi // (2 * phi_hat) * safeguard)
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

    N is confined to the prior support, N_min = floor(pi/(2 phi_max)) .. N_max = floor(pi/(2 phi_min))
    — the constant-C form has neither bound, so a badly under-shooting pilot could send it past any
    admissible depth, and an over-shooting one below the depth its own pilot was taken at.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    if trace is not None:
        trace.nominal_budget = float(budget)
    if m_exploration * N_min > budget:
        if trace is not None:
            trace.set_exploration(0.0, status="refused_pilot_unaffordable")
        return np.inf, budget
    phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
    budget_used = m_exploration * N_min
    if trace is not None:
        # reverse engineering has no overshoot detector: declared_overshoot stays None
        trace.probe(N_min, m_exploration, phi_hat)
        trace.opening_pilot_N, trace.opening_pilot_phi_hat = int(N_min), float(phi_hat)
        trace.accepted_pilot_N, trace.accepted_pilot_phi_hat = int(N_min), float(phi_hat)
        # the thesis definition: the raw depth the pilot implies, before the safeguard. A pilot of
        # exactly zero implies an unbounded depth, which is reported at the deepest admissible N.
        N_guess = int(N_max) if phi_hat <= 0 else max(int(np.pi // (2 * phi_hat)), 1)
        trace.set_exploration(budget_used, N_guess=N_guess, status="ok")
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        if trace is not None:
            trace.status = "no_exploitation_budget_exhausted"
        return phi_hat, budget_used
    N = risk_optimal_depth(phi_hat, pilot_sd(N_min, m_exploration), remaining_budget,
                           eps_target, N_min=max(int(N_min), 1),
                           N_max=max(int(N_max), int(N_min), 1), support=(phi_min, phi_max))
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
    # one pilot batch, as in Algorithm 7: the posterior reads the hit count directly, so a pilot of
    # exactly zero needs no retry.
    phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
    budget_used = m_exploration * N_min
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
