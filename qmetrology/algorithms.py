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


def find_phi_fixed_budget_oracle_alias(rng, phi, phi_max, phi_min, budget):
    """Oracle without a safeguard: N = floor(pi/(2 phi)), the deepest non-aliasing circuit.

    Reverse engineering (Algorithm 6) with a perfect pilot and C_safe = 1. It falls short of the true
    oracle at tight eps because the estimator is boundary-clamped at N phi ~ pi/2, which shows that
    backing off the inferred depth is required even under perfect information.
    """
    N = alias_depth(phi, budget)
    m = int(budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, m * N


# Separable baseline (N = 1)
def find_phi_fixed_budget_separable(rng, phi, phi_max, phi_min, budget):
    """Separable protocol: a single N=1 circuit measured `budget` times."""
    N = 1
    m = int(budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, m * N


# Brute force (Algorithm 3): fix N = N_min, spend all budget on m
def find_phi_fixed_budget_brute_force(rng, phi, phi_max, phi_min, budget):
    N_min = max(np.pi // (2 * phi_max), 1)
    m = int(budget / N_min)
    phi_hat = simulate_errors(rng, phi, m, N_min)
    return phi_hat, m * N_min


# Linear search (Algorithm 4): scan N upward, detect overshoot via lookback
def _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window, inc):
    """Exploration scan of Algorithm 4 — probe N = N_min, N_min+inc, ... until the running mean of
    the estimates falls `lookback_window` times in a row (the overshoot verdict).

    Returns (phi_hats, Ns, budget_used, overshot) or None if the first probe already exceeds the
    budget. Consumes the RNG in exactly the same order as the published algorithm, so the constant-`s`
    and statistical-safeguard variants stay paired trial-by-trial.
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

        r_mean_new = np.mean(phi_hat_list)
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


def find_phi_fixed_budget_linear_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        lookback_window=5, safeguard=1, inc=1):
    out = _linear_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget,
                                 lookback_window, inc)
    if out is None:
        return np.inf, budget
    phi_hat_list, N_list, budget_used, overshot = out
    N = N_list[-1] - lookback_window * inc if overshot else N_list[-1]

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hat_list) - lookback_window)
        return phi_hat_list[idx], budget_used

    N = max(1, N - safeguard)
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += m * N
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
def _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf):
    """Exploration phase of Algorithm 5 — bisection on N with the Eq.(3.4) overshoot test.

    Returns (phi_hat, N, budget_used, phi_acc, N_acc, phi_0, N_0, L) or None if the first probe
    already exceeds the budget.

      (phi_acc, N_acc)  the deepest probe *not* classified as an overshoot
      (phi_0, N_0)      the OPENING probe at N_min -- the pilot Eq. (3.8) uses (see below)
      L                 the bisection's lower bound (== N_acc; kept explicit for the pseudocode)
      history           [(N_i, phi_hat_i)] for EVERY probe, including the flagged ones
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
        if phi_hat < phi_1:
            N -= (N - lb) // 2
            ub = temp_N
        else:
            phi_acc, N_acc = phi_hat, temp_N
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        if temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True

    return phi_hat, N, budget_used, phi_acc, N_acc, phi_0, N_0, lb, history


def find_phi_fixed_budget_binary_search(rng, phi, phi_max, phi_min, m_exploration, budget,
                                        safeguard=1, conf=0.95):
    out = _binary_search_explore(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    phi_hat, N, budget_used, _pa, _Na, _p0, _N0, _L, _h = out

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
    phi_hat, _N_bisect, budget_used, _phi_acc, _N_acc, phi_0, N_0, L, _hist = out

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
                                                   budget, eps_target):
    """Algorithm 6 with the tuned safety factor C_safe (Eq. 3.6) replaced by the risk-optimal depth.

    N is capped by the prior support, N_max = floor(pi/(2 phi_min)) — the constant-C form has no such
    cap, so a badly under-shooting pilot could send it past any admissible depth.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    if m_exploration * N_min > budget:
        return np.inf, budget
    phi_hat = 0
    budget_used = 0
    while phi_hat == 0:
        if budget_used + m_exploration * N_min > budget:
            return np.inf, budget_used  # cannot afford another pilot retry
        phi_hat = simulate_errors(rng, phi, m_exploration, N_min)
        budget_used += m_exploration * N_min
    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        return phi_hat, budget_used
    N = risk_optimal_depth(phi_hat, pilot_sd(N_min, m_exploration), remaining_budget,
                           eps_target, N_max=max(int(N_max), 1), support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    budget_used += N * m
    return phi_hat, budget_used


def find_phi_fixed_budget_reverse_engineering_share(rng, phi, phi_max, phi_min, budget, eps_target,
                                                    pilot_share=0.02, m_floor=20):
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
        eps_target=eps_target)


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
    phi_hat, _Nb, budget_used, _pa, _Na, _p0, _N0, _L, history = out
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
    "oracle_alias": find_phi_fixed_budget_oracle_alias,
    "separable": find_phi_fixed_budget_separable,
    "brute": find_phi_fixed_budget_brute_force,
    "linear": find_phi_fixed_budget_linear_search,
    "binary": find_phi_fixed_budget_binary_search,
    "reverse_eng": find_phi_fixed_budget_reverse_engineering,
    "linear_risk": find_phi_fixed_budget_linear_search_risk,
    "binary_risk": find_phi_fixed_budget_binary_search_risk,
    "reverse_eng_risk": find_phi_fixed_budget_reverse_engineering_risk,
    "reverse_eng_share": find_phi_fixed_budget_reverse_engineering_share,
    "binary_post": find_phi_fixed_budget_binary_search_post,
    "reverse_eng_post": find_phi_fixed_budget_reverse_engineering_post,
    "linear_post": find_phi_fixed_budget_linear_search_post,
}

# linear_risk is deliberately absent: it is a diagnostic, not a reported variant (see its docstring)
RISK_OF = {"binary": "binary_risk", "reverse_eng": "reverse_eng_risk"}
