"""Two open questions about the exploration phase.

(1) LINEAR SEARCH + STATISTICAL SAFEGUARD.  Linear search is the one algorithm still carrying a
    grid-tuned safeguard (the decrement `s`). Its scan produces a whole *history* of estimates
    (phi_hat_i at depth N_i, m' shots each), not a single pilot, so the natural version of the rule
    pools them by inverse variance: with Var_i = 1/(4 N_i^2 m') the weights are w_i ~ N_i^2 and

        phi_pool = sum(N_i^2 phi_hat_i) / sum(N_i^2),     sigma_pool = 1/(2 sqrt(m' sum N_i^2)).

    Both the pooled and the single-deepest-probe pilot are measured against the tuned `s`.

(2) REVERSE ENGINEERING, PILOT AS A BUDGET SHARE.  Instead of a fixed shot count m', spend a fixed
    fraction rho of the budget on the pilot: m' = rho*B/N_min. Two questions, and they have
    different answers, so both are reported:
      (a) per-budget tuned  -- is the share parameterisation as good as the shot count?
      (b) ONE value for all budgets -- does rho transfer across budgets better than m' does?
    (b) is the only reason to prefer a share: it is a claim about robustness, not about the peak.

    python analysis/exploration_study.py [--quick]
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_reverse_engineering_risk as RE_R,
)

QUICK = "--quick" in sys.argv
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
N_BUDGETS = 3 if QUICK else 7


# --------------------------------------------------------------------------------------------
# (1) linear search with the statistical safeguard
# --------------------------------------------------------------------------------------------
def _linear_explore(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window, inc):
    """The scan of Algorithm 4, verbatim, but returning the probe history instead of one depth.

    Consumes the RNG in exactly the same order as find_phi_fixed_budget_linear_search, so the two
    are paired trial-by-trial. Returns (phi_hats, Ns, budget_used, overshot) or None.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)
    if m_exploration * N > budget:
        return None

    phi_hat_list, N_list = [], []
    counter, budget_used, r_mean, done = 0, 0, 0, False
    while not done:
        if budget_used + N * m_exploration > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_list.append(N)

        r_mean_new = np.mean(phi_hat_list)
        counter = counter + 1 if (len(phi_hat_list) >= 2 and r_mean_new < r_mean) else 0
        r_mean = r_mean_new

        if counter >= lookback_window:
            done = True
        elif N >= N_max or budget_used >= budget:
            done = True
        else:
            N += inc
    return phi_hat_list, N_list, budget_used, counter >= lookback_window


def _linear_risk(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target,
                 lookback_window=5, inc=1, pool=True):
    out = _linear_explore(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window, inc)
    if out is None:
        return np.inf, budget
    phi_hats, Ns, budget_used, overshot = out

    # drop the probes that triggered the overshoot verdict -- they are the aliased ones. This is the
    # same window the published algorithm backs off by (N -= lookback_window*inc).
    keep = max(1, len(phi_hats) - lookback_window) if overshot else len(phi_hats)
    ph, Nk = np.asarray(phi_hats[:keep], float), np.asarray(Ns[:keep], float)
    ok = np.isfinite(ph)
    if not ok.any():
        return np.inf, budget_used
    ph, Nk = ph[ok], Nk[ok]

    remaining_budget = budget - budget_used
    if remaining_budget <= 0:
        idx = max(0, len(phi_hats) - lookback_window)
        return phi_hats[idx], budget_used

    if pool:
        w = Nk ** 2
        phi_pilot = float((w * ph).sum() / w.sum())
        sigma = 1.0 / (2.0 * np.sqrt(m_exploration * (Nk ** 2).sum()))
    else:
        phi_pilot, sigma = float(ph[-1]), pilot_sd(Nk[-1], m_exploration)

    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), N_min)
    N = risk_optimal_depth(phi_pilot, sigma, remaining_budget, eps_target,
                           N_min=N_min, N_max=N_max, support=(phi_min, phi_max))
    m = int(remaining_budget / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


def linear_risk_pooled(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target,
                       lookback_window=5, inc=1):
    """Pilot = inverse-variance pooled over every non-aliased probe of the scan."""
    return _linear_risk(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target,
                        lookback_window, inc, pool=True)


def linear_risk_last(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target,
                     lookback_window=5, inc=1):
    """Pilot = the deepest non-aliased probe only (the analogue of the binary-search pilot)."""
    return _linear_risk(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target,
                        lookback_window, inc, pool=False)


# --------------------------------------------------------------------------------------------
# (2) reverse engineering with the pilot sized as a share of the budget
# --------------------------------------------------------------------------------------------
def re_risk_share(rng, phi, phi_max, phi_min, budget, eps_target, pilot_share=0.02, m_floor=20):
    """Algorithm 6 + statistical safeguard, with m' = pilot_share * budget / N_min."""
    N_min = max(np.pi // (2 * phi_max), 1)
    m_exploration = max(int(m_floor), int(pilot_share * budget / N_min))
    return RE_R(rng, phi, phi_max, phi_min, m_exploration=m_exploration, budget=budget,
                eps_target=eps_target)


# --------------------------------------------------------------------------------------------
def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


SCENARIOS = [(0.01, 0.1, 1e-3), (0.01, 0.1, 1e-4), (0.001, 0.01, 1e-4)]
if QUICK:
    SCENARIOS = [(0.01, 0.1, 1e-3)]


def budgets_for(pmax, eps):
    c = brute90(pmax, eps)
    return np.unique(np.geomspace(c / 12, c * 6, N_BUDGETS).astype(np.int64))


def tuned(fn, grid, b, pmin, pmax, eps):
    """Grid-tune on seed 42, evaluate the winner on seed 2024. Returns (test_rate, cfg)."""
    res = E.grid_full(fn, {**grid, "budget": [int(b)]}, R_TUNE, pmin, pmax, eps, SEED_TUNE)
    _r, _bud, cfg = max(res, key=lambda x: x[0])
    return E.success_rate(fn, cfg, R_TEST, pmin, pmax, eps, SEED_TEST), cfg


RHO = [0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2, 0.3]


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for pmin, pmax, eps in SCENARIOS:
        name = f"U({pmin:g},{pmax:g}), eps={eps:.0e}"
        c = brute90(pmax, eps)
        m_hi = int(np.clip(c, 200, 300_000))
        m_lin = np.unique(np.geomspace(3, m_hi, 16).astype(int))
        m_re = np.unique(np.geomspace(20, m_hi, 20).astype(int))
        print(f"\n=== {name}   N_min={n_min(pmax)} ===", flush=True)

        for b in budgets_for(pmax, eps):
            rec = {"setting": name, "phi_min": pmin, "phi_max": pmax, "eps": eps, "budget": int(b)}
            rec["brute"] = 100 * E.success_rate(BF, {"budget": int(b)}, R_TEST, pmin, pmax, eps, SEED_TEST)

            # --- linear: tuned s  vs  statistical safeguard (two pilots)
            r, cfg = tuned(LIN, {"m_exploration": m_lin, "lookback_window": [1, 2, 3, 5],
                                 "safeguard": [0, 1, 2, 5], "inc": [1, 2, 5]}, b, pmin, pmax, eps)
            rec["linear"], rec["linear_cfg"] = 100 * r, cfg
            g_lr = {"m_exploration": m_lin, "lookback_window": [1, 2, 3, 5], "inc": [1, 2, 5],
                    "eps_target": [eps]}
            r, cfg = tuned(linear_risk_pooled, g_lr, b, pmin, pmax, eps)
            rec["linear_risk_pool"], rec["linear_risk_pool_cfg"] = 100 * r, cfg
            r, cfg = tuned(linear_risk_last, g_lr, b, pmin, pmax, eps)
            rec["linear_risk_last"], rec["linear_risk_last_cfg"] = 100 * r, cfg

            # --- reverse engineering: fixed m'  vs  budget share
            r, cfg = tuned(RE_R, {"m_exploration": m_re, "eps_target": [eps]}, b, pmin, pmax, eps)
            rec["re_fixed"], rec["re_fixed_cfg"] = 100 * r, cfg
            r, cfg = tuned(re_risk_share, {"pilot_share": RHO, "eps_target": [eps]}, b, pmin, pmax, eps)
            rec["re_share"], rec["re_share_cfg"] = 100 * r, cfg

            # --- transfer: ONE hyperparameter for every budget in this scenario. Evaluated on the
            #     test seed for each candidate; the best single value is picked afterwards, so this
            #     measures the best achievable *fixed* setting, not a lucky one.
            rec["re_share_curve"] = {str(rho): 100 * E.success_rate(
                re_risk_share, {"budget": int(b), "eps_target": eps, "pilot_share": rho},
                R_TEST, pmin, pmax, eps, SEED_TEST) for rho in RHO}
            rec["re_fixed_curve"] = {str(int(mm)): 100 * E.success_rate(
                RE_R, {"budget": int(b), "eps_target": eps, "m_exploration": int(mm)},
                R_TEST, pmin, pmax, eps, SEED_TEST) for mm in m_re}

            rows.append(rec)
            print(f"  B={b:>13,}  brute {rec['brute']:5.1f} | lin {rec['linear']:5.1f} "
                  f"pool {rec['linear_risk_pool']:5.1f} last {rec['linear_risk_last']:5.1f} "
                  f"| RE m' {rec['re_fixed']:5.1f} rho {rec['re_share']:5.1f}", flush=True)

    with open("results/exploration_study.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    keys = ["setting", "budget", "brute", "linear", "linear_risk_pool", "linear_risk_last",
            "re_fixed", "re_share"]
    with open("results/exploration_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], str) else round(float(r[k]), 2) for k in keys])
    print("\nwrote results/exploration_study.{csv,json}")


if __name__ == "__main__":
    main()
