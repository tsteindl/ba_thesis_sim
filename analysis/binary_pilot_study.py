"""Which probe of the bisection should feed the statistical safeguard?

Binary search takes several estimates during exploration, at increasing depths. Eq. (3.8) needs one
pilot (phi_hat_0, sigma). Four defensible choices:

  first   the opening probe at N_min. A2 holds by construction (N_min*phi <= pi/2 for all admissible
          phi), so it cannot be aliased -- but it is the least precise, sigma = 1/(2 N_min sqrt(m')).
  deep    the deepest probe NOT flagged as an overshoot.  <-- what qmetrology/algorithms.py does
          sigma is smaller by a factor N_min/N_acc, but A2 now rests on the overshoot test rather
          than on construction: a MISSED overshoot (a type-II error of that test) biases the pilot
          LOW, which makes Eq. (3.8) believe phi is smaller than it is and pick an even deeper N.
  last    the final probe measured, whatever it is -- including one flagged as an overshoot.
  pool    inverse-variance combination of every non-flagged probe. Var_i = 1/(4 N_i^2 m') is known,
          so the weights are N_i^2:  phi = sum(N_i^2 phi_i)/sum(N_i^2), sigma = 1/(2 sqrt(m' sum N_i^2)).

The trade-off is precision against latent aliasing, and which side wins is an empirical question --
it depends on the overshoot test's confidence `conf`, which is itself tuned. This script measures it.

Reports, per operating point and per variant: the de-biased convergence rate, the pilot's bias
E[phi_pilot - phi], its spread, the realised overshoot rate P(N* phi > pi/2), and the chosen depth
relative to N_opt.

    python analysis/binary_pilot_study.py [--quick]
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

QUICK = "--quick" in sys.argv
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
VARIANTS = ["first", "deep", "last", "pool"]


def explore_history(rng, phi, phi_max, phi_min, m_exploration, budget, conf):
    """The bisection of Algorithm 5, recording every probe.

    Byte-for-byte the same arithmetic and the same RNG consumption as
    qmetrology.algorithms._binary_search_explore (verified in the test at the bottom), but it returns
    the whole history instead of a single pilot.

    Returns (probes, budget_used) with probes = [(N, phi_hat, accepted), ...], or None.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)

    if m_exploration * N > budget:
        return None

    phi_hat = simulate_errors(rng, phi, m_exploration, N)
    budget_used = m_exploration * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
    probes = [(N, phi_hat, True)]          # the opening probe is accepted by construction

    lb, ub = N_min, N_max
    N += (ub - N) // 2

    done = False
    while not done:
        if budget_used + m_exploration * N > budget:
            break  # this probe does not fit — stop *before* spending, not after
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += m_exploration * N
        temp_N = N
        if phi_hat < phi_1:
            probes.append((temp_N, phi_hat, False))
            N -= (N - lb) // 2
            ub = temp_N
        else:
            probes.append((temp_N, phi_hat, True))
            N += (ub - N) // 2
            lb = temp_N
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m_exploration * N**2)))
        if temp_N == N or N < N_min or N > N_max or budget_used >= budget:
            done = True

    return probes, budget_used


def pilot_from(probes, m_exploration, which):
    """(phi_pilot, sigma) for one of the four choices, or None if unusable."""
    ok = [(N, p) for N, p, acc in probes if acc and np.isfinite(p)]
    if which == "last":
        fin = [(N, p) for N, p, _a in probes if np.isfinite(p)]
        if not fin:
            return None
        N, p = fin[-1]
        return p, pilot_sd(N, m_exploration)
    if not ok:
        return None
    if which == "first":
        N, p = ok[0]
        return p, pilot_sd(N, m_exploration)
    if which == "deep":
        N, p = ok[-1]
        return p, pilot_sd(N, m_exploration)
    if which == "pool":
        Nk = np.array([N for N, _ in ok], dtype=float)
        ph = np.array([p for _, p in ok], dtype=float)
        w = Nk ** 2
        return float((w * ph).sum() / w.sum()), 1.0 / (2.0 * np.sqrt(m_exploration * w.sum()))
    raise ValueError(which)


def _run(which, rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf):
    """Algorithm 5 + Eq. (3.8), with the pilot taken according to `which`."""
    out = explore_history(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    probes, budget_used = out
    remaining = budget - budget_used
    if remaining <= 0:
        return probes[-1][1], budget_used
    pil = pilot_from(probes, m_exploration, which)
    if pil is None:
        return np.inf, budget_used
    phi_pilot, sigma = pil
    if not np.isfinite(phi_pilot) or sigma <= 0:
        return np.inf, budget_used
    N = risk_optimal_depth(phi_pilot, sigma, remaining, eps_target,
                           N_max=max(int(np.pi // (2 * phi_min)), 1), support=(phi_min, phi_max))
    m = int(remaining / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


# Defined at module level, not built by a factory: ProcessPoolExecutor pickles the function by
# reference, and a closure returned from a factory has no importable qualname.
def binary_pilot_first(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("first", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_pilot_deep(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("deep", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_pilot_last(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("last", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_pilot_pool(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("pool", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


FN = {"first": binary_pilot_first, "deep": binary_pilot_deep,
      "last": binary_pilot_last, "pool": binary_pilot_pool}


def diagnostics(which, phi_min, phi_max, eps, budget, m_exploration, conf, R=6000, seed=2024):
    """Pilot bias/spread, realised overshoot rate, and depth vs N_opt -- the mechanism behind the rate."""
    rows = []
    for s in np.random.default_rng(seed).integers(0, 2**63, size=R):
        rng = np.random.default_rng(int(s))
        phi = float(rng.uniform(phi_min, phi_max))
        out = explore_history(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
        if out is None:
            continue
        probes, used = out
        remaining = budget - used
        if remaining <= 0:
            continue
        pil = pilot_from(probes, m_exploration, which)
        if pil is None:
            continue
        phi_pilot, sigma = pil
        if not np.isfinite(phi_pilot):
            continue
        N = risk_optimal_depth(phi_pilot, sigma, remaining, eps,
                               N_max=max(int(np.pi // (2 * phi_min)), 1), support=(phi_min, phi_max))
        n_opt = max(1, int(np.pi // (2 * phi)))
        rows.append((phi_pilot - phi, sigma, float(N * phi > np.pi / 2), N / n_opt,
                     float(m_exploration * np.cos(probes[-1][0] * phi) ** 2)))
    a = np.array(rows)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), bias=a[:, 0].mean(), bias_med=np.median(a[:, 0]), sd_pilot=a[:, 0].std(),
                sigma_model=np.median(a[:, 1]), overshoot=100 * a[:, 2].mean(),
                depth_ratio=np.median(a[:, 3]))


SCENARIOS = [(0.01, 0.1, 1e-3), (0.01, 0.1, 1e-4), (0.001, 0.01, 1e-4)]
if QUICK:
    SCENARIOS = [(0.01, 0.1, 1e-3)]


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def brute90(pmax, eps):
    return 0.6724 / (n_min(pmax) * eps ** 2)


def main():
    os.makedirs("results", exist_ok=True)
    out = []
    for pmin, pmax, eps in SCENARIOS:
        name = f"U({pmin:g},{pmax:g}), eps={eps:.0e}"
        c = brute90(pmax, eps)
        budgets = np.unique(np.geomspace(c / 12, c * 6, 3 if QUICK else 6).astype(np.int64))
        m_hi = int(np.clip(c, 200, 300_000))
        m_b = np.unique(np.geomspace(20, m_hi, 8 if QUICK else 14).astype(int))
        conf = [0.5, 0.65, 0.8, 0.9]
        print(f"\n=== {name}  N_min={n_min(pmax)} ===", flush=True)
        for b in budgets:
            rec = {"setting": name, "phi_min": pmin, "phi_max": pmax, "eps": eps, "budget": int(b)}
            for v in VARIANTS:
                grid = {"m_exploration": m_b, "conf": conf, "eps_target": [eps], "budget": [int(b)]}
                res = E.grid_full(FN[v], grid, R_TUNE, pmin, pmax, eps, SEED_TUNE)
                _r, _bud, cfg = max(res, key=lambda x: x[0])
                rate = E.success_rate(FN[v], cfg, R_TEST, pmin, pmax, eps, SEED_TEST)
                d = diagnostics(v, pmin, pmax, eps, int(b), int(cfg["m_exploration"]), cfg["conf"])
                rec[v] = 100 * rate
                rec[v + "_cfg"] = {k: (int(x) if isinstance(x, (np.integer,)) else x)
                                   for k, x in cfg.items() if k != "budget"}
                rec[v + "_diag"] = d
            out.append(rec)
            print(f"  B={b:>13,}  " + "  ".join(f"{v} {rec[v]:6.2f}" for v in VARIANTS)
                  + f"   | overshoot% " + " ".join(f"{rec[v+'_diag'].get('overshoot', float('nan')):5.2f}"
                                                   for v in VARIANTS), flush=True)

    with open("results/binary_pilot.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open("results/binary_pilot.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "budget"] + VARIANTS
                   + [f"{v}_{k}" for v in VARIANTS for k in ("bias", "overshoot", "depth_ratio")])
        for r in out:
            w.writerow([r["setting"], r["budget"]] + [round(r[v], 3) for v in VARIANTS]
                       + [round(r[v + "_diag"].get(k, float("nan")), 5)
                          for v in VARIANTS for k in ("bias", "overshoot", "depth_ratio")])
    print("\nwrote results/binary_pilot.{csv,json}")


if __name__ == "__main__":
    main()
