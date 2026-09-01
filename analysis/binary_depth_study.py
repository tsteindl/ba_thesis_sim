"""Should binary search just exploit at L, the bisection's lower bound?

After the bisection, `lb` (call it L) is the deepest depth that was probed and *not* flagged as an
overshoot. It is therefore an empirically verified-safe depth, which suggests an appealingly simple
rule: skip Eq. (3.8) entirely and exploit at N = L.

Note L is the same quantity as the `deep` pilot's depth: the loop sets `lb = temp_N` and
`phi_acc, N_acc = phi_hat, temp_N` in the same branch, and N never drops below lb, so L is the
largest accepted depth. The question here is not which probe to trust but whether the bisection's own
bound should *be* the answer.

Four depth rules, all sharing the identical exploration (same seeds, same probes):

  risk        N* from Eq. (3.8), searched over the full prior support   <-- what is shipped
  L           N = L. No safeguard at all: trust the overshoot test.
  risk_capL   min(N*, L). The safeguard may only reduce the bisection's bound.
  risk_maxL   max(N*, L). Take the deeper of the two.

L is NOT guaranteed safe: the overshoot test has a type-I error, so a probe at N > N_opt can be
accepted by chance. The diagnostics report P(L > N_opt) alongside the convergence rates.

    python analysis/binary_depth_study.py [--quick]
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors
from binary_pilot_study import explore_history, pilot_from

QUICK = "--quick" in sys.argv
R_TUNE = 400 if QUICK else 2000
R_TEST = 4000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024
VARIANTS = ["risk", "L", "risk_capL", "risk_maxL"]


def _L(probes):
    """The bisection's lower bound: the deepest probe not flagged as an overshoot."""
    acc = [N for N, p, a in probes if a and np.isfinite(p)]
    return max(acc) if acc else None


def _depth(which, probes, m_exploration, remaining, eps, phi_min, phi_max):
    L = _L(probes)
    if L is None:
        return None
    if which == "L":
        return int(L)
    pil = pilot_from(probes, m_exploration, "deep")
    if pil is None or not np.isfinite(pil[0]) or pil[1] <= 0:
        return None
    N_star = risk_optimal_depth(pil[0], pil[1], remaining, eps,
                                N_min=max(int(np.pi // (2 * phi_max)), 1),
                                N_max=max(int(np.pi // (2 * phi_min)), 1),
                                support=(phi_min, phi_max))
    if which == "risk":
        return int(N_star)
    if which == "risk_capL":
        return int(max(min(N_star, L), 1))
    if which == "risk_maxL":
        return int(max(N_star, L))
    raise ValueError(which)


def _run(which, rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf):
    out = explore_history(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
    if out is None:
        return np.inf, budget
    probes, budget_used = out
    remaining = budget - budget_used
    if remaining <= 0:
        return probes[-1][1], budget_used
    N = _depth(which, probes, m_exploration, remaining, eps_target, phi_min, phi_max)
    if N is None:
        return np.inf, budget_used
    m = int(remaining / N)
    phi_hat = simulate_errors(rng, phi, m, N)
    return phi_hat, budget_used + m * N


def binary_depth_risk(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("risk", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_depth_L(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("L", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_depth_capL(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("risk_capL", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


def binary_depth_maxL(rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf=0.95):
    return _run("risk_maxL", rng, phi, phi_max, phi_min, m_exploration, budget, eps_target, conf)


FN = {"risk": binary_depth_risk, "L": binary_depth_L,
      "risk_capL": binary_depth_capL, "risk_maxL": binary_depth_maxL}


def diagnostics(phi_min, phi_max, eps, budget, m_exploration, conf, R=6000, seed=2024):
    """How L and N* sit relative to N_opt -- the mechanism behind the rates."""
    rows = []
    for s in np.random.default_rng(seed).integers(0, 2**63, size=R):
        rng = np.random.default_rng(int(s))
        phi = float(rng.uniform(phi_min, phi_max))
        out = explore_history(rng, phi, phi_max, phi_min, m_exploration, budget, conf)
        if out is None:
            continue
        probes, used = out
        rem = budget - used
        if rem <= 0:
            continue
        L = _L(probes)
        if L is None:
            continue
        d = {w: _depth(w, probes, m_exploration, rem, eps, phi_min, phi_max) for w in VARIANTS}
        if any(v is None for v in d.values()):
            continue
        n_opt = max(1, int(np.pi // (2 * phi)))
        rows.append([L / n_opt, d["risk"] / n_opt, float(L > n_opt), float(d["risk"] > n_opt),
                     float(d["risk"] > L), len(probes)])
    a = np.array(rows)
    if not len(a):
        return {}
    return dict(n=len(a), L_over_Nopt=float(np.median(a[:, 0])),
                Nstar_over_Nopt=float(np.median(a[:, 1])),
                L_overshoot=100 * float(a[:, 2].mean()),
                Nstar_overshoot=100 * float(a[:, 3].mean()),
                Nstar_deeper_than_L=100 * float(a[:, 4].mean()),
                probes=float(np.median(a[:, 5])))


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
        confs = [0.5, 0.65, 0.8, 0.9]
        print(f"\n=== {name}  N_min={n_min(pmax)} ===", flush=True)
        for b in budgets:
            rec = {"setting": name, "phi_min": pmin, "phi_max": pmax, "eps": eps, "budget": int(b)}
            for v in VARIANTS:
                grid = {"m_exploration": m_b, "conf": confs, "eps_target": [eps], "budget": [int(b)]}
                res = E.grid_full(FN[v], grid, R_TUNE, pmin, pmax, eps, SEED_TUNE)
                _r, _bud, cfg = max(res, key=lambda x: x[0])
                rec[v] = 100 * E.success_rate(FN[v], cfg, R_TEST, pmin, pmax, eps, SEED_TEST)
                rec[v + "_cfg"] = {k: (int(x) if isinstance(x, np.integer) else x)
                                   for k, x in cfg.items() if k != "budget"}
            # diagnostics at the SHIPPED rule's tuned config, so L and N* are compared like for like
            cf = rec["risk_cfg"]
            rec["diag"] = diagnostics(pmin, pmax, eps, int(b), int(cf["m_exploration"]), cf["conf"])
            out.append(rec)
            dg = rec["diag"]
            print(f"  B={b:>13,}  " + "  ".join(f"{v} {rec[v]:6.2f}" for v in VARIANTS)
                  + f"   | L/Nopt {dg.get('L_over_Nopt', float('nan')):.2f}"
                  f"  N*/Nopt {dg.get('Nstar_over_Nopt', float('nan')):.2f}"
                  f"  L over {dg.get('L_overshoot', float('nan')):5.2f}%"
                  f"  N* deeper than L {dg.get('Nstar_deeper_than_L', float('nan')):5.1f}%", flush=True)

    with open("results/binary_depth.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    keys = ["n", "L_over_Nopt", "Nstar_over_Nopt", "L_overshoot", "Nstar_overshoot",
            "Nstar_deeper_than_L", "probes"]
    with open("results/binary_depth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "budget"] + VARIANTS + keys)
        for r in out:
            w.writerow([r["setting"], r["budget"]] + [round(r[v], 3) for v in VARIANTS]
                       + [round(r["diag"].get(k, float("nan")), 4) for k in keys])
    print("\nwrote results/binary_depth.{csv,json}")


if __name__ == "__main__":
    main()
