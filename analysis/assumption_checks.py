"""Numerical verification of the safeguard's assumptions A1, A5 and A7.

A1  Var(phi_hat) = 1/(4 N^2 m), independent of phi. Simulated across a (N, m, phi) grid and compared
    with the prediction; the ratio is reported against m*p0, the quantity that controls the boundary
    censoring which breaks it.
A5  "overshoot => the trial fails". Measured directly: among trials whose chosen depth exceeds N_opt,
    how many converge anyway (a marginal overshoot still converges while phi - pi/2N < eps).
A7  "the chosen operating point stays inside A1's validity region". Measured as the share of trials
    whose exploitation shot lands at m*p0 < 10.

    python analysis/assumption_checks.py [--quick]
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.sim import simulate_errors

QUICK = "--quick" in sys.argv
R_A1 = 4000 if QUICK else 20_000
R_OP = 4000 if QUICK else 20_000


# ----------------------------------------------------------------------------- A1
def a1_cell(task):
    N, m, phi, R, seed = task
    rng = np.random.default_rng(seed)
    p0 = np.cos(N * phi) ** 2
    k = rng.binomial(m, p0, size=R)
    ph = np.arccos(np.sqrt(k / m)) / N
    emp = float(np.std(ph))
    pred = 1.0 / (2.0 * N * np.sqrt(m))
    # share of draws pinned to either end of the estimator's range
    atom = float(np.mean((k == 0) | (k == m)))
    return dict(N=N, m=m, phi=phi, m_p0=m * p0, emp_sd=emp, pred_sd=pred,
                ratio=emp / pred, atom=atom)


def check_a1():
    tasks = []
    seed = 0
    for N in (15, 157, 1570, 15707):
        for m in (5, 20, 100, 1000, 10000):
            for frac in (0.05, 0.25, 0.5, 0.75, 0.9, 0.98):
                phi = frac * np.pi / (2 * N)     # phi as a fraction of the aliasing limit
                seed += 1
                tasks.append((N, m, phi, R_A1, seed))
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        rows = list(ex.map(a1_cell, tasks, chunksize=4))
    with open("results/assumption_a1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    r = np.array([x["ratio"] for x in rows])
    mp = np.array([x["m_p0"] for x in rows])
    print("A1  empirical sd / predicted sd = 1/(2N sqrt(m)):")
    for lo, hi, lab in ((0, 1, "m*p0 < 1"), (1, 10, "1 <= m*p0 < 10"), (10, 100, "10 <= m*p0 < 100"),
                        (100, np.inf, "m*p0 >= 100")):
        s = (mp >= lo) & (mp < hi)
        if s.sum():
            print(f"      {lab:18s} n={s.sum():3d}  ratio median {np.median(r[s]):5.2f}  "
                  f"range {r[s].min():.2f}-{r[s].max():.2f}")
    ok = mp >= 10
    print(f"      -> inside the stated validity region (m*p0 >= 10) the law is accurate to "
          f"{100*abs(np.median(r[ok])-1):.1f}% (median)")
    return rows


# ----------------------------------------------------------------------------- A5 / A7
def _op_trial(task):
    pmin, pmax, eps, B, m, seeds = task
    n = over = over_conv = low_info = conv = 0
    nmin = max(int(np.pi // (2 * pmax)), 1)
    nsup = max(int(np.pi // (2 * pmin)), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = float(rng.uniform(pmin, pmax))
            if m * nmin > B:
                continue
            ph0 = simulate_errors(rng, phi, m, nmin)
            rem = B - m * nmin
            if rem <= 0 or not np.isfinite(ph0):
                continue
            N = risk_optimal_depth(ph0, pilot_sd(nmin, m), rem, eps,
                                   N_min=min(nmin, nsup), N_max=nsup, support=(pmin, pmax))
            mm = int(rem / N)
            if mm < 1:
                continue
            est = simulate_errors(rng, phi, mm, N)
            c = abs(est - phi) < eps
            nopt = max(1, int(np.pi // (2 * phi)))
            n += 1
            conv += c
            if N > nopt:
                over += 1
                over_conv += c
            if mm * np.cos(N * phi) ** 2 < 10:
                low_info += 1
    return n, conv, over, over_conv, low_info


def check_a5_a7():
    rows = list(csv.DictReader(open("results/binary_diagnostics.csv")))
    out = []
    print(f"\nA5/A7  {'scenario':28s} {'budget':>14s} {'over%':>6s} {'of those, converge':>19s} "
          f"{'m*p0<10':>8s}")
    for r in rows:
        pmin, pmax, eps = float(r["phi_min"]) if "phi_min" in r else None, None, None
        break
    # binary_diagnostics.csv has no phi columns; recover them from story_curves
    meta = {}
    with open("results/story_curves.csv", newline="") as f:
        for x in csv.DictReader(f):
            meta[x["setting"]] = (float(x["phi_min"]), float(x["phi_max"]), float(x["eps"]))
    for r in rows:
        pmin, pmax, eps = meta[r["setting"]]
        B, m = int(r["budget"]), int(r["m"])
        seeds = np.random.default_rng(2024).integers(0, 2 ** 63, size=R_OP)
        chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
        with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
            res = list(ex.map(_op_trial, [(pmin, pmax, eps, B, m, c) for c in chunks], chunksize=1))
        n = sum(x[0] for x in res)
        if not n:
            continue
        over, over_conv, low = sum(x[2] for x in res), sum(x[3] for x in res), sum(x[4] for x in res)
        d = dict(setting=r["setting"], budget=B, n=n, overshoot=over / n,
                 overshoot_converged=(over_conv / over) if over else float("nan"),
                 low_info_share=low / n)
        out.append(d)
        print(f"       {r['setting'][:28]:28s} {B:14,d} {100*d['overshoot']:5.2f}% "
              f"{100*d['overshoot_converged']:18.1f}% {100*d['low_info_share']:7.1f}%")
    with open("results/assumption_a5_a7.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    ov = np.array([d["overshoot"] for d in out])
    oc = np.array([d["overshoot_converged"] for d in out])
    li = np.array([d["low_info_share"] for d in out])
    print(f"\n       A5: overshoot happens in {100*ov.mean():.2f}% of trials (max {100*ov.max():.2f}%); "
          f"of those, {100*np.nanmean(oc):.1f}% still converge -> A5 is conservative, not wrong")
    print(f"       A7: {100*li.mean():.1f}% of trials land at m*p0 < 10 "
          f"(median {100*np.median(li):.1f}%, max {100*li.max():.1f}%)")
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    check_a1()
    check_a5_a7()
