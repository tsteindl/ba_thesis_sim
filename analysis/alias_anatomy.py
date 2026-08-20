"""Why does a pilot that is aliased in ~23% of trials not destroy convergence?

The deepest ACCEPTED probe sits above N_opt in about a quarter of trials, and an aliased estimate is
not merely noisy -- it is the folded value d(N phi)/N. Since N > N_opt implies pi/(2N) < phi and the
estimator can never exceed pi/(2N), an aliased probe ALWAYS reads low. Feeding a low pilot to Eq. (3.8)
should make it believe phi is smaller than it is and pick a depth that is too deep.

This script measures what actually happens, splitting every trial by whether the pilot was aliased:
the size of the depth overshoot N_acc/N_opt, the resulting bias phi_acc/phi, the depth the safeguard
then chooses, whether that depth aliases, and whether the trial converges.

    python analysis/alias_anatomy.py [--quick]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_story_study import explore

QUICK = "--quick" in sys.argv
R = 4000 if QUICK else 30_000


def _task(t):
    pmin, pmax, eps, B, m, conf, seeds = t
    rec = []
    n_sup = max(int(np.pi // (2 * pmin)), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = float(rng.uniform(pmin, pmax))
            o = explore(rng, phi, pmax, pmin, m, B, conf)
            if o is None:
                continue
            _p, _N, used, phi_acc, N_acc, _p0, _N0, L, U, probes = o
            rem = B - used
            if rem <= 0 or not np.isfinite(phi_acc):
                continue
            n_opt = max(1, int(np.pi // (2 * phi)))
            N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m), rem, eps, N_max=n_sup,
                                   support=(pmin, pmax))
            mm = int(rem / N)
            if mm < 1:
                continue
            est = simulate_errors(np.random.default_rng([int(s) % (2 ** 32), 15485863]), phi, mm, N)
            rec.append((float(N_acc > n_opt), N_acc / n_opt, phi_acc / phi, N / n_opt,
                        float(N > n_opt), float(abs(est - phi) < eps), len(probes)))
    return rec


def run_point(pmin, pmax, eps, B, m, conf):
    seeds = np.random.default_rng(2024).integers(0, 2 ** 63, size=R)
    chunks = [c for c in np.array_split(seeds, E.N_JOBS * 2) if len(c)]
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as ex:
        out = list(ex.map(_task, [(pmin, pmax, eps, B, m, conf, c) for c in chunks], chunksize=1))
    return np.array([r for part in out for r in part])


def main():
    rows = list(csv.DictReader(open("results/binary_deep.csv")))
    keep = [r for r in rows if float(r["deep_1s_probes"]) >= 2]
    agg = []
    print(f"{'scenario':24s} {'alias%':>7s} | {'N_acc/N_opt':>12s} {'phi_hat/phi':>12s} "
          f"{'N*/N_opt':>9s} {'final over%':>11s} {'converge%':>10s}")
    for r in keep:
        pmin, pmax, eps = float(r["phi_min"]), float(r["phi_max"]), float(r["eps"])
        B, m, conf = int(r["budget"]), int(r["deep_1s_m"]), float(r["deep_1s_conf"])
        a = run_point(pmin, pmax, eps, B, m, conf)
        if not len(a):
            continue
        al = a[:, 0] > 0.5
        agg.append((a, al))
        for lab, sel in (("aliased pilot", al), ("clean pilot", ~al)):
            if not sel.sum():
                continue
            print(f"{(r['setting'][:22] if lab.startswith('aliased') else ''):24s} "
                  f"{100*al.mean() if lab.startswith('aliased') else 0:6.1f}% | "
                  f"{a[sel,1].mean():12.3f} {a[sel,2].mean():12.3f} {a[sel,3].mean():9.3f} "
                  f"{100*a[sel,4].mean():10.2f}% {100*a[sel,5].mean():9.2f}%   {lab}")
    A = np.vstack([x[0] for x in agg])
    al = A[:, 0] > 0.5
    print(f"\n=== pooled over {len(agg)} operating points, {len(A):,} trials ===")
    print(f"pilot aliased in {100*al.mean():.1f}% of trials")
    for lab, sel in (("ALIASED pilot", al), ("clean pilot", ~al)):
        print(f"  {lab:14s} n={sel.sum():7,d}  N_acc/N_opt {A[sel,1].mean():.3f} "
              f"(median {np.median(A[sel,1]):.3f}, p95 {np.percentile(A[sel,1],95):.3f})  "
              f"phi_hat/phi {A[sel,2].mean():.3f}  ->  N*/N_opt {A[sel,3].mean():.3f}  "
              f"final overshoot {100*A[sel,4].mean():.2f}%  converge {100*A[sel,5].mean():.2f}%")
    q = A[al, 1]
    print(f"\n  of the aliased pilots: {100*(q<1.05).mean():.1f}% are within 5% of N_opt, "
          f"{100*(q<1.2).mean():.1f}% within 20%, {100*(q>2).mean():.1f}% beyond 2x")
    print(f"  convergence penalty of an aliased pilot: "
          f"{100*(A[al,5].mean()-A[~al,5].mean()):+.2f} pp")
    np.save("results/alias_anatomy.npy", A)


if __name__ == "__main__":
    main()
