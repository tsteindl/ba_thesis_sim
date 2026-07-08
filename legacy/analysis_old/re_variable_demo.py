"""Does making m_exploitation HUGE let variable RE beat brute? Direct demonstration.

(1) Sweep m_exploitation over a very wide range; show (convergence, mean budget). Bigger
    exploitation -> MORE budget, not less. The 90% crossing sits above brute and never drops.
(2) Decompose the near-90% config's budget by phi-tercile -> variable RE OVER-spends on the
    easy small-phi trials (large N) and under-spends on the hard large-phi ones: the mean is
    inflated because a single m_exploitation cannot hold the per-trial budget constant.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology.sim import simulate_errors
from qmetrology import experiments as E
from qmetrology.algorithms import find_phi_fixed_budget_brute_force as BF, find_phi_reverse_engineering as RE

pmin, pmax, eps, R = 0.01, 0.1, 1e-4, 15000
brute90 = E.rate_and_budget(BF, {"budget": 4_516_540}, R, pmin, pmax, eps, 2024)[1]
print(f"brute @90% budget = {brute90:,.0f}\n")


def run(mexp, mfin, sg=0.9, R=R, record=False):
    seeds = np.random.default_rng(2024).integers(0, 2**63, R)
    s = 0
    phis = np.empty(R); buds = np.empty(R); ok = np.zeros(R, bool)
    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd)); phi = float(rng.uniform(pmin, pmax))
        ph, b = RE(rng, phi, pmax, pmin, m_exploration=mexp, m_exploitation=int(mfin), safeguard=sg)
        phis[i] = phi; buds[i] = b; ok[i] = abs(ph - phi) < eps
    return ok.mean(), buds.mean(), phis, buds, ok


print("(1) sweep m_exploitation (m_exploration=1000, safeguard=0.9):")
print(f"  {'m_exploitation':>15} | {'conv rate':>9} | {'mean budget':>14}")
for mfin in np.geomspace(3_000, 1_000_000_000, 16):
    r, b, *_ = run(1000, mfin)
    flag = "  <- first >=90%" if r >= 0.90 and b > 0 else ""
    print(f"  {int(mfin):>15,} | {r:9.3f} | {b:14,.0f}{flag}")

print("\n(2) budget misallocation for a ~90% config (m_exploration=1000, m_exploitation tuned):")
# pick the smallest m_exploitation reaching ~90% from a fine local sweep
best = None
for mfin in np.geomspace(50_000, 5_000_000, 24):
    r, b, phis, buds, ok = run(1000, mfin, R=20000)
    if r >= 0.90:
        best = (mfin, r, b, phis, buds); break
if best:
    mfin, r, b, phis, buds = best
    print(f"  config: m_exploitation={int(mfin):,}  -> rate {r:.3f}, mean budget {b:,.0f}  (brute {brute90:,.0f})")
    edges = np.quantile(phis, [0, 1/3, 2/3, 1.0])
    for lo, hi, name in [(edges[0], edges[1], "small phi (EASY)"), (edges[1], edges[2], "mid phi"),
                         (edges[2], edges[3], "large phi (HARD)")]:
        m = (phis >= lo) & (phis < hi + 1e-12)
        print(f"    {name:18s} phi in [{lo:.3f},{hi:.3f}]  mean budget {buds[m].mean():14,.0f}   ({'over' if buds[m].mean()>brute90 else 'under'} brute)")
