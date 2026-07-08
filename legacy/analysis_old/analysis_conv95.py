"""Validate the cached >95%-convergence winners by re-running them at high R with a
fresh seed (the metric 'min budget s.t. success>=0.95 over a grid at R=1000' is
itself a winner's-curse selection -> reported budgets may be optimistic)."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (find_phi_linear_search, find_phi_binary_search,
                                    find_phi_reverse_engineering)

def validate(tag, fn, params, phi_min, phi_max, eps, paper_budget, R, seed=2024):
    r, b = E.rate_and_budget(fn, params, R, phi_min, phi_max, eps, seed)
    lo, hi = E.wilson(r, R)
    ok = "OK (>=95%)" if lo >= 0.95 else ("borderline" if hi >= 0.95 else "FAILS 95%")
    print(f"{tag:30s} success={r*100:5.1f}% CI[{lo*100:.1f},{hi*100:.1f}]  budget={b:11.0f}  (paper {paper_budget})  -> {ok}")

if __name__ == "__main__":
    print("=== Col 1: eps=1e-3, U(0.01,0.1), R=20000 ===")
    validate("linear (cached 66707)", find_phi_linear_search,
             {"m_exploration":126,"m_exploitation":4291,"lookback_window":1,"safeguard":2,"inc":1},
             0.01, 0.1, 1e-3, 66707, 20000)
    validate("binary (cached 128984)", find_phi_binary_search,
             {"m_exploration":120,"m_exploitation":3393,"conf":0.5,"safeguard":2},
             0.01, 0.1, 1e-3, 117194, 20000)
    validate("reverse_eng (cached 87977)", find_phi_reverse_engineering,
             {"m_exploration":732,"m_exploitation":2222,"safeguard":0.85},
             0.01, 0.1, 1e-3, 87976, 20000)
    print("\n=== Col 2 spot-check: eps=1e-4, U(0.01,0.1), R=3000 ===")
    validate("reverse_eng (cached 7.72M)", find_phi_reverse_engineering,
             {"m_exploration":10811,"m_exploitation":209910,"safeguard":0.9},
             0.01, 0.1, 1e-4, 6733795, 3000)


def corrected_budget(tag, fn, base, key, vals, phi_min, phi_max, eps, R=20000, seed=2024):
    """Sweep `key` upward; report the smallest value whose held-out success >= 95%."""
    print(f"\n{tag}: sweeping {key} for validated 95% (R={R})")
    hit = None
    for v in vals:
        p = dict(base); p[key] = v
        r, b = E.rate_and_budget(fn, p, R, phi_min, phi_max, eps, seed)
        lo, hi = E.wilson(r, R)
        mark = ""
        if r >= 0.95 and hit is None:
            hit = (v, b, r); mark = "  <-- validated 95%"
        print(f"   {key}={v:>8} : success={r*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}]  budget={b:11.0f}{mark}")
    return hit

if __name__ == "__main__":
    print("\n\n=== Corrected col-1 budgets (eps=1e-3, narrow): budget for a VALIDATED 95% ===")
    h = corrected_budget("linear", find_phi_linear_search,
        {"m_exploration":126,"lookback_window":1,"safeguard":2,"inc":1}, "m_exploitation",
        [4291,6000,8000,10000,13000,16000], 0.01, 0.1, 1e-3)
    print(f"   -> linear honest 95% budget ~ {h[1]:.0f} (paper 66707)" if h else "   -> not reached")
    h = corrected_budget("reverse_eng", find_phi_reverse_engineering,
        {"m_exploration":732,"safeguard":0.85}, "m_exploitation",
        [2222,3000,4000,5000,6500,8000], 0.01, 0.1, 1e-3)
    print(f"   -> RE honest 95% budget ~ {h[1]:.0f} (paper 87976)" if h else "   -> not reached")
    h = corrected_budget("binary", find_phi_binary_search,
        {"m_exploration":120,"conf":0.5,"safeguard":2}, "m_exploitation",
        [3393,5000,7000,9000,12000,16000], 0.01, 0.1, 1e-3)
    print(f"   -> binary honest 95% budget ~ {h[1]:.0f} (paper 117194)" if h else "   -> not reached")
