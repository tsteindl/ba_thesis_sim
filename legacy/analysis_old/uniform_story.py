"""The best HONEST story under the fixed UNIFORM assumption phi ~ U(phi_min, phi_max).

A) de-biasing sanity: is R large enough? (re-validate the headline at R up to 200k)
B) why U(.,pi/2) fails but U(0.001,0.1) works, under uniform (small-phi mass + (phi_max/phi)^2)
C) the recommended story: budget-ratio + a convergence-vs-budget figure

    python analysis/uniform_story.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_linear_search as LIN,
    find_phi_fixed_budget_reverse_engineering as RE,
)

GRID_LIN = {"m_exploration": np.unique(np.geomspace(5, 3000, 10).astype(int)),
            "lookback_window": [1, 2, 5], "safeguard": [1, 2], "inc": [1, 2]}
GRID_RE = {"m_exploration": np.unique(np.geomspace(20, 8000, 12).astype(int)), "safeguard": [0.8, 0.85, 0.9]}


def brute(b, pmin, pmax, eps, R=30000, dist="uniform"):
    return 100 * E.success_rate(BF, {"budget": int(b)}, R, pmin, pmax, eps, 2024, phi_dist=dist)


def best_adaptive(b, pmin, pmax, eps, R=30000, dist="uniform", which=False):
    out = {}
    for name, fn, g in [("linear", LIN, GRID_LIN), ("reverse_eng", RE, GRID_RE)]:
        res = E.grid_full(fn, {**g, "budget": [int(b)]}, 700, pmin, pmax, eps, 42, phi_dist=dist)
        win = max(res, key=lambda r: r[0])[2]
        out[name] = 100 * E.success_rate(fn, win, R, pmin, pmax, eps, 2024, phi_dist=dist)
    best = max(out, key=out.get)
    return (out[best], best) if which else out[best]


def crossing(budgets, rates, target):
    r = np.asarray(rates)
    if target <= r[0] or target >= r[-1]:
        return float("nan")
    return float(np.exp(np.interp(target, r, np.log(budgets))))


def part_A():
    print("== A) De-biasing sanity: headline = reverse_eng, U(0.01,0.1), eps=1e-4, budget=1e6 ==")
    print("   (tune on seed 42; VALIDATE the winner on fresh seed 2024 at increasing R)")
    res = E.grid_full(RE, {**GRID_RE, "budget": [1_000_000]}, 700, 0.01, 0.1, 1e-4, 42)
    win = max(res, key=lambda r: r[0])[2]
    for R in (20_000, 50_000, 100_000, 200_000):
        b = 100 * E.success_rate(BF, {"budget": 1_000_000}, R, 0.01, 0.1, 1e-4, 2024)
        a = 100 * E.success_rate(RE, win, R, 0.01, 0.1, 1e-4, 2024)
        se = 100 * np.sqrt(0.25 / R) * np.sqrt(2)
        print(f"   R={R:>7,}: brute={b:5.2f}%  reverse_eng={a:5.2f}%  margin={a-b:+5.2f}pp  (SE_margin~{se:.2f}pp)")


def part_B():
    print("\n== B) Why 'up to pi/2' fails but 'down to small phi' works, under UNIFORM (eps=1e-3) ==")
    print("   small-phi mass = fraction of a uniform draw with phi < 0.01 (where adaptive has big edge)")
    cases = [("U(0.001, 0.1)   [phi_max small, wide DOWN]", 0.001, 0.1, 30_000),
             ("U(0.157, pi/2)  [~1 order, UP]", 0.157, np.pi / 2, 100_000),
             ("U(0.00157, pi/2)[3 orders, spans, UP]", 0.00157, np.pi / 2, 100_000),
             ("U(0.0001, 0.1)  [3 orders, wide DOWN]", 0.0001, 0.1, 30_000)]
    for name, pmin, pmax, b in cases:
        mass = max(0.0, (min(0.01, pmax) - pmin)) / (pmax - pmin) * 100
        br = brute(b, pmin, pmax, 1e-3)
        ad, who = best_adaptive(b, pmin, pmax, 1e-3, which=True)
        print(f"   {name:42s} budget={b:>7,}: small-phi mass={mass:5.1f}%  brute={br:5.1f}%  "
              f"best-adaptive={ad:5.1f}% ({who})  margin={ad-br:+5.1f}pp")


def part_C():
    print("\n== C) Recommended UNIFORM story: budget ratio + figure (U(0.01,0.1)) ==")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, eps, budgets in [(axes[0], 1e-3, np.geomspace(3e3, 3e6, 12)),
                             (axes[1], 1e-4, np.geomspace(3e5, 3e8, 12))]:
        br = [brute(b, 0.01, 0.1, eps) for b in budgets]
        ad = [best_adaptive(b, 0.01, 0.1, eps) for b in budgets]
        ax.plot(budgets, br, "o-", label="brute force", color="gray")
        ax.plot(budgets, ad, "s-", label="best adaptive", color="steelblue")
        ax.set_xscale("log"); ax.set_xlabel("budget = N·m"); ax.set_ylabel("% converged")
        ax.set_title(f"eps=1e-{int(-np.log10(eps))},  phi~U(0.01,0.1)")
        ax.axhline(50, color="k", ls=":", lw=0.7); ax.axhline(90, color="k", ls=":", lw=0.7)
        ax.legend()
        print(f"   eps={eps:.0e}:")
        for T in (50, 90):
            bb, ba = crossing(budgets, br, T), crossing(budgets, ad, T)
            if np.isfinite(bb) and np.isfinite(ba):
                print(f"      reach {T}%: brute {bb:12,.0f} | adaptive {ba:12,.0f}  ->  {bb/ba:4.2f}x less budget")
    fig.suptitle("Adaptive reaches the same reliability at lower budget (uniform prior, de-biased)")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/fig_budget_ratio.png", dpi=130)
    print("   -> wrote results/fig_budget_ratio.png")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
