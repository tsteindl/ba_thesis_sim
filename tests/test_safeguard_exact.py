"""Focused tests for the EXACT statistical safeguard (EXACT_SAFEGUARD_RERUN_HANDOFF.md).

    python tests/test_safeguard_exact.py

Runs standalone (no pytest needed) and is pytest-compatible if pytest is installed. These cover the
properties the exhaustive-enumeration rewrite is supposed to guarantee, as opposed to the pipeline
properties tests/test_consolidated.py covers:

 1 vectorised argmax == a deliberately naive scalar reference loop, truncated support included
 2 N_min <= N* <= N_max always, and N_min > 1 never collapses to 1
 3 an exact tie resolves to the SMALLEST N
 4 affordability boundaries: B < N_min, B == N_min, and the budgets where floor(B/N) steps
 5 the score really uses floor(B/N) -- a case where the old smooth m = B/N picks a different N
 6 invalid inputs and a collapsed truncated posterior
 7 binary_deep / reverse_eng_risk never overspend, and their recorded N_star stays in [N_min, N_max]
 8 every live call site passes BOTH bounds explicitly (AST scan of the repository)
"""
import ast
import os
import pathlib
import sys

import numpy as np
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qmetrology import algorithms as ALG
from qmetrology.safeguard import _p_safe, _score, effective_C, pilot_sd, risk_optimal_depth
from qmetrology.trace import AlgorithmTrace

# the three (N_min, N_max) pairs the active thesis matrix actually uses
PRIORS = [(0.01, 0.1), (0.001, 0.01), (0.001, 0.1)]


def bounds(phi_min, phi_max):
    lo = max(int(np.pi // (2 * phi_max)), 1)
    return lo, max(int(np.pi // (2 * phi_min)), lo)


# ------------------------------------------------------------------ 1. scalar reference loop
def _reference(phi_hat, sigma, budget, eps, N_min, N_max, support=None):
    """Deliberately naive: one Python iteration per candidate, no vectorisation, no shortcuts.

    Written to be obviously correct rather than fast -- it is the thing the production
    implementation is checked against, so it must not share its structure.
    """
    best_N, best_s = None, -np.inf
    for N in range(int(N_min), int(N_max) + 1):
        m = int(budget) // N
        if m >= 1:
            p_acc = 2.0 * float(ndtr(2.0 * eps * N * np.sqrt(m))) - 1.0
        else:
            p_acc = 0.0
        t = np.pi / (2.0 * N)
        if support is None:
            p_safe = float(ndtr((t - phi_hat) / sigma))
        else:
            lo, hi = support
            Fa = float(ndtr((lo - phi_hat) / sigma))
            Fb = float(ndtr((hi - phi_hat) / sigma))
            den = Fb - Fa
            if not np.isfinite(den) or den < 1e-12:
                p_safe = float(t > min(max(phi_hat, lo), hi))
            else:
                p_safe = min(max((float(ndtr((t - phi_hat) / sigma)) - Fa) / den, 0.0), 1.0)
        s = p_safe * p_acc
        if s > best_s:                      # strict '>' == keep the SMALLEST N on a tie
            best_N, best_s = N, s
    return best_N


def test_matches_scalar_reference():
    rng = np.random.default_rng(20260901)
    n = 0
    for (pmin, pmax) in PRIORS:
        N_min, N_max = bounds(pmin, pmax)
        for _ in range(120):
            m_pilot = int(rng.integers(1, 4000))
            sigma = pilot_sd(N_min, m_pilot)
            # pilots inside AND outside the prior support -- the boundary layer is where the
            # truncated and untruncated posteriors disagree
            phi_hat = float(rng.uniform(0.5 * pmin, 1.5 * pmax))
            budget = int(rng.integers(1, 5_000_000))
            eps = float(10.0 ** rng.uniform(-8, -2))
            for support in (None, (pmin, pmax)):
                got = risk_optimal_depth(phi_hat, sigma, budget, eps,
                                         N_min=N_min, N_max=N_max, support=support)
                want = _reference(phi_hat, sigma, budget, eps, N_min, N_max, support)
                assert got == want, (phi_hat, sigma, budget, eps, support, got, want)
                n += 1
    assert n == 3 * 120 * 2


# ------------------------------------------------------------------------ 2. range invariant
def test_result_always_inside_bounds_and_never_1():
    rng = np.random.default_rng(7)
    for (pmin, pmax) in PRIORS:
        N_min, N_max = bounds(pmin, pmax)
        assert N_min > 1, (pmin, pmax)          # otherwise the "never 1" claim is vacuous
        for _ in range(400):
            phi_hat = float(rng.uniform(0.2 * pmin, 3.0 * pmax))
            sigma = pilot_sd(N_min, int(rng.integers(1, 20_000)))
            budget = int(rng.integers(1, 10_000_000))
            eps = float(10.0 ** rng.uniform(-8, -2))
            support = (pmin, pmax) if rng.random() < 0.5 else None
            N = risk_optimal_depth(phi_hat, sigma, budget, eps,
                                   N_min=N_min, N_max=N_max, support=support)
            assert N_min <= N <= N_max, (phi_hat, sigma, budget, eps, N)
            assert N != 1


def test_narrow_interval_and_reversed_bounds():
    """N_min == N_max leaves exactly one admissible depth; N_max < N_min degenerates to N_min."""
    assert risk_optimal_depth(0.05, 0.001, 10_000, 1e-3, N_min=20, N_max=20) == 20
    assert risk_optimal_depth(0.05, 0.001, 10_000, 1e-3, N_min=20, N_max=3) == 20


# ------------------------------------------------------------------------ 3. the tie rule
def _saturated(N_min, N_max):
    """Inputs where BOTH factors are exactly 1.0 in double precision over the whole interval, so
    every candidate scores identically and only the tie rule decides."""
    return dict(phi_hat=1e-9, sigma=1e-12, budget=10 ** 9, eps=1.0, N_min=N_min, N_max=N_max)


def test_exact_tie_returns_smallest_N():
    for (N_min, N_max) in [(15, 157), (157, 1570), (1, 40)]:
        kw = _saturated(N_min, N_max)
        s = _score(np.arange(N_min, N_max + 1), kw["phi_hat"], kw["sigma"],
                   kw["budget"], kw["eps"])
        assert np.all(s == s[0]), "inputs are not an exact tie any more"   # guard the guard
        assert (s == s.max()).sum() > 1
        assert risk_optimal_depth(**kw) == N_min


# --------------------------------------------------------- 4. affordability / floor boundaries
def test_budget_below_at_and_around_N_min():
    N_min, N_max = 157, 1570
    kw = dict(phi_hat=0.003, sigma=pilot_sd(N_min, 200), eps=1e-4,
              N_min=N_min, N_max=N_max, support=(0.001, 0.01))

    # B < N_min: nothing is affordable, the score is identically zero, N_min comes back and the
    # CALLER is the one that must decline to spend (tested for the algorithms in test 7).
    for B in (1, N_min - 1):
        s = _score(np.arange(N_min, N_max + 1), kw["phi_hat"], kw["sigma"], B, kw["eps"],
                   kw["support"])
        assert np.all(s == 0.0)
        assert risk_optimal_depth(budget=B, **kw) == N_min
        assert B // N_min == 0

    # B == N_min: exactly one shot at exactly one depth is affordable
    N = risk_optimal_depth(budget=N_min, **kw)
    assert N == N_min and N_min // N == 1

    # every candidate remains affordable at its own depth or scores zero -- never in between
    for B in (N_min, N_min + 1, 2 * N_min, 2 * N_min - 1, 10 * N_min):
        N = risk_optimal_depth(budget=B, **kw)
        assert N_min <= N <= N_max
        assert B // N >= 1, (B, N)


def test_score_steps_where_floor_steps():
    """floor(B/N) is a step function; the score must inherit the step, not smooth it away."""
    B, N = 1000, 100
    assert B // N == 10 and B // (N + 1) == 9
    a = float(_score(N, 1e-9, 1e-12, B, 1e-4))
    b = float(_score(N + 1, 1e-9, 1e-12, B, 1e-4))
    # p_safe is 1.0 for both, so the ratio is exactly the accuracy factor's: N sqrt(10) vs
    # (N+1) sqrt(9) -- the deeper candidate is WORSE, which a smooth m = B/N can never produce.
    assert N * np.sqrt(10) > (N + 1) * np.sqrt(9)
    assert a > b


# -------------------------------------------------- 5. regression: floor(B/N), not the old B/N
def _smooth_argmax(phi_hat, sigma, budget, eps, N_min, N_max, support):
    """The score this implementation REPLACED: m = B/N, so p_acc = 2 Phi(2 eps sqrt(N B)) - 1."""
    c = np.arange(N_min, N_max + 1, dtype=np.int64)
    p_acc = 2.0 * ndtr(2.0 * eps * np.sqrt(c.astype(float) * float(budget))) - 1.0
    return int(c[int(np.argmax(_p_safe(c.astype(float), phi_hat, sigma, support) * p_acc))])


def test_uses_floor_not_smooth_budget_over_N():
    # (phi_min, phi_max, eps, budget, m_pilot, phi_hat, support?, floor answer, smooth answer)
    CASES = [
        (0.01, 0.1, 1e-3, 300, 5, 0.01, False, 60, 64),
        (0.01, 0.1, 1e-3, 300, 5, 0.01, True, 50, 44),
        (0.01, 0.1, 1e-3, 300, 5, 0.01375, False, 50, 51),
        (0.01, 0.1, 1e-3, 300, 5, 0.0325, True, 30, 28),
    ]
    for (pmin, pmax, eps, B, mp, ph, trunc, want_floor, want_smooth) in CASES:
        N_min, N_max = bounds(pmin, pmax)
        sup = (pmin, pmax) if trunc else None
        sd = pilot_sd(N_min, mp)
        got = risk_optimal_depth(ph, sd, B, eps, N_min=N_min, N_max=N_max, support=sup)
        assert got == want_floor, (pmin, pmax, eps, B, mp, ph, trunc, got, want_floor)
        assert _smooth_argmax(ph, sd, B, eps, N_min, N_max, sup) == want_smooth
        assert want_floor != want_smooth       # the case would prove nothing otherwise


# ----------------------------------------------------- 6. invalid inputs / collapsed posterior
def test_invalid_inputs_return_N_min():
    N_min, N_max = 157, 1570
    for kw in (dict(phi_hat=np.nan, sigma=1e-4, budget=10_000, eps=1e-4),
               dict(phi_hat=np.inf, sigma=1e-4, budget=10_000, eps=1e-4),
               dict(phi_hat=0.005, sigma=0.0, budget=10_000, eps=1e-4),
               dict(phi_hat=0.005, sigma=-1e-4, budget=10_000, eps=1e-4),
               dict(phi_hat=0.005, sigma=1e-4, budget=0, eps=1e-4),
               dict(phi_hat=0.005, sigma=1e-4, budget=-5, eps=1e-4),
               dict(phi_hat=0.005, sigma=1e-4, budget=10_000, eps=0.0),
               dict(phi_hat=0.005, sigma=1e-4, budget=10_000, eps=-1e-4)):
        for support in (None, (0.001, 0.01)):
            N = risk_optimal_depth(N_min=N_min, N_max=N_max, support=support, **kw)
            assert N == N_min, (kw, support, N)
            assert N != 1


def test_collapsed_truncated_posterior():
    """A pilot far outside the prior makes Fb - Fa underflow; _p_safe falls back to the indicator
    P(phi < pi/2N) = 1{pi/2N > clamp(phi_hat)}, and the rule must still return an admissible N."""
    pmin, pmax = 0.001, 0.01
    N_min, N_max = bounds(pmin, pmax)
    sigma = 1e-9
    for phi_hat in (10.0, -5.0, 0.5):
        c = np.arange(N_min, N_max + 1, dtype=float)
        Fa, Fb = ndtr((pmin - phi_hat) / sigma), ndtr((pmax - phi_hat) / sigma)
        assert not (Fb - Fa >= 1e-12), "posterior did not actually collapse"
        p = _p_safe(c, phi_hat, sigma, (pmin, pmax))
        assert set(np.unique(p)) <= {0.0, 1.0}
        N = risk_optimal_depth(phi_hat, sigma, 1_000_000, 1e-4,
                               N_min=N_min, N_max=N_max, support=(pmin, pmax))
        assert N_min <= N <= N_max
    # phi_hat above phi_max collapses the posterior onto phi_max, where only N_min is safe
    assert risk_optimal_depth(10.0, 1e-9, 1_000_000, 1e-4, N_min=N_min, N_max=N_max,
                              support=(pmin, pmax)) == N_min


def test_effective_C_requires_bounds():
    N_min, N_max = bounds(0.01, 0.1)
    c = effective_C(0.05, pilot_sd(N_min, 100), 9_000, 1e-3, N_min=N_min, N_max=N_max)
    assert 0.0 < c <= N_max
    for fn in (risk_optimal_depth, effective_C):
        try:
            fn(0.05, 1e-3, 9_000, 1e-3)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted a call with no bounds")


# ---------------------------------------------- 7. the two reported algorithms end to end
def test_reported_algorithms_budget_and_depth():
    CASES = [(0.01, 0.1, 1e-3, 10_000, 120, 0.5),
             (0.01, 0.1, 1e-8, 3_000_000, 800, 0.62),
             (0.001, 0.01, 1e-4, 1_000_000, 300, 0.8),
             (0.001, 0.1, 1e-4, 200_000, 60, 0.5),
             (0.001, 0.01, 1e-4, 400, 3, 0.5),          # budget below one pilot probe
             (0.001, 0.1, 1e-4, 2_000, 12, 0.5)]        # exploitation barely affordable
    seen_star = 0
    for (pmin, pmax, eps, B, m, conf) in CASES:
        N_min, N_max = bounds(pmin, pmax)
        for s in range(80):
            for algo in ("binary_deep", "reverse_eng_risk"):
                tr = AlgorithmTrace()
                rng = np.random.default_rng(s)
                phi = float(rng.uniform(pmin, pmax))
                if algo == "binary_deep":
                    _ph, used = ALG.find_phi_fixed_budget_binary_search_deep(
                        rng, phi, pmax, pmin, m_exploration=m, budget=B, eps_target=eps,
                        conf=conf, trace=tr)
                else:
                    _ph, used = ALG.find_phi_fixed_budget_reverse_engineering_risk(
                        rng, phi, pmax, pmin, m_exploration=m, budget=B, eps_target=eps, trace=tr)
                assert used <= B, (algo, pmin, pmax, B, s, used)
                if tr.N_star is not None:
                    seen_star += 1
                    assert N_min <= tr.N_star <= N_max, (algo, pmin, pmax, B, s, tr.N_star)
                    assert tr.m_final >= 1
                    # the exploitation was actually paid for out of what was left
                    assert tr.N_star * tr.m_final <= B - tr.budget_exploration + 1e-9
    assert seen_star > 0


# ----------------------------------------------------------------- 8. call-site bound audit
def test_every_call_site_passes_both_bounds():
    live, missing = 0, []
    for p in sorted(pathlib.Path(ROOT).rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("results/", "legacy/")) or "__pycache__" in rel:
            continue
        for node in ast.walk(ast.parse(p.read_text(), rel)):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) in ("risk_optimal_depth", "effective_C")):
                continue
            if any(k.arg is None for k in node.keywords):
                # a ** unpacking cannot be checked statically. Allowed only in this file, whose
                # own parametrised cases carry the bounds inside the dict.
                assert rel == "tests/test_safeguard_exact.py", f"{rel}:{node.lineno} hides its bounds behind **kwargs"
                continue
            live += 1
            kw = {k.arg for k in node.keywords}
            if not {"N_min", "N_max"} <= kw:
                missing.append(f"{rel}:{node.lineno} passes {sorted(kw)}")
    assert live >= 30, f"only found {live} call sites -- did the scan break?"
    assert not missing, "call sites without explicit bounds:\n  " + "\n  ".join(missing)


def test_no_search_schedule_constants_remain():
    src = pathlib.Path(ROOT, "qmetrology", "safeguard.py").read_text()
    for dead in ("_N_SEARCH", "_N_ROUNDS", "_HARD_CAP", "geomspace", "5.0 * sigma"):
        assert dead not in src, f"{dead} still present in safeguard.py"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as ex:
            bad += 1
            print(f"  FAIL  {name}: {ex}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
