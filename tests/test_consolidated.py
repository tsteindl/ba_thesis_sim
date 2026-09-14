"""Validation suite for the results pipeline.

    python tests/test_consolidated.py            # everything except the two slow end-to-end checks
    python tests/test_consolidated.py --slow     # + smoke sweep, determinism and resume

Runs standalone (no pytest needed) and is pytest-compatible if pytest is installed.

Covered:
 1 trace neutrality          6 performance consistency
 2 budget compliance         7 CI consistency (row-specific R, no hard-coded 40,000/50,000)
 3 determinism               8 winner provenance
 4 definition checks         9 thesis/code parity
 5 eligibility accounting   10 quick/full modes and resume

Plus the replay invariants of analysis/linear_detector_study.py, whose whole argument
is that replaying a recorded probe stream reproduces Algorithm 4 exactly.
"""
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from qmetrology import algorithms as ALG
from qmetrology import diagnostics as D
from qmetrology import manifest as M
from qmetrology import oracle as O
from qmetrology import pipeline as P
from qmetrology.trace import AlgorithmTrace, ProbeTrace, n_opt, run_record

SLOW = "--slow" in sys.argv

CASES = [(0.01, 0.1, 1e-3, 10_000), (0.001, 0.01, 1e-4, 1_000_000),
         (0.01, float(np.pi / 2), 1e-3, 77_459), (0.001, 0.1, 1e-4, 200_000)]
PARAMS = {
    "brute": {},
    "linear": {"m_exploration": 12, "lookback_window": 5, "safeguard": 1, "inc": 1},
    "binary_deep": {"m_exploration": 120, "conf": 0.5},
    "reverse_eng_risk": {"m_exploration": 180},
}


def _params(algo, scen_eps, budget):
    p = dict(PARAMS[algo])
    if algo in ("binary_deep", "reverse_eng_risk"):
        p["eps_target"] = scen_eps
    p["budget"] = budget
    return p


def _run(algo, seed, pmin, pmax, eps, B, trace=None):
    fn = M.ALGORITHMS[algo]["fn"]
    rng = np.random.default_rng(seed)
    phi = float(rng.uniform(pmin, pmax))
    kw = _params(algo, eps, B)
    if trace is not None:
        kw["trace"] = trace
        trace.nominal_budget = float(B)
    out = fn(rng, phi, pmax, pmin, **kw)
    if trace is not None:
        trace.finalize(phi, out[0], out[1], eps)
    return phi, out


def test_oracle_is_analytic_and_uses_whole_shots():
    """The reported Oracle has no trial runner and its analytic integer B90 is minimal."""
    target, eps, N = 0.9, 1e-3, 31
    m = O.oracle_shots_for_rate(target, eps, N)
    p = lambda shots: 2 * O.ndtr(2 * eps * N * np.sqrt(shots)) - 1
    assert p(m) >= target and (m == 1 or p(m - 1) < target)

    b90 = O.oracle_budget_for_rate(target, eps, 0.01, 0.1)
    assert b90 == 24_435
    assert O.oracle_rate(b90, eps, 0.01, 0.1) >= target
    assert O.oracle_rate(b90 - 1, eps, 0.01, 0.1) < target

    # This narrow interval has N_opt=31 throughout.  The prior average must therefore reduce to the
    # same single closed-form probability with the whole-shot count floor(B/N_opt).
    budget = 10_000
    expected = p(budget // N)
    assert abs(O.oracle_rate(budget, eps, 0.05, 0.0505) - expected) < 1e-14
    sigma = 1.0 / (2.0 * N * np.sqrt(budget // N))
    expected_median = sigma * O.ndtri(0.75)
    assert abs(O.oracle_abs_error_quantile(budget, 0.5, 0.05, 0.0505)
               - expected_median) < 1e-14
    assert abs(O.oracle_error_variance(budget, 0.05, 0.0505) - sigma ** 2) < 1e-18
    assert M.ALGORITHMS["oracle_hl"]["fn"] is None


# ---------------------------------------------------------------------------- 1. neutrality
def test_trace_neutrality():
    for pmin, pmax, eps, B in CASES:
        for algo in M.ORDER:
            for s in range(150):
                _phi, a = _run(algo, s, pmin, pmax, eps, B)
                _phi2, b = _run(algo, s, pmin, pmax, eps, B, trace=AlgorithmTrace())
                assert np.allclose(a[0], b[0], equal_nan=True), (algo, s, a, b)
                assert a[1] == b[1], (algo, s, a, b)


# ------------------------------------------------------------------------ 2. budget compliance
def test_budget_compliance():
    """No fixed-budget algorithm may spend more than its nominal budget. Violations are counted and
    reported, never clipped away afterwards."""
    viol = []
    for pmin, pmax, eps, B in CASES:
        for algo in M.ORDER:
            for s in range(300):
                tr = AlgorithmTrace()
                _phi, (_ph, used) = _run(algo, s, pmin, pmax, eps, B, trace=tr)
                if used > B + 1e-6:
                    viol.append((algo, pmin, pmax, B, s, used))
                # the exploitation shot must fit in what the exploration left
                if tr.N_star is not None and tr.m_final is not None:
                    assert tr.budget_exploration + tr.N_star * tr.m_final <= B + 1e-6, \
                        (algo, s, tr.budget_exploration, tr.N_star, tr.m_final, B)
    assert not viol, f"{len(viol)} budget violations, e.g. {viol[:3]}"


# -------------------------------------------------------------------------- 4. definition checks
def test_run_record_detector_definitions():
    """Hand-constructed traces: false alarm = a SAFE probe declared an overshoot; miss = an
    OVERSHOOTING probe not flagged. Both are trial-level (at least one)."""
    def mk(probes):
        tr = AlgorithmTrace(N_opt=10)
        tr.probes = [ProbeTrace(i, N, 5, 0.1, dec, not dec, N > 10)
                     for i, (N, dec) in enumerate(probes)]
        return tr
    r = run_record(mk([(5, False), (8, False)]), True)          # all safe, none flagged
    assert r["false_alarm"] == 0 and r["miss"] == 0
    assert (r["tn"], r["fp"], r["tp"], r["fn"]) == (2, 0, 0, 0)
    r = run_record(mk([(5, True), (8, False)]), True)           # safe probe flagged -> false alarm
    assert r["false_alarm"] == 1 and r["miss"] == 0 and r["fp"] == 1
    r = run_record(mk([(5, False), (20, False)]), True)         # overshoot accepted -> miss
    assert r["false_alarm"] == 0 and r["miss"] == 1 and r["fn"] == 1
    r = run_record(mk([(20, True)]), True)                      # overshoot flagged -> TP
    assert r["false_alarm"] == 0 and r["miss"] == 0 and r["tp"] == 1
    r = run_record(mk([(5, False)]), False)                     # no detector -> N/A, not zero
    assert np.isnan(r["false_alarm"]) and np.isnan(r["miss"])
    assert n_opt(0.1) == 15 and n_opt(0.01) == 157 and n_opt(10.0) == 1


def test_linear_scan_clamps_final_increment_to_N_max():
    """An increment that does not divide the interval must not create an out-of-range probe."""
    out = ALG._linear_search_explore(
        np.random.default_rng(7), 0.05, 0.1, 0.01,
        m_exploration=1, budget=100_000, lookback_window=10_000, inc=3)
    assert out is not None
    _phi_hats, depths, _budget_used, overshot = out
    assert not overshot
    assert depths[-2:] == [156, 157]
    assert max(depths) <= 157


def test_algorithm_definitions():
    """N_guess, N_star and B_exploration must equal their handoff definitions, recomputed
    independently from the probe list of the very same run."""
    for pmin, pmax, eps, B in CASES:
        N_min = max(int(np.pi // (2 * pmax)), 1)
        for s in range(200):
            # ---- brute force: no guess, no exploration
            tr = AlgorithmTrace()
            _run("brute", s, pmin, pmax, eps, B, trace=tr)
            assert tr.N_guess is None and tr.budget_exploration == 0.0
            assert tr.N_star == N_min and len(tr.probes) == 0

            # ---- linear search
            tr = AlgorithmTrace()
            _run("linear", s, pmin, pmax, eps, B, trace=tr)
            if tr.probes:
                lw, inc, sg = (PARAMS["linear"]["lookback_window"], PARAMS["linear"]["inc"],
                               PARAMS["linear"]["safeguard"])
                m = PARAMS["linear"]["m_exploration"]
                fired = tr.status in ("detector_fired",) or any(p.declared_overshoot
                                                                for p in tr.probes)
                last = tr.probes[-1].N
                assert tr.N_guess == (last - lw * inc if fired else last), (s, tr.status)
                assert tr.budget_exploration == sum(p.N * m for p in tr.probes)
                if tr.N_star is not None:
                    assert tr.N_star == max(1, tr.N_guess - sg)
                # the flagged probes are exactly the ones backtracked over
                flagged = [p.stage_index for p in tr.probes if p.declared_overshoot]
                keep = max(0, len(tr.probes) - lw) if fired else len(tr.probes)
                assert flagged == list(range(keep, len(tr.probes)))

            # ---- binary search: N_guess is the DEEPEST ACCEPTED probe
            tr = AlgorithmTrace()
            _run("binary_deep", s, pmin, pmax, eps, B, trace=tr)
            if tr.probes:
                m = PARAMS["binary_deep"]["m_exploration"]
                acc = [p.N for p in tr.probes if p.accepted]
                assert tr.N_guess == max(max(acc), 1), (s, tr.N_guess, acc)
                assert tr.N_guess == tr.accepted_pilot_N
                assert tr.opening_pilot_N == N_min
                assert tr.budget_exploration == sum(p.N * m for p in tr.probes)
                if tr.N_star is not None:
                    assert 1 <= tr.N_star <= max(int(np.pi // (2 * pmin)), 1)

            # ---- reverse engineering: N_guess = floor(pi / (2 phi_hat_0)); retries are charged
            tr = AlgorithmTrace()
            _run("reverse_eng_risk", s, pmin, pmax, eps, B, trace=tr)
            if tr.probes:
                m = PARAMS["reverse_eng_risk"]["m_exploration"]
                assert all(p.N == N_min for p in tr.probes)
                assert tr.budget_exploration == len(tr.probes) * m * N_min
                if tr.N_guess is not None:
                    ph = tr.probes[-1].phi_hat
                    assert tr.N_guess == max(int(np.pi // (2 * ph)), 1)
                    assert all(p.declared_overshoot is None for p in tr.probes)


def _reference_bisection(rng, phi, phi_max, phi_min, m, budget, conf):
    """An INDEPENDENT transcription of Algorithm 5's exploration loop, deliberately written out
    in full rather than importing qmetrology.algorithms._binary_search_explore.

    test_binary_deep_matches_fine_sweep pairs it with a direct call to the reported algorithm on
    the same seed: if the two ever disagree about the accepted pilot, one of them has drifted.
    A reference that shared code with the implementation could not detect that.

    Returns (used_budget, phi_acc, N_acc), or None if the opening probe is unaffordable.
    """
    N_min = max(np.pi // (2 * phi_max), 1)
    N_max = max(np.pi // (2 * phi_min), 1)
    N = max(1, N_min)
    if m * N > budget:
        return None
    phi_hat = ALG.simulate_errors(rng, phi, m, N)
    used = m * N
    phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N ** 2)))
    phi_acc, N_acc = phi_hat, N
    lb, ub = N_min, N_max
    N += (ub - N) // 2
    while True:
        if used + m * N > budget:
            break
        phi_hat = ALG.simulate_errors(rng, phi, m, N)
        used += m * N
        old_N = N
        if phi_hat < phi_1:
            N -= (N - lb) // 2
            ub = old_N
        else:
            phi_acc, N_acc = phi_hat, old_N
            lb = old_N
            N += (ub - N) // 2
            phi_1 = norm.ppf(1 - conf, phi_hat, np.sqrt(1 / (4 * m * N ** 2)))
        if old_N == N or N < N_min or N > N_max or used >= budget:
            break
    return used, phi_acc, N_acc


def test_binary_deep_matches_fine_sweep():
    """The reported binary search must reproduce, trial for trial, the depth decision of the
    independent transcription of Algorithm 5 in `_reference_bisection`."""
    from qmetrology.safeguard import pilot_sd, risk_optimal_depth
    n = bad = 0
    for (pmin, pmax, eps, B, m, conf) in [(0.01, 0.1, 1e-3, 10_000, 100, 0.5),
                                          (0.001, 0.01, 1e-4, 1_000_000, 300, 0.8),
                                          (0.01, float(np.pi / 2), 1e-3, 77_459, 978, 0.5)]:
        n_min, n_sup = max(int(np.pi // (2 * pmax)), 1), max(int(np.pi // (2 * pmin)), 1)
        for s in range(300):
            rng = np.random.default_rng(s)
            phi = float(rng.uniform(pmin, pmax))
            o = _reference_bisection(rng, phi, pmax, pmin, m, B, conf)
            ref = None
            if o is not None:
                used, phi_acc, N_acc = o
                rem = B - used
                if rem > 0 and np.isfinite(phi_acc):
                    N = risk_optimal_depth(phi_acc, pilot_sd(N_acc, m), rem, eps,
                                           N_min=min(n_min, n_sup), N_max=n_sup,
                                           support=(pmin, pmax))
                    ref = N if int(rem / N) >= 1 else None
            tr = AlgorithmTrace()
            rng2 = np.random.default_rng(s)
            phi2 = float(rng2.uniform(pmin, pmax))
            ALG.find_phi_fixed_budget_binary_search_deep(
                rng2, phi2, pmax, pmin, m_exploration=m, budget=B, eps_target=eps, conf=conf,
                trace=tr)
            n += 1
            bad += int(tr.N_star != ref)
    assert bad == 0, f"{bad}/{n} trials disagree with the independent reference"


# ---------------------------------------------------------- 3./5./6. one held-out evaluation
def _one_point(mode="smoke"):
    scen = [s for s in M.SCENARIOS if s["id"] == "narrow_e3"][0]
    cfg = {"m_exploration": 120, "conf": 0.5, "eps_target": scen["eps"]}
    A, st, _ = P.heldout("binary_deep", scen, 10_000, cfg, mode)
    return scen, cfg, A, st


def test_determinism():
    scen, cfg, A1, s1 = _one_point()
    _s, _c, A2, s2 = _one_point()
    for k in A1:
        assert np.allclose(A1[k], A2[k], equal_nan=True), k
    assert s1 == s2
    w1, _ = P.tune("binary_deep", scen, 10_000, "smoke")
    w2, _ = P.tune("binary_deep", scen, 10_000, "smoke")
    assert w1 == w2, (w1, w2)


def test_eligibility_accounting():
    scen, cfg, A, st = _one_point()
    d = D.point_diagnostics(A, st, M.ALGORITHMS["binary_deep"], 50, M.SEED_BOOT)
    R = d["R"]
    assert d["guess_eligible_n"] + d["guess_ineligible_n"] == R
    assert d["star_eligible_n"] + d["star_ineligible_n"] == R
    for name in ("guess_exact", "guess_within5", "guess_within10", "guess_overshoot"):
        assert d[name + "_n"] == d["guess_eligible_n"], name
        assert 0 <= d[name + "_k"] <= d[name + "_n"], name
    for name in ("star_overshoot", "star_exact", "star_within5", "star_within10"):
        assert d[name + "_n"] == d["star_eligible_n"], name
    for name in ("single_probe", "no_exploitation", "no_exploitation_budget"):
        assert d[name + "_n"] == R, name
    assert d["conv_given_safe_depth_n"] <= d["star_eligible_n"]
    assert d["rescue_share_n"] <= d["guess_eligible_n"]
    assert sum(int(x.split("=")[1]) for x in d["termination_reasons"].split(";")) == R
    # the non-detector algorithms must report N/A, never a fabricated zero
    A2, st2, _ = P.heldout("reverse_eng_risk", scen, 10_000,
                           {"m_exploration": 180, "eps_target": scen["eps"]}, "smoke")
    d2 = D.point_diagnostics(A2, st2, M.ALGORITHMS["reverse_eng_risk"], 50, M.SEED_BOOT)
    assert np.isnan(d2["false_alarm_rate"]) and np.isnan(d2["miss_rate"])
    assert d2["detector_eligible_n"] == 0
    A3, st3, _ = P.heldout("brute", scen, 10_000, {}, "smoke")
    d3 = D.point_diagnostics(A3, st3, M.ALGORITHMS["brute"], 50, M.SEED_BOOT)
    assert d3["guess_eligible_n"] == 0 and np.isnan(d3["guess_ratio_median"])
    assert d3["star_eligible_n"] == len(A3["converged"])


def test_performance_consistency():
    """The convergence count derived from the traced runs must equal the count the untraced
    evaluation produces on the same seeds -- i.e. the diagnostics and the performance curve are the
    same evaluation, not two."""
    scen, cfg, A, st = _one_point()
    seeds = P.seeds_for(M.SEED_TEST, M.MODES["smoke"]["R_test"])
    succ, n = P._task(("binary_deep", scen, 10_000, cfg, seeds, False, False))
    assert n == len(A["converged"])
    assert succ == int(A["converged"].sum()), (succ, int(A["converged"].sum()))
    d = D.point_diagnostics(A, st, M.ALGORITHMS["binary_deep"], 50, M.SEED_BOOT)
    assert d["rate_k"] == succ and d["rate_n"] == n
    assert abs(d["rate"] - succ / n) < 1e-12


# ------------------------------------------------------------------------- 7. CI consistency
def test_ci_uses_row_R():
    from qmetrology.uncertainty import wilson
    for R in (200, 1500, 40_000):
        lo, hi = wilson(0.5, R)
        assert hi - lo > 0
    a = wilson(0.5, 1500)[1] - wilson(0.5, 1500)[0]
    b = wilson(0.5, 40_000)[1] - wilson(0.5, 40_000)[0]
    assert a > b, "a smaller R must give a wider interval"
    # qmetrology/manifest.py is the ONE place a trial count may be written down; every other module
    # must read it from the manifest or from the data row. A literal R anywhere else is the bug this
    # guards against (the pre-existing scripts disagree: 50,000 / 40,000 / 30,000 / 20,000).
    consumers = ["qmetrology/diagnostics.py", "qmetrology/pipeline.py",
                 "analysis/run.py", "analysis/finalize.py"]
    for fn in consumers:
        src = open(os.path.join(ROOT, fn)).read()
        for bad in ("20000", "20_000", "30000", "30_000", "40000", "40_000",
                    "50000", "50_000", "60000", "60_000"):
            assert bad not in src, f"hard-coded trial count {bad} in {fn}"
    # ... and the manifest must actually declare one per mode, so nothing can fall back to a default
    for mode, cfg in M.MODES.items():
        for k in ("R_test", "R_tune", "R_tune2"):
            assert isinstance(cfg.get(k), int) and cfg[k] > 0, (mode, k)
    # the uncertainty layer must take R as an argument, never read a module constant
    import inspect
    from qmetrology import uncertainty as U
    for fname in ("crossing_ci", "ratio_ci"):
        assert "R" in inspect.signature(getattr(U, fname)).parameters, fname


def test_ci_metadata_present():
    scen, cfg, A, st = _one_point()
    d = D.point_diagnostics(A, st, M.ALGORITHMS["binary_deep"], 50, M.SEED_BOOT)
    for k in ("R", "ci_level", "n_boot", "boot_seed", "ci_method_proportion",
              "ci_method_quantile", "ci_method_mean"):
        assert k in d and d[k] not in ("", None), k
    assert d["ci_level"] == 95 and d["boot_seed"] == M.SEED_BOOT


# ------------------------------------------------------------- 9. thesis/code parity (audit)
def test_code_audit_names_the_implementations():
    import audit
    txt = audit.build(run_comparison=False)
    for algo in M.ORDER:
        assert M.ALGORITHMS[algo]["variant"] in txt, algo
    assert "find_phi_fixed_budget_binary_search_deep" in txt
    assert "find_phi_fixed_budget_binary_search_risk" in txt
    assert "deepest" in txt.lower() and "opening probe" in txt
    # the manifest must point at the deepest-probe binary search, not the opening-probe one
    assert M.ALGORITHMS["binary_deep"]["fn"] is ALG.find_phi_fixed_budget_binary_search_deep
    assert set(M.ORDER) == {"brute", "linear", "binary_deep", "reverse_eng_risk"}


# ------------------------------------------------ 8./10. end-to-end: provenance, modes, resume
def _read(d, name):
    with open(os.path.join(d, name), newline="") as f:
        return list(csv.DictReader(f))


def _sweep(outdir, extra=()):
    env = dict(os.environ, RESULTS_OUT=outdir)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis/run.py"),
                        "--smoke", *extra], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "FAILED" not in r.stdout, r.stdout[-3000:]
    return r


def test_end_to_end_smoke_and_provenance():
    if not SLOW:
        print("   (skipped: pass --slow)")
        return
    d = tempfile.mkdtemp(prefix="consolidated_")
    try:
        _sweep(d)
        perf, wins = _read(d, "performance_curves.csv"), _read(d, "winners.csv")
        assert perf and wins
        # 8. winner provenance: exactly one frozen winner per held-out row, same params
        key = lambda r: (r["scenario_id"], r["algorithm"], r["budget"])
        wmap = {}
        for w in wins:
            assert key(w) not in wmap, f"duplicate winner for {key(w)}"
            wmap[key(w)] = w
        for r in perf:
            if r["algorithm"] == "oracle_hl":
                assert key(r) not in wmap, "the analytic oracle must not have a simulated winner"
                assert r["R"] == "" and r["rate_lo"] == "" and r["rate_hi"] == ""
                continue
            w = wmap[key(r)]
            assert w["params"] == r["params"], (key(r), w["params"], r["params"])
            assert abs(float(w["heldout_rate"]) - float(r["rate"])) < 1e-12
            assert w["seed_tune_blocks"] == ",".join(str(x) for x in M.SEED_TUNE_BLOCKS)
            if w["tuned"] == "True":
                assert json.loads(w["block_rates"]), key(r)
        # every reported R is the manifest's, and appears on every derived row
        assert {int(r["R"]) for r in perf if r["R"]} == {M.MODES["smoke"]["R_test"]}
        for name in ("budget_crossings.csv", "detector_confusion.csv", "budget_audit.csv",
                     "diagnostics_by_point.csv", "diagnostics_headline.csv", "optimal_params.csv",
                     "diagnostics_by_phase.csv"):
            assert _read(d, name), name
        for name in ("REPORT.md", "experiment_manifest.json", "algorithm_code_audit.md"):
            assert os.path.exists(os.path.join(d, name)) or name == "algorithm_code_audit.md"
        cross = _read(d, "budget_crossings.csv")
        perf_R = {int(r["R"]) for r in perf if r["R"]}
        assert {int(c["R"]) for c in cross if int(c["R"]) > 0} <= perf_R, \
            "crossing CI must use the curve's own R"
        assert all(c["crossing_rule"] for c in cross)
        # appendix rows must state which tested budget a B_90 configuration belongs to
        par = _read(d, "optimal_params.csv")
        tags = {p["reported_for"] for p in par}
        assert {"tested_budget", "B90_nearest", "B90_bracket_lo", "B90_bracket_hi"} <= tags, tags
        for p in par:
            if p["reported_for"] != "tested_budget":
                assert "B_90" in p["budget_note"], p["budget_note"]
        # budget audit must be able to state compliance
        aud = _read(d, "budget_audit.csv")
        assert sum(int(a["budget_violations"]) for a in aud) == 0
        assert max(float(a["budget_util_max"]) for a in aud) <= 1.0 + 1e-9
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reproducibility_and_resume():
    if not SLOW:
        print("   (skipped: pass --slow)")
        return
    d1, d2 = tempfile.mkdtemp(prefix="c1_"), tempfile.mkdtemp(prefix="c2_")
    try:
        _sweep(d1)
        _sweep(d2)
        a, b = _read(d1, "performance_curves.csv"), _read(d2, "performance_curves.csv")
        assert a == b, "same manifest + same seeds must reproduce identical outputs"
        assert _read(d1, "diagnostics_by_point.csv") == _read(d2, "diagnostics_by_point.csv")
        # resume: drop the last scenario's rows, re-run with --resume, expect the same file back
        done = {r["scenario_id"] for r in a}
        last = [s["id"] for s in M.SCENARIOS if s["id"] in done][-1]
        for name in ("performance_curves.csv", "winners.csv", "diagnostics_by_point.csv",
                     "detector_confusion.csv", "budget_audit.csv", "diagnostics_by_phase.csv"):
            rows = _read(d2, name)
            keep = [r for r in rows if r.get("scenario_id") != last]
            with open(os.path.join(d2, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(keep)
        _sweep(d2, extra=("--resume",))
        c = _read(d2, "performance_curves.csv")
        assert sorted(map(str, a), key=str) == sorted(map(str, c), key=str), \
            "--resume must reproduce the dropped scenario exactly"
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


# ------------------------------------------------------- linear-search detector replay (11)
def _lds():
    import linear_detector_study as L
    return L


def test_lds_conv_prob_matches_oracle():
    """The analytic exploitation must be the same quantity the oracle rows are built from."""
    L = _lds()
    rng = np.random.default_rng(11)
    for _ in range(80):
        phi = float(rng.uniform(0.001, 0.5))
        N, B = int(rng.integers(1, 400)), int(rng.integers(100, 10 ** 6))
        eps = float(10.0 ** rng.uniform(-6, -2))
        a = float(O.convergence_prob(N, phi, B, eps)[0])
        b = float(L.conv_prob(np.array([N]), np.array([phi]), np.array([B]), eps)[0])
        assert abs(a - b) < 1e-12, f"conv_prob disagrees with the oracle at N={N}, phi={phi}"


def test_lds_moving_window_is_a_lagged_comparison():
    """Once the trailing window is full, 'the window mean fell' IS 'phi_hat_k < phi_hat_{k-w}'.

    The study relies on this to describe the supervisor's moving-window rule as a member of the
    Eq. (3.6) family rather than as an averaging rule.
    """
    L = _lds()
    P = np.random.default_rng(3).random((500, 30))
    for w in (1, 2, 3, 5, 8):
        fell = L._falls(L._trailing_mean(P, w))
        direct = np.zeros_like(fell)
        direct[:, w:] = P[:, w:] < P[:, :-w]
        assert (fell[:, w:] == direct[:, w:]).all(), f"window identity fails at w={w}"


def test_lds_replay_reproduces_algorithm_4():
    """Replaying a recorded scan must give Algorithm 4's own N_guess and N_star, exactly.

    This is the load-bearing assumption of the bake-off: the depths a scan can afford are fixed
    before any datum is seen, so every candidate rule can be scored on one recorded probe stream.
    """
    L = _lds()
    pmin, pmax, eps = 0.01, 0.1, 1e-3
    checked = 0
    for B in (10_000, 41_326):
        for cfg in ({"m_exploration": 2, "lookback_window": 4, "safeguard": 1, "inc": 1},
                    {"m_exploration": 15, "lookback_window": 3, "safeguard": 2, "inc": 2},
                    {"m_exploration": 9, "lookback_window": 1, "safeguard": 0, "inc": 3}):
            depths = L.scan_depths(max(1, int(np.pi // (2 * pmax))),
                                   max(1, int(np.pi // (2 * pmin))),
                                   cfg["inc"], cfg["m_exploration"], B)
            C = L.context(depths, cfg["m_exploration"], cfg["inc"])
            for seed in range(40):
                tr = AlgorithmTrace(algorithm="linear")
                rng = np.random.default_rng(seed)
                phi = float(rng.uniform(pmin, pmax))
                tr.nominal_budget = float(B)
                out = ALG.find_phi_fixed_budget_linear_search(
                    rng, phi, pmax, pmin, trace=tr, budget=B, **cfg)
                tr.finalize(phi, out[0], out[1], eps)
                n = len(tr.probes)
                assert list(depths[:n]) == [p.N for p in tr.probes], \
                    "scan_depths does not reproduce the depths the algorithm probes"
                # pad the unseen tail with +inf: it can neither fall nor fire, so a rule that did
                # not fire within the recorded probes still does not fire here
                Pm = np.full((1, len(depths)), np.inf)
                Pm[0, :n] = [p.phi_hat for p in tr.probes]
                fire, back = L.rule_cumulative(Pm, C, cfg["lookback_window"])
                o = L.outcome(fire, back, C, np.array([phi]), Pm, cfg["safeguard"], eps, B)
                assert int(o["N_guess"][0]) == int(tr.N_guess), \
                    f"replayed N_guess {o['N_guess'][0]} != {tr.N_guess} (B={B}, cfg={cfg})"
                if tr.N_star is not None:
                    assert int(o["N_star"][0]) == int(tr.N_star), \
                        f"replayed N_star {o['N_star'][0]} != {tr.N_star} (B={B}, cfg={cfg})"
                assert bool(o["false_alarm"][0]) == bool(run_record(tr, True)["false_alarm"]), \
                    "replayed false-alarm verdict differs from the trace's"
                assert bool(o["miss"][0]) == bool(run_record(tr, True)["miss"]), \
                    "replayed miss verdict differs from the trace's"
                checked += 1
    assert checked >= 200, "the replay check did not run over enough trials"


def test_lds_replay_reproduces_lagged_variant():
    """The sweep-ready lagged variant and the bake-off's model of it must be the same algorithm.

    `find_phi_fixed_budget_linear_search_lagged` is what a production sweep would run if the
    stopping rule were replaced; `rule_threshold` is what the bake-off scored. If they ever drift
    apart, the measured gain would not be the gain a re-run would deliver.
    """
    L = _lds()
    pmin, pmax, eps = 0.01, 0.1, 1e-3
    checked = 0
    for B in (10_000, 41_326):
        for cfg in ({"m_exploration": 2, "lookback_window": 4, "safeguard": 1, "inc": 1,
                     "lag": 2, "conf": 0.5},
                    {"m_exploration": 9, "lookback_window": 6, "safeguard": 2, "inc": 2,
                     "lag": 3, "conf": 0.95}):
            depths = L.scan_depths(max(1, int(np.pi // (2 * pmax))),
                                   max(1, int(np.pi // (2 * pmin))),
                                   cfg["inc"], cfg["m_exploration"], B)
            C = L.context(depths, cfg["m_exploration"], cfg["inc"])
            for seed in range(40):
                tr = AlgorithmTrace(algorithm="linear_lagged")
                rng = np.random.default_rng(seed)
                phi = float(rng.uniform(pmin, pmax))
                tr.nominal_budget = float(B)
                out = ALG.find_phi_fixed_budget_linear_search_lagged(
                    rng, phi, pmax, pmin, trace=tr, budget=B, **cfg)
                tr.finalize(phi, out[0], out[1], eps)
                n = len(tr.probes)
                Pm = np.full((1, len(depths)), np.inf)
                Pm[0, :n] = [p.phi_hat for p in tr.probes]
                fire, back = L.rule_threshold(Pm, C, alpha=1 - cfg["conf"],
                                              l=cfg["lookback_window"], lag=cfg["lag"])
                o = L.outcome(fire, back, C, np.array([phi]), Pm, cfg["safeguard"], eps, B)
                assert int(o["N_guess"][0]) == int(tr.N_guess), \
                    f"lagged replay N_guess {o['N_guess'][0]} != {tr.N_guess} (B={B}, cfg={cfg})"
                if tr.N_star is not None:
                    assert int(o["N_star"][0]) == int(tr.N_star), \
                        f"lagged replay N_star {o['N_star'][0]} != {tr.N_star}"
                checked += 1
    assert checked >= 150, "the lagged replay check did not run over enough trials"


def test_linear_mean_window_default_is_cumulative():
    """`mean_window` must be a strict generalisation: w = 0 reproduces the published Algorithm 4.

    The parameter was added to the tuning grid, not substituted for the old rule, so the tuner can
    never select something worse than what the thesis already reports. That only holds if w = 0 is
    bit-for-bit the previous behaviour.
    """
    pmin, pmax, eps = 0.01, 0.1, 1e-3
    for B in (10_000, 41_326):
        for cfg in ({"m_exploration": 2, "lookback_window": 4, "safeguard": 1, "inc": 1},
                    {"m_exploration": 15, "lookback_window": 3, "safeguard": 2, "inc": 2}):
            for seed in range(60):
                r1 = np.random.default_rng(seed)
                phi = float(r1.uniform(pmin, pmax))
                a = ALG.find_phi_fixed_budget_linear_search(r1, phi, pmax, pmin, budget=B, **cfg)
                r2 = np.random.default_rng(seed)
                phi2 = float(r2.uniform(pmin, pmax))
                b = ALG.find_phi_fixed_budget_linear_search(r2, phi2, pmax, pmin, budget=B,
                                                            mean_window=0, **cfg)
                assert a == b, f"mean_window=0 changed the published behaviour (B={B}, cfg={cfg})"
    # and the grid really does contain it, so the tuner can fall back on it
    assert 0 in M.ALGORITHMS["linear"]["discrete"]["mean_window"], \
        "the mean_window grid must contain 0, the cumulative-mean rule"


def test_linear_mean_window_matches_the_bakeoff_rule():
    """A finite `mean_window` must be the same rule the detector study scored as `window`."""
    L = _lds()
    pmin, pmax, eps = 0.01, 0.1, 1e-3
    B, checked = 10_000, 0
    for w, l, s_, m, inc in ((3, 6, 0, 1, 1), (2, 4, 0, 5, 2), (6, 12, 3, 1, 2)):
        depths = L.scan_depths(max(1, int(np.pi // (2 * pmax))), max(1, int(np.pi // (2 * pmin))),
                               inc, m, B)
        C = L.context(depths, m, inc)
        for seed in range(40):
            tr = AlgorithmTrace(algorithm="linear")
            rng = np.random.default_rng(seed)
            phi = float(rng.uniform(pmin, pmax))
            tr.nominal_budget = float(B)
            out = ALG.find_phi_fixed_budget_linear_search(
                rng, phi, pmax, pmin, trace=tr, budget=B, m_exploration=m, lookback_window=l,
                safeguard=s_, inc=inc, mean_window=w)
            tr.finalize(phi, out[0], out[1], eps)
            n = len(tr.probes)
            Pm = np.full((1, len(depths)), np.inf)
            Pm[0, :n] = [p.phi_hat for p in tr.probes]
            fire, back = L.rule_window(Pm, C, w=w, l=l)
            o = L.outcome(fire, back, C, np.array([phi]), Pm, s_, eps, B)
            assert int(o["N_guess"][0]) == int(tr.N_guess), \
                f"mean_window={w} disagrees with the bake-off's `window` rule"
            checked += 1
    assert checked >= 100


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        sys.stdout.write(f"  {name:52s} ")
        sys.stdout.flush()
        try:
            fn()
            print("ok")
        except AssertionError as e:
            fails += 1
            print(f"FAIL\n      {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed"
          + ("" if SLOW else "   (run with --slow for the end-to-end checks)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
