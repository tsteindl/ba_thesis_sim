"""Instrumentation for the phase-search algorithms (ALGORITHM_DIAGNOSTICS_HANDOFF.md).

The algorithms keep their public `(phi_hat, budget_used)` signature. Passing `trace=Trace(...)`
makes them *additionally* record what they did; nothing in here draws a random number, so a
same-seed run with and without a trace returns identical `phi_hat` and budget use
(tests/test_consolidated.py::test_trace_neutrality asserts it).

Two objects:

  ProbeTrace       one exploration measurement -- its depth, shot count, estimate, whether the
                   algorithm's own detector declared it an overshoot, and (filled in afterwards by
                   `AlgorithmTrace.finalize`, which is simulation-side) whether it really was one.
  AlgorithmTrace   one trial -- the budget split, the exploration guess N_guess, the exploitation
                   depth N_star, and the probe list.

`N_guess` is the exploration phase's answer to "how deep should the exploitation go", recorded
BEFORE any safeguard is applied; `N_star` is the depth the exploitation actually ran at. The two
differ by exactly the safeguard, which is the point of measuring both. Their per-algorithm
definitions live with the algorithms and are asserted in tests/test_consolidated.py.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


def n_opt(phi):
    """N_opt = floor(pi / (2 phi)), floored at 1 -- the deepest non-aliasing depth for phi.

    Same definition (and same lower bound) the simulation uses for N_min/N_max, so `N_guess/N_opt`
    is comparable across algorithms and scenarios.
    """
    if not np.isfinite(phi) or phi <= 0:
        return 1
    return max(int(np.pi // (2.0 * phi)), 1)


@dataclass
class ProbeTrace:
    stage_index: int
    N: int
    m: int
    phi_hat: float
    declared_overshoot: Optional[bool] = None   # None where the algorithm has no detector
    accepted: Optional[bool] = None             # None where "accepted" is not meaningful
    true_overshoot: Optional[bool] = None       # N > N_opt; simulation ground truth, filled later


@dataclass
class AlgorithmTrace:
    algorithm: str = ""
    implementation_variant: str = ""
    phi: float = float("nan")
    phi_hat_final: float = float("nan")
    converged: Optional[bool] = None
    nominal_budget: float = float("nan")
    budget_used_total: float = 0.0
    budget_exploration: float = 0.0
    N_opt: Optional[int] = None
    N_guess: Optional[int] = None          # null for the non-search baselines
    N_star: Optional[int] = None           # null if the exploitation never ran
    m_final: Optional[int] = None
    status: str = ""                       # termination reason
    opening_pilot_N: Optional[int] = None
    opening_pilot_phi_hat: Optional[float] = None
    accepted_pilot_N: Optional[int] = None
    accepted_pilot_phi_hat: Optional[float] = None
    probes: List[ProbeTrace] = field(default_factory=list)

    # -- recording API used by the algorithms -------------------------------------------------
    def probe(self, N, m, phi_hat, declared_overshoot=None, accepted=None):
        self.probes.append(ProbeTrace(len(self.probes), int(N), int(m), float(phi_hat),
                                      declared_overshoot, accepted))

    def set_exploration(self, budget_exploration, N_guess=None, status=None):
        self.budget_exploration = float(budget_exploration)
        if N_guess is not None:
            self.N_guess = int(N_guess)
        if status is not None:
            self.status = status

    def set_exploitation(self, N_star, m_final):
        self.N_star = int(N_star)
        self.m_final = int(m_final)

    # -- simulation-side completion -----------------------------------------------------------
    def finalize(self, phi, phi_hat, budget_used, eps):
        """Attach the ground truth the algorithm is not allowed to see."""
        self.phi = float(phi)
        self.phi_hat_final = float(phi_hat)
        self.budget_used_total = float(budget_used)
        self.N_opt = n_opt(phi)
        self.converged = bool(np.isfinite(phi_hat) and abs(phi_hat - phi) < eps)
        for p in self.probes:
            p.true_overshoot = bool(p.N > self.N_opt)
        if not self.status:
            self.status = "ok"
        return self

    def to_dict(self):
        return asdict(self)


# Per-run scalars extracted from a trace. This is what the production sweep keeps in memory:
# storing full probe lists for tens of millions of trials is neither necessary nor affordable
# (--keep-traces writes them for selected points instead).
RUN_FIELDS = [
    "converged", "N_opt", "N_guess", "N_star", "m_final",
    "budget_used_total", "budget_exploration", "nominal_budget",
    "n_probes", "n_accepted", "phi", "phi_hat_final",
    "opening_pilot_N", "accepted_pilot_N",
    "false_alarm", "miss", "tp", "fp", "tn", "fn", "status",
]


def run_record(tr, has_detector):
    """Collapse one AlgorithmTrace into the scalars every diagnostic is computed from.

    `has_detector` says whether the algorithm classifies probes at all (linear and binary do;
    reverse engineering and the baselines do not, and get NaN rather than a fabricated zero).
    """
    nan = float("nan")
    tp = fp = tn = fn = 0
    fa = miss = nan
    if has_detector and tr.probes:
        fa_hit = miss_hit = False
        for p in tr.probes:
            if p.declared_overshoot is None:
                continue
            if p.declared_overshoot and p.true_overshoot:
                tp += 1
            elif p.declared_overshoot and not p.true_overshoot:
                fp += 1
                fa_hit = True
            elif (not p.declared_overshoot) and p.true_overshoot:
                fn += 1
                miss_hit = True
            else:
                tn += 1
        fa, miss = float(fa_hit), float(miss_hit)
    return {
        "converged": float(bool(tr.converged)),
        "N_opt": float(tr.N_opt if tr.N_opt is not None else nan),
        "N_guess": float(tr.N_guess) if tr.N_guess is not None else nan,
        "N_star": float(tr.N_star) if tr.N_star is not None else nan,
        "m_final": float(tr.m_final) if tr.m_final is not None else nan,
        "budget_used_total": float(tr.budget_used_total),
        "budget_exploration": float(tr.budget_exploration),
        "nominal_budget": float(tr.nominal_budget),
        "n_probes": float(len(tr.probes)),
        "n_accepted": float(sum(1 for p in tr.probes if p.accepted)),
        "phi": float(tr.phi),
        "phi_hat_final": float(tr.phi_hat_final),
        "opening_pilot_N": float(tr.opening_pilot_N) if tr.opening_pilot_N is not None else nan,
        "accepted_pilot_N": float(tr.accepted_pilot_N) if tr.accepted_pilot_N is not None else nan,
        "false_alarm": fa, "miss": miss,
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "status": tr.status,
    }
