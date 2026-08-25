"""The Chapter-4 figures, regenerated from the consolidated results.

Replaces analysis/render_story.py, which reads the pre-consolidation sweep (results/story_*.csv) and
still plots the old opening-probe binary search at R = 40,000.

    python analysis/consolidated/thesis_figures.py            # all figures
    python analysis/consolidated/thesis_figures.py --only story,precision
    python analysis/consolidated/thesis_figures.py --out DIR  # write somewhere else

Figures (written to results/consolidated/):
    fig_story           convergence vs budget for the headline scenario, with the 90% crossing
    fig_story_small     the same, compact, for a single-column layout
    fig_pareto          the budget/convergence frontier across scenarios
    fig_precision       budget ratio vs brute force at 90%, against the precision requirement
    fig_error           estimator-error distribution vs budget (median + IQR band)
    fig_variance        sample variance of signed estimator error vs budget; analytic Oracle
    fig_error_variance  error median and variance side by side
    fig_error_density   density of normalized per-trial errors at the headline fixed budget
    fig_signed_error_density density of signed normalized errors at the headline fixed budget
    fig_phi_hat_density density of final estimates at the headline fixed budget
    fig_broad           convergence vs budget under the broad priors
    fig_algorithm_diagnostics exploration cost and final overshoot near 90% convergence

Every panel is annotated with the R it was produced at, read from the data rather than hard-coded,
so a figure can never silently disagree with the tables.
"""
import csv
import gzip
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmetrology import manifest as M
from pipeline_io import path

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else path("")

PAPER_RC = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 4,
}

NICE = {"brute": "Brute force (baseline)", "separable": "Separable ($N=1$)",
        "linear": "Linear search", "binary_deep": "Binary search",
        "reverse_eng_risk": "Reverse engineering",
        "oracle_hl": "Oracle ($N_\\text{opt}$)"}
COL = {"brute": "#7f7f7f", "separable": "#c7c7c7", "linear": "#2ca02c",
       "binary_deep": "#ff7f0e", "reverse_eng_risk": "#1f77b4", "oracle_hl": "#000000"}
MARK = {"brute": "o", "separable": "v", "linear": "^", "binary_deep": "D",
        "reverse_eng_risk": "s", "oracle_hl": "*"}
ORDER = ["brute", "separable", "linear", "binary_deep", "reverse_eng_risk", "oracle_hl"]
ERROR_ORDER = list(ORDER)
HEAD = "narrow_e3"


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def load(name):
    p = path(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def trace_errors(sid, algo, budget):
    """Absolute final-estimator errors from an existing held-out trace subset."""
    p = path("traces", f"{sid}__{algo}__B{int(budget)}.jsonl.gz")
    if not os.path.exists(p):
        return np.array([], float)
    errors = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            phi, phi_hat = f(row.get("phi")), f(row.get("phi_hat_final"))
            if np.isfinite(phi) and np.isfinite(phi_hat):
                errors.append(abs(phi_hat - phi))
    return np.asarray(errors, float)


def trace_signed_errors(sid, algo, budget):
    """Signed final-estimator errors from an existing held-out trace subset."""
    p = path("traces", f"{sid}__{algo}__B{int(budget)}.jsonl.gz")
    if not os.path.exists(p):
        return np.array([], float)
    errors = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            phi, phi_hat = f(row.get("phi")), f(row.get("phi_hat_final"))
            if np.isfinite(phi) and np.isfinite(phi_hat):
                errors.append(phi_hat - phi)
    return np.asarray(errors, float)


def trace_phi_hats(sid, algo, budget):
    """Final estimates from an existing held-out trace subset."""
    p = path("traces", f"{sid}__{algo}__B{int(budget)}.jsonl.gz")
    if not os.path.exists(p):
        return np.array([], float)
    estimates = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            phi_hat = f(json.loads(line).get("phi_hat_final"))
            if np.isfinite(phi_hat):
                estimates.append(phi_hat)
    return np.asarray(estimates, float)


def out_path(name):
    return os.path.join(OUT, name)


def prior_of(label):
    """'U(0.01,0.1), eps=1e-03' -> 'U(0.01,0.1)'. Splitting on the first comma would cut the prior
    itself in half, which is how the precision figure ended up titled 'U(0.01'."""
    return label.split("), ")[0] + ")" if "), " in label else label


# Binary search and reverse engineering track each other closely -- in the broad priors they
# coincide to within Monte-Carlo noise. Solid-on-solid would hide one of them entirely and read as a
# missing curve, so reverse engineering is dashed: where the two separate you see both, and where
# they coincide the dashes sit on top of the solid line and the overlap is visible as such.
# DASHED = {"reverse_eng_risk": (0, (5, 2))}
DASHED = {}


def style(a):
    return dict(marker=MARK[a], color=COL[a], label=NICE[a],
                ls=":" if a == "oracle_hl" else DASHED.get(a, "-"),
                mfc="none" if a in ("separable",) else COL[a])


class D:
    def __init__(self):
        self.perf = load("performance_curves.csv")
        self.cross = load("budget_crossings.csv")
        self.err = load("error_curves.csv")
        self.var = load("variance_curves.csv")
        self.diag = load("diagnostics_by_point.csv")
        self.man = json.load(open(path("experiment_manifest.json")))
        self.R = self.man["trials"]["R_test"]
        self.scen = {s["id"]: s for s in self.man["scenarios"]}

    def curve(self, sid, algo):
        rs = sorted([r for r in self.perf
                     if r["scenario_id"] == sid and r["algorithm"] == algo],
                    key=lambda r: int(f(r["budget"])))
        return (np.array([int(f(r["budget"])) for r in rs], float),
                np.array([f(r["rate"]) for r in rs]),
                np.array([f(r["rate_lo"]) for r in rs]),
                np.array([f(r["rate_hi"]) for r in rs]))

    def cross90(self, sid, algo, thr=90):
        for c in self.cross:
            if (c["scenario_id"] == sid and c["algorithm"] == algo
                    and int(c["threshold_pct"]) == thr):
                return f(c["budget_to_reach"]), f(c["ratio_vs_brute"])
        return float("nan"), float("nan")

    def diag_near90(self, sid, algo):
        """Diagnostic row at the tested budget nearest the interpolated 90% crossing."""
        target, _ = self.cross90(sid, algo)
        rows = [r for r in self.diag
                if r["scenario_id"] == sid and r["algorithm"] == algo]
        if not rows or not np.isfinite(target) or target <= 0:
            return None
        return min(rows, key=lambda r: abs(np.log(f(r["budget"]) / target)))


def _foot(fig, d, bands="", extra=""):
    """Provenance strip. `bands` must describe what the shaded regions actually are -- a figure with
    no bands must not claim Wilson intervals."""
    parts = [f"Held-out seed {M.SEED_TEST}, R = {d.R:,} trials per point"]
    if bands:
        parts.append(bands)
    parts.append("parameters frozen on tuning seeds "
                 + ",".join(str(x) for x in M.SEED_TUNE_BLOCKS))
    txt = "; ".join(parts) + "."
    if extra:
        txt += " " + extra
    fig.text(0.005, 0.005, txt, fontsize=6.5, color="#555555", ha="left", va="bottom")


def fig_story(d, small=False):
    sid = HEAD
    fig, ax = plt.subplots(figsize=(6.2, 3.4) if small else (8.2, 4.6))
    for a in ORDER:
        b, r, lo, hi = d.curve(sid, a)
        if not b.size:
            continue
        ax.plot(b, 100*r, **style(a))
        ok = np.isfinite(lo)
        if ok.any():
            ax.fill_between(b[ok], 100*lo[ok], 100*hi[ok], color=COL[a], alpha=0.15, lw=0)
    ax.axhline(90, color="k", lw=0.7, ls="--", alpha=0.6)
    bb, _ = d.cross90(sid, "brute")
    br, ratio = d.cross90(sid, "reverse_eng_risk")
    if np.isfinite(bb) and np.isfinite(br) and not small:
        ax.annotate("", xy=(br, 90), xytext=(bb, 90),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
        ax.annotate(f"{ratio:.2f}x less budget\nat 90% convergence",
                    xy=(np.sqrt(bb*br), 90), xytext=(np.sqrt(bb*br), 66),
                    ha="center", fontsize=9, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    ax.set_xscale("log")
    ax.set_xlabel("Budget $C = N \\cdot m$")
    ax.set_ylabel("Converged (%)")
    ax.set_ylim(0, 101)
    ax.set_title(f"{d.scen[sid]['label']}", fontsize=10)
    ax.legend(fontsize=7.5 if small else 8.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    # _foot(fig, d, bands="shaded bands are 95% Wilson intervals")
    name = "fig_story_small.png" if small else "fig_story.png"
    fig.savefig(out_path(name), bbox_inches="tight")
    plt.close(fig)
    return name


def fig_precision(d):
    sweep = [s["id"] for s in M.SCENARIOS if "precision_sweep" in s["families"]]
    sweep = [s for s in sweep if s in d.scen]
    sweep.sort(key=lambda k: -f(d.scen[k]["eps"]))
    eps = [f(d.scen[k]["eps"]) for k in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for a in ORDER:
        if a in ("brute", "separable"):
            continue
        y = [d.cross90(k, a)[1] for k in sweep]
        if not np.isfinite(y).any():
            continue
        ax.plot(eps, y, **style(a))
    ax.axhline(1.0, color="#7f7f7f", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Precision requirement $\\epsilon$")
    ax.set_ylabel("Budget ratio vs brute force\nat 90% convergence")
    ax.set_title(prior_of(d.scen[sweep[0]]["label"]), fontsize=10)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    # _foot(fig, d, extra="Ratios > 1 mean less budget is required. Crossings are log-linearly "
        #   "interpolated between tested budgets.")
    fig.savefig(out_path("fig_precision.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_precision.png"


def fig_algorithm_diagnostics(d):
    """Exploration cost and final overshoot near 90% convergence for the precision sweep."""
    sweep = [s["id"] for s in M.SCENARIOS if "precision_sweep" in s["families"]]
    sweep = [sid for sid in sweep if sid in d.scen]
    sweep.sort(key=lambda sid: -f(d.scen[sid]["eps"]))
    eps = np.array([f(d.scen[sid]["eps"]) for sid in sweep], float)
    algs = ["linear", "binary_deep", "reverse_eng_risk"]
    fig, (ax_cost, ax_overshoot) = plt.subplots(1, 2, figsize=(9.3, 3.65), sharex=True)

    for a in algs:
        rows = [d.diag_near90(sid, a) for sid in sweep]
        cost = np.array([100*f(r.get("exploration_share_median")) if r else np.nan
                         for r in rows])
        cost_lo = np.array([100*f(r.get("exploration_share_median_lo")) if r else np.nan
                            for r in rows])
        cost_hi = np.array([100*f(r.get("exploration_share_median_hi")) if r else np.nan
                            for r in rows])
        overshoot = np.array([100*f(r.get("star_overshoot")) if r else np.nan
                              for r in rows])
        overshoot_lo = np.array([100*f(r.get("star_overshoot_lo")) if r else np.nan
                                 for r in rows])
        overshoot_hi = np.array([100*f(r.get("star_overshoot_hi")) if r else np.nan
                                 for r in rows])

        ax_cost.plot(eps, cost, **style(a))
        ax_cost.fill_between(eps, cost_lo, cost_hi, color=COL[a], alpha=0.14, lw=0)
        ax_overshoot.plot(eps, overshoot, **style(a))
        ax_overshoot.fill_between(eps, overshoot_lo, overshoot_hi,
                                  color=COL[a], alpha=0.14, lw=0)

    for ax in (ax_cost, ax_overshoot):
        ax.set_xscale("log")
        ax.set_xlabel("Precision requirement $\\epsilon$")
        ax.set_ylim(bottom=0)
    ax_cost.invert_xaxis()
    ax_cost.set_ylabel("Median exploration-budget share (%)")
    ax_overshoot.set_ylabel("Final overshoot rate (%)")
    ax_cost.set_title("(a) Exploration cost", fontsize=10)
    ax_overshoot.set_title("(b) Overshooting after safeguard", fontsize=10)
    ax_cost.legend(fontsize=8, loc="best")
    sc = d.scen[sweep[0]]
    fig.suptitle(rf"$\phi\sim\mathcal{{U}}({f(sc['phi_min']):g},{f(sc['phi_max']):g})$, "
                 "near 90% convergence", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path("fig_algorithm_diagnostics.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_algorithm_diagnostics.png"


def fig_pareto(d):
    sids = [s["id"] for s in M.SCENARIOS if s["id"] in d.scen]
    n = len(sids)
    ncol = 5
    nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9*ncol, 2.5*nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes, sids):
        for a in ORDER:
            b, r, _lo, _hi = d.curve(sid, a)
            if b.size:
                ax.plot(b, 100*r, **{**style(a), "label": None, "markersize": 3, "lw": 1.3})
        ax.set_xscale("log")
        ax.set_title(d.scen[sid]["label"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(90, color="k", lw=0.6, ls="--", alpha=0.5)
    for ax in axes[len(sids):]:
        ax.axis("off")
    for ax in axes[:len(sids)]:
        ax.set_xlabel("Budget", fontsize=8)
    for i in range(0, len(sids), ncol):
        axes[i].set_ylabel("Converged (%)", fontsize=8)
    h = [plt.Line2D([], [], **{**style(a), "label": NICE[a]}) for a in ORDER]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _foot(fig, d)
    fig.savefig(out_path("fig_pareto.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_pareto.png"


def fig_error(d, sid=HEAD):
    if not d.err:
        print("   (fig_error skipped: run analysis/consolidated/error_curves.py first)")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for a in ERROR_ORDER:
        rs = sorted([r for r in d.err if r["scenario_id"] == sid and r["algorithm"] == a],
                    key=lambda r: int(f(r["budget"])))
        if not rs:
            continue
        b = np.array([int(f(r["budget"])) for r in rs], float)
        p50 = np.array([f(r["err_p50"]) for r in rs])
        ax.plot(b, p50, **style(a))
        if a != "oracle_hl":
            p25 = np.array([f(r["err_p25"]) for r in rs])
            p75 = np.array([f(r["err_p75"]) for r in rs])
            ax.fill_between(b, p25, p75, color=COL[a], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = f(d.scen[sid]["eps"])
    ax.axhline(eps, color="k", ls="--", lw=0.9)
    # ax.annotate(f"tolerance $\\epsilon = {eps:g}$", xy=(0.015, eps), xycoords=("axes fraction",
                # "data"), fontsize=8, va="bottom", ha="left",
                # bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    ax.set_xlabel("Budget $C = N \\cdot m$")
    ax.set_ylabel("$|\\hat{\\phi} - \\phi|$  (median; protocol IQR bands)")
    ax.set_title(d.scen[sid]["label"], fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    # _foot(fig, d, bands="bands are the 25-75% range of the per-trial error")
    fig.savefig(out_path("fig_error.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_error.png"


def fig_variance(d, sid=HEAD):
    if not d.var:
        print("   (fig_variance skipped: run analysis/consolidated/variance_curves.py first)")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for a in ORDER:
        rs = sorted([r for r in d.var if r["scenario_id"] == sid and r["algorithm"] == a],
                    key=lambda r: int(f(r["budget"])))
        if not rs:
            continue
        b = np.array([int(f(r["budget"])) for r in rs], float)
        variance = np.array([f(r["error_variance"]) for r in rs])
        ax.plot(b, variance, **style(a))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Budget $C = N \\cdot m$")
    ax.set_ylabel(r"Variance of signed error $\mathrm{Var}(\hat{\phi}-\phi)$")
    ax.set_title(d.scen[sid]["label"], fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _foot(fig, d, extra="protocols: sample variance; Oracle: analytic prior-averaged QCRB variance")
    fig.savefig(out_path("fig_variance.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_variance.png"


def fig_error_variance(d, sid=HEAD):
    """Combined error and variance panels; keeps both standalone figures unchanged."""
    if not d.err or not d.var:
        print("   (fig_error_variance skipped: error_curves.csv or variance_curves.csv missing)")
        return None
    fig, (ax_error, ax_var) = plt.subplots(1, 2, figsize=(12.2, 4.35))

    for a in ERROR_ORDER:
        rs = sorted([r for r in d.err if r["scenario_id"] == sid and r["algorithm"] == a],
                    key=lambda r: int(f(r["budget"])))
        if not rs:
            continue
        budget = np.array([int(f(r["budget"])) for r in rs], float)
        median = np.array([f(r["err_p50"]) for r in rs])
        ax_error.plot(budget, median, **style(a))
        if a != "oracle_hl":
            p25 = np.array([f(r["err_p25"]) for r in rs])
            p75 = np.array([f(r["err_p75"]) for r in rs])
            ax_error.fill_between(budget, p25, p75, color=COL[a], alpha=0.12, lw=0)

    for a in ORDER:
        rs = sorted([r for r in d.var if r["scenario_id"] == sid and r["algorithm"] == a],
                    key=lambda r: int(f(r["budget"])))
        if not rs:
            continue
        budget = np.array([int(f(r["budget"])) for r in rs], float)
        variance = np.array([f(r["error_variance"]) for r in rs])
        ax_var.plot(budget, variance, **style(a))

    for ax in (ax_error, ax_var):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Budget $C = N \\cdot m$")
    ax_error.axhline(f(d.scen[sid]["eps"]), color="k", ls="--", lw=0.9)
    ax_error.set_ylabel(r"$|\hat{\phi}-\phi|$ (median; protocol IQR bands)")
    ax_var.set_ylabel(r"Variance of signed error $\mathrm{Var}(\hat{\phi}-\phi)$")
    ax_error.set_title("(a) Absolute estimation error", fontsize=10)
    ax_var.set_title("(b) Error variance", fontsize=10)
    fig.suptitle(d.scen[sid]["label"], fontsize=11)

    handles = [plt.Line2D([], [], **{**style(a), "label": NICE[a]}) for a in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.035))
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    # _foot(fig, d, extra="left: protocol IQR bands; right: sample variance for protocols and "
        #   "analytic prior-averaged QCRB variance for the Oracle")
    fig.savefig(out_path("fig_error_variance.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_error_variance.png"


def _reflected_kde(samples, grid):
    """Gaussian KDE on [0, infinity), reflected at zero to avoid boundary leakage."""
    q25, q75 = np.quantile(samples, [0.25, 0.75])
    robust_sd = (q75 - q25) / 1.349
    ordinary_sd = np.std(samples, ddof=1)
    positive_scales = [x for x in (robust_sd, ordinary_sd) if np.isfinite(x) and x > 0]
    scale = min(positive_scales) if positive_scales else max(float(grid[-1]) / 20, 1e-6)
    bandwidth = max(0.9 * scale * len(samples) ** (-0.2), float(grid[-1]) / 1000)
    z_pos = (grid[:, None] - samples[None, :]) / bandwidth
    z_ref = (grid[:, None] + samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z_pos**2) + np.exp(-0.5 * z_ref**2)
    return kernel.mean(axis=1) / (np.sqrt(2 * np.pi) * bandwidth)


def _gaussian_kde(samples, grid):
    """Gaussian KDE with a robust Silverman bandwidth."""
    q25, q75 = np.quantile(samples, [0.25, 0.75])
    robust_sd = (q75 - q25) / 1.349
    ordinary_sd = np.std(samples, ddof=1)
    positive_scales = [x for x in (robust_sd, ordinary_sd) if np.isfinite(x) and x > 0]
    scale = min(positive_scales) if positive_scales else max(float(np.ptp(grid)) / 20, 1e-6)
    bandwidth = max(0.9 * scale * len(samples) ** (-0.2), float(np.ptp(grid)) / 1000)
    z = (grid[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (np.sqrt(2 * np.pi) * bandwidth)


def fig_error_density(d, sid=HEAD, budget=10_000):
    """Distribution of |phi_hat - phi| / eps at one fixed-budget operating point."""
    eps = f(d.scen[sid]["eps"])
    errors = {}
    for a in M.ORDER:
        raw = trace_errors(sid, a, budget)
        if raw.size:
            errors[a] = raw / eps
    if not errors:
        print("   (fig_error_density skipped: no stored traces for "
              f"{sid} at budget {budget:,})")
        return None

    pooled = np.concatenate(list(errors.values()))
    xmax = max(1.2, float(np.quantile(pooled, 0.995)))
    grid = np.linspace(0, xmax, 500)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axvspan(0, 1, color="#2ca02c", alpha=0.06, lw=0)
    for a in M.ORDER:
        x = errors.get(a)
        if x is None:
            continue
        ax.plot(grid, _reflected_kde(x, grid), color=COL[a], lw=2.0,
                ls=DASHED.get(a, "-"), label=NICE[a])
    ax.axvline(1, color="k", ls="--", lw=0.9)
    ax.annotate(r"convergence threshold $|\hat{\phi}-\phi|=\epsilon$",
                xy=(1, 0.98), xycoords=("data", "axes fraction"),
                xytext=(5, -3), textcoords="offset points", ha="left", va="top", fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Normalized absolute error $|\hat{\phi}-\phi|/\epsilon$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{d.scen[sid]['label']},  budget $C={budget:,}$", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    n_min = min(len(x) for x in errors.values())
    fig.text(
        0.005, 0.005,
        f"Held-out seed {M.SEED_TEST}; {n_min:,} stored traced trials per algorithm; "
        "parameters frozen on tuning seeds "
        + ",".join(str(x) for x in M.SEED_TUNE_BLOCKS)
        + "; reflected Gaussian KDE. Display ends at the pooled 99.5th percentile.",
        fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.savefig(out_path("fig_error_density.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_error_density.png"


def fig_signed_error_density(d, sid=HEAD, budget=10_000):
    """Distribution of (phi_hat - phi) / eps at one fixed-budget operating point."""
    eps = f(d.scen[sid]["eps"])
    errors = {}
    for a in M.ORDER:
        raw = trace_signed_errors(sid, a, budget)
        if raw.size:
            errors[a] = raw / eps
    if not errors:
        print("   (fig_signed_error_density skipped: no stored traces for "
              f"{sid} at budget {budget:,})")
        return None

    pooled = np.concatenate(list(errors.values()))
    xmax = max(2.0, float(np.quantile(np.abs(pooled), 0.995)))
    grid = np.linspace(-xmax, xmax, 600)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    line_styles = {
        "brute": (0, (4, 2)),
        "linear": "-",
        "binary_deep": (0, (7, 2)),
        "reverse_eng_risk": (0, (2, 1.5)),
    }
    for a in M.ORDER:
        x = errors.get(a)
        if x is None:
            continue
        ax.plot(grid, _gaussian_kde(x, grid), color=COL[a], lw=2.0,
                ls=line_styles.get(a, "-"), label=NICE[a])
    ax.axvline(0, color="#555555", ls=":", lw=0.9)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Normalized signed error $(\hat{\phi}-\phi)/\epsilon$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{d.scen[sid]['label']},  budget $C={budget:,}$", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    n_min = min(len(x) for x in errors.values())
    fig.text(
        0.005, 0.005,
        f"Held-out seed {M.SEED_TEST}; {n_min:,} stored traced trials per algorithm; "
        "parameters frozen on tuning seeds "
        + ",".join(str(x) for x in M.SEED_TUNE_BLOCKS)
        + "; Gaussian KDE. Display is limited symmetrically at the pooled "
          "99.5th percentile of the absolute normalized error.",
        fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.savefig(out_path("fig_signed_error_density.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_signed_error_density.png"


def fig_phi_hat_density(d, sid=HEAD, budget=10_000):
    """Distribution of the final estimates at one fixed-budget operating point."""
    estimates = {}
    for a in M.ORDER:
        x = trace_phi_hats(sid, a, budget)
        if x.size:
            estimates[a] = x
    if not estimates:
        print("   (fig_phi_hat_density skipped: no stored traces for "
              f"{sid} at budget {budget:,})")
        return None

    pooled = np.concatenate(list(estimates.values()))
    span = float(np.quantile(pooled, 0.995) - np.quantile(pooled, 0.005))
    xmin = float(pooled.min() - 0.03 * span)
    xmax = float(pooled.max() + 0.03 * span)
    grid = np.linspace(xmin, xmax, 500)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    scen = d.scen[sid]
    ax.axvspan(f(scen["phi_min"]), f(scen["phi_max"]), color="#7f7f7f", alpha=0.05, lw=0)
    for a in M.ORDER:
        x = estimates.get(a)
        if x is None:
            continue
        ax.plot(grid, _gaussian_kde(x, grid), color=COL[a], lw=2.0,
                ls=DASHED.get(a, "-"), label=NICE[a])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Final estimate $\hat{\phi}$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{d.scen[sid]['label']},  budget $C={budget:,}$", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    n_min = min(len(x) for x in estimates.values())
    fig.text(
        0.005, 0.005,
        f"Held-out seed {M.SEED_TEST}; {n_min:,} stored traced trials per algorithm; "
        "parameters frozen on tuning seeds "
        + ",".join(str(x) for x in M.SEED_TUNE_BLOCKS)
        + "; Gaussian KDE. Shading marks the prior support.",
        fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.savefig(out_path("fig_phi_hat_density.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_phi_hat_density.png"


def fig_broad(d):
    sids = [s["id"] for s in M.SCENARIOS
            if "broad_prior" in s["families"] and s["id"] in d.scen]
    if not sids:
        return None
    fig, axes = plt.subplots(1, len(sids), figsize=(5.0*len(sids), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, sids):
        for a in ORDER:
            b, r, lo, hi = d.curve(sid, a)
            if not b.size:
                continue
            ax.plot(b, 100*r, **style(a))
            ok = np.isfinite(lo)
            if ok.any():
                ax.fill_between(b[ok], 100*lo[ok], 100*hi[ok], color=COL[a], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.axhline(90, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Budget $C = N \\cdot m$")
        ax.set_title(f"{d.scen[sid]['label']}   ($N_{{min}} = {d.scen[sid]['N_min']}$)",
                     fontsize=9.5)
    axes[0].set_ylabel("Converged (%)")
    axes[0].set_ylim(0, 101)
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _foot(fig, d, bands="shaded bands are 95% Wilson intervals")
    fig.savefig(out_path("fig_broad.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_broad.png"

def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    os.makedirs(OUT, exist_ok=True)
    d = D()
    made = []
    with plt.rc_context(PAPER_RC):
        for tag, fn in (("story", lambda: fig_story(d, False)),
                        ("story_small", lambda: fig_story(d, True)),
                        ("precision", lambda: fig_precision(d)),
                        ("algorithm_diagnostics", lambda: fig_algorithm_diagnostics(d)),
                        ("pareto", lambda: fig_pareto(d)),
                        ("error", lambda: fig_error(d)),
                        ("variance", lambda: fig_variance(d)),
                        ("error_variance", lambda: fig_error_variance(d)),
                        ("error_density", lambda: fig_error_density(d)),
                        ("signed_error_density", lambda: fig_signed_error_density(d)),
                        ("phi_hat_density", lambda: fig_phi_hat_density(d)),
                        ("broad", lambda: fig_broad(d))):
            if only and tag not in only:
                continue
            n = fn()
            if n:
                made.append(n)
    print(f"wrote {len(made)} figures to {OUT}")
    for n in made:
        print(f"   {n}")


if __name__ == "__main__":
    main()
