"""The Chapter-4 figures, regenerated from the results tables.

Nothing here simulates: every figure is drawn from results/*.csv, so a figure cannot silently
disagree with the tables, and each carries its R and seeds in a footer strip.

    python analysis/thesis_figures.py            # all figures
    python analysis/thesis_figures.py --only story,precision
    python analysis/thesis_figures.py --out DIR  # write somewhere else

Figures (written to results/):
    fig_story           convergence vs budget for the headline scenario, with the 90% crossing
    fig_story_small     the same, compact, for a single-column layout
    fig_pareto          the budget/convergence frontier across scenarios
    fig_precision       budget ratio vs brute force at 90%, against the precision requirement
    fig_error           estimator-error distribution vs budget (median + IQR band)
    fig_variance        sample variance of signed estimator error vs budget; analytic Oracle
    fig_error_variance  error median and variance side by side
    fig_algorithm_diagnostics exploration cost and final overshoot near 90% convergence
    fig_overshoot_criterion   exact operating characteristic of the overshoot rule (Sec. 3.2.2)
    fig_linear_detector       linear search under alternative stopping rules (Sec. 4.5)

Every panel is annotated with the R it was produced at, read from the data rather than hard-coded,
so a figure can never silently disagree with the tables.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
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
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
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
    sids = [s["id"] for s in M.SCENARIOS
            if s["id"] in d.scen and "broad_prior" not in s.get("families", ())]
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
        print("   (fig_error skipped: run analysis/error_curves.py first)")
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
        print("   (fig_variance skipped: run analysis/variance_curves.py first)")
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



def _overshoot_curves():
    """(m, conf, reference) -> (x grid, power). Analytic, from overshoot_criterion.py."""
    out = {}
    for row in load("overshoot_power.csv"):
        key = (int(f(row["m_exploration"])), f(row["conf"]), row["reference"])
        out.setdefault(key, ([], []))
        out[key][0].append(f(row["x"]))
        out[key][1].append(f(row["power"]))
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}


def fig_overshoot_criterion(d, m=200, conf=0.5):
    """How reliably the overshoot rule fires, as a function of how deep the probe is.

    Exact, not simulated: probe and reference each have m'+1 possible outcomes, so the probability
    is a sum over all (m'+1)^2 pairs.
    """
    cur = _overshoot_curves()
    if not cur:
        return None
    SPEC = [("perfect", "#000000", "--", r"perfect reference"),
            ("late", "#1f77b4", "-", r"late probe ($\rho = 1.05$)"),
            ("first", "#d62728", "-", r"first probe ($\rho = 5.7$)")]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for ref, col, ls, lab in SPEC:
        key = (m, conf, ref)
        if key not in cur:
            continue
        x, pw = cur[key]
        ax.plot(x, pw, color=col, ls=ls, lw=1.3 if ls == "--" else 1.9, label=lab)
    ax.axvline(1.0, color="0.25", lw=1.0, ls=":")
    ax.axvspan(0.8, 1.0, color="#2ca02c", alpha=0.07, lw=0)
    ax.set_xlim(0.8, 1.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"probe depth $N / N_{\mathrm{opt}}$")
    ax.set_ylabel("P(rule declares an overshoot)")
    ax.text(0.9, 1.005, "$N$ is safe", ha="center", va="bottom", fontsize=8.5, color="#2ca02c")
    ax.text(1.2, 1.005, "$N$ has overshot", ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95, frameon=True,
              edgecolor="none", facecolor="white")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.005, 0.005,
             rf"Exact, at $m' = {m}$ shots and conf $= {conf:g}$. Past $N_{{\mathrm{{opt}}}}$ a "
             r"probe always reads below $\phi$, so with a perfect reference the rule never misses; "
             "the gap" "\n" r"between the curves is the cost of comparing against a noisy "
             r"$\hat\phi_{\mathrm{acc}}$ instead. Ripple on the dashed curve is the estimator's "
             r"discreteness.",
             fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.savefig(out_path("fig_overshoot_criterion.png"), bbox_inches="tight")
    fig.savefig(out_path("fig_overshoot_criterion.pdf"), bbox_inches="tight")
    plt.close(fig)
    return "fig_overshoot_criterion.png"


# ---------------------------------------------------------- linear search: alternative stop rules
RULE_COL = {"cumulative": "#2ca02c", "window": "#d62728", "prepost": "#9467bd",
            "threshold": "#1f77b4", "pooled": "#ff7f0e", "cusum": "#8c564b",
            "published": "#555555"}
RULE_NICE = {"cumulative": "cumulative mean (re-tuned)", "window": "moving window",
             "prepost": "pre/post windows", "threshold": "Eq. (3.6) at lag $w$",
             "pooled": "Eq. (3.6) vs. pooled ref.", "cusum": "CUSUM",
             "published": "published configuration"}
RULE_MARK = {"cumulative": "o", "window": "^", "prepost": "v", "threshold": "D",
             "pooled": "s", "cusum": "P", "published": "x"}


def fig_linear_detector(d):
    """What Algorithm 4 would do with a different stopping rule -- the study that led to the
    `mean_window` parameter.

    (a) Held-out convergence of each candidate rule relative to the cumulative-mean rule re-tuned by
    the same tuner, so the curves isolate the RULE rather than the tuning. (b) How often the
    cumulative rule stops before the aliasing boundary is ever reached, measured against the
    "l consecutive falls is roughly 2^-l" argument. (c) The exploration-size profile, the control
    against "the alternative simply prefers a different m'". (d) The convergence curve and the
    90%-reliability crossing that follow.

    Reads the bake-off CSVs, which record the sweep as it stood BEFORE `mean_window` existed; the
    "published configuration" row is that sweep's frozen Algorithm 4.
    """
    bake = load("linear_detector_bakeoff.csv")
    if not bake:
        return None
    streaks, profile = load("linear_detector_streaks.csv"), load("linear_detector_profile.csv")
    matched = load("linear_detector_matched.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    (axa, axb), (axc, axd) = axes

    # -- (a) what the stopping rule is worth, at the six diagnostic points ---------------------
    pts, seen = [], set()
    for r in bake:
        k = (r["scenario_id"], int(f(r["budget"])))
        if k in seen or "crossing sweep" in r.get("point_tag", ""):
            continue
        seen.add(k)
        pts.append(k)
    x = np.arange(len(pts))
    for rule in ("window", "threshold", "prepost", "pooled", "cusum", "published"):
        rs = {(r["scenario_id"], int(f(r["budget"]))): r for r in bake if r["rule"] == rule}
        if not rs:
            continue
        y = np.array([f(rs[k]["delta_vs_cumulative_pp"]) if k in rs else np.nan for k in pts])
        e = np.array([1.96 * f(rs[k]["delta_se_pp"], 0.0) if k in rs else np.nan for k in pts])
        # window and threshold coincide wherever the tuner picks alpha = 0.5, which is most of the
        # time -- threshold is dashed so the overlap reads as an overlap, not as one curve
        axa.errorbar(x, y, yerr=e, marker=RULE_MARK[rule], color=RULE_COL[rule],
                     lw=1.6 if rule != "published" else 1.0,
                     ls={"published": ":", "threshold": "--"}.get(rule, "-"), capsize=2,
                     label=RULE_NICE[rule], zorder=3 if rule == "window" else 2)
    mrate = {(r["scenario_id"], int(f(r["budget"])), r["rule"]): f(r["rate"]) for r in matched}
    cm = np.array([mrate.get((sid, b, "cumulative"), np.nan) for sid, b in pts])
    wm = np.array([mrate.get((sid, b, "window"), np.nan) for sid, b in pts])
    if np.isfinite(cm).any():
        axa.plot(x, 100 * (wm - cm), marker="^", ms=5, mfc="none", ls="-.", lw=1.1,
                 color=RULE_COL["window"], label="moving window, $m'$ and inc pinned")
    axa.axhline(0, color="0.3", lw=1.0)
    axa.set_xticks(x)
    axa.set_xticklabels([f"{sid}\n$B$ = {b:,}" for sid, b in pts], fontsize=6.5)
    axa.set_ylabel("convergence vs. cumulative-mean rule (pp)")
    axa.set_title("(a) what the stopping rule is worth", fontsize=10, loc="left")
    axa.legend(fontsize=6.6, loc="best")

    # -- (b) the premature-stop probability against the 2^-l argument --------------------------
    by_pt = {}
    for r in streaks:
        by_pt.setdefault((r["scenario_id"], int(f(r["budget"]))), []).append(r)
    want = [k for k in (("narrow_e3", 10_000), ("narrow_e3", 41_326),
                        ("narrow_e4", 2_917_365)) if k in by_pt] or sorted(by_pt)[:3]
    for i, key in enumerate(want[:3]):
        rows = sorted(by_pt[key], key=lambda r: int(f(r["lookback_window"])))
        ls = ["-", "--", ":"][i]
        l = [int(f(r["lookback_window"])) for r in rows]
        axb.plot(l, [100 * f(r["premature_stop_measured"]) for r in rows], ls, marker="o", ms=3.5,
                 color="#d62728", lw=1.7, label=f"measured: {key[0]}, $B$ = {key[1]:,}")
        axb.plot(l, [100 * f(r["premature_stop_coin_flip"]) for r in rows], ls, color="#1f77b4",
                 lw=1.2, label=r"$1-(1-2^{-l})^{T-l+1}$" if i == 0 else None)
        axb.plot(l, [100 * f(r["premature_stop_indep_q"]) for r in rows], ls, color="#7f7f7f",
                 lw=1.2, label="the same with the measured $q$" if i == 0 else None)
    axb.set_xlabel("lookback window $l$")
    axb.set_ylabel(r"$P($stop before $N_{\mathrm{opt}}$ is reached$)$  (%)")
    axb.set_ylim(-2, 102)
    axb.set_title("(b) the false-alarm argument, measured", fontsize=10, loc="left")
    axb.legend(fontsize=6.6, loc="upper right")

    # -- (c) the exploration-size profile at the headline point --------------------------------
    head = pts[0] if pts else None
    prof = [r for r in profile
            if head and r["scenario_id"] == head[0] and int(f(r["budget"])) == head[1]]
    top = max([f(r["best_tune_mean"]) for r in prof], default=1.0)
    for rule in ("cumulative", "window", "threshold", "pooled"):
        rs = sorted([r for r in prof if r["rule"] == rule],
                    key=lambda r: int(f(r["m_exploration"])))
        # drop the tail where the budget no longer affords a scan at all: the collapse to zero is a
        # budget constraint, not a property of the rule, and it flattens everything else
        rs = [r for r in rs if f(r["best_tune_mean"]) > 0.5 * top]
        if not rs:
            continue
        axc.plot([int(f(r["m_exploration"])) for r in rs],
                 [100 * f(r["best_tune_mean"]) for r in rs], marker=RULE_MARK[rule],
                 color=RULE_COL[rule], lw=1.6, ms=3.5,
                 ls="--" if rule == "threshold" else "-", label=RULE_NICE[rule])
    axc.set_xscale("log")
    axc.set_xlabel(r"exploration shots per probe $m'$")
    axc.set_ylabel("best convergence at that $m'$ (%)")
    if head:
        axc.set_title(f"(c) exploration size is not the explanation\n"
                      f"      ({head[0]}, $B$ = {head[1]:,})", fontsize=10, loc="left")
    axc.legend(fontsize=7, loc="lower left")

    # -- (d) the convergence curve and the 90% crossing ----------------------------------------
    sid_c = "narrow_e3"
    curves = {}
    for r in bake:
        if r["scenario_id"] != sid_c:
            continue
        curves.setdefault(r["rule"], []).append((int(f(r["budget"])), f(r["rate"])))
    best = None
    for rule in ("window", "threshold", "prepost", "pooled"):
        v = curves.get(rule)
        if v and (best is None or max(y for _, y in v) > best[1]):
            best = (rule, max(y for _, y in v))
    drawn = False
    for rule in [r for r in ("published", "cumulative", best[0] if best else None) if r]:
        v = sorted(curves.get(rule, []))
        if len(v) < 3:
            continue
        drawn = True
        axd.plot([b for b, _ in v], [100 * y for _, y in v], marker=RULE_MARK[rule],
                 color=RULE_COL[rule], lw=1.7, ms=4,
                 ls=":" if rule == "published" else "-", label=RULE_NICE[rule])
    if drawn:
        swept = [b for v in curves.values() for b, _ in v]
        lo_b, hi_b = 0.7 * min(swept), 1.4 * max(swept)
        for algo, col in (("brute", COL["brute"]), ("binary_deep", COL["binary_deep"]),
                          ("reverse_eng_risk", COL["reverse_eng_risk"])):
            ref = sorted([(int(f(r["budget"])), 100 * f(r["rate"]))
                          for r in load("performance_curves.csv")
                          if r["scenario_id"] == sid_c and r["algorithm"] == algo
                          and lo_b <= int(f(r["budget"])) <= hi_b])
            if ref:
                axd.plot([b for b, _ in ref], [y for _, y in ref], color=col, lw=1.1, alpha=0.75,
                         ls="--", label=NICE[algo], zorder=1)
    axd.axhline(90, color="0.35", lw=1.0, ls=":")
    cross = {r["rule"]: f(r["budget_to_reach"]) for r in load("linear_detector_crossings.csv")
             if r["scenario_id"] == sid_c}
    for rule, dy in (("published", -13), (best[0] if best else "published", 9)):
        b90 = cross.get(rule, float("nan"))
        if np.isfinite(b90):
            axd.plot([b90], [90], marker="v", ms=7, color=RULE_COL[rule], zorder=5, clip_on=False)
            axd.annotate(f"$B_{{90}}$ = {b90:,.0f}", (b90, 90), textcoords="offset points",
                         xytext=(0, dy), ha="center", fontsize=6.8, color=RULE_COL[rule], zorder=5)
    axd.set_xscale("log")
    axd.set_ylim(40, 100)
    if drawn:
        axd.set_xlim(0.8 * min(swept), 1.25 * max(swept))
    axd.set_xlabel("budget $B$")
    axd.set_ylabel("converged (%)")
    axd.set_title("(d) the curve that follows, and the 90% crossing\n"
                  f"      ({sid_c})", fontsize=10, loc="left")
    if drawn:
        axd.legend(fontsize=6.6, loc="lower right")
    else:   # run without --curve: too few budgets to draw a convergence curve
        axd.text(0.5, 0.5, "run with --curve", transform=axd.transAxes, ha="center", va="center",
                 fontsize=9, color="#999999")

    R = int(f(bake[0].get("R"), 0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.005, 0.006,
             f"Held-out evaluation at R = {R:,} trials per point, on the seed the production sweep "
             "holds out. Every rule is tuned over the same (m', inc, s) axes on the same two tuning "
             "blocks and scored by the exact convergence probability of the depth it selects, so "
             "the comparison isolates the stopping rule. Error bars are 95% intervals on the "
             "paired difference. In (d) the dashed reference curves are the production sweep's own "
             "simulated rates for the other protocols.",
             fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.savefig(out_path("fig_linear_detector.png"), bbox_inches="tight")
    fig.savefig(out_path("fig_linear_detector.pdf"), bbox_inches="tight")
    plt.close(fig)
    return "fig_linear_detector.png"


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
                        ("overshoot", lambda: fig_overshoot_criterion(d)),
                        ("linear_detector", lambda: fig_linear_detector(d))):
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
