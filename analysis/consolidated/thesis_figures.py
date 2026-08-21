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
    fig_broad           convergence vs budget under the broad priors

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
        "oracle_hl": "Attainable ceiling ($N_{opt}$, Eq. 3.4 law)"}
COL = {"brute": "#7f7f7f", "separable": "#c7c7c7", "linear": "#2ca02c",
       "binary_deep": "#ff7f0e", "reverse_eng_risk": "#1f77b4", "oracle_hl": "#000000"}
MARK = {"brute": "o", "separable": "v", "linear": "^", "binary_deep": "D",
        "reverse_eng_risk": "s", "oracle_hl": "*"}
ORDER = ["brute", "separable", "linear", "binary_deep", "reverse_eng_risk", "oracle_hl"]
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
DASHED = {"reverse_eng_risk": (0, (5, 2))}


def style(a):
    return dict(marker=MARK[a], color=COL[a], label=NICE[a],
                ls=":" if a == "oracle_hl" else DASHED.get(a, "-"),
                mfc="none" if a in ("separable",) else COL[a])


class D:
    def __init__(self):
        self.perf = load("performance_curves.csv")
        self.cross = load("budget_crossings.csv")
        self.err = load("error_curves.csv")
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
    _foot(fig, d, bands="shaded bands are 95% Wilson intervals")
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
    _foot(fig, d, extra="Ratios > 1 mean less budget is required. Crossings are log-linearly "
          "interpolated between tested budgets.")
    fig.savefig(out_path("fig_precision.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_precision.png"


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
    for a in ORDER:
        rs = sorted([r for r in d.err if r["scenario_id"] == sid and r["algorithm"] == a],
                    key=lambda r: int(f(r["budget"])))
        if not rs:
            continue
        b = np.array([int(f(r["budget"])) for r in rs], float)
        p25 = np.array([f(r["err_p25"]) for r in rs])
        p50 = np.array([f(r["err_p50"]) for r in rs])
        p75 = np.array([f(r["err_p75"]) for r in rs])
        ax.plot(b, p50, **style(a))
        ax.fill_between(b, p25, p75, color=COL[a], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = f(d.scen[sid]["eps"])
    ax.axhline(eps, color="k", ls="--", lw=0.9)
    ax.annotate(f"tolerance $\\epsilon = {eps:g}$", xy=(0.015, eps), xycoords=("axes fraction",
                "data"), fontsize=8, va="bottom", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    ax.set_xlabel("Budget $C = N \\cdot m$")
    ax.set_ylabel("$|\\hat{\\phi} - \\phi|$  (median, IQR band)")
    ax.set_title(d.scen[sid]["label"], fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _foot(fig, d, bands="bands are the 25-75% range of the per-trial error")
    fig.savefig(out_path("fig_error.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_error.png"


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
                        ("pareto", lambda: fig_pareto(d)),
                        ("error", lambda: fig_error(d)),
                        ("broad", lambda: fig_broad(d))):
            if only and tag not in only:
                continue
            n = fn()
            if n:
                made.append(n)
    print(f"wrote {len(made)} figures to {OUT} at R = {d.R:,}")
    for n in made:
        print(f"   {n}")


if __name__ == "__main__":
    main()
