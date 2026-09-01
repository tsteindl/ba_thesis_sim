"""Grid-tuned safeguard constant vs. the statistically derived one.

Isolates the single change described in qmetrology/safeguard.py: for reverse engineering the tuned
safety factor C_safe (Eq. 3.6), for binary search the tuned decrement s, are replaced by the depth
that maximises P(no overshoot) x P(converge) under the asymptotic law phi_hat ~ N(phi, 1/(4N^2 m)).

Everything else is held fixed, and both arms are de-biased the same way: the remaining parameters
are grid-tuned on seed 42 and the winning config is re-validated on seed 2024.

Writes results/safeguard_compare.csv, results/SAFEGUARD.md and results/fig_safeguard.png.
    python analysis/safeguard_study.py [--quick]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E
from qmetrology.safeguard import pilot_sd, risk_optimal_depth
from qmetrology.algorithms import (
    find_phi_fixed_budget_brute_force as BF,
    find_phi_fixed_budget_binary_search as BIN,
    find_phi_fixed_budget_reverse_engineering as RE,
    find_phi_fixed_budget_binary_search_risk as BIN_R,
    find_phi_fixed_budget_reverse_engineering_risk as RE_R,
)

QUICK = "--quick" in sys.argv
R_TUNE = 500 if QUICK else 2000
R_TEST = 5000 if QUICK else 40_000
SEED_TUNE, SEED_TEST = 42, 2024

# (phi_min, phi_max, eps, budget, label) — the operating points the thesis actually reports
SCENARIOS = [
    (0.01, 0.1, 1e-3, 10_000, "T3.1  U(0.01,0.1) e-3 @10k"),
    (0.01, 0.1, 1e-3, 45_000, "T3.2  U(0.01,0.1) e-3 @45k"),
    (0.01, 0.1, 1e-4, 3_000_000, "T3.2  U(0.01,0.1) e-4 @3M"),
    (0.001, 0.01, 1e-4, 400_000, "T3.2  U(0.001,0.01) e-4 @400k"),
    (0.001, 0.1, 1e-4, 3_500_000, "T3.2  U(0.001,0.1) e-4 @3.5M"),
    (0.01, np.pi / 16, 1e-3, 10_000, "T3.3  U(0.01,pi/16) e-3 @10k"),
    (0.01, np.pi / 8, 1e-3, 10_000, "T3.3  U(0.01,pi/8) e-3 @10k"),
    (0.01, np.pi / 4, 1e-3, 10_000, "T3.3  U(0.01,pi/4) e-3 @10k"),
    (0.01, np.pi / 2, 1e-3, 10_000, "T3.3  U(0.01,pi/2) e-3 @10k"),
    (0.01, np.pi / 2, 1e-3, 174_608, "T3.4  U(0.01,pi/2) e-3 @175k"),
    (0.01, np.pi / 2, 1e-3, 887_240, "T3.4  U(0.01,pi/2) e-3 @887k"),
]
if QUICK:
    SCENARIOS = [SCENARIOS[0], SCENARIOS[8]]

COL = {"binary": "#ff7f0e", "reverse_eng": "#1f77b4", "brute": "#7f7f7f"}
NICE = {"binary": "Binary search", "reverse_eng": "Reverse engineering"}


def n_min(pmax):
    return max(1, int(np.floor(np.pi / (2 * pmax))))


def m_grid(pmax, eps, n):
    """Same exploration-size grid the budget sweep uses — deliberately not capped low."""
    hi = int(np.clip(0.6724 / (n_min(pmax) * eps**2), 200, 300_000))
    return np.unique(np.geomspace(20, hi, n).astype(int))


def debias(fn, grid, pmin, pmax, eps, B):
    """Tune on seed 42, validate the winner on seed 2024. Returns (percent, winning config)."""
    _m, arg, _ = E.grid_search_max(fn, {**grid, "budget": [int(B)]}, R_TUNE, pmin, pmax, eps, SEED_TUNE)
    return 100 * E.success_rate(fn, arg, R_TEST, pmin, pmax, eps, SEED_TEST), arg


def run_scenario(pmin, pmax, eps, B, name):
    m_re, m_bin = m_grid(pmax, eps, 24), m_grid(pmax, eps, 16)
    conf = [0.5, 0.65, 0.8, 0.9]
    out = {"brute": (100 * E.success_rate(BF, {"budget": int(B)}, R_TEST, pmin, pmax, eps, SEED_TEST), {})}
    out["reverse_eng"] = debias(RE, {"m_exploration": m_re, "safeguard": [0.8, 0.85, 0.9, 0.925, 0.95, 0.975]},
                                pmin, pmax, eps, B)
    out["reverse_eng_risk"] = debias(RE_R, {"m_exploration": m_re, "eps_target": [eps]}, pmin, pmax, eps, B)
    out["binary"] = debias(BIN, {"m_exploration": m_bin, "safeguard": [0, 1, 2], "conf": conf},
                           pmin, pmax, eps, B)
    out["binary_risk"] = debias(BIN_R, {"m_exploration": m_bin, "conf": conf, "eps_target": [eps]},
                                pmin, pmax, eps, B)
    print(f"  {name:34s} brute {out['brute'][0]:5.1f}% | "
          f"RE {out['reverse_eng'][0]:5.1f}->{out['reverse_eng_risk'][0]:5.1f} | "
          f"BIN {out['binary'][0]:5.1f}->{out['binary_risk'][0]:5.1f}", flush=True)
    return out


def _clean(cfg):
    return {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in cfg.items() if k != "budget"}


# ---------------------------------------------------------------- figure
def figure(rows):
    """Left: paired const-vs-statistical outcome per operating point. Right: the mechanism."""
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                         "legend.frameon": False})
    names = [r["scenario"] for r in rows][::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(names) + 3.4),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    y = np.arange(len(names))
    for algo, off in (("reverse_eng", 0.18), ("binary", -0.18)):
        a = np.array([r[f"{algo}_pct"] for r in rows])[::-1]
        b = np.array([r[f"{algo}_risk_pct"] for r in rows])[::-1]
        ax.hlines(y + off, a, b, color=COL[algo], lw=2.4, alpha=0.5, zorder=1)
        ax.scatter(a, y + off, s=46, facecolor="white", edgecolor=COL[algo], lw=1.8, zorder=3)
        ax.scatter(b, y + off, s=46, color=COL[algo], zorder=3, label=NICE[algo])
    ax.scatter([r["brute_pct"] for r in rows][::-1], y, marker="|", s=150, color=COL["brute"],
               lw=1.6, zorder=2, label="Brute force (baseline)")
    # identity (hue) and treatment (fill) are separate encodings, so each gets its own legend entry
    ax.scatter([], [], s=46, facecolor="white", edgecolor="0.35", lw=1.8, label="open = tuned constant")
    ax.scatter([], [], s=46, color="0.35", label="filled = statistical safeguard")
    ax.set(yticks=y, yticklabels=names, xlabel="% of trials converged (validated, seed 2024)",
           xlim=(0, 104), ylim=(-1.9, len(names) - 0.4), title="Same algorithm, safeguard replaced")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2, columnspacing=1.0)

    # mechanism: the implied multiplicative safeguard, vs the tuned constant it replaces
    phis = np.geomspace(0.01, 0.1, 60)
    shades = ["#c6dbef", "#6baed6", "#2171b5"]
    for (B, m_e, eps), c in zip([(1e4, 76, 1e-3), (4.5e4, 409, 1e-3), (3e6, 1308, 1e-4)], shades):
        nmin, ceff = 15, []
        sd = pilot_sd(nmin, m_e)
        for p in phis:
            naive = max(int(np.pi // (2 * p)), 1)
            ceff.append(risk_optimal_depth(p, sd, B - m_e * nmin, eps,
                                           N_min=nmin, N_max=314) / naive)
        ax2.plot(phis, ceff, lw=2, color=c, label=f"B={B:,.0f}, ε={eps:.0e}, m'={m_e}")
    ax2.axhline(0.85, ls="--", lw=1.4, color="#d62728", label="grid-tuned constant C = 0.85")
    ax2.set(xscale="log", xlabel="true phase φ", ylabel="implied safeguard  $C_{eff}=N^*/\\lfloor π/2\\hatφ\\rfloor$",
            ylim=(0.4, 1.05), title="Why it wins: the rule is φ- and budget-adaptive")
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig("results/fig_safeguard.png")
    print("wrote results/fig_safeguard.png")


def _rows_from_csv(path="results/safeguard_compare.csv"):
    """Re-render the figure from an existing run (python analysis/safeguard_study.py --figure-only)."""
    import ast
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append({"scenario": d["scenario"], "brute_pct": float(d["brute_pct"]),
                         "reverse_eng_pct": float(d["re_const_pct"]),
                         "reverse_eng_risk_pct": float(d["re_stat_pct"]),
                         "binary_pct": float(d["bin_const_pct"]),
                         "binary_risk_pct": float(d["bin_stat_pct"]),
                         **{f"{k}_params": ast.literal_eval(d[c]) for k, c in
                            [("reverse_eng", "re_const_params"), ("reverse_eng_risk", "re_stat_params"),
                             ("binary", "bin_const_params"), ("binary_risk", "bin_stat_params")]}})
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    if "--figure-only" in sys.argv:
        figure(_rows_from_csv())
        return
    print(f"safeguard study: {len(SCENARIOS)} operating points, R_tune={R_TUNE}, R_test={R_TEST:,}\n")
    rows = []
    for pmin, pmax, eps, B, name in SCENARIOS:
        r = run_scenario(pmin, pmax, eps, B, name)
        rows.append({"scenario": name, "phi_min": pmin, "phi_max": pmax, "eps": eps, "budget": B,
                     "brute_pct": r["brute"][0],
                     **{f"{k}_pct": r[k][0] for k in
                        ("reverse_eng", "reverse_eng_risk", "binary", "binary_risk")},
                     **{f"{k}_params": _clean(r[k][1]) for k in
                        ("reverse_eng", "reverse_eng_risk", "binary", "binary_risk")}})

    with open("results/safeguard_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "phi_min", "phi_max", "eps", "budget", "brute_pct",
                    "re_const_pct", "re_stat_pct", "re_delta_pp",
                    "bin_const_pct", "bin_stat_pct", "bin_delta_pp",
                    "re_const_params", "re_stat_params", "bin_const_params", "bin_stat_params"])
        for r in rows:
            w.writerow([r["scenario"], r["phi_min"], r["phi_max"], r["eps"], r["budget"],
                        round(r["brute_pct"], 2),
                        round(r["reverse_eng_pct"], 2), round(r["reverse_eng_risk_pct"], 2),
                        round(r["reverse_eng_risk_pct"] - r["reverse_eng_pct"], 2),
                        round(r["binary_pct"], 2), round(r["binary_risk_pct"], 2),
                        round(r["binary_risk_pct"] - r["binary_pct"], 2),
                        r["reverse_eng_params"], r["reverse_eng_risk_params"],
                        r["binary_params"], r["binary_risk_params"]])

    se = 100 * np.sqrt(0.25 / R_TEST)
    L = ["# Safeguard: grid-tuned constant vs. statistically derived depth\n",
         "_One change only: the tuned safety factor `C_safe` (reverse engineering, Eq. 3.6) and the tuned "
         "decrement `s` (binary search) are replaced by the depth maximising "
         "`P(no overshoot) x P(converge)` under `phi_hat ~ N(phi, 1/(4 N^2 m))` (Eq. 3.4). Both arms are "
         f"grid-tuned on seed 42 and validated on seed 2024 at R = {R_TEST:,}; worst-case SE ≈ {se:.2f} pp._\n",
         "| Operating point | brute | RE const-C | RE stat. | Δ | Binary const-s | Binary stat. | Δ |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        dre = r["reverse_eng_risk_pct"] - r["reverse_eng_pct"]
        dbi = r["binary_risk_pct"] - r["binary_pct"]
        L.append(f"| {r['scenario']} | {r['brute_pct']:.1f}% | {r['reverse_eng_pct']:.1f}% | "
                 f"**{r['reverse_eng_risk_pct']:.1f}%** | {dre:+.1f} | {r['binary_pct']:.1f}% | "
                 f"**{r['binary_risk_pct']:.1f}%** | {dbi:+.1f} |")
    d_re = np.array([r["reverse_eng_risk_pct"] - r["reverse_eng_pct"] for r in rows])
    d_bi = np.array([r["binary_risk_pct"] - r["binary_pct"] for r in rows])
    L += ["", f"**Reverse engineering:** {(d_re > 0).sum()}/{len(d_re)} operating points improve, "
              f"median {np.median(d_re):+.1f} pp, max {d_re.max():+.1f} pp.",
          f"**Binary search:** {(d_bi > 0).sum()}/{len(d_bi)} improve, median {np.median(d_bi):+.1f} pp, "
          f"max {d_bi.max():+.1f} pp.", "",
          "The statistical rule also removes a tuned parameter from each algorithm: the winning configs "
          "below contain only `m_exploration` (and `conf` for binary search).\n"]
    ratio = np.array([r["reverse_eng_params"]["m_exploration"] / r["reverse_eng_risk_params"]["m_exploration"]
                      for r in rows])
    L += [f"**It also needs a much cheaper pilot.** Reverse engineering's winning `m_exploration` is "
          f"{np.median(ratio):.1f}x smaller with the statistical safeguard (max {ratio.max():.0f}x): because the "
          "rule scales the backoff to the pilot's own spread, it can act safely on a coarse estimate, "
          "whereas a constant C has to be bought with exploration shots.\n",
          "![const vs statistical safeguard](fig_safeguard.png)\n",
          "## Winning configurations\n",
          "| Operating point | RE const-C | RE stat. | Binary const-s | Binary stat. |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scenario']} | `{r['reverse_eng_params']}` | `{r['reverse_eng_risk_params']}` | "
                 f"`{r['binary_params']}` | `{r['binary_risk_params']}` |")
    with open("results/SAFEGUARD.md", "w") as f:
        f.write("\n".join(L) + "\n")

    figure(rows)
    print("wrote results/safeguard_compare.csv and results/SAFEGUARD.md")


if __name__ == "__main__":
    main()
