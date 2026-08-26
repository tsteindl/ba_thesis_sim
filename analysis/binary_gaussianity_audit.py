"""Finite-sample audit of the Gaussian approximation used by the binary-search overshoot rule.

The entangled estimator is phi_hat = arccos(sqrt(K/m))/N with K ~ Bin(m, cos^2(N phi)), and the
thesis models it as phi_hat ~ N(phi, 1/(4 m N^2)).  That law is a delta-method limit; this script
measures how far the *finite-sample* law is from it, both deep inside the non-aliasing branch and
at the first aliasing boundary N phi = pi/2.

phi_hat has a discrete distribution supported on the m+1 atoms K = 0..m, so its exact CDF, variance
and KS distance follow from enumeration -- no Monte Carlo is needed for them.  Monte Carlo is run
anyway, at the settings the thesis quotes, so the reported sample variance is a genuine simulation
result and cross-checks the enumeration.

Everything is reported in units of the model SD sigma = 1/(2 N sqrt(m)).  Standardising,

    (phi_hat - phi) / sigma = 2 sqrt(m) [ arccos(sqrt(K/m)) - N phi ],

so the standardised law depends on (N, phi) only through theta = N phi, i.e. only through the
fraction r = theta / (pi/2) of the way to the first aliasing boundary.  The figure uses that.

    python analysis/binary_gaussianity_audit.py
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binom, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = 0.05          # the fixed true parameter of Section 2.7
R = 50_000          # Monte Carlo repetitions
SEED = 20260825
OUT = "results/binary_gaussianity"

# (label, short LaTeX label, N, m) -- four interior rows showing low-m convergence at a fixed
# operating point, then three tracking the approach to and crossing of the first aliasing boundary.
# The short labels keep the typeset table inside \textwidth; the long ones go to the CSV.
CONFIGS = [
    ("very low shots, interior",      r"very low shots",          15, 5),
    ("low shots, interior",           r"low shots",               15, 20),
    ("selected Table 4.1 shot count", r"selected shot count",     15, 114),
    ("old moderate-shot comparison",  r"moderate shots",          15, 400),
    ("near boundary",                 r"near boundary",           30, 114),
    ("immediately below boundary",    r"at $N_{\mathrm{opt}}$",   31, 114),
    ("immediately past boundary",     r"past boundary",           32, 114),
]

# The five exploration shot counts the parameter sweep selects (Tables C.1 and C.2).
SELECTED_M = (114, 289, 334, 391, 2635)

# The corresponding tuned operating points, for the second-probe affordability check.
SELECTED_POINTS = [
    ("narrow_e3", 10_000, "Table 4.1"),
    ("narrow_e3", 29_173, "near B90"),
    ("narrow_e4", 2_917_365, "near B90"),
    ("small_e4", 394_843, "near B90"),
    ("wide_e4", 2_059_436, "near B90"),
]
PARAMS_CSV = "results/consolidated/optimal_params.csv"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


# --------------------------------------------------------------------------- exact enumeration
def exact_law(m, theta):
    """Atoms (in units of sigma) and probabilities of the exact estimator law, sorted increasing."""
    k = np.arange(m + 1)
    w = binom.pmf(k, m, np.cos(theta) ** 2)
    z = 2.0 * np.sqrt(m) * (np.arccos(np.sqrt(k / m)) - theta)
    order = np.argsort(z)                       # arccos decreases in k, so this reverses
    return z[order], w[order]


def exact_ks(z, w):
    """sup_x |F(x) - Phi(x)|.  F steps only at the atoms, Phi is continuous and increasing, so the
    supremum is attained at an atom, approached either from the left or from the right."""
    upper = np.cumsum(w)                        # F(z_i)
    lower = upper - w                           # F(z_i-)
    g = norm.cdf(z)
    return float(max(np.max(np.abs(upper - g)), np.max(np.abs(lower - g))))


def exact_moments(z, w):
    """Mean, variance and fourth central moment of the exact law, in units of sigma."""
    mean = float(z @ w)
    d = z - mean
    return mean, float(w @ d ** 2), float(w @ d ** 4)


# --------------------------------------------------------------------------- the seven-row table
def build_table():
    seeds = np.random.SeedSequence(SEED).spawn(len(CONFIGS))
    rows = []
    for (label, tex_label, N, m), ss in zip(CONFIGS, seeds):
        theta = N * PHI
        p0 = float(np.cos(theta) ** 2)
        z, w = exact_law(m, theta)
        mass = float(w.sum())
        mean_z, var_z, mu4_z = exact_moments(z, w)

        sigma2 = 1.0 / (4.0 * m * N ** 2)               # asymptotic variance
        k = np.random.default_rng(ss).binomial(m, p0, size=R)
        phi_hat = np.arccos(np.sqrt(k / m)) / N
        s2 = float(phi_hat.var(ddof=1))

        # is the MC variance consistent with the exact one?  Var(s^2) ~ (mu4 - var^2)/R.
        se_ratio = np.sqrt(max(mu4_z - var_z ** 2, 0.0) / R)
        z_var = (s2 / sigma2 - var_z) / se_ratio if se_ratio > 0 else 0.0

        rows.append(dict(
            label=label, tex_label=tex_label, N=N, m=m, phi=PHI,
            r=theta / (np.pi / 2), p0=p0, m_min_p0=m * min(p0, 1 - p0),
            s2_mc=s2, var_asym=sigma2, ratio_mc=s2 / sigma2,
            ratio_exact=var_z, ks_exact=exact_ks(z, w),
            bias_exact_sigma=mean_z, sd_exact_sigma=np.sqrt(var_z),
            pmf_mass=mass, mc_var_z=z_var,
        ))
    return rows


# --------------------------------------------------------------------------- formatting
def _mantissa_exp(x, n=3):
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10.0 ** e
    if round(abs(mant), n - 1) >= 10.0:          # e.g. 9.999e-6 rounds up to 10.00e-6
        mant, e = mant / 10.0, e + 1
    return mant, e


def fixed3(x):
    """3 significant figures, fixed notation (for r, m*min(p0,1-p0), the ratio and the KS)."""
    if x == 0:
        return "0"
    _, e = _mantissa_exp(x)
    return f"{x:.{max(2 - e, 0)}f}"


def sci3(x):
    """3 significant figures, plain scientific notation (CSV / Markdown)."""
    mant, e = _mantissa_exp(x)
    return f"{mant:.2f}e{e:+03d}"


def sci3_tex(x):
    """3 significant figures, scientific notation for LaTeX math mode."""
    mant, e = _mantissa_exp(x)
    return rf"{mant:.2f}\times 10^{{{e}}}"


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def markdown_table(rows):
    head = ("| Configuration | $N$ | $m$ | $r=N\\phi/(\\pi/2)$ | $m\\min(p_0,1-p_0)$ | "
            "$s_R^2$ | $1/(4mN^2)$ | ratio | KS |")
    out = [head, "|" + "---|" * 9]
    for r in rows:
        out.append("| {label} | {N} | {m} | {r} | {mm} | {s2} | {va} | {ra} | {ks} |".format(
            label=r["label"], N=r["N"], m=r["m"], r=fixed3(r["r"]), mm=fixed3(r["m_min_p0"]),
            s2=sci3(r["s2_mc"]), va=sci3(r["var_asym"]), ra=fixed3(r["ratio_mc"]),
            ks=fixed3(r["ks_exact"])))
    return "\n".join(out)


def latex_table(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Finite-sample accuracy of the Gaussian approximation "
        r"$\hat\phi\sim\mathcal{N}(\phi,\,1/(4mN^2))$ for the entangled estimator at "
        r"$\phi=0.05$. The sample variance $s_R^2$ comes from $R=50{,}000$ simulated trials; "
        r"the Kolmogorov--Smirnov distance is computed exactly by enumerating the $m+1$ possible "
        r"outcomes $K=0,\dots,m$ and therefore carries no Monte-Carlo error. The first four rows "
        r"hold the operating point fixed and increase the number of shots $m$: the variance ratio "
        r"and the KS distance both approach the Gaussian prediction once both expected outcome "
        r"counts $m\min(p_0,1-p_0)$ reach about ten. The last three rows move $N$ towards and "
        r"across the first aliasing boundary $r=N\phi/(\pi/2)=1$, where $p_0\to 0$, the outcome "
        r"$K=0$ takes almost all of the probability mass and the approximation fails outright.}",
        r"  \label{tab:gaussianity}",
        r"  \begin{tabular}{lrrrrrrrr}",
        r"    \toprule",
        r"    Configuration & $N$ & $m$ & $r$ & $m\min(p_0,1-p_0)$ & $s_R^2$ & $1/(4mN^2)$ "
        r"& ratio & KS \\",
        r"    \midrule",
    ]
    for i, r in enumerate(rows):
        if i == 4:
            lines.append(r"    \midrule")
        lines.append(
            "    {label} & {N} & {m} & ${r}$ & ${mm}$ & ${s2}$ & ${va}$ & ${ra}$ & ${ks}$ \\\\".format(
                label=r["tex_label"], N=r["N"], m=r["m"], r=fixed3(r["r"]),
                mm=fixed3(r["m_min_p0"]), s2=sci3_tex(r["s2_mc"]), va=sci3_tex(r["var_asym"]),
                ra=fixed3(r["ratio_mc"]), ks=fixed3(r["ks_exact"])))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the optional figure
def regular_region(m, thresh=10.0):
    """Range of r on which m*min(p0, 1-p0) >= thresh, or None when no r qualifies."""
    q = thresh / m
    if q > 0.5:
        return None
    lo = np.arccos(np.sqrt(1 - q)) / (np.pi / 2)     # p0 = 1 - q
    hi = np.arccos(np.sqrt(q)) / (np.pi / 2)         # p0 = q
    return lo, hi


def make_figure(ms=(20, 114, 334), path_stem=None):
    rr = np.linspace(0.05, 1.10, 421)
    colors = {ms[0]: BLUE, ms[1]: ORANGE, ms[2]: AQUA}
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    curves = {}
    for m in ms:
        ks = np.empty_like(rr)
        sd = np.empty_like(rr)
        for i, r in enumerate(rr):
            z, w = exact_law(m, r * np.pi / 2)
            ks[i] = exact_ks(z, w)
            sd[i] = np.sqrt(exact_moments(z, w)[1])
        curves[m] = (ks, sd)
        axl.plot(rr, ks, color=colors[m], lw=1.8, label=f"$m={m}$", zorder=3)
        axr.plot(rr, sd, color=colors[m], lw=1.8, label=f"$m={m}$", zorder=3)

    for ax in (axl, axr):
        ax.axvline(1.0, color=INK2, lw=1.0, ls="--", zorder=1)
        # sits above the axes, so it never collides with the curves
        ax.text(1.0, 1.015, "first aliasing boundary", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xlabel(r"$r = N\phi\,/\,(\pi/2)$")
        ax.set_xlim(0.05, 1.10)
        ax.grid(True, alpha=0.6)
        ax.set_axisbelow(True)

    # band under the axis marking where the two-sided marker m*min(p0, 1-p0) >= 10 holds
    axl.set_ylim(-0.11, 1.03)
    axl.axhline(0.0, color="#8a8a85", lw=0.8, zorder=2)
    for j, m in enumerate(ms):
        reg = regular_region(m)
        y = -0.035 - 0.030 * j
        if reg is None:
            continue
        if reg[1] - reg[0] < 0.02:      # m = 20 qualifies only at the single point p0 = 1/2
            axl.plot([0.5 * (reg[0] + reg[1])], [y], marker="|", ms=7, mew=3.0,
                     color=colors[m], zorder=3)
            continue
        axl.plot(reg, [y, y], color=colors[m], lw=3.4, solid_capstyle="butt", zorder=3)
    axl.text(1.09, -0.035, r"$m\min(p_0,1-p_0)\geq 10$", fontsize=7.5, color=INK2,
             ha="right", va="center")

    axl.set_ylabel("exact KS distance from the Gaussian model")
    axl.set_title("exact CDF vs. $\\mathcal{N}(\\phi,\\,1/(4mN^2))$", fontsize=9.5, loc="left")
    axl.legend(loc="upper center", ncol=3, fontsize=9)

    axr.axhline(1.0, color=INK2, lw=1.0, zorder=2)
    axr.set_ylim(-0.05, 1.42)
    axr.set_ylabel(r"exact SD $/\;[1/(2N\sqrt{m})]$")
    axr.set_title("exact SD vs. the asymptotic SD", fontsize=9.5, loc="left")
    axr.legend(loc="lower center", ncol=3, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return curves


# --------------------------------------------------------------------------- regular region
def regular_region_summary(ms=SELECTED_M, n_grid=97):
    """Worst-case departure from the Gaussian law over the regular region of the principal branch.

    Sweeps r over (0, 1) and keeps the grid points satisfying m*min(p0, 1-p0) >= 10. On the
    principal branch the folded centre arccos(sqrt(p0))/N coincides with phi, so this summary does
    not depend on which of the two centring conventions is used.
    """
    rr = np.linspace(0.02, 0.98, n_grid)
    rows = []
    for m in ms:
        ks, sd_err, bias = [], [], []
        for r in rr:
            theta = r * np.pi / 2
            p0 = float(np.cos(theta) ** 2)
            if m * min(p0, 1 - p0) < 10:
                continue
            z, w = exact_law(m, theta)
            mean_z, var_z, _ = exact_moments(z, w)
            ks.append(exact_ks(z, w))
            sd_err.append(abs(np.sqrt(var_z) - 1.0))
            bias.append(abs(mean_z))
        rows.append(dict(m=m, n_regular=len(ks), max_ks=max(ks), median_ks=float(np.median(ks)),
                         max_rel_sd_error=max(sd_err), max_abs_bias_sd=max(bias)))
    return rows


# --------------------------------------------------------------------------- second probe
def second_probe_summary(path=PARAMS_CSV):
    """Can each tuned operating point afford the opening pilot plus one bisection probe?

    Binary search opens at N_min and, if it can, probes the midpoint of {N_min, ..., N_max}. Both
    probes cost m' shots, so the minimum budget for two probes is m' * (N_min + N_second).
    """
    with open(path, newline="") as f:
        table = [r for r in csv.DictReader(f)
                 if r["algorithm"] == "binary_deep" and r["reported_for"] == "tested_budget"]
    rows = []
    for scenario_id, budget, purpose in SELECTED_POINTS:
        match = [r for r in table
                 if r["scenario_id"] == scenario_id and int(r["budget"]) == budget]
        if len(match) != 1:
            raise RuntimeError(f"expected one row for {scenario_id}, B={budget}; got {len(match)}")
        row = match[0]
        m = int(json.loads(row["params"])["m_exploration"])
        n_min = max(int(np.pi // (2 * float(row["phi_max"]))), 1)
        n_max = max(int(np.pi // (2 * float(row["phi_min"]))), 1)
        n_second = n_min + (n_max - n_min) // 2
        need = m * (n_min + n_second)
        rows.append(dict(scenario_id=scenario_id, purpose=purpose, budget=budget, m=m,
                         N_min=n_min, N_second=n_second, second_probe_min_budget=need,
                         second_probe_fits=int(need <= budget)))
    return rows


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build_table()

    print("verification")
    worst_mass = max(abs(r["pmf_mass"] - 1.0) for r in rows)
    worst_z = max(abs(r["mc_var_z"]) for r in rows)
    print(f"  every enumerated PMF sums to 1: max |sum - 1| = {worst_mass:.2e}")
    print(f"  MC vs exact variance:           max |z| = {worst_z:.2f} "
          f"(|z| < 3 means agreement within sampling error)")
    assert worst_mass < 1e-9, "an enumerated PMF does not sum to one"
    assert worst_z < 4.0, "a Monte-Carlo variance disagrees with the exact variance"

    print("\nseven-row table (phi = 0.05, R = 50,000, seed = 20260825)")
    hdr = f"  {'N':>3} {'m':>5} {'r':>7} {'m*min':>9} {'s2_MC':>11} {'asym':>11} " \
          f"{'ratio_MC':>9} {'ratio_ex':>9} {'KS':>7} {'bias/sd':>8}"
    print(hdr)
    for r in rows:
        print(f"  {r['N']:>3} {r['m']:>5} {r['r']:>7.4f} {r['m_min_p0']:>9.3f} "
              f"{r['s2_mc']:>11.4e} {r['var_asym']:>11.4e} {r['ratio_mc']:>9.3f} "
              f"{r['ratio_exact']:>9.3f} {r['ks_exact']:>7.4f} {r['bias_exact_sigma']:>8.3f}")

    write_csv(rows, f"{OUT}/gaussianity_table.csv")
    with open(f"{OUT}/gaussianity_table.md", "w") as f:
        f.write(markdown_table(rows) + "\n")
    with open(f"{OUT}/gaussianity_table.tex", "w") as f:
        f.write(latex_table(rows) + "\n")

    print("\nregular region of the principal branch, at the selected exploration shot counts")
    reg_rows = regular_region_summary()
    print(f"  {'m':>5} {'points':>7} {'max KS':>8} {'median KS':>10} {'max SD err':>11} {'max |bias|':>11}")
    for r in reg_rows:
        print(f"  {r['m']:>5} {r['n_regular']:>7} {r['max_ks']:>8.4f} {r['median_ks']:>10.4f} "
              f"{100 * r['max_rel_sd_error']:>10.2f}% {r['max_abs_bias_sd']:>10.4f}s")
    write_csv(reg_rows, f"{OUT}/regular_region_summary.csv")

    print("\nsecond bisection probe: affordable at the tuned operating points?")
    probe_rows = second_probe_summary()
    print(f"  {'scenario':>10} {'purpose':>10} {'B':>11} {'m':>6} {'need':>12} {'fits':>5}")
    for r in probe_rows:
        print(f"  {r['scenario_id']:>10} {r['purpose']:>10} {r['budget']:>11,} {r['m']:>6} "
              f"{r['second_probe_min_budget']:>12,} {'yes' if r['second_probe_fits'] else 'NO':>5}")
    write_csv(probe_rows, f"{OUT}/second_probe_budget.csv")

    print("\nfigure (optional supporting material)")
    make_figure(path_stem=f"{OUT}/fig_gaussianity_boundary")
    for m in (20, 114, 334):
        reg = regular_region(m)
        print(f"  m={m:4d}: m*min(p0,1-p0) >= 10 on r in "
              + ("(empty)" if reg is None else f"[{reg[0]:.3f}, {reg[1]:.3f}]"))

    print(f"\nwrote {OUT}/gaussianity_table.{{csv,md,tex}}, {OUT}/regular_region_summary.csv,\n"
          f"      {OUT}/second_probe_budget.csv and {OUT}/fig_gaussianity_boundary.{{pdf,png}}")


if __name__ == "__main__":
    main()
