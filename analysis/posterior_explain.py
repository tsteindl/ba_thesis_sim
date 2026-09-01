"""Figures for docs/POSTERIOR.md — what the exact-posterior depth criterion does and what it buys.

Four figures, no new simulation sweeps: figures 1-2 replay ONE bisection trial to show the mechanism,
figures 3-4 read the existing result CSVs.

    python analysis/posterior_explain.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from pipeline_io import path
from qmetrology.posterior import posterior, depth_from_posterior
from qmetrology.safeguard import pilot_sd, risk_optimal_depth

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})

# ------------------------------------------------------------------ one representative trial
PMIN, PMAX, EPS = 0.001, 0.01, 1e-4
M, CONF, SEED = 60, 0.8, 11
PHI = 0.0032
BUDGET = 900_000


def bisect(rng, phi, pmax, pmin, m, budget, conf):
    """Algorithm 5's exploration, recording (N, hits) and the accept/reject flag of every probe."""
    N_min, N_max = max(int(np.pi // (2 * pmax)), 1), max(int(np.pi // (2 * pmin)), 1)
    N = N_min
    hits = int(rng.binomial(m, np.cos(N * phi) ** 2))
    ph = float(np.arccos(np.sqrt(hits / m)) / N)
    used = m * N
    phi_1 = norm.ppf(1 - conf, ph, np.sqrt(1 / (4 * m * N ** 2)))
    probes = [(N, hits, True)]
    lb, ub = N_min, N_max
    N += (ub - N) // 2
    while True:
        if used + m * N > budget:
            break
        hits = int(rng.binomial(m, np.cos(N * phi) ** 2))
        ph = float(np.arccos(np.sqrt(hits / m)) / N)
        used += m * N
        keep = ph >= phi_1
        probes.append((N, hits, keep))
        t = N
        if not keep:
            N -= (N - lb) // 2
            ub = t
        else:
            N += (ub - N) // 2
            lb = t
            phi_1 = norm.ppf(1 - conf, ph, np.sqrt(1 / (4 * m * t ** 2)))
        if t == N or N < N_min or N > N_max or used >= budget:
            break
    return probes, used, lb


def trial():
    rng = np.random.default_rng(SEED)
    probes, used, L = bisect(rng, PHI, PMAX, PMIN, M, BUDGET, CONF)
    return probes, used, L



# ------------------------------------------------------------------ fig 0: the fold the estimator applies
def fig0(out, N=510):
    """What phi_hat converges to as m -> infinity, as a function of the true phi.

    E[k]/m = cos^2(N phi) exactly, so the noiseless estimator returns
        arccos|cos(N phi)| / N = d(N phi) / N,   d(theta) = min_j |theta - j pi|,
    which is the sawtooth below: an identity on [0, pi/2N] and a reflection thereafter.
    """
    phi = np.linspace(PMIN, PMAX, 6000)
    fold = np.abs(((N * phi + np.pi / 2) % np.pi) - np.pi / 2) / N
    lim = np.pi / (2 * N)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.axvspan(lim, PMAX, color="#f2d9cf", alpha=0.55, lw=0)
    ax.plot(phi, fold, lw=2.6, color=ORANGE, zorder=2,
            label=r"$\hat\varphi$ as $m\to\infty$ at $N=%d$" % N)
    ax.plot(phi, phi, lw=1.6, ls=(0, (5, 4)), color=INK2, zorder=3,
            label="identity: what an estimator should return")
    ax.axvline(lim, lw=1.4, color="#8a2f2f")
    ax.text(lim * 1.05, PMAX * 0.66, r"$\pi/2N$", color="#8a2f2f", fontsize=10)
    ax.text(lim * 1.05, PMAX * 0.53,
            "beyond here the estimator is\nnot merely noisy - it is wrong,\nand nothing in its output says so",
            color="#8a2f2f", fontsize=9)
    ax.set_xlabel(r"true $\varphi$")
    ax.set_ylabel(r"value $\hat\varphi$ converges to")
    ax.set_title("The fold: $\\hat\\varphi$ cannot leave $[0,\\pi/2N]$", fontsize=11, loc="left")
    ax.set_xlim(PMIN, PMAX)
    ax.set_ylim(0, PMAX)
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 1: the two readings of a probe
def fig1(probes, out):
    n_opt = int(np.pi // (2 * PHI))
    phi = np.linspace(PMIN, PMAX, 4000)

    # a shallow probe (the RE pilot) and the deepest probe the overshoot test rejected
    shallow = probes[0]
    rejected = [p for p in probes if not p[2]]
    deep = max(rejected, key=lambda p: p[0]) if rejected else max(probes, key=lambda p: p[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    ax = axes[0]
    for (N, hits, _keep), c, lab in ((shallow, BLUE, "opening probe"), (deep, ORANGE, "deep probe")):
        ax.plot(phi, np.cos(N * phi) ** 2, lw=2, color=c, label=f"{lab}: $N={N}$")
        ax.axhline(hits / M, lw=1.2, ls=":", color=c)
        # every phi consistent with the reading
        sol = phi[np.abs(np.cos(N * phi) ** 2 - hits / M) < 0.004]
        ax.plot(sol, np.full_like(sol, hits / M), ls="none", marker="o", ms=6,
                mfc="white", mec=c, mew=1.8, zorder=4)
    ax.axvline(PHI, lw=1.4, color=INK2, ls="--")
    ax.text(PHI * 1.05, 1.02, r"true $\varphi$", color=INK2, fontsize=9)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$p_0(\varphi)=\cos^2(N\varphi)$")
    ax.set_title("A single probe is a curve, not a point", fontsize=11, loc="left")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(axis="y")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.06, "open circles = every $\\varphi$ that explains the reading\n"
                        "deep probe: many; opening probe: one",
            transform=ax.transAxes, fontsize=9, color=INK2)

    ax = axes[1]
    counts = [(N, h) for N, h, _ in probes]
    g, w = posterior(counts, M, PMIN, PMAX, 8000)
    for N, h, keep in probes:
        _g, wi = posterior([(N, h)], M, PMIN, PMAX, 8000)
        ax.plot(_g, wi / wi.max(), lw=1.0, color=MUTED, zorder=1)
    ax.plot(g, w / w.max(), lw=2.4, color=BLUE, label="exact posterior, all probes", zorder=3)
    ax.fill_between(g, 0, w / w.max(), color=BLUE, alpha=0.12, zorder=2)
    # normal approximation from the opening probe alone
    N0, h0, _ = probes[0]
    ph0 = float(np.arccos(np.sqrt(h0 / M)) / N0)
    nd = np.exp(-0.5 * ((g - ph0) / pilot_sd(N0, M)) ** 2)
    ax.plot(g, nd, lw=2.2, color=ORANGE, ls="--", label="normal law, opening probe only", zorder=3)
    ax.axvline(PHI, lw=1.4, color=INK2, ls="--")
    ax.axvline(np.pi / (2 * n_opt), lw=0, color="none")
    ax.set_xlim(PMIN, 0.0055)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel("density (scaled to peak)")
    ax.set_title("What the whole history says about $\\varphi$", fontsize=11, loc="left")
    ax.plot([], [], lw=1.0, color=MUTED, label="one probe's likelihood")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(PHI * 1.04, 0.55, r"true $\varphi$", color=INK2, fontsize=9)
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return ph0


# ------------------------------------------------------------------ fig 2: the criterion
def fig2(probes, used, L, out):
    rem = BUDGET - used
    n_opt = int(np.pi // (2 * PHI))
    n_sup = int(np.pi // (2 * PMIN))
    counts = [(N, h) for N, h, _ in probes]
    N0, h0, _ = probes[0]
    ph0 = float(np.arccos(np.sqrt(h0 / M)) / N0)
    sd0 = pilot_sd(N0, M)

    n_min = min(max(int(np.pi // (2 * PMAX)), 1), n_sup)
    Ns = np.arange(1, n_sup + 1)
    thr = np.pi / (2.0 * Ns)

    g, w = posterior(counts, M, PMIN, PMAX, 20000)
    cdf = np.cumsum(w)
    idx = np.searchsorted(g, thr, side="left") - 1
    p_post = np.where(idx < 0, 0.0, cdf[np.clip(idx, 0, len(cdf) - 1)])
    p_post = np.where(thr >= g[-1], 1.0, p_post)
    p_norm = ndtr((thr - ph0) / sd0)
    # the safeguard's own accuracy factor: whole shots, m = floor(B'/N)
    p_conv = 2.0 * ndtr(2.0 * EPS * Ns * np.sqrt(int(rem) // Ns)) - 1.0

    N_post = depth_from_posterior(counts, M, PMIN, PMAX, rem, EPS, N_max=n_sup)
    N_norm = risk_optimal_depth(ph0, sd0, rem, EPS, N_min=n_min, N_max=n_sup,
                                support=(PMIN, PMAX))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    ax = axes[0]
    ax.plot(Ns, p_post, lw=2.4, color=BLUE, label=r"$P(\varphi<\pi/2N\mid\mathrm{all\ probes})$, exact")
    ax.plot(Ns, p_norm, lw=2.2, ls="--", color=ORANGE,
            label=r"$\Phi\left((\pi/2N-\hat\varphi_0)/\sigma\right)$, normal")
    ax.plot(Ns, p_conv, lw=2.0, color=AQUA, label=r"$2\Phi(2\varepsilon N\sqrt{\lfloor B'/N\rfloor})-1$, precision")
    ax.axvline(n_opt, lw=1.4, ls="--", color=INK2)
    ax.text(n_opt * 1.02, 0.5, r"$N_{\rm opt}$", color=INK2, fontsize=9)
    ax.set_xlabel("exploitation depth $N$")
    ax.set_ylabel("probability")
    ax.set_title("The two factors", fontsize=11, loc="left")
    ax.set_xscale("log")
    ax.grid(axis="y")
    ax.legend(loc="center left", fontsize=8.5)

    ax = axes[1]
    ax.plot(Ns, p_post * p_conv, lw=2.4, color=BLUE, label="exact posterior")
    ax.plot(Ns, p_norm * p_conv, lw=2.2, ls="--", color=ORANGE, label="normal, one pilot")
    for N, c, dy in ((N_norm, ORANGE, -26), (N_post, BLUE, 12)):
        y = (p_post if c == BLUE else p_norm)[N - 1] * p_conv[N - 1]
        ax.plot([N], [y], marker="o", ms=9, mfc="white", mec=c, mew=2.2, zorder=5)
        ax.annotate(f"$N^*={N}$", (N, y), textcoords="offset points", xytext=(-4, dy),
                    ha="right", color=c, fontsize=9.5)
    ax.axvline(n_opt, lw=1.4, ls="--", color=INK2)
    ax.text(n_opt - 6, 0.30, r"$N_{\rm opt}=%d$" % n_opt, color=INK2, fontsize=9,
            ha="right", rotation=90, va="bottom")
    ax.axvline(L, lw=1.4, ls=":", color="#8a2f2f")
    ax.text(L + 6, 0.30, r"$L=%d$ (bisection's bound)" % L, color="#8a2f2f", fontsize=9,
            rotation=90, va="bottom")
    ax.set_xlabel("exploitation depth $N$")
    ax.set_ylabel("objective")
    ax.set_title("Their product picks the depth", fontsize=11, loc="left")
    ax.set_xlim(300, 700)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return N_post, N_norm, n_opt, rem


# ------------------------------------------------------------------ fig 3: what it buys
def fig3(out):
    rows = list(csv.DictReader(open(path("posterior_all.csv"))))
    pairs = [("linear_s", "linear_post", "linear search"),
             ("re_normal", "re_post", "reverse engineering"),
             ("binary_first", "binary_post", "binary search")]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for i, (a, b, lab) in enumerate(pairs):
        d = np.array([float(r[b + "_rate"]) - float(r[a + "_rate"]) for r in rows])
        y = len(pairs) - 1 - i
        jit = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.26
        ax.plot(d, y + jit, ls="none", marker="o", ms=5, mfc=BLUE, mec="white", mew=0.8,
                alpha=0.65, zorder=3)
        ax.plot([d.mean()], [y], marker="D", ms=10, color=ORANGE, mec="white", mew=1.4, zorder=4)
        ax.annotate(f"mean {d.mean():+.2f} pp", (d.mean(), y), textcoords="offset points",
                    xytext=(0, 16), ha="center", color=ORANGE, fontsize=9.5)
        ax.text(-1.4, y, lab, ha="right", va="center", fontsize=10, color=INK)
    ax.axvline(0, lw=1.4, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(pairs) - 0.25)
    ax.set_xlim(-1.4, 10)
    ax.set_xlabel("convergence rate, exact posterior − normal approximation (pp)")
    ax.set_title("Each dot is one operating point (40 points, R = 30,000, paired)",
                 fontsize=11, loc="left")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4: cost of the bisection's cap
def fig4(out):
    rows = list(csv.DictReader(open(path("binary_rescue.csv"))))
    g = lambda r, a: float(r[a + "_rate"])
    base = np.array([g(r, "first") for r in rows])
    series = [("post", BLUE, "posterior, capped only by the prior support"),
              ("postL", ORANGE, r"posterior, capped at $L$"),
              ("L1", AQUA, r"exploit at $N=L-1$ (bisection's own limit)")]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    q = np.linspace(0, 100, len(rows))
    for a, c, lab in series:
        d = np.sort(np.array([g(r, a) for r in rows]) - base)
        ax.plot(q, d, lw=2.2, color=c, label=lab)
        ax.annotate(f"mean {d.mean():+.1f}", (100, d[-1]), textcoords="offset points",
                    xytext=(6, -2), color=c, fontsize=9, va="center")
    ax.axhline(0, lw=1.4, color=INK2)
    ax.set_xlabel("operating points, sorted by loss (%)")
    ax.set_ylabel("convergence rate vs. uncapped rule (pp)")
    ax.set_title("What capping the depth at the bisection's bound costs (167 points, R = 30,000)",
                 fontsize=11, loc="left")
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 108)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    probes, used, L = trial()
    print("trial: phi=%.5f  N_opt=%d  probes=%s  used=%d  L=%d"
          % (PHI, int(np.pi // (2 * PHI)), [(N, h, k) for N, h, k in probes], used, L))
    fig0(path("fig_posterior_fold.png"))
    ph0 = fig1(probes, path("fig_posterior_mechanism.png"))
    N_post, N_norm, n_opt, rem = fig2(probes, used, L, path("fig_posterior_criterion.png"))
    print("phi_hat_0=%.5f  N*_post=%d  N*_norm=%d  N_opt=%d  L=%d  rem=%d"
          % (ph0, N_post, N_norm, n_opt, L, rem))
    fig3(path("fig_posterior_gain.png"))
    fig4(path("fig_binary_cap_cost.png"))
    print("wrote 4 figures to results/")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ numbers quoted in POSTERIOR.md
def grid_resolution_check(out_csv=path("posterior_grid_check.csv"), R=300, n_grid=4096):
    """Is the 4096-point grid fine enough to resolve the sharpest ridge the exploration can produce?

    A probe of m shots at depth N reading k ~ 0 hits contributes sin^{2m}(N phi). Writing
    delta = pi/2 - N phi, sin(N phi) = cos(delta) and sin^{2m} = exp(2m log cos delta) ~ exp(-m delta^2),
    so the ridge is Gaussian in delta with sd 1/sqrt(2m), i.e. sd 1/(N sqrt(2m)) in phi. The grid
    spacing is (phi_max - phi_min)/(n_grid - 1). The ratio is the number of grid points across the ridge;
    below ~5 the quadrature under-resolves the posterior.

    Replays the real bisection at every operating point of results/binary_rescue.csv, using that point's
    tuned (m', conf), and reports the DEEPEST probe actually taken (95th percentile over trials).
    pts_per_ridge is grid points per ridge SD.
    """
    rows = list(csv.DictReader(open(path("binary_rescue.csv"))))
    seen, out = set(), []
    for r in rows:
        key = (r["setting"], r["budget"])
        if key in seen:
            continue
        seen.add(key)
        pmin, pmax = float(r["phi_min"]), float(r["phi_max"])
        m, conf, B = int(r["post_m"]), float(r["post_conf"]), int(r["budget"])
        deepest = []
        for s in range(R):
            rng = np.random.default_rng(s)
            phi = float(rng.uniform(pmin, pmax))
            try:
                probes, _u, _L = bisect(rng, phi, pmax, pmin, m, B, conf)
            except (ValueError, ZeroDivisionError):
                continue
            deepest.append(max(N for N, _h, _k in probes))
        if not deepest:
            continue
        N95 = float(np.percentile(deepest, 95))
        spacing = (pmax - pmin) / (n_grid - 1)
        width = 1.0 / (N95 * np.sqrt(2.0 * m))   # one sd of the ridge, in phi
        out.append(dict(setting=r["setting"], budget=B, m=m, N95=N95,
                        spacing=spacing, ridge_width=width, pts_per_ridge=width / spacing))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    worst = sorted(out, key=lambda d: d["pts_per_ridge"])
    print("\n--- grid resolution (points across the sharpest ridge, n_grid=4096) ---")
    print(f"{'setting':30s} {'budget':>12s} {'m':>7s} {'N95':>8s} {'pts/ridge':>10s}")
    for d in worst[:6]:
        print(f"{d['setting']:30s} {d['budget']:12d} {d['m']:7d} {d['N95']:8.0f} {d['pts_per_ridge']:10.1f}")
    print("  ...")
    for d in worst[-3:]:
        print(f"{d['setting']:30s} {d['budget']:12d} {d['m']:7d} {d['N95']:8.0f} {d['pts_per_ridge']:10.1f}")
    p = np.array([d["pts_per_ridge"] for d in out])
    print(f"  {len(p)} points: min {p.min():.1f}, 5th pct {np.percentile(p,5):.1f}, "
          f"median {np.median(p):.1f}, frac below 5: {(p<5).mean():.1%}")
    return out


def posterior_sharpness(R=400, n_grid=4096):
    """How much tighter is the exact posterior than the opening probe's normal law?

    Reports sd(posterior)/sd(pilot) over R trials at the operating point of figures 1-2.
    """
    ratios, covered_post, covered_norm = [], 0, 0
    for s in range(R):
        rng = np.random.default_rng(1000 + s)
        phi = float(rng.uniform(PMIN, PMAX))
        probes, _u, _L = bisect(rng, phi, PMAX, PMIN, M, BUDGET, CONF)
        counts = [(N, h) for N, h, _ in probes]
        g, w = posterior(counts, M, PMIN, PMAX, n_grid)
        mu = float((g * w).sum())
        sd = float(np.sqrt(((g - mu) ** 2 * w).sum()))
        N0, h0, _ = probes[0]
        ph0 = float(np.arccos(np.sqrt(h0 / M)) / N0)
        sd0 = pilot_sd(N0, M)
        ratios.append(sd0 / sd)
        covered_post += abs(phi - mu) < 2 * sd
        covered_norm += abs(phi - ph0) < 2 * sd0
    r = np.array(ratios)
    print("\n--- posterior sharpness vs the opening pilot (R=%d) ---" % R)
    print(f"  sd(pilot)/sd(posterior): median {np.median(r):.1f}x, "
          f"10th-90th pct {np.percentile(r,10):.1f}x-{np.percentile(r,90):.1f}x")
    print(f"  2-sd interval covers the truth: posterior {covered_post/R:.1%}, pilot {covered_norm/R:.1%}")
    return r


def worked_example():
    """The 11 probes of the figure-1/2 trial, with each one's contribution."""
    probes, used, L = trial()
    counts = [(N, h) for N, h, _ in probes]
    g, w = posterior(counts, M, PMIN, PMAX, 20000)
    mu = float((g * w).sum())
    sd = float(np.sqrt(((g - mu) ** 2 * w).sum()))
    n_opt = int(np.pi // (2 * PHI))
    print("\n--- worked example: phi=%.5f, N_opt=%d, m'=%d, conf=%.2f ---" % (PHI, n_opt, M, CONF))
    print(f"{'i':>2s} {'N_i':>6s} {'k_i':>4s} {'k/m':>6s} {'phi_hat':>9s} {'flag':>8s} "
          f"{'N_i>N_opt':>10s} {'pi/2N_i':>9s}")
    for i, (N, h, keep) in enumerate(probes, 1):
        ph = float(np.arccos(np.sqrt(h / M)) / N)
        print(f"{i:2d} {N:6d} {h:4d} {h/M:6.3f} {ph:9.5f} {'accept' if keep else 'REJECT':>8s} "
              f"{'yes' if N > n_opt else 'no':>10s} {np.pi/(2*N):9.5f}")
    N0, h0, _ = probes[0]
    ph0 = float(np.arccos(np.sqrt(h0 / M)) / N0)
    print(f"  pilot  : phi_hat_0 = {ph0:.5f}, sigma = {pilot_sd(N0, M):.5e}  "
          f"(error {abs(ph0-PHI)/pilot_sd(N0,M):.2f} sigma)")
    print(f"  posterior: mean = {mu:.5f}, sd = {sd:.5e}  (error {abs(mu-PHI)/sd:.2f} sd), "
          f"{pilot_sd(N0,M)/sd:.1f}x tighter")
    print(f"  budget: explored {used:,} of {BUDGET:,}; L = {L}")
