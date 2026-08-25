"""Finite-sample audit of the Gaussian approximation used by binary search.

The audit has two parts:

1. Compute the *exact* distribution of

       phi_hat = arccos(sqrt(K/m)) / N,   K ~ Bin(m, cos(N phi)^2),

   at the exploration shot counts selected by the production sweep.  This is
   stronger and less noisy than a Monte Carlo normality check.
2. Replay the complete reported binary-search algorithm at the corresponding
   operating points.  This tests whether the raw approximation survives the
   adaptive selection of the deepest accepted probe used by the safeguard.

No thesis source is changed.  Run from the repository root with

    python analysis/binary_gaussianity_audit.py

Outputs are written to results/binary_gaussianity/.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, kstest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qmetrology.algorithms import find_phi_fixed_budget_binary_search_deep
from qmetrology.safeguard import pilot_sd
from qmetrology.trace import AlgorithmTrace

OUT = ROOT / "results" / "binary_gaussianity"
R_TRIALS = int(os.environ.get("BINARY_GAUSSIANITY_R", "20000"))
SEED = 20260824


@dataclass(frozen=True)
class Configuration:
    scenario_id: str
    purpose: str
    budget: int
    phi_min: float
    phi_max: float
    eps: float
    m: int
    conf: float
    heldout_rate: float

    @property
    def short_label(self) -> str:
        if self.purpose == "Table 3.1":
            return "Table 3.1"
        return self.scenario_id.replace("_", " ") + " near B90"


SELECTED = (
    ("narrow_e3", 10_000, "Table 3.1"),
    ("narrow_e3", 29_173, "near B90"),
    ("narrow_e4", 2_917_365, "near B90"),
    ("small_e4", 394_843, "near B90"),
    ("wide_e4", 2_059_436, "near B90"),
)


def load_configurations() -> list[Configuration]:
    """Read the selected parameters rather than duplicating them in this audit."""
    path = ROOT / "results" / "consolidated" / "optimal_params.csv"
    data = pd.read_csv(path)
    data = data[(data.algorithm == "binary_deep") & (data.reported_for == "tested_budget")]
    configs = []
    for scenario_id, budget, purpose in SELECTED:
        row = data[(data.scenario_id == scenario_id) & (data.budget == budget)]
        if len(row) != 1:
            raise RuntimeError(f"expected one parameter row for {scenario_id}, B={budget}; got {len(row)}")
        row = row.iloc[0]
        params = json.loads(row.params)
        configs.append(Configuration(
            scenario_id=scenario_id,
            purpose=purpose,
            budget=budget,
            phi_min=float(row.phi_min),
            phi_max=float(row.phi_max),
            eps=float(row.eps),
            m=int(params["m_exploration"]),
            conf=float(params["conf"]),
            heldout_rate=float(row.heldout_rate),
        ))
    return configs


def exact_metrics(m: int, r: float) -> dict[str, float]:
    """Exact transformed-binomial law; N cancels after standardisation.

    r = N phi / (pi/2).  The Gaussian is centred on the folded, identifiable
    phase acos(|cos(N phi)|)/N, not on phi after the aliasing boundary.
    """
    theta = np.pi * r / 2.0
    p = float(np.cos(theta) ** 2)
    k = np.arange(m + 1)
    mass = binom.pmf(k, m, p)
    estimate = np.arccos(np.sqrt(k / m))
    center = float(np.arccos(np.sqrt(p)))
    sigma = 1.0 / (2.0 * np.sqrt(m))

    order = np.argsort(estimate)
    x, q = estimate[order], mass[order]
    cdf_right = np.cumsum(q)
    cdf_left = cdf_right - q
    gaussian_cdf = norm.cdf(x, loc=center, scale=sigma)
    d_ks = max(float(np.max(np.abs(cdf_right - gaussian_cdf))),
               float(np.max(np.abs(cdf_left - gaussian_cdf))))

    mean = float(np.sum(q * x))
    sd = float(np.sqrt(np.sum(q * (x - mean) ** 2)))
    return {
        "m": m,
        "r": r,
        "p0": p,
        "expected_smaller_count": m * min(p, 1.0 - p),
        "ks_distance": d_ks,
        "bias_in_model_sd": (mean - center) / sigma,
        "sd_ratio": sd / sigma,
        "endpoint_atom": float(mass[0] + mass[-1]),
    }


def controlled_audit(configs: list[Configuration]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_grid = np.round(np.linspace(0.02, 1.98, 197), 8)
    shot_counts = sorted({c.m for c in configs})
    rows = [exact_metrics(m, float(r)) for m in shot_counts for r in r_grid]
    curves = pd.DataFrame(rows)

    summaries = []
    for m, group in curves.groupby("m"):
        regular = group[group.expected_smaller_count >= 10]
        summaries.append({
            "m": m,
            "regular_grid_points": len(regular),
            "max_ks_regular": regular.ks_distance.max(),
            "median_ks_regular": regular.ks_distance.median(),
            "max_abs_sd_error_regular": np.max(np.abs(regular.sd_ratio - 1.0)),
            "max_abs_bias_sd_regular": np.max(np.abs(regular.bias_in_model_sd)),
        })
    return curves, pd.DataFrame(summaries)


def _distribution_stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {k: np.nan for k in ("n", "mean_z", "sd_z", "ks_z", "coverage_95")}
    return {
        "n": len(a),
        "mean_z": float(np.mean(a)),
        "sd_z": float(np.std(a, ddof=1)),
        "ks_z": float(kstest(a, norm.cdf).statistic),
        "coverage_95": float(np.mean(np.abs(a) <= norm.ppf(0.975))),
    }


def replay_configuration(config: Configuration, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    opening_z: list[float] = []
    selected_safe_z: list[float] = []
    probe_z: list[float] = []
    regular_probe_z: list[float] = []
    opening_regular = selected_regular = selected_safe = only_opening = converged = 0
    false_alarm_trials = miss_trials = 0
    guess_overshoot = final_overshoot = 0
    exploration_shares: list[float] = []
    probes_total = 0

    for _ in range(R_TRIALS):
        phi = float(rng.uniform(config.phi_min, config.phi_max))
        tr = AlgorithmTrace(algorithm="binary_deep")
        estimate, spent = find_phi_fixed_budget_binary_search_deep(
            rng, phi, config.phi_max, config.phi_min, config.m, config.budget,
            config.eps, config.conf, trace=tr,
        )
        tr.finalize(phi, estimate, spent, config.eps)
        converged += int(bool(tr.converged))
        guess_overshoot += int(int(tr.N_guess) > int(tr.N_opt))
        final_overshoot += int(int(tr.N_star) > int(tr.N_opt))
        exploration_shares.append(float(tr.budget_exploration) / config.budget)
        probes_total += len(tr.probes)
        only_opening += int(len(tr.probes) == 1)

        n_open = int(tr.opening_pilot_N)
        ph_open = float(tr.opening_pilot_phi_hat)
        opening_z.append((ph_open - phi) / pilot_sd(n_open, config.m))
        p_open = float(np.cos(n_open * phi) ** 2)
        opening_regular += int(config.m * min(p_open, 1.0 - p_open) >= 10)

        n_acc = int(tr.accepted_pilot_N)
        ph_acc = float(tr.accepted_pilot_phi_hat)
        p_acc = float(np.cos(n_acc * phi) ** 2)
        selected_regular += int(config.m * min(p_acc, 1.0 - p_acc) >= 10)
        safe_acc = n_acc <= int(tr.N_opt)
        selected_safe += int(safe_acc)
        if safe_acc:
            selected_safe_z.append((ph_acc - phi) / pilot_sd(n_acc, config.m))

        false_alarm = miss = False
        for probe in tr.probes:
            n = int(probe.N)
            p0 = float(np.cos(n * phi) ** 2)
            folded_center = float(np.arccos(abs(np.cos(n * phi))) / n)
            z = (float(probe.phi_hat) - folded_center) / pilot_sd(n, config.m)
            probe_z.append(z)
            if config.m * min(p0, 1.0 - p0) >= 10:
                regular_probe_z.append(z)
            safe = n <= int(tr.N_opt)
            false_alarm |= bool(probe.declared_overshoot and safe)
            miss |= bool((not probe.declared_overshoot) and (not safe))
        false_alarm_trials += int(false_alarm)
        miss_trials += int(miss)

    opening = _distribution_stats(opening_z)
    selected = _distribution_stats(selected_safe_z)
    probes = _distribution_stats(probe_z)
    regular_probes = _distribution_stats(regular_probe_z)
    n_min = max(int(np.pi // (2 * config.phi_max)), 1)
    n_max = max(int(np.pi // (2 * config.phi_min)), 1)
    n_second = n_min + (n_max - n_min) // 2
    second_probe_minimum_budget = config.m * (n_min + n_second)
    summary = {
        "scenario_id": config.scenario_id,
        "purpose": config.purpose,
        "budget": config.budget,
        "m_exploration": config.m,
        "conf": config.conf,
        "reported_heldout_rate": config.heldout_rate,
        "replay_convergence": converged / R_TRIALS,
        "mean_probes": probes_total / R_TRIALS,
        "only_opening_share": only_opening / R_TRIALS,
        "N_min": n_min,
        "N_second": n_second,
        "second_probe_minimum_budget": second_probe_minimum_budget,
        "second_probe_fits": int(second_probe_minimum_budget <= config.budget),
        "exploration_share_median": float(np.median(exploration_shares)),
        "guess_overshoot_rate": guess_overshoot / R_TRIALS,
        "final_overshoot_rate": final_overshoot / R_TRIALS,
        "opening_regular_share": opening_regular / R_TRIALS,
        "selected_regular_share": selected_regular / R_TRIALS,
        "selected_safe_share": selected_safe / R_TRIALS,
        "all_probes_regular_share": len(regular_probe_z) / len(probe_z),
        "false_alarm_trial_rate": false_alarm_trials / R_TRIALS,
        "miss_trial_rate": miss_trials / R_TRIALS,
    }
    for prefix, stats in (("opening", opening), ("selected_safe", selected),
                          ("all_probes", probes), ("regular_probes", regular_probes)):
        summary.update({f"{prefix}_{key}": value for key, value in stats.items()})
    arrays = {
        "opening_z": np.asarray(opening_z),
        "selected_safe_z": np.asarray(selected_safe_z),
        "probe_z": np.asarray(probe_z),
        "regular_probe_z": np.asarray(regular_probe_z),
    }
    return summary, arrays


def full_algorithm_audit(configs: list[Configuration]) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    arrays = {}
    for i, config in enumerate(configs):
        print(f"replaying {config.short_label}: B={config.budget:,}, m={config.m}, conf={config.conf}")
        row, samples = replay_configuration(config, SEED + 10_000 * i)
        rows.append(row)
        arrays[config.short_label] = samples
    return pd.DataFrame(rows), arrays


def plot_controlled(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, curves.m.nunique()))
    for color, (m, group) in zip(colors, curves.groupby("m")):
        axes[0].plot(group.r, group.ks_distance, lw=1.8, color=color, label=f"m = {m}")
        axes[1].plot(group.r, group.sd_ratio, lw=1.8, color=color)
    for ax in axes:
        ax.axvline(1.0, color="0.2", ls="--", lw=1.1)
        ax.grid(alpha=0.2)
    axes[0].axhline(0.10, color="0.5", ls=":", lw=1.0)
    axes[0].set_ylabel("exact KS distance")
    axes[0].set_ylim(0, 0.72)
    axes[0].legend(ncol=2, frameon=False)
    axes[0].text(1.015, 0.67, "first aliasing boundary", fontsize=8)
    axes[1].axhline(1.0, color="0.5", ls=":", lw=1.0)
    axes[1].set_ylabel("exact SD / asymptotic SD")
    axes[1].set_xlabel(r"normalised phase-gate position  $r=N\phi/(\pi/2)$")
    axes[1].set_ylim(0, 1.55)
    fig.savefig(OUT / "fig_exact_normality_map.pdf")
    fig.savefig(OUT / "fig_exact_normality_map.png", dpi=220)
    plt.close(fig)


def plot_exact_densities() -> None:
    shot_counts = (114, 334)
    positions = (0.50, 0.75, 0.90)
    edges = np.linspace(-5, 5, 26)
    width = edges[1] - edges[0]
    grid = np.linspace(-5, 5, 600)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.5), sharex=True, constrained_layout=True)
    for row, m in enumerate(shot_counts):
        for col, r in enumerate(positions):
            ax = axes[row, col]
            theta = np.pi * r / 2
            p = float(np.cos(theta) ** 2)
            k = np.arange(m + 1)
            mass = binom.pmf(k, m, p)
            z = 2 * np.sqrt(m) * (np.arccos(np.sqrt(k / m)) - np.arccos(np.sqrt(p)))
            binned, _ = np.histogram(z, bins=edges, weights=mass)
            ax.stairs(binned / width, edges, fill=True, color="#4C78A8", alpha=0.5,
                      label="exact, binned PMF")
            ax.plot(grid, norm.pdf(grid), color="#D62728", lw=1.8, label="standard normal")
            ax.set_title(f"m={m}, r={r:.2f}\nexpected smaller count={m * min(p, 1-p):.1f}", fontsize=9)
            ax.grid(alpha=0.16)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"standardised error  $(\hat\phi-\phi_{fold})/\sigma$")
    for ax in axes[:, 0]:
        ax.set_ylabel("density-scaled mass")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_exact_density_examples.pdf")
    fig.savefig(OUT / "fig_exact_density_examples.png", dpi=220)
    plt.close(fig)


def _plot_ecdf(ax, values: np.ndarray, label: str, color) -> None:
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    ax.plot(values, y, lw=1.25, color=color, label=label)


def plot_pilot_cdfs(arrays: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(arrays)))
    for color, (label, sample) in zip(colors, arrays.items()):
        _plot_ecdf(axes[0], sample["opening_z"], label, color)
        _plot_ecdf(axes[1], sample["selected_safe_z"], label, color)
    grid = np.linspace(-4, 4, 600)
    for ax in axes:
        ax.plot(grid, norm.cdf(grid), color="black", ls="--", lw=1.5, label="standard normal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("standardised pilot error")
        ax.set_xlim(-4, 4)
    axes[0].set_title("Opening pilot (safe by construction)")
    axes[1].set_title("Deepest accepted pilot (safe trials only)")
    axes[0].set_ylabel("empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_operating_pilot_cdfs.pdf")
    fig.savefig(OUT / "fig_operating_pilot_cdfs.png", dpi=220)
    plt.close(fig)


def _pct(x: float, digits: int = 1) -> str:
    return f"{100*x:.{digits}f}%"


def write_report(configs: list[Configuration], controlled: pd.DataFrame,
                 operating: pd.DataFrame) -> None:
    lines = [
        "# Finite-sample Gaussianity audit for binary search",
        "",
        "## Bottom line",
        "",
        "The simulations support a narrower claim than 'the exploration estimator is Gaussian': "
        "at the shot counts selected by the reported sweep, the raw one-shot estimator is close to "
        "its asymptotic normal law whenever both binomial outcomes have reasonable expected counts. "
        "The approximation still fails in a shrinking neighbourhood of every turning point, including "
        "the first aliasing boundary. Adaptive selection of the deepest accepted probe creates an "
        "additional distortion that increasing the shot count alone does not justify away.",
        "",
        "This is therefore useful evidence for the local sampling model, but not a proof of the whole "
        "binary-search derivation. The algorithm should be defended by combining this finite-sample "
        "audit with the end-to-end detector and final-convergence diagnostics.",
        "",
        "## What was tested",
        "",
        f"The full algorithm was replayed for {R_TRIALS:,} independent trials per operating point "
        f"(seed family {SEED}). The parameters were read from `results/consolidated/optimal_params.csv`.",
        "",
        "The exact one-shot distribution was also enumerated. With "
        "`K ~ Bin(m, cos^2(N phi))`, there are only `m+1` possible estimates, so no Monte Carlo "
        "error is needed for the Kolmogorov--Smirnov (KS) distance or variance ratio. The comparison "
        "normal is centred at the folded phase `acos(|cos(N phi)|)/N`. Beyond the boundary it is not "
        "centred at the true phase.",
        "",
        "A finite-sample regularity marker is `m min(p0, 1-p0) >= 10`. This is deliberately two-sided: "
        "near `p0=0` the estimator piles up at `K=0`, while near `p0=1` it piles up at `K=m`.",
        "",
        "## Exact one-shot result at the selected shot counts",
        "",
        "| exploration shots m | max KS in regular region | median KS | max relative SD error | max absolute bias / model SD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in controlled.iterrows():
        lines.append(
            f"| {int(row.m)} | {row.max_ks_regular:.3f} | {row.median_ks_regular:.3f} | "
            f"{100*row.max_abs_sd_error_regular:.1f}% | {row.max_abs_bias_sd_regular:.3f} |"
        )

    lines += [
        "",
        "The 0.10 KS line in the figure is a visual reference, not a formal acceptance threshold. "
        "The more interpretable variance result is that the asymptotic standard deviation is close "
        "to the exact standard deviation throughout the regular region. Close to `r=1`, the exact law "
        "becomes discrete and eventually degenerate, so no shot count makes the approximation uniform "
        "over the boundary itself.",
        "",
        "## Critical implementation finding",
        "",
        "At four of the five selected operating points the tuned exploration shot count makes the "
        "second bisection probe unaffordable. The algorithm therefore records exactly one probe in "
        "every replayed trial: the opening pilot at `N_min`. At those points neither the binary-search "
        "branch nor its overshoot criterion is exercised. The reported performance is produced by the "
        "opening pilot followed by the statistical safeguard.",
        "",
        "| operating point | B | minimum B for opening + second probe | second probe fits? | opening-only trials |",
        "|---|---:|---:|:---:|---:|",
    ]
    for _, row in operating.iterrows():
        label = "Table 3.1" if row.purpose == "Table 3.1" else row.scenario_id.replace("_", " ") + " near B90"
        lines.append(
            f"| {label} | {int(row.budget):,} | {int(row.second_probe_minimum_budget):,} | "
            f"{'yes' if row.second_probe_fits else 'no'} | {_pct(row.only_opening_share)} |"
        )

    lines += [
        "",
        "This is not a Gaussianity failure, but it changes what the headline settings can demonstrate. "
        "They support the opening-pilot safeguard, not the binary-search mechanism. Only the "
        "`narrow_e4` near-B90 point in this audit actually tests repeated bisection decisions.",
        "",
        "## Connection to the best-performing parameters",
        "",
        "| operating point | m | conf | replay conv. | mean probes | regular probes | guess overshoot | final overshoot | opening KS | opening mean/SD | selected-safe KS | selected-safe mean/SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in operating.iterrows():
        label = "Table 3.1" if row.purpose == "Table 3.1" else row.scenario_id.replace("_", " ") + " near B90"
        lines.append(
            f"| {label} | {int(row.m_exploration)} | {row.conf:.2f} | "
            f"{_pct(row.replay_convergence)} | {row.mean_probes:.2f} | "
            f"{_pct(row.all_probes_regular_share)} | {_pct(row.guess_overshoot_rate)} | "
            f"{_pct(row.final_overshoot_rate, 2)} | "
            f"{row.opening_ks_z:.3f} | {row.opening_mean_z:+.2f}/{row.opening_sd_z:.2f} | "
            f"{row.selected_safe_ks_z:.3f} | {row.selected_safe_mean_z:+.2f}/{row.selected_safe_sd_z:.2f} |"
        )

    conf_half = operating[np.isclose(operating.conf, 0.5)]
    lines += [
        "",
        "`opening` is the unselected pilot at the smallest number of phase gates and is safe by "
        "construction. `selected-safe` is the deepest accepted pilot, restricted to trials in which "
        "that number of phase gates was in fact non-aliasing. A standard normal would have mean 0, "
        "SD 1 and KS distance 0.",
        "",
        f"At {len(conf_half)} of the {len(operating)} selected operating points, `conf=0.5`. There the "
        "normal quantile in the branch comparison is exactly zero, so the variance formula does not "
        "affect the accept/reject branch. It remains relevant to the downstream statistical safeguard.",
        "",
    ]
    active = operating[operating.only_opening_share < 1].iloc[0]
    lines += [
        "At the one active bisection point (`narrow_e4` near B90), the fresh, regular probe residuals "
        f"are close to standard normal (mean {active.regular_probes_mean_z:+.3f}, SD "
        f"{active.regular_probes_sd_z:.3f}, KS {active.regular_probes_ks_z:.3f}). However, only "
        f"{_pct(active.all_probes_regular_share)} of probes satisfy the finite-sample regularity marker. "
        "More importantly, selecting the deepest accepted safe pilot shifts its standardised error to "
        f"mean {active.selected_safe_mean_z:+.2f} with SD {active.selected_safe_sd_z:.2f} and KS "
        f"{active.selected_safe_ks_z:.3f}. The trial-level false-alarm and missed-overshoot rates are "
        f"{_pct(active.false_alarm_trial_rate)} and {_pct(active.miss_trial_rate)}, respectively. "
        "Thus the raw Gaussian approximation is good where its regularity condition holds, but the "
        "selected pilot is not an unbiased Gaussian observation.",
        "",
        "## Recommended thesis claim",
        "",
        "> Exact enumeration at the exploration shot counts selected by the parameter sweep shows "
        "that the transformed-binomial estimator is well approximated by the delta-method normal law "
        "away from the binomial endpoints. In the region with at least ten expected observations of "
        "each outcome, its exact standard deviation remains close to `1/(2 N sqrt(m))` and the exact "
        "CDF has a small KS distance from the corresponding normal CDF. The approximation is not "
        "uniform: it fails near the turning points, in particular at the aliasing boundary where the "
        "estimator becomes discrete or degenerate. We therefore treat the normal law as a local design "
        "approximation and validate the adaptive search separately by Monte Carlo.",
        "",
        "Do not say that these plots prove Gaussianity at the boundary, or that they calibrate the "
        "deepest accepted pilot exactly. The latter is selected using the same noisy observations and "
        "must be described as an approximation checked by the operating diagnostics.",
        "",
        "## Figures",
        "",
        "- `fig_exact_normality_map.pdf`: exact KS distance and SD ratio over the first fold.",
        "- `fig_exact_density_examples.pdf`: density-scaled exact PMFs with normal overlays.",
        "- `fig_operating_pilot_cdfs.pdf`: empirical CDFs for the opening and deepest accepted pilots.",
        "",
        "The density panels show binned probability mass because the estimator is discrete at finite "
        "`m`; strictly speaking it has a PMF, not a continuous PDF. The PNG versions are convenient "
        "for notebook-style inspection; the PDFs are vector graphics for a thesis draft.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = load_configurations()
    curves, controlled = controlled_audit(configs)
    operating, arrays = full_algorithm_audit(configs)

    curves.to_csv(OUT / "exact_normality_curves.csv", index=False)
    controlled.to_csv(OUT / "exact_normality_summary.csv", index=False)
    operating.to_csv(OUT / "operating_point_summary.csv", index=False)
    plot_controlled(curves)
    plot_exact_densities()
    plot_pilot_cdfs(arrays)
    write_report(configs, controlled, operating)
    print(f"wrote audit to {OUT}")


if __name__ == "__main__":
    main()
