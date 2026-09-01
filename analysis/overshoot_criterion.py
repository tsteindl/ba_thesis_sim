"""Theoretical justification of binary search's one-shot overshoot criterion (Section 3.2.2).

The rule, exactly as implemented in qmetrology/algorithms.py:

    phi_1 = norm.ppf(1 - conf, phi_hat_acc, 1/(2 N sqrt(m')))     # threshold from the reference
    declare overshoot   <=>   phi_hat_N < phi_1

The question is what justifies calling a probe "significantly below" the reference. The answer here
is that the rule does NOT need the Gaussian assumption to be valid -- it needs it only to place the
threshold, and that use is checkable against an exact calculation.

THREE SEPARATE CLAIMS, only the third of which involves a normal approximation.

1. VALIDITY -- the rule is exactly a one-sided binomial test, no approximation involved.
   phi_hat = arccos(sqrt(K/m'))/N is strictly decreasing in K, so for any threshold phi_1 in the
   estimator's range,

       phi_hat_N < phi_1     <=>     K > m' cos^2(N phi_1).

   The rule is therefore a cut on the raw count. Under the null "N is safe and the phase is phi",
   K ~ Bin(m', cos^2(N phi)), so the rule's size is a binomial tail probability -- exact, in closed
   form, with no distributional assumption anywhere. verify_equivalence() checks this identity.

2. POWER -- guaranteed by an exact inequality, not by a limit.
   arccos returns a value in [0, pi/2], so every estimate obeys N phi_hat <= pi/2, while overshooting
   means N phi > pi/2. Hence at ANY overshooting N and for EVERY outcome K,

       phi_hat <= pi/(2N) < phi,

   i.e. an overshooting probe always reads below the truth. This replaces the asymptotic
   "phi_hat -> 0" identity with a statement that holds at the finite N where the rule actually
   fires. verify_power_bound() checks it.

3. CALIBRATION -- the only place the normal approximation is used, and it is quantified.
   The normal quantile chooses where to put the cut. exact_size() computes what size that cut
   actually achieves, from the binomial. Inside the two-sided regularity region
   m' min(p0, 1-p0) >= 10 the achieved size tracks the nominal one closely; outside it, at the
   shallow end where p0 -> 1, it can drift, and near the aliasing boundary it errs conservatively.

    python analysis/overshoot_criterion.py

Writes overshoot_size.csv, overshoot_power.csv and overshoot_bracket_walk.csv.
"""
import csv
import os

import numpy as np
from scipy.stats import binom, norm

from pipeline_io import path

PHI = 0.05                       # only fixes the scale; every result depends on N only via N/N_opt
SHOTS = (50, 200, 800)
CONFS = (0.5, 0.66, 0.95)        # 0.5 and 0.66 are the values the sweep selected
REGULARITY = 10.0                # the two-sided marker m' min(p0, 1-p0) >= 10
SAFE_GRID = np.linspace(0.05, 0.999, 400)


# --------------------------------------------------------------------------- claim 1: validity
def verify_equivalence(n=300_000, seed=3):
    """{phi_hat_N < phi_1} and {K > m' cos^2(N phi_1)} are the same event. Returns #disagreements."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n):
        m = int(rng.integers(2, 600))
        N = int(rng.integers(1, 200))
        phi_1 = rng.uniform(0, np.pi / (2 * N))
        k = int(rng.integers(0, m + 1))
        if (np.arccos(np.sqrt(k / m)) / N < phi_1) != (k > m * np.cos(N * phi_1) ** 2):
            bad += 1
    return bad


def cut(m, N, phi_1):
    """The count above which the rule fires: k* = m' cos^2(N phi_1)."""
    return m * np.cos(N * phi_1) ** 2


def exact_size(m, x, conf, phi=PHI):
    """Exact P(rule fires) at a SAFE depth x = N/N_opt, against a perfect reference.

    This is the rule's true false-alarm rate: a binomial tail, with no normal approximation.
    """
    theta = x * np.pi / 2
    N = theta / phi
    phi_1 = phi + norm.ppf(1 - conf) / (2 * N * np.sqrt(m))
    return float(1 - binom.cdf(np.floor(cut(m, N, phi_1)), m, np.cos(theta) ** 2))


# --------------------------------------------------------------------------- claim 2: power
def verify_power_bound(n=200_000, seed=7):
    """At any overshooting N, phi_hat < phi for every outcome. Returns max(phi_hat - phi)."""
    rng = np.random.default_rng(seed)
    worst = -np.inf
    for _ in range(n):
        phi = rng.uniform(0.001, 0.5)
        N = int(rng.uniform(1.0001, 4.0) * (np.pi / (2 * phi))) + 1
        if N * phi <= np.pi / 2:
            continue
        m = int(rng.integers(1, 500))
        k = rng.binomial(m, np.cos(N * phi) ** 2)
        worst = max(worst, np.arccos(np.sqrt(k / m)) / N - phi)
    return float(worst)


# --------------------------------------------------------------------------- claim 3: calibration
def regular(m, x, thresh=REGULARITY):
    p0 = np.cos(x * np.pi / 2) ** 2
    return m * min(p0, 1 - p0) >= thresh


def size_summary(m, conf, grid=SAFE_GRID):
    """Achieved size across the safe region, split by the regularity marker."""
    vals = np.array([exact_size(m, float(x), conf) for x in grid])
    keep = np.array([regular(m, float(x)) for x in grid])
    nominal = 1 - conf
    out = dict(m_exploration=m, conf=conf, nominal_alpha=round(nominal, 4),
               n_regular=int(keep.sum()))
    if keep.any():
        v = vals[keep]
        out.update(size_min=round(float(v.min()), 4), size_max=round(float(v.max()), 4),
                   max_deviation=round(float(np.abs(v - nominal).max()), 4))
    else:
        out.update(size_min=float("nan"), size_max=float("nan"), max_deviation=float("nan"))
    near = grid > 0.97
    out["size_near_boundary_max"] = round(float(vals[near].max()), 4)
    out["size_all_max"] = round(float(vals.max()), 4)
    return out



# --------------------------------------------------------------------------- Berry-Esseen
BE_C = 0.4748          # Shevtsova (2011), best known constant for the i.i.d. Berry-Esseen bound


def berry_esseen(m, p):
    """Rigorous bound on |P((K - mp)/sqrt(mpq) <= x) - Phi(x)| for K ~ Bin(m, p).

    K is a sum of m i.i.d. Bernoulli(p) variables, for which the third absolute central moment is
    rho = pq(p^2 + q^2) and sigma^2 = pq, so the classical bound C rho / (sigma^3 sqrt(m)) becomes

        C (p^2 + q^2) / sqrt(m p q).

    This is non-asymptotic: it holds for every m and every p, with no appeal to a limit.
    """
    q = 1.0 - p
    return BE_C * (p ** 2 + q ** 2) / np.sqrt(m * p * q)


def binomial_normal_ks(m, p):
    """Exact sup-distance between the standardised Bin(m, p) CDF and the normal CDF."""
    k = np.arange(m + 1)
    w = binom.pmf(k, m, p)
    upper = np.cumsum(w)
    g = norm.cdf((k - m * p) / np.sqrt(m * p * (1 - p)))
    return float(max(np.max(np.abs(upper - g)), np.max(np.abs(upper - w - g))))


def normal_approximation_quality(m, thresh=REGULARITY, n=600):
    """Worst-case Berry-Esseen bound and true KS distance over the regularity region."""
    ps = np.linspace(1e-9, 0.5, n)
    ps = ps[m * ps >= thresh]
    if not len(ps):
        return None
    return dict(m_exploration=m,
                be_bound_max=round(float(max(berry_esseen(m, float(p)) for p in ps)), 4),
                true_ks_max=round(float(max(binomial_normal_ks(m, float(p)) for p in ps[::20])), 4))




def ks_phi_hat(m, x):
    """Exact KS distance between phi_hat's law at depth x = N/N_opt and its delta-method normal."""
    theta = x * np.pi / 2
    k = np.arange(m + 1)
    w = binom.pmf(k, m, np.cos(theta) ** 2)
    z = 2.0 * np.sqrt(m) * (np.arccos(np.sqrt(k / m)) - theta)
    o = np.argsort(z)
    z, w = z[o], w[o]
    upper = np.cumsum(w)
    g = norm.cdf(z)
    return float(max(np.max(np.abs(upper - g)), np.max(np.abs(upper - w - g))))


def transfer_check(ms=SHOTS, xs=(0.2, 0.4, 0.5, 0.6, 0.8, 0.88)):
    """Does 'K is nearly Gaussian' carry over to 'phi_hat is nearly Gaussian' at finite m'?

    Berry-Esseen bounds the distance for the COUNT. phi_hat's distance is measured against a
    different normal (the delta-method one), so the two are formally distinct quantities. This
    reports both, plus the bound, so the gap can be seen rather than assumed.
    """
    rows = []
    for m in ms:
        for x in xs:
            theta = x * np.pi / 2
            p0 = float(np.cos(theta) ** 2)
            rows.append(dict(m_exploration=m, x=x, p0=round(p0, 6),
                             regularity=round(m * min(p0, 1 - p0), 3),
                             ks_count=round(binomial_normal_ks(m, p0), 5),
                             ks_phi_hat=round(ks_phi_hat(m, x), 5),
                             be_bound=round(berry_esseen(m, p0), 5)))
    return rows


# --------------------------------------------------------------------------- threshold map
def threshold_offset(theta, delta):
    """Exact displacement of the cut from m'p0, as a fraction of m'.

        k*/m' - p0 = cos^2(theta + delta) - cos^2(theta) = -sin(2 theta + delta) sin(delta)

    from cos A - cos B = -2 sin((A+B)/2) sin((A-B)/2). No approximation.
    """
    return -np.sin(2 * theta + delta) * np.sin(delta)


def standardised_cut(m, z, theta):
    """The Gaussian threshold, expressed on the count's own standardised scale.

        u = (k* - m'p0)/sqrt(m' p0 q0) = -z A(m',z) B(theta, delta)

    with A = 2 sqrt(m') sin(delta)/z  (the sin d ~ d error, -> 1 as m' grows) and
    B = sin(2 theta + delta)/sin(2 theta)  (-> 1 as delta -> 0, singular as sin 2theta -> 0).
    A perfectly placed cut would give u = -z exactly, and hence size exactly Phi(z).
    """
    delta = z / (2 * np.sqrt(m))
    A = 1.0 if z == 0 else 2 * np.sqrt(m) * np.sin(delta) / z
    B = np.sin(2 * theta + delta) / np.sin(2 * theta)
    return -z * A * B, A, B


def size_error_bound(m, conf, x, phi=PHI):
    """Two-part rigorous bound on |achieved size - nominal alpha| at a safe depth x = N/N_opt.

        |size - alpha|  <=  |Phi(-u) - Phi(z)|   +   BerryEsseen(m', p0)
                            ^ threshold placement  ^ normal vs binomial

    The first term is the delta method's own error, in closed form; the second is Berry-Esseen.
    """
    z = norm.ppf(1 - conf)
    theta = x * np.pi / 2
    p0 = float(np.cos(theta) ** 2)
    u, A, B = standardised_cut(m, z, theta)
    placement = abs(norm.cdf(-u) - norm.cdf(z))
    be = berry_esseen(m, p0)
    return dict(m_exploration=m, conf=conf, x=round(x, 4), p0=round(p0, 6),
                regularity=round(m * min(p0, 1 - p0), 3),
                nominal=round(float(norm.cdf(z)), 4),
                achieved=round(exact_size(m, x, conf, phi), 4),
                term_placement=round(float(placement), 4),
                term_berry_esseen=round(float(be), 4),
                bound=round(float(placement + be), 4),
                A=round(float(A), 5), B=round(float(B), 5))


# --------------------------------------------------------------------------- power vs depth
def probe_law(m, x):
    theta = x * np.pi / 2
    k = np.arange(m + 1)
    w = binom.pmf(k, m, np.cos(theta) ** 2)
    z = 2.0 * np.sqrt(m) * (np.arccos(np.sqrt(k / m)) - theta)
    o = np.argsort(z)
    return z[o], w[o]


def fire_probability(m, x, rho, conf):
    """Exact P(rule fires) at depth x against a reference rho = N/N_acc times shallower.

    rho = None is the perfect reference phi_hat_acc = phi.
    """
    zp, wp = probe_law(m, x)
    z_alpha = norm.ppf(1 - conf)
    if rho is None:
        return float(wp[zp < z_alpha].sum())
    zr, wr = probe_law(m, x / rho)
    cut_z = rho * zr + z_alpha
    cum = np.cumsum(wp)
    i = np.searchsorted(zp, cut_z, side="left")
    return float(wr @ np.where(i > 0, cum[np.clip(i - 1, 0, None)], 0.0))


def bracket_walk(phi, n_min=15, n_max=157, steps=6):
    """The probes Algorithm 6 takes, and the reference quality rho = N/N_acc at each."""
    n_opt = int(np.pi // (2 * phi))
    lo, hi, n_acc = n_min, n_max, n_min
    n = n_min + (n_max - n_min) // 2
    out = []
    for step in range(steps):
        safe = n <= n_opt
        out.append(dict(phi=phi, N_opt=n_opt, step=step + 1, N=n, N_acc=n_acc,
                        rho=round(n / n_acc, 3), x=round(n * phi / (np.pi / 2), 3),
                        truth="safe" if safe else "overshoot"))
        if safe:
            n_acc, lo = n, n
            n = n + (hi - n) // 2
        else:
            hi = n
            n = n - (n - lo) // 2
        if hi - lo <= 1:
            break
    return out


def _write(name, rows):
    with open(path(name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(path(""), exist_ok=True)

    print("claim 1 -- the rule is exactly a one-sided binomial test on the count K")
    bad = verify_equivalence()
    print(f"  {{phi_hat < phi_1}} vs {{K > m' cos^2(N phi_1)}}: {bad} disagreements in 300,000 draws")
    assert bad == 0, "the binomial restatement is not exact"

    print("\nclaim 2 -- an overshooting probe always reads below the truth")
    worst = verify_power_bound()
    print(f"  max(phi_hat - phi) over 200,000 overshooting draws = {worst:.3e} (never positive)")
    assert worst <= 0.0, "the power bound failed"

    print("\nclaim 3 -- what the normal quantile costs, measured against the exact binomial")
    rows = [size_summary(m, c) for m in SHOTS for c in CONFS]
    _write("overshoot_size.csv", rows)
    print(f"  {'m':>5} {'conf':>5} {'nominal':>8} {'exact size (regular region)':>29} "
          f"{'max dev':>8} {'near boundary':>14}")
    for r in rows:
        print(f"  {r['m_exploration']:>5} {r['conf']:>5.2f} {r['nominal_alpha']:>8.2f} "
              f"{'[' + format(r['size_min'], '.3f') + ', ' + format(r['size_max'], '.3f') + ']':>29} "
              f"{r['max_deviation']:>8.3f} {r['size_near_boundary_max']:>14.3f}")

    print("\nnormal approximation to the COUNT, bounded rigorously (Berry-Esseen)")
    bes = [normal_approximation_quality(m) for m in SHOTS]
    bes = [b for b in bes if b]
    print(f"  {'m':>5} {'BE bound':>10} {'true KS':>9}   (worst case over m' min(p,1-p) >= 10)")
    for b in bes:
        print(f"  {b['m_exploration']:>5} {b['be_bound_max']:>10.4f} {b['true_ks_max']:>9.4f}")
        assert b["true_ks_max"] <= b["be_bound_max"], "Berry-Esseen bound violated"
    by_m = {b["m_exploration"]: b for b in bes}
    for r in rows:
        b = by_m.get(r["m_exploration"])
        if b:
            r["be_bound_max"] = b["be_bound_max"]
            r["true_ks_max"] = b["true_ks_max"]
    _write("overshoot_size.csv", rows)

    print("\ndoes Gaussianity of K carry over to phi_hat? (delta method, at finite m')")
    tr = [r for r in transfer_check() if r["regularity"] >= REGULARITY]
    _write("overshoot_transfer.csv", tr)
    ratio = max(r["ks_phi_hat"] / r["ks_count"] for r in tr)
    covered = sum(r["ks_phi_hat"] <= r["be_bound"] for r in tr)
    print(f"  inside the regularity region: max KS(phi_hat)/KS(K) = {ratio:.3f}, "
          f"Berry-Esseen covers phi_hat at {covered}/{len(tr)} points")
    print("  outside it, they separate:")
    for x in (0.90, 0.95, 0.98):
        theta = x * np.pi / 2
        p0 = float(np.cos(theta) ** 2)
        print(f"    m'=200, N/N_opt={x:.2f} (m*min={200*min(p0,1-p0):5.2f}): "
              f"KS(K)={binomial_normal_ks(200, p0):.4f}  KS(phi_hat)={ks_phi_hat(200, x):.4f}")

    print("\ntwo-part error bound on the achieved size (placement + Berry-Esseen)")
    bnd = [size_error_bound(m, c, float(x))
           for m in SHOTS for c in CONFS
           for x in np.linspace(0.15, 0.90, 40)]
    bnd = [b for b in bnd if b["regularity"] >= REGULARITY]
    _write("overshoot_error_bound.csv", bnd)
    viol = [b for b in bnd if abs(b["achieved"] - b["nominal"]) > b["bound"] + 1e-9]
    print(f"  {len(bnd)} points in the regularity region; bound violated at {len(viol)}")
    assert not viol, "the two-part error bound failed"
    for m in SHOTS:
        sub = [b for b in bnd if b["m_exploration"] == m]
        print(f"  m'={m:>4}: placement term <= {max(b['term_placement'] for b in sub):.4f}, "
              f"Berry-Esseen term <= {max(b['term_berry_esseen'] for b in sub):.4f}, "
              f"actual error <= {max(abs(b['achieved']-b['nominal']) for b in sub):.4f}")

    print("\nerror budget: normal approximation vs reference noise (m' = 200)")
    print(f"  {'nominal':>8} {'normal approx':>14} {'late ref':>10} {'first ref':>10}")
    for conf in CONFS:
        a = 1 - conf
        e1 = max(abs(exact_size(200, float(x), conf) - a)
                 for x in np.linspace(0.2, 0.85, 60))
        print(f"  {a:>8.2f} {e1:>14.3f} "
              f"{abs(fire_probability(200, 0.9, 1.05, conf) - a):>10.3f} "
              f"{abs(fire_probability(200, 0.9, 5.73, conf) - a):>10.3f}")

    print("\npower against depth (exact), for the reference qualities binary search produces")
    grid = np.round(np.linspace(0.80, 1.40, 241), 6)
    prows = []
    for m in SHOTS:
        for conf in CONFS:
            for rho, tag in ((None, "perfect"), (1.05, "late"), (5.73, "first")):
                for x in grid:
                    prows.append(dict(m_exploration=m, conf=conf, reference=tag,
                                      rho="" if rho is None else rho, x=float(x),
                                      power=round(fire_probability(m, float(x), rho, conf), 6)))
    _write("overshoot_power.csv", prows)
    for tag in ("perfect", "late", "first"):
        v = [r["power"] for r in prows
             if r["m_exploration"] == 200 and r["conf"] == 0.5
             and r["reference"] == tag and r["x"] > 1.0]
        print(f"  m'=200, conf=0.50, {tag:>7} reference: min power past the boundary = {min(v):.3f}")

    _write("overshoot_bracket_walk.csv",
           [r for phi in (0.02, 0.05) for r in bracket_walk(phi)])
    print("\nhow the bracket narrows (Algorithm 6, N in {15,...,157})")
    for phi in (0.02, 0.05):
        print(f"  phi={phi}:  " + "  ".join(
            f"N/N_opt={w['x']:.3f} (rho={w['rho']:.2f})" for w in bracket_walk(phi)))

    print(f"\nwrote overshoot_size.csv, overshoot_power.csv, overshoot_bracket_walk.csv "
          f"to {path('')}")


if __name__ == "__main__":
    main()
