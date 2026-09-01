"""Old vs new convergence rate, cell by cell, for the exact-safeguard rerun.

    python analysis/compare_safeguard_rerun.py \\
        --old results --new results_partial

Both sides are held-out evaluations of a freshly tuned winner on the SAME seed (SEED_TEST) at the
same R, so the two rates are paired at the level of the operating point but not of the trial: the
tuner can pick a different configuration under the new depth rule, which is the whole reason the
rerun was necessary. A difference is therefore "this operating point's best achievable rate moved",
not "these trials went differently".

The uncertainty on each rate is the Wilson half-width already stored in the row (rate_lo/rate_hi);
two independent binomials at R = 50,000 give an SE on a DIFFERENCE of at most ~0.32 pp, so a
regression under about 0.6 pp is not distinguishable from noise at one point. What matters is
whether regressions are systematic -- concentrated in a scenario, a budget range, or an algorithm --
which the per-scenario and per-regime breakdowns below are for.

Only cells present on BOTH sides are compared; the broad-prior scenarios are reported separately and
excluded from the verdict, since the rerun deliberately does not cover them.
"""
import argparse
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qmetrology import manifest as M

# SE on a difference of two independent binomial proportions, worst case p = 0.5 at this R
NOISE_PP = 2 * 100 * math.sqrt(0.25 / 50_000 + 0.25 / 50_000)   # ~0.63 pp, a 2-SE band


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def cells(rows, algos):
    out = {}
    for r in rows:
        if r["algorithm"] not in algos or not r.get("rate"):
            continue
        out[(r["scenario_id"], r["algorithm"], int(float(r["budget"])))] = r
    return out


def pp(x):
    return f"{x:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="results")
    ap.add_argument("--new", required=True)
    ap.add_argument("--algorithms", default="binary_deep,reverse_eng_risk")
    ap.add_argument("--threshold-pp", type=float, default=NOISE_PP,
                    help="a regression larger than this is counted as material (default: a 2-SE "
                         "band on a difference at R=50,000)")
    ap.add_argument("--csv", default="", help="also write the per-cell table here")
    a = ap.parse_args()

    algos = [x.strip() for x in a.algorithms.split(",") if x.strip()]
    broad = {s["id"] for s in M.SCENARIOS if "broad_prior" in s.get("families", ())}
    old = cells(read(os.path.join(a.old, "performance_curves.csv")), algos)
    new = cells(read(os.path.join(a.new, "performance_curves.csv")), algos)

    common = sorted(set(old) & set(new))
    only_old = sorted(k for k in set(old) - set(new) if k[0] not in broad)
    only_new = sorted(set(new) - set(old))
    print(f"old {a.old}: {len(old)} cells\nnew {a.new}: {len(new)} cells")
    print(f"compared {len(common)}; only-old (non-broad) {len(only_old)}; only-new {len(only_new)}")
    if only_old:
        print(f"  WARNING missing from the rerun: {only_old[:6]}")
    if only_new:
        print(f"  WARNING not in the base: {only_new[:6]}")
    print()

    rows = []
    for k in common:
        o, n = old[k], new[k]
        do, dn = 100 * float(o["rate"]), 100 * float(n["rate"])
        # half-widths of the two Wilson intervals, for the per-cell "is it resolvable" flag
        try:
            ho = 100 * (float(o["rate_hi"]) - float(o["rate_lo"])) / 2
            hn = 100 * (float(n["rate_hi"]) - float(n["rate_lo"])) / 2
        except (ValueError, KeyError):
            ho = hn = float("nan")
        rows.append(dict(scenario_id=k[0], algorithm=k[1], budget=k[2],
                         rate_old_pct=round(do, 4), rate_new_pct=round(dn, 4),
                         delta_pp=round(dn - do, 4),
                         ci_halfwidth_old_pp=round(ho, 4), ci_halfwidth_new_pp=round(hn, 4),
                         resolvable=bool(abs(dn - do) > math.hypot(ho, hn)),
                         params_old=o.get("params", ""), params_new=n.get("params", "")))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}\n")

    def summarise(label, sub):
        if not sub:
            return
        d = [r["delta_pp"] for r in sub]
        worse = [r for r in sub if r["delta_pp"] < -a.threshold_pp]
        better = [r for r in sub if r["delta_pp"] > a.threshold_pp]
        print(f"{label:<22} n={len(sub):>4}  mean {pp(sum(d)/len(d)):>7}  "
              f"min {pp(min(d)):>7}  max {pp(max(d)):>7}  "
              f"worse {len(worse):>3} ({100*len(worse)/len(sub):4.1f}%)  "
              f"better {len(better):>3} ({100*len(better)/len(sub):4.1f}%)")

    print(f"delta = new - old, in percentage points. "
          f"material = |delta| > {a.threshold_pp:.2f} pp\n")
    summarise("ALL", rows)
    print()
    for al in algos:
        summarise(al, [r for r in rows if r["algorithm"] == al])
    print()
    for sid in sorted({r["scenario_id"] for r in rows}):
        summarise(sid, [r for r in rows if r["scenario_id"] == sid])

    # regressions only bite where there is room to regress: a cell already at 100% cannot improve,
    # and one at 0% cannot get worse, so split by where the old rate sat.
    print()
    live = [r for r in rows if 2.0 < r["rate_old_pct"] < 98.0]
    summarise("live (2-98% old)", live)
    summarise("saturated (>=98%)", [r for r in rows if r["rate_old_pct"] >= 98.0])
    summarise("floor (<=2%)", [r for r in rows if r["rate_old_pct"] <= 2.0])

    worse = sorted((r for r in rows if r["delta_pp"] < -a.threshold_pp),
                   key=lambda r: r["delta_pp"])
    print(f"\n{len(worse)} material regression(s) of {len(rows)} cells "
          f"({100*len(worse)/max(len(rows),1):.1f}%)")
    if worse:
        print(f"\n{'scenario':<12} {'algorithm':<18} {'budget':>18} {'old':>7} {'new':>7} "
              f"{'delta':>7}  resolvable")
        for r in worse[:40]:
            print(f"{r['scenario_id']:<12} {r['algorithm']:<18} {r['budget']:>18,} "
                  f"{r['rate_old_pct']:>7.2f} {r['rate_new_pct']:>7.2f} "
                  f"{pp(r['delta_pp']):>7}  {'yes' if r['resolvable'] else 'no'}")
        if len(worse) > 40:
            print(f"... and {len(worse)-40} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
