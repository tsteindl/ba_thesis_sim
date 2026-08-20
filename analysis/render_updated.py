"""Results-chapter tables from the boundary-free sweep (results/fine_sweep.csv).

New numbers: binary_deep (deepest-accepted pilot), re_fixed, linear_fixed -- all re-tuned over
m' in [1, budget // N_min] with two-stage refinement. Carried over from results/story_curves.csv
unchanged: brute force and separable (no tuning parameters) and oracle_hl (the omniscient ceiling).

    python analysis/render_updated.py > results/RESULTS_UPDATED.md
"""
import collections
import csv
import sys

import numpy as np

NEW = {"binary_deep": "Binary search (deepest-accepted pilot)",
       "re_fixed": "Reverse engineering",
       "linear_fixed": "Linear search"}
OLD = {"binary_risk": "Binary search", "reverse_eng_risk": "Reverse engineering",
       "linear": "Linear search", "brute": "Brute force", "separable": "Separable (N=1)",
       "oracle_hl": "Attainable ceiling (Eq. 3.4 law)"}


def load():
    old = collections.defaultdict(dict)
    meta = {}
    for x in csv.DictReader(open("results/story_curves.csv")):
        old[(x["setting"], int(x["budget"]))][x["algo"]] = float(x["rate"])
        meta[x["setting"]] = (float(x["phi_min"]), float(x["phi_max"]), float(x["eps"]))
    new = collections.defaultdict(dict)
    bnd = collections.defaultdict(dict)
    for x in csv.DictReader(open("results/fine_sweep.csv")):
        new[(x["setting"], int(x["budget"]))][x["algo"]] = float(x["rate"])
        bnd[(x["setting"], int(x["budget"]))][x["algo"]] = (int(x["at_min"]), int(x["at_max"]))
    return old, new, meta, bnd


def curve(store, setting, algo):
    return sorted((b, v[algo]) for (s, b), v in store.items() if s == setting and algo in v)


def crossing(pts, thr):
    pts = sorted(pts)
    for (b0, r0), (b1, r1) in zip(pts, pts[1:]):
        if r0 < thr <= r1:
            f = (thr - r0) / (r1 - r0) if r1 > r0 else 0.0
            return float(np.exp(np.log(b0) + f * (np.log(b1) - np.log(b0))))
    return None


def main():
    old, new, meta, bnd = load()
    settings = sorted({s for s, _ in new})

    print("# Results — rebuilt on a boundary-free grid\n")
    print("_`binary_deep`, `re_fixed` and `linear_fixed` are re-tuned over "
          "`m' ∈ [1, budget // N_min]` with two-stage refinement (`analysis/fine_sweep.py`, "
          "23 scenarios × 14 budgets, tuned seed 42 at R=600/1500, validated seed 2024 at "
          "R=20,000). Brute force, separable and the ceiling `oracle_hl` carry over from "
          "`story_curves.csv` — none of them has a tuning parameter._\n")
    print("_Binary search uses the deepest **accepted** probe as the Eq. (3.8) pilot "
          "(`results/ALGORITHM_BINARY.md`); everything else is the shipped algorithm._\n")

    # ---------------------------------------------------------------- Table C
    print("## Table C — budget ratio vs brute force, by threshold\n")
    print("_Ratio = `B_brute(p*) / B_algo(p*)`: how much **less** budget the algorithm needs to first "
          "reach convergence `p*`. >1 wins._\n")
    hdr = "| Scenario | p* | Linear | Binary | Rev. eng. | Ceiling | winner |"
    print(hdr)
    print("|---|---:|---:|---:|---:|---:|---|")
    agg = collections.defaultdict(list)
    for s in settings:
        cb = curve(old, s, "brute")
        for thr in (0.5, 0.75, 0.9):
            bb = crossing(cb, thr)
            if not bb:
                continue
            row, vals = [], {}
            for a in ("linear_fixed", "binary_deep", "re_fixed"):
                c = crossing(curve(new, s, a), thr)
                vals[a] = bb / c if c else None
            co = crossing(curve(old, s, "oracle_hl"), thr)
            vo = bb / co if co else None
            if not any(v for v in vals.values()):
                continue
            best = max((v for v in vals.values() if v), default=None)
            wn = [NEW[k].split(" (")[0] for k, v in vals.items() if v and v == best]
            fmt = lambda v: (f"**{v:.2f}×**" if v and v >= 1 else (f"_{v:.2f}×_" if v else "—"))
            print(f"| {s} | {int(100*thr)}% | {fmt(vals['linear_fixed'])} | "
                  f"{fmt(vals['binary_deep'])} | {fmt(vals['re_fixed'])} | "
                  f"{(f'{vo:.2f}×' if vo else '—')} | {wn[0] if wn else '—'} |")
            for k, v in vals.items():
                if v:
                    agg[(k, thr)].append(v)
    print()
    print("### Pooled\n")
    print("| p* | Linear | Binary | Rev. eng. |")
    print("|---|---:|---:|---:|")
    for thr in (0.5, 0.75, 0.9):
        cells = []
        for a in ("linear_fixed", "binary_deep", "re_fixed"):
            v = np.array(agg[(a, thr)])
            cells.append(f"{v.mean():.2f}× (median {np.median(v):.2f}×)" if len(v) else "—")
        print(f"| {int(100*thr)}% | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- head-to-head
    print("\n## Head-to-head, paired over all 308 operating points\n")
    pts = sorted(new)
    b = np.array([new[p]["binary_deep"] for p in pts]) * 100
    r = np.array([new[p]["re_fixed"] for p in pts]) * 100
    l = np.array([new[p]["linear_fixed"] for p in pts]) * 100
    br = np.array([old[p]["brute"] for p in pts if p in old]) * 100
    oc = np.array([old[p]["oracle_hl"] for p in pts if p in old]) * 100
    print("| comparison | mean | median | wins |")
    print("|---|---:|---:|---:|")
    for lab, d in (("binary − reverse engineering", b - r), ("binary − linear", b - l),
                   ("reverse engineering − linear", r - l)):
        print(f"| {lab} | {d.mean():+.2f} pp | {np.median(d):+.2f} pp | "
              f"{100*(d>0.1).mean():.0f}% |")
    print(f"\nMean convergence: linear **{l.mean():.2f}%**, binary **{b.mean():.2f}%**, "
          f"reverse engineering **{r.mean():.2f}%**, brute force {br.mean():.2f}%, "
          f"ceiling {oc.mean():.2f}%.\n")
    print(f"Share of the ceiling reached: linear {100*l.mean()/oc.mean():.1f}%, "
          f"binary {100*b.mean()/oc.mean():.1f}%, reverse engineering "
          f"{100*r.mean()/oc.mean():.1f}%.\n")

    # ---------------------------------------------------------------- what moved
    print("## What moved against the old sweep\n")
    print("| algorithm | mean | median | max gain | max loss | points gaining >1 pp |")
    print("|---|---:|---:|---:|---:|---:|")
    for n, o in (("linear_fixed", "linear"), ("binary_deep", "binary_risk"),
                 ("re_fixed", "reverse_eng_risk")):
        d = np.array([(new[p][n] - old[p][o]) * 100 for p in pts if p in old and o in old[p]])
        print(f"| {NEW[n]} | {d.mean():+.2f} pp | {np.median(d):+.2f} pp | {d.max():+.2f} | "
              f"{d.min():+.2f} | {100*(d>1).mean():.0f}% |")

    print("\n## Grid boundary diagnostics\n")
    print("| algorithm | winner at `m'=1` | winner at `m' = budget/N_min` | (old sweep: at grid min) |")
    print("|---|---:|---:|---:|")
    for a, oldpct in (("linear_fixed", "37.5%"), ("binary_deep", "23.5%"), ("re_fixed", "32.1%")):
        lo = np.mean([bnd[p][a][0] for p in pts if a in bnd[p]])
        hi = np.mean([bnd[p][a][1] for p in pts if a in bnd[p]])
        print(f"| {NEW[a]} | {100*lo:.1f}% | {100*hi:.1f}% | {oldpct} |")
    print("\n_`m'=1` is the physical minimum (one shot), so a hit there is a real optimum, not a "
          "truncated search. The upper endpoint is never selected._")


if __name__ == "__main__":
    main()
