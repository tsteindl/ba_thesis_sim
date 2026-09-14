"""Merge an affected-only partial rerun into a complete staging dataset.

    python analysis/merge_partial.py \\
        --partial results_partial \\
        --base    results \\
        --out     results_staging

WHY THIS EXISTS. A partial rerun (`run.py --algorithms ... --scenarios ...`) covers only the cells
whose algorithm actually changed. Its output directory is therefore NOT a dataset: it has no
`brute`, no `linear`, no reference rows, and no derived files. Promoting it, or pointing `finalize`
at it, would produce a report describing a fraction of the matrix. This script builds the complete
thesis dataset instead:

  1. copy the BASE (the current complete production data) into a fresh staging directory;
  2. drop every row belonging to a dropped scenario family (the broad priors, removed from the
     thesis) -- so no stale scenario survives into the merged tables;
  3. drop the base's rows for the REPLACED algorithms in the retained scenarios, and put the
     partial run's rows in their place -- never merge the two implementations' rows;
  4. keep `brute`, `linear`, `separable` and the analytic oracle rows exactly as they were;
  5. assert, before writing anything: key uniqueness on both sides, identical scenario/budget
     coverage for the replaced cells, matching seeds and trial counts between kept and replaced
     rows, and that the partial manifest really carries the new safeguard implementation metadata.

The cell key is `(scenario_id, algorithm, budget)`, plus `phase_bin` for the phase-stratified table.

Nothing derived is produced here. `add_baselines`, `finalize`, the error/variance
curves, `thesis_tables.py` and `thesis_figures.py` are run afterwards against the staging directory
via RESULTS_OUT, and only a checked staging directory is ever promoted.
"""
import argparse
import csv
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qmetrology import manifest as M

# name -> extra key columns beyond (scenario_id, algorithm, budget)
TABLES = {
    "performance_curves.csv": (),
    "winners.csv": (),
    "diagnostics_by_point.csv": (),
    "detector_confusion.csv": (),
    "budget_audit.csv": (),
    "diagnostics_by_phase.csv": ("phase_bin",),
}

# Rows carrying these must agree between the kept base rows and the incoming partial rows, or the
# two halves of the merged dataset were not produced under the same protocol.
PROVENANCE = ("seed_test", "seed_tune_blocks", "R")

EXPECTED_SELECTION_RULE = "exact_enumeration_integer_shots"

# Outputs of studies that run their own simulations and never read the replaced algorithms' rows:
# analysis/linear_detector_study.py (Algorithm 4's detector) and
# overshoot_criterion.py (the bisection's accept/reject test). Changing the exploitation DEPTH rule
# moves neither, and both cost far more to rebuild than the merge, so they are carried over.
INDEPENDENT_PREFIXES = ("linear_detector_", "overshoot_")
INDEPENDENT_EXACT = {
    "linear_degenerate_zone.csv",   # linear_detector_study.py
    "linear_search_claims.csv",     # linear_search_claims.py -- see the note it prints
}

# Written fresh by the partial run itself, so its copy is the current one.
FROM_PARTIAL = ("algorithm_code_audit.md",)


def is_independent(name):
    return name in INDEPENDENT_EXACT or name.startswith(INDEPENDENT_PREFIXES)


def read(p):
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write(p, fields, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def key(row, extra):
    return (row["scenario_id"], row["algorithm"], str(int(float(row["budget"]))),
            *(row.get(k, "") for k in extra))


def die(msg):
    raise SystemExit(f"MERGE REFUSED: {msg}")


def check_manifests(base_dir, part_dir, algos, keep_ids):
    """Everything that must line up before a single row is moved."""
    bm = json.load(open(os.path.join(base_dir, "experiment_manifest.json")))
    pm = json.load(open(os.path.join(part_dir, "experiment_manifest.json")))

    if bm["mode"] != pm["mode"]:
        die(f"mode differs: base {bm['mode']!r} vs partial {pm['mode']!r}")
    if bm["seeds"] != pm["seeds"]:
        die(f"seeds differ:\n  base    {bm['seeds']}\n  partial {pm['seeds']}")
    if bm["trials"] != pm["trials"]:
        die(f"trial counts differ:\n  base    {bm['trials']}\n  partial {pm['trials']}")

    sg = pm.get("safeguard")
    if not sg:
        die("the partial manifest has no `safeguard` block -- it was produced by a build that "
            "predates the exact-safeguard metadata, so its rows cannot be identified")
    if sg.get("selection_rule") != EXPECTED_SELECTION_RULE:
        die(f"partial safeguard selection_rule is {sg.get('selection_rule')!r}, "
            f"expected {EXPECTED_SELECTION_RULE!r}")
    if bm.get("safeguard", {}).get("selection_rule") == EXPECTED_SELECTION_RULE:
        print("  note: the BASE already claims the exact safeguard. Check you are not merging a "
              "partial run into data it has already been merged into.")

    sel = pm.get("selection") or {}
    if not sel.get("partial"):
        die("the partial manifest is not marked as a partial run")
    if set(sel.get("algorithms", ())) != set(algos):
        die(f"--algorithms {sorted(algos)} does not match the partial run's "
            f"{sorted(sel.get('algorithms', ()))}")
    missing = set(keep_ids) - set(sel.get("scenarios", ()))
    if missing:
        die(f"the partial run does not cover retained scenario(s) {sorted(missing)}")
    return bm, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", required=True, help="the affected-only rerun's output directory")
    ap.add_argument("--base", default=os.path.join(ROOT, "results"),
                    help="the complete production dataset to build on")
    ap.add_argument("--out", required=True, help="staging directory to create (must not exist)")
    ap.add_argument("--algorithms", default="binary_deep,reverse_eng_risk",
                    help="the algorithms whose rows the partial run replaces")
    ap.add_argument("--drop-families", default="broad_prior",
                    help="scenario families to delete from the merged dataset ('' keeps all)")
    ap.add_argument("--scenarios", default="",
                    help="restrict the merged dataset to these scenario ids (default: every "
                         "scenario not in a dropped family)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out directory")
    a = ap.parse_args()

    algos = [x.strip() for x in a.algorithms.split(",") if x.strip()]
    drop_fams = {x.strip() for x in a.drop_families.split(",") if x.strip()}
    unknown = [x for x in algos if x not in M.ALGORITHMS]
    if unknown:
        die(f"unknown algorithms {unknown}")

    keep_ids = [s["id"] for s in M.SCENARIOS if not (set(s.get("families", ())) & drop_fams)]
    if a.scenarios.strip():
        want = [x.strip() for x in a.scenarios.split(",") if x.strip()]
        unknown = [x for x in want if x not in {s["id"] for s in M.SCENARIOS}]
        if unknown:
            die(f"--scenarios: unknown {unknown}")
        excluded = [x for x in want if x not in keep_ids]
        if excluded:
            die(f"--scenarios names {excluded}, which --drop-families removes")
        keep_ids = [x for x in keep_ids if x in want]
    drop_ids = [s["id"] for s in M.SCENARIOS if s["id"] not in keep_ids]

    base, part, out = (os.path.abspath(a.base), os.path.abspath(a.partial), os.path.abspath(a.out))
    live = os.path.abspath(os.path.join(ROOT, "results"))
    if out == live:
        die("--out is the live production directory. Stage the merge somewhere else and promote "
            "it deliberately.")
    if out in (base, part):
        die("--out must differ from --base and --partial")
    if os.path.exists(out):
        if not a.force:
            die(f"{out} already exists (pass --force to replace it)")
        shutil.rmtree(out)

    print(f"base    {base}\npartial {part}\nout     {out}")
    print(f"replacing {algos} in {len(keep_ids)} scenario(s); dropping {drop_ids or 'nothing'}\n")
    check_manifests(base, part, algos, keep_ids)

    # ---------------------------------------------------------------- copy, then rebuild tables
    shutil.copytree(base, out)

    total_kept = total_new = 0
    for name, extra in TABLES.items():
        b = read(os.path.join(base, name))
        p = read(os.path.join(part, name))
        if b is None:
            die(f"base is missing {name}")
        if p is None:
            die(f"partial run is missing {name} -- all six raw tables are required")
        bfields, brows = b
        pfields, prows = p
        if bfields != pfields:
            die(f"{name}: header differs between base and partial\n  base    {bfields}\n"
                f"  partial {pfields}")

        # rows the partial run is allowed to contribute
        new = [r for r in prows if r["scenario_id"] in keep_ids and r["algorithm"] in algos]
        stray = [r for r in prows if r not in new]
        if stray:
            die(f"{name}: the partial run holds {len(stray)} row(s) outside the merge scope, "
                f"first = {key(stray[0], extra)}")

        kept = [r for r in brows
                if r["scenario_id"] in keep_ids and r["algorithm"] not in algos]

        for label, rows in (("base(kept)", kept), ("partial", new)):
            ks = [key(r, extra) for r in rows]
            if len(ks) != len(set(ks)):
                dupe = next(k for k in ks if ks.count(k) > 1)
                die(f"{name}: duplicate cell key in {label}: {dupe}")

        # the replaced cells must cover exactly what the base covered -- no gained or lost points
        old_keys = {key(r, extra) for r in brows
                    if r["scenario_id"] in keep_ids and r["algorithm"] in algos}
        new_keys = {key(r, extra) for r in new}
        if old_keys != new_keys:
            miss, extra_k = sorted(old_keys - new_keys)[:5], sorted(new_keys - old_keys)[:5]
            die(f"{name}: replaced-cell coverage differs.\n  missing from partial: {miss}\n"
                f"  not in base:          {extra_k}\n"
                f"  ({len(old_keys)} base vs {len(new_keys)} partial)")

        # seeds and trial counts must match between the halves being stitched together
        for col in PROVENANCE:
            if col not in bfields:
                continue
            bv = {r[col] for r in kept if r.get(col) not in ("", None)}
            pv = {r[col] for r in new if r.get(col) not in ("", None)}
            if bv and pv and bv != pv:
                die(f"{name}: {col} differs between kept and replaced rows: "
                    f"base {sorted(bv)[:4]} vs partial {sorted(pv)[:4]}")

        merged = kept + new
        merged.sort(key=lambda r: (r["scenario_id"], r["algorithm"], float(r["budget"]),
                                   *(r.get(k, "") for k in extra)))
        write(os.path.join(out, name), bfields, merged)
        total_kept += len(kept)
        total_new += len(new)
        print(f"  {name:<28} kept {len(kept):>5}  replaced {len(new):>5}  "
              f"dropped {len(brows) - len(kept) - len(old_keys):>5}  -> {len(merged):>5}")

    # --------------------------------------------------------------------------- the manifest
    pm = json.load(open(os.path.join(part, "experiment_manifest.json")))
    pm["selection"] = {
        "partial": False,
        "algorithms": list(M.ORDER),
        "scenarios": list(keep_ids),
        "note": "merged dataset: complete over the retained scenarios",
    }
    pm["merge_provenance"] = {
        "built_by": "analysis/merge_partial.py",
        "base": os.path.relpath(base, ROOT),
        "partial": os.path.relpath(part, ROOT),
        "algorithms_replaced": algos,
        "scenario_families_dropped": sorted(drop_fams),
        "scenarios_dropped": drop_ids,
        "rows_kept_from_base": total_kept,
        "rows_taken_from_partial": total_new,
    }
    with open(os.path.join(out, "experiment_manifest.json"), "w") as f:
        json.dump(pm, f, indent=2, default=str)

    # Files copied from the base fall into three groups, and blanket-deleting them would throw away
    # hours of unrelated simulation:
    #
    #   DERIVED    read the six raw tables, so they describe the OLD numbers. Deleted here and
    #              rewritten by the regeneration step, which is the only thing that may recreate
    #              them -- nothing stale can be read or promoted in between.
    #   INDEPENDENT own simulations that never touch the replaced algorithms' rows (the linear-search
    #              detector bake-off and the overshoot-criterion study). The depth rule cannot move
    #              them, and they cost far more to rebuild than this merge. Carried over as-is.
    #   FROM_PARTIAL  written fresh by the rerun itself; the partial run's copy wins.
    stale, kept_files = [], []
    for fn in sorted(os.listdir(out)):
        fp = os.path.join(out, fn)
        if not os.path.isfile(fp) or fn in TABLES or fn == "experiment_manifest.json":
            continue
        if is_independent(fn):
            kept_files.append(fn)
            continue
        os.remove(fp)
        stale.append(fn)
    shutil.rmtree(os.path.join(out, "tex"), ignore_errors=True)
    os.makedirs(os.path.join(out, "tex"), exist_ok=True)   # finalize writes into it, but does not
                                                           # create it (run.py's ensure_dirs does)

    from_partial = []
    for fn in FROM_PARTIAL:
        src = os.path.join(part, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, fn))
            from_partial.append(fn)

    print(f"\n{total_kept} rows kept, {total_new} replaced")
    print(f"removed {len(stale)} derived file(s) -- regenerate them against {out}")
    print(f"carried over {len(kept_files)} file(s) from independent studies "
          f"(linear detector, overshoot criterion) that the depth rule cannot affect")
    print(f"took {len(from_partial)} file(s) from the partial run: {from_partial}")
    if any(fn == "linear_search_claims.csv" for fn in kept_files):
        print("  NOTE linear_search_claims.csv reads the raw tables. It is carried over because "
              "rebuilding it re-simulates; re-run linear_search_claims.py if its numbers are "
              "quoted anywhere that must reflect the new rows.")
    print("\nnext:\n"
          f"  export RESULTS_OUT={out}\n"
          "  python analysis/add_baselines.py\n"
          "  python -c \"import sys;sys.path.insert(0,'analysis');"
          "import finalize;finalize.main('max')\"\n"
          "  python analysis/error_curves.py\n"
          "  python analysis/variance_curves.py\n"
          "  python analysis/thesis_tables.py\n"
          "  python analysis/thesis_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
