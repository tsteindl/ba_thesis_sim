"""Tests for the selective (affected-only) rerun mode and its merge step.

    python tests/test_partial_pipeline.py

The point of the selective mode is that a rerun of two algorithms cannot damage or half-write the
complete production dataset. These tests pin the properties that make that true:

 * the filters are validated against the manifest and recorded in experiment_manifest.json
 * a partial run refuses to run add_baselines/finalize
 * resume is per (scenario, algorithm): an unrelated algorithm's rows never mark a scenario done
 * merge_partial refuses every way the two datasets can fail to line up, and produces a complete
   merged dataset when they do
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from qmetrology import manifest as M

RUN = os.path.join(ROOT, "analysis", "run.py")
MERGE = os.path.join(ROOT, "analysis", "merge_partial.py")
ALGOS = "binary_deep,reverse_eng_risk"
SCENS = "narrow_e3,small_e4"
TABLES = ["performance_curves.csv", "winners.csv", "diagnostics_by_point.csv",
          "detector_confusion.csv", "budget_audit.csv", "diagnostics_by_phase.csv"]


def _run(out, *args, expect=0):
    env = dict(os.environ, RESULTS_OUT=out)
    p = subprocess.run([sys.executable, RUN, *args], env=env, capture_output=True, text=True)
    assert p.returncode == expect, f"exit {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p.stdout + p.stderr


def _rows(d, name):
    with open(os.path.join(d, name), newline="") as f:
        return list(csv.DictReader(f))


def _smoke_partial(out):
    return _run(out, "--smoke", "--algorithms", ALGOS, "--scenarios", SCENS)


# --------------------------------------------------------------------- filter validation
def test_filters_are_validated():
    import run as R
    for bad in (["--algorithms", "nope"], ["--scenarios", "nope"],
                ["--algorithms", "binary_deep,binary_deep"], ["--algorithms", ""]):
        try:
            R.resolve_selection(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")
    algos, scens, partial = R.resolve_selection(["--algorithms", ALGOS, "--scenarios", SCENS])
    assert algos == ["binary_deep", "reverse_eng_risk"]
    assert [s["id"] for s in scens] == ["narrow_e3", "small_e4"]
    assert partial is True
    # no filters at all is not a partial run
    assert R.resolve_selection([])[2] is False
    # naming every algorithm and every scenario explicitly is also not partial
    assert R.resolve_selection(["--algorithms", ",".join(M.ORDER),
                                "--scenarios", ",".join(s["id"] for s in M.SCENARIOS)])[2] is False
    # the typed order does not leak into the run order
    assert R.resolve_selection(["--algorithms", "reverse_eng_risk,binary_deep"])[0] == \
        ["binary_deep", "reverse_eng_risk"]


# --------------------------------------------------- a partial run's shape and its refusals
def test_partial_run_writes_all_six_tables_and_skips_finalize():
    with tempfile.TemporaryDirectory() as d:
        log = _smoke_partial(d)
        assert "skipping add_baselines and finalize" in log

        for name in TABLES:
            rows = _rows(d, name)
            assert rows, f"{name} is empty"
            assert {r["scenario_id"] for r in rows} == {"narrow_e3", "small_e4"}
            want = {"binary_deep"} if name == "detector_confusion.csv" \
                else {"binary_deep", "reverse_eng_risk"}
            assert {r["algorithm"] for r in rows} == want, name

        # finalize's outputs must NOT exist -- that is what makes the partial directory safe
        for derived in ("REPORT.md", "budget_crossings.csv", "optimal_params.csv",
                        "diagnostics_headline.csv"):
            assert not os.path.exists(os.path.join(d, derived)), derived

        man = json.load(open(os.path.join(d, "experiment_manifest.json")))
        sel = man["selection"]
        assert sel["partial"] is True
        assert sel["algorithms"] == ["binary_deep", "reverse_eng_risk"]
        assert sel["scenarios"] == ["narrow_e3", "small_e4"]
        assert sel["algorithms_omitted"] == ["brute", "linear"]
        assert "broad_pi2_e3" in sel["scenarios_omitted"]
        assert man["safeguard"]["selection_rule"] == "exact_enumeration_integer_shots"


def test_resume_is_per_scenario_and_algorithm():
    with tempfile.TemporaryDirectory() as d:
        _smoke_partial(d)
        assert "--resume: 2 scenario(s) already complete" in _run(
            d, "--smoke", "--resume", "--algorithms", ALGOS, "--scenarios", SCENS)

        # strip ONE algorithm from ONE scenario: the other algorithm's rows are still there, and
        # must not be enough to call the scenario done.
        rows = _rows(d, "performance_curves.csv")
        fields = list(rows[0])
        keep = [r for r in rows
                if not (r["scenario_id"] == "small_e4" and r["algorithm"] == "binary_deep")]
        assert len(keep) < len(rows)
        with open(os.path.join(d, "performance_curves.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
        log = _run(d, "--smoke", "--resume", "--algorithms", ALGOS, "--scenarios", SCENS)
        assert "--resume: 1 scenario(s) already complete" in log, log[-800:]
        assert "[small_e4]" in log, "the incomplete scenario was not rerun"
        assert "[narrow_e3]" not in log, "the complete scenario was rerun anyway"

        # a partial budget grid is likewise not "complete"
        rows = _rows(d, "performance_curves.csv")
        b0 = next(r["budget"] for r in rows if r["scenario_id"] == "narrow_e3")
        keep = [r for r in rows
                if not (r["scenario_id"] == "narrow_e3" and r["budget"] == b0)]
        assert len(keep) < len(rows)
        with open(os.path.join(d, "performance_curves.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
        log = _run(d, "--smoke", "--resume", "--algorithms", ALGOS, "--scenarios", SCENS)
        assert "--resume: 1 scenario(s) already complete" in log, log[-800:]
        assert "[narrow_e3]" in log, "a short budget grid was treated as complete"


# ------------------------------------------------------------------------ the merge step
def _merge(*args, expect=0):
    p = subprocess.run([sys.executable, MERGE, *args], capture_output=True, text=True)
    assert p.returncode == expect, f"exit {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p.stdout + p.stderr


def _fake_base(base, part, algos=("binary_deep", "reverse_eng_risk")):
    """A 'complete' dataset built from a partial smoke run: the same cells, plus brute/linear rows
    standing in for the algorithms the rerun does not touch, plus a broad-prior scenario to drop."""
    os.makedirs(base, exist_ok=True)
    for name in TABLES:
        rows = _rows(part, name)
        fields = list(rows[0])
        out = list(rows)
        for r in rows:
            if r["algorithm"] != algos[0]:
                continue
            for other in ("brute", "linear"):
                if name == "detector_confusion.csv" and other == "brute":
                    continue
                stub = dict(r, algorithm=other)
                if "implementation_variant" in fields:
                    stub["implementation_variant"] = f"stub.{other}"
                out.append(stub)
            out.append(dict(r, scenario_id="broad_pi2_e3"))     # a row the merge must delete
        with open(os.path.join(base, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
    man = json.load(open(os.path.join(part, "experiment_manifest.json")))
    man.pop("safeguard", None)
    man["selection"] = {"partial": False}
    with open(os.path.join(base, "experiment_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    # a derived file the merge must strip, because it describes the OLD numbers
    open(os.path.join(base, "REPORT.md"), "w").write("stale report\n")


def test_merge_builds_a_complete_dataset():
    with tempfile.TemporaryDirectory() as d:
        part, base, out = (os.path.join(d, x) for x in ("part", "base", "stage"))
        _smoke_partial(part)
        _fake_base(base, part)

        log = _merge("--partial", part, "--base", base, "--out", out,
                     "--algorithms", ALGOS, "--drop-families", "broad_prior",
                     "--scenarios", SCENS)
        assert "rows kept" in log

        for name in TABLES:
            rows = _rows(out, name)
            assert "broad_pi2_e3" not in {r["scenario_id"] for r in rows}, name
            want = {"binary_deep", "linear"} if name == "detector_confusion.csv" \
                else {"binary_deep", "reverse_eng_risk", "brute", "linear"}
            assert {r["algorithm"] for r in rows} == want, (name, {r["algorithm"] for r in rows})
            extra = ("phase_bin",) if name == "diagnostics_by_phase.csv" else ()
            ks = [(r["scenario_id"], r["algorithm"], r["budget"], *(r[k] for k in extra))
                  for r in rows]
            assert len(ks) == len(set(ks)), f"{name} has duplicate cell keys"
            # the replaced rows are the PARTIAL run's, not the base's
            for r in rows:
                if r["algorithm"] in ("brute", "linear") and "implementation_variant" in r:
                    assert r["implementation_variant"].startswith("stub."), name

        man = json.load(open(os.path.join(out, "experiment_manifest.json")))
        assert man["selection"]["partial"] is False
        assert man["safeguard"]["selection_rule"] == "exact_enumeration_integer_shots"
        assert man["merge_provenance"]["algorithms_replaced"] == ["binary_deep",
                                                                  "reverse_eng_risk"]
        assert set(man["merge_provenance"]["scenarios_dropped"]) >= {"broad_pi4_e3",
                                                                     "broad_pi2_e3"}
        assert not os.path.exists(os.path.join(out, "REPORT.md")), "stale derived file survived"


def test_merge_refuses_mismatches():
    with tempfile.TemporaryDirectory() as d:
        part, base = os.path.join(d, "part"), os.path.join(d, "base")
        _smoke_partial(part)
        _fake_base(base, part)
        args = ["--partial", part, "--base", base, "--algorithms", ALGOS,
                "--scenarios", SCENS]

        def refuse(tag, mutate, restore):
            o = os.path.join(d, "out_" + tag)
            mutate()
            try:
                log = _merge(*args, "--out", o, expect=1)
                assert "MERGE REFUSED" in log, (tag, log[-400:])
            finally:
                restore()
                shutil.rmtree(o, ignore_errors=True)

        pman = os.path.join(part, "experiment_manifest.json")
        orig = open(pman).read()

        def putman(fn):
            m = json.loads(orig)
            fn(m)
            json.dump(m, open(pman, "w"), indent=2)

        refuse("nosafeguard", lambda: putman(lambda m: m.pop("safeguard")),
               lambda: open(pman, "w").write(orig))
        refuse("oldrule", lambda: putman(
            lambda m: m["safeguard"].update(selection_rule="geometric_refine_smooth_shots")),
            lambda: open(pman, "w").write(orig))
        refuse("notpartial", lambda: putman(lambda m: m["selection"].update(partial=False)),
               lambda: open(pman, "w").write(orig))
        refuse("seeds", lambda: putman(lambda m: m["seeds"].update(test=999)),
               lambda: open(pman, "w").write(orig))
        refuse("trials", lambda: putman(lambda m: m["trials"].update(R_test=7)),
               lambda: open(pman, "w").write(orig))
        refuse("scencover", lambda: putman(
            lambda m: m["selection"].update(scenarios=["narrow_e3"])),
            lambda: open(pman, "w").write(orig))

        # a missing replaced cell must be caught by the coverage check, not silently dropped
        pc = os.path.join(part, "performance_curves.csv")
        pcorig = open(pc).read()

        def drop_one():
            rows = _rows(part, "performance_curves.csv")
            fields = list(rows[0])
            with open(pc, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows[1:])
        refuse("coverage", drop_one, lambda: open(pc, "w").write(pcorig))

        # and the live production directory is never a legal destination
        log = _merge(*args, "--out", os.path.join(ROOT, "results"), expect=1)
        assert "live production directory" in log


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}", flush=True)
        except AssertionError as ex:
            bad += 1
            print(f"  FAIL  {name}: {ex}", flush=True)
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
