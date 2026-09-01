"""CSV/JSON writing helpers for the results pipeline.

Every file is tidy (one row per fact) and self-describing: each row repeats the scenario keys, the
algorithm and its implementation variant, the budget, the seeds, the trial count R, and -- for every
proportion -- its eligible denominator. Nothing downstream has to know a column-prefix convention or
a global default R.
"""
import csv
import json
import os

# overridable so the test-suite can drive the whole pipeline into a scratch directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("RESULTS_OUT", os.path.join(REPO_ROOT, "results"))


def path(*parts):
    return os.path.join(OUT, *parts)


def ensure_dirs():
    os.makedirs(path("tex"), exist_ok=True)
    os.makedirs(path("traces"), exist_ok=True)


def write_json(name, obj):
    with open(path(name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


class Table:
    """Append-only CSV with a fixed header, so an interrupted run keeps its finished scenarios."""

    def __init__(self, name, fields, reset):
        self.p = path(name)
        self.fields = list(fields)
        if reset or not os.path.exists(self.p):
            with open(self.p, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def rows(self, rows):
        if not rows:
            return
        with open(self.p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self.fields})

    def read(self):
        if not os.path.exists(self.p):
            return []
        with open(self.p, newline="") as f:
            return list(csv.DictReader(f))


def scenario_keys(scen):
    return {
        "setting": scen["label"], "scenario_id": scen["id"],
        "phi_min": scen["phi_min"], "phi_max": scen["phi_max"],
        "phi_distribution": scen["phi_dist"], "eps": scen["eps"],
    }


def live_points(algorithm, lo=0.02, hi=0.98):
    """The (scenario, budget) cells where `algorithm` is neither floored nor saturated.

    Side studies use this to spend their trials where a difference can still show up: at a
    saturated point every arm converges, so comparing them there measures nothing.
    """
    pts = []
    with open(path("performance_curves.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["algorithm"] == algorithm and lo < float(r["rate"]) < hi:
                pts.append(dict(setting=r["setting"], phi_min=float(r["phi_min"]),
                                phi_max=float(r["phi_max"]), eps=float(r["eps"]),
                                budget=int(r["budget"])))
    return pts
