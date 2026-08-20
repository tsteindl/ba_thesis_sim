"""Replay the saved binary-search winners with exact depth-attainment diagnostics.

``binary_diagnostics.py`` originally stored only a within-5% indicator.  This script avoids repeating
the tuning step: it reads every saved winning (m, conf) configuration, evaluates it on a fresh seed,
and writes exact / 1% / 5% / 10% depth-hit rates to ``results/binary_diagnostics_replay.csv``.
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmetrology import experiments as E

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_diagnostics import FIELDS, _task


R = 5_000
SEED = 8_675_309
SOURCE = "results/binary_diagnostics.csv"
OUT = "results/binary_diagnostics_replay.csv"


def main():
    meta = {}
    with open("results/story_curves.csv", newline="") as f:
        for row in csv.DictReader(f):
            meta[row["setting"]] = (float(row["phi_min"]), float(row["phi_max"]), float(row["eps"]))

    with open(SOURCE, newline="") as f:
        saved = list(csv.DictReader(f))
    seeds = np.random.default_rng(SEED).integers(0, 2**63, size=(len(saved), R))
    tasks = []
    for row, trial_seeds in zip(saved, seeds):
        pmin, pmax, eps = meta[row["setting"]]
        cfg = {"m_exploration": int(row["m"]), "conf": float(row["conf"])}
        tasks.append((pmin, pmax, eps, int(row["budget"]), cfg, trial_seeds))

    print(f"replaying {len(tasks)} saved winners at R={R}", flush=True)
    with ProcessPoolExecutor(max_workers=E.N_JOBS, mp_context=E._CTX) as executor:
        evaluated = list(executor.map(_task, tasks, chunksize=1))

    header = ["setting", "budget", "m", "conf"] + FIELDS
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row, (sums, n) in zip(saved, evaluated):
            values = {field: sums[i] / n for i, field in enumerate(FIELDS)}
            writer.writerow({"setting": row["setting"], "budget": row["budget"],
                             "m": row["m"], "conf": row["conf"], **values})

    exact = np.mean([sums[FIELDS.index("p_exact")] / n for sums, n in evaluated])
    within1 = np.mean([sums[FIELDS.index("p_within_1")] / n for sums, n in evaluated])
    within5 = np.mean([sums[FIELDS.index("p_at_opt")] / n for sums, n in evaluated])
    within10 = np.mean([sums[FIELDS.index("p_within_10")] / n for sums, n in evaluated])
    print(f"mean across operating points: exact={100*exact:.2f}%  within1={100*within1:.2f}%  "
          f"within5={100*within5:.2f}%  within10={100*within10:.2f}%", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
