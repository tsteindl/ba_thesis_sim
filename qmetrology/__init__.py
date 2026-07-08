"""Code for the thesis "Evaluating adaptive quantum metrology protocols for parameter search
under resource constraints".

  sim.py          circuit simulation + estimator
  algorithms.py   the fixed-budget phase-search algorithms (+ RE-lite for Table 3.1)
  experiments.py  parallel Monte-Carlo evaluator
  config.py       seeds, settings, params, grids, published values
  tables.py       the Chapter-3 %-converged tables (3.1, 3.3)

The budget-to-target results (Table 3.2 and the follow-up study) come from
analysis/extensive_sweep.py. Entry point: python -m qmetrology.reproduce.
"""
from . import algorithms, config, experiments, sim, tables
from .experiments import grid_search_max, rate_and_budget, success_rate, wilson

__all__ = ["sim", "algorithms", "experiments", "config", "tables",
           "success_rate", "grid_search_max", "rate_and_budget", "wilson"]
