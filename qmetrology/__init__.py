"""Code for the thesis "Evaluating adaptive quantum metrology protocols for parameter search
under resource constraints".

  sim.py          circuit simulation + estimator
  algorithms.py   the fixed-budget phase-search algorithms (Chapter 3)
  safeguard.py    exploitation depth derived from the asymptotic law
  posterior.py    the exact-posterior depth criterion
  oracle.py       the analytic omniscient ceilings
  manifest.py     scenarios, budget grids, tuning grids, seeds and trial counts -- one source
  pipeline.py     tune -> freeze -> held-out evaluation, in parallel
  experiments.py  the Monte-Carlo evaluator the pipeline and the side studies share
  diagnostics.py  exploration / safeguard / detector telemetry
  trace.py        the neutral per-probe instrumentation the diagnostics read
  uncertainty.py  Wilson + bootstrap intervals and the curve crossing

Entry point for every reported number: python analysis/run.py --max
"""
