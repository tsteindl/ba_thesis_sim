# Affected-only exact-safeguard rerun

Started : 2026-09-01T17:53:10+02:00
PID     : 82328   (also in sweep.pid)
Host    : tsteindl-X870-Taichi-Creator   cores: 24
Python  : /home/tsteindl/.pyenv/versions/3.11.9/bin/python3 3.11.9
Git     : e70668c (working tree modified -- see the diff for this change)

Command
-------
    CONSOLIDATED_OUT=results/consolidated_exact_safeguard_partial \
      python3 -u analysis/consolidated/run.py --max --resume --keep-traces \
        --algorithms binary_deep,reverse_eng_risk \
        --scenarios narrow_e3,narrow_e4,narrow_e5,narrow_e6,narrow_e7,narrow_e8,small_e4,wide_e4

Logs    : sweep_20260901_175310.out / sweep_20260901_175310.err  (resumable: rerun the same command, --resume
          skips only scenarios where EVERY selected algorithm already has a full budget grid)

Expected output
---------------
177 scenario/budget points x 2 algorithms = 354 rows in each one-row-per-cell raw table
(performance_curves, winners, diagnostics_by_point, budget_audit); detector_confusion holds
the 177 binary_deep rows only; diagnostics_by_phase has several phase-bin rows per cell.

add_baselines and finalize are deliberately NOT run here -- this directory is a partial
dataset. Merge with analysis/consolidated/merge_partial.py before regenerating anything.

Gates passed before launch
--------------------------
  tests/test_safeguard_exact.py     13/13
  tests/test_partial_pipeline.py     5/5
  tests/test_consolidated.py --slow 20/20
  results/consolidated/thesis_code/verify.py  200/200 on all four algorithms
  isolated --smoke partial run       ok
  timing preflight                   see timing_preflight.log
