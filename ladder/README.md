# Ladder — a phase-unwrapping protocol beyond the thesis's Table 3.2

Self-contained addition, kept separate from `qmetrology/`/`analysis/`/`results/` on request:
imports `qmetrology` read-only, registers nothing in its `FIXED_BUDGET`, and writes only to
`ladder/results/`.

**Why:** every protocol in the thesis inverts a single measurement batch via
`phi_hat = arccos(sqrt(p_hat))/N`, valid only while `N*phi < pi/2`. That caps the usable
circuit depth, so cost-to-precision is `~1/eps^2` (standard quantum limit) for brute force
*and* every adaptive protocol alike — the ~1.72× ceiling in `results/RESULTS.md` is this
depth cap, not a physical Heisenberg limit. Sequencing measurements at geometrically
deepening `N`, each unwrapped by the previous (coarser) estimate, removes the cap and
recovers Heisenberg scaling (`~1/eps`) — the budget-ratio advantage then *grows* with
precision instead of saturating. See `algorithms.py`'s docstring for the mechanism and
references (Kitaev; Higgins et al. 2007; Berry et al. 2009).

```
ladder/
  algorithms.py    # find_phi_fixed_budget_ladder (+ _capped wrapper), same signature as qmetrology
  poc.py           # deterministic per-stage traces + a quick R=2,000 sanity comparison
  study.py         # de-biased tune/validate budget sweep -> results/ladder_{curves,cube,winners,t31}.csv
  make_results.py  # assembles results/LADDER.md + fig_ladder_scaling.png
  results/         # LADDER.md, the CSVs, the figure — separate from the top-level results/
```

## Reproduce

```bash
python ladder/poc.py           # ~1 min: traces + quick comparison
python ladder/study.py         # ~30-90 min: full de-biased sweep (--quick for a smoke test)
python ladder/make_results.py  # assembles results/LADDER.md
```

## Two variants

- **`ladder_capped`** — depth capped at `pi/(2*phi_min)`, the same maximum depth the existing
  protocols already use. Apples-to-apples: isolates the gain to the *inference* strategy.
- **`ladder_unbounded`** — depth grows freely. The information-theoretic ceiling; honestly,
  its GHZ circuit size grows without bound as `eps` shrinks (see the traces in `poc.py`).
