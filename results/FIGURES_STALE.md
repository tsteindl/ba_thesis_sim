# Stale figures in `results/*.png`

`fig_story.png`, `fig_story_small.png`, `fig_pareto.png`, `fig_error.png`, `fig_error_small.png`,
`fig_precision.png` and `fig_broad.png` in **this** directory were produced by
`analysis/render_story.py` / `analysis/broad_dist_study.py` from the pre-consolidation sweep
(2026-08-17). They plot the **old opening-probe binary search** at **R = 40,000**.

Current versions are in **`results/consolidated/`**, at **R = 50,000**, with the deepest-probe
binary search and the widened tuning grids. Regenerate with:

```bash
python analysis/consolidated/thesis_figures.py           # all six
python analysis/consolidated/thesis_figures.py --only story,error
```

Every current figure carries its R and seeds in a footer strip, so a figure cannot silently
disagree with the tables.

The remaining `.png` files here (`fig_safeguard`, `fig_pilot_share`, `fig_posterior_*`,
`fig_levers`, `fig_binary_*`) belong to studies outside the consolidated pipeline and are unchanged.
