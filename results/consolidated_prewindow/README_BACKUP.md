# Snapshot: the sweep behind the current thesis draft

This is a verbatim copy of `results/consolidated/` as it stood on 2026-08-25, immediately before
the re-run that added the `mean_window` axis to Algorithm 4
(see [`../consolidated/LINEAR_SEARCH.md`](../consolidated/LINEAR_SEARCH.md)).

Every number in the thesis draft as reviewed, and every number quoted in
`linear_search_revision_notes.md` and in `linear_search_claims.csv`, refers to **this** directory.
Keep it until the new sweep has been checked and the thesis text has been updated to match.

The re-run was `python analysis/consolidated/run.py --max --keep-traces` with
`qmetrology.manifest.ALGORITHMS["linear"]["discrete"]["mean_window"] = [0, 1, 2, 3, 4, 6, 8, 12]`.
`mean_window = 0` is the previously published cumulative-mean rule, so the new tuning grid contains
the old algorithm and the new winners cannot be worse than these except by tuning noise.
