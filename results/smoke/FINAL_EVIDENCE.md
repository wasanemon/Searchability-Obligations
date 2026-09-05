# Current offline smoke evidence

The current post-audit smoke run is:

```text
offline-smoke-20260905T121406.178089Z-43d3397eca
```

It has `COMPLETED.json`, implementation-tree SHA-256
`8c4581cf733bf25b3e27c115be1dbdad1285564f03846c0d0e93b822c2c80f31`,
8/8 checkpointed blocks, 384 raw query-method rows, zero contract violations,
zero optimized-baseline validation failures, and four of four independent
Fraction-oracle matches. `results/smoke/final_analysis/summary.json` and
`FINAL_EVIDENCE.generated.md` were regenerated only from this run's
checksummed raw shards with `evidence_role=calibration`.

The detached clean-worktree acceptance rerun is also preserved verbatim as:

```text
offline-smoke-20260905T122801.383599Z-43d3397eca
```

It used the same implementation-tree and config hashes, completed 8/8 blocks
and 384 raw rows, and has checkpoint SHA-256
`a3450087326a30b32901db792c594f4cf9bb9fc4ff430823a0b1dd16fb469982`.
This second run is clean-environment reproduction evidence; it is not pooled
with the designated run to manufacture a larger scientific sample.

For beta=0, group skip rates were 56.25% (clustered), 0% (isotropic), 0%
(Delta-near-query), and 75% (outlier-radius). Exact recall was 1.0 in all four
distributions, but pruning was slower than Delta Flat in every one. This is a
deliberately tiny pipeline/correctness smoke, not a practical-usefulness test.

The earlier post-audit run ending in `062705...` and the pre-audit run identified
in `PREAUDIT.md` remain preserved. They are development history and are not
substituted for the current source-frozen smoke above.
