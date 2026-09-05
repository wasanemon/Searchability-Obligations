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

For beta=0, group skip rates were 56.25% (clustered), 0% (isotropic), 0%
(Delta-near-query), and 75% (outlier-radius). Exact recall was 1.0 in all four
distributions, but pruning was slower than Delta Flat in every one. This is a
deliberately tiny pipeline/correctness smoke, not a practical-usefulness test.

The earlier post-audit run ending in `062705...` and the pre-audit run identified
in `PREAUDIT.md` remain preserved. They are development history and are not
substituted for the current source-frozen smoke above.
