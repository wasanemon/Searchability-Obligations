# Post-audit offline smoke evidence

The current post-audit smoke run is:

```text
offline-smoke-20260905T062705.177285Z-43d3397eca
```

It has `COMPLETED.json`, implementation-tree SHA-256
`d2e55be3b4f19ddff046d4cdedbfeafe07ad12ccdf5a9603726eab612a269095`,
8/8 checkpointed blocks, 384 raw query-method rows, zero contract violations,
zero optimized-baseline validation failures, and four of four independent
Fraction-oracle matches. `results/smoke/analysis/summary.json` was regenerated
only from checksummed raw shards with `evidence_role=calibration`.

This is final evidence that the offline pipeline and core comparisons execute;
its small synthetic sample is not final evidence for practical usefulness.
The earlier run identified in `PREAUDIT.md` remains preserved but excluded.
