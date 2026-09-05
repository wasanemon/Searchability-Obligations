# Final evidence snapshot

`summary.json` is the deterministic, path-independent projection produced by
`make report` after every accepted `final` run has passed raw-shard checksum,
row-count, identity, and `COMPLETED.json` validation. `manifest.json` records
its SHA-256 and byte size. The large per-query JSONL files remain under the
ignored `results/runs/` tree as required by `AGENTS.md`; their individual
SHA-256 values and row counts are retained in this snapshot.

Regenerate against an explicit raw-data location with:

```bash
make report REPORT_INPUT=/absolute/path/to/results/runs
```
