# Issue #3 native recheck evidence

This directory contains small immutable evidence for GitHub Issue #3. Large
per-query rows, profiles, and indexes remain under the ignored
`results/native_recheck_runs/` namespace. Every tracked summary identifies the
corresponding run and checksums; it must not be pooled with Issue #1 evidence.

## Audited superseded decision

The most recent fully audited source-frozen formal validation run was
`native-recheck-validation-20260905T175114361812Z-227a918c57`. It completed
84/84 blocks and contains 84 uncompressed JSONL shards, 42,064 rows, and
288,072,929 bytes at:

```text
results/native_recheck_runs/superseded/
  validation-pre-complete-report-audit-20260905T175114Z/
    native-recheck-validation-20260905T175114361812Z-227a918c57/raw/
```

Its raw-inventory SHA-256 is
`10cf04e63b8a699253d84462463b6e213a69cbd590505625732dca1ab0c563de`;
`COMPLETED.json` has SHA-256
`da0daff64d90878b5ef24d48dde6c1d0e6adfbf7b87c62946c5aea135a19b35d`.
The run is bound to config hash
`227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925`,
implementation-tree hash
`d6a9b505d0a3fe75a93ee1810ec69d69c7ea2e8607ecb7181d84b23388593003`,
and native shared-object hash
`eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.

The validation gate is `NOT_PASSED`. The stopping-rule decision is
`NOT_SUPPORTED_IN_TESTED_REGIME` / `NO_GO`, with
`final_status=NOT_RUN_GATE_NOT_PASSED`. No candidate lock, final-run config,
fresh SIFT holdout timing, final summary/gate, pre-HNSW authorization, or final
HNSW directory was created. This is a completed negative result, not an
unfinished final run. It is no longer the current publication run because an
independent audit required additional numeric material in the hashed report
generator. The run remains valid for its frozen source and is preserved rather
than pooled with its replacement. A new formal run will populate the current
validation and tracked analysis paths after that reporting correction passes
tests and native correctness.

## Publication files

- `development_profile_old.json`: hash-bound development-only profile summary
  for the old Python path; its timings are not pooled with validation.
- `holdout_manifest.json`: pre-registration record for the untouched SIFT
  query interval `[1200,2200)` and the historically used query-ID audit.
- `native_correctness.json` and `native_correctness_junit.xml`: the actual
  native 10,000-case and regression gate.
- `validation_summary.json`: deterministic aggregate, accepted-run identity,
  comparison statistics, and the complete raw-shard path/row/checksum list.
- `validation_gate.json`: all 28 candidate decisions and the durable
  `NOT_PASSED` reason.
- `validation_representative_raw.json`: deterministic 96-row inspection sample
  bound to the exact summary, gate, and accepted run.
- `final_decision.json`: the validation stopping-rule outcome and proof that
  the fresh-final path was not authorized.

Audited superseded-output SHA-256 values (these identify the preserved audit
copy, not the temporarily absent current publication paths):

| artifact | SHA-256 |
|---|---|
| `native_correctness.json` | `5bddea79c03b763a8bf498b367e3a81b8b7a001b331d4601dd5a1c99c64601c4` |
| `native_correctness_junit.xml` | `247f0f4cbc691b6cd40e22efb9423eaf0eba7dc8c836a56e24c9c654e0dd5cee` |
| `validation_summary.json` | `4511edaaa2f056623cbda7c13f4ceea1ecde6a8908467c6383ba3c9eaff20415` |
| `validation_gate.json` | `8c109ac4a844d60493d337e2bcdb6616fd0ba416d447eb4d9ad87d89c766ce45` |
| `validation_representative_raw.json` | `6c01fd2b9a5fed9424e4d96ebaaad81b16a7a0543bfab2a56e3b19c0b2f15c81` |
| `final_decision.json` | `b0cf8944a2d3c9f2dd4650d6424f426a5f4e1cab1163eb99818f56693a679180` |

The preserved generated Japanese synthesis is under the superseded audit
directory. Its SHA-256 is
`95488e87f14f51cac94e2f7c0451d3491937b49e8b7a3fbb5d774fffb32466a7`.

## Regeneration versus re-execution

After the replacement formal run completes, this command will revalidate every
current raw and ancillary checksum and regenerate the Japanese report from the
tracked, hash-bound summary/gate/representative files without rerunning timed
search:

```bash
make native-report \
  NATIVE_VALIDATION_INPUT=results/native_recheck_runs/validation
```

To independently rebuild the three analysis files from raw before generating
the report, run the analyzer directly with the frozen bootstrap settings:

```bash
.venv/bin/python scripts/analyze_native_recheck.py \
  --input results/native_recheck_runs/validation \
  --output results/native_recheck_evidence/validation_summary.json \
  --gate-output results/native_recheck_evidence/validation_gate.json \
  --bootstrap-resamples 5000 --bootstrap-seed 6202052 \
  --representative-raw-output \
    results/native_recheck_evidence/validation_representative_raw.json
make native-report
```

`make native-validate` is different: it reruns the approximately half-hour
one-thread benchmark and creates a new run identity. Never combine rows from a
new run with a superseded run, and never copy values from a partial run. The
first complete validation, whose report generator later failed an independent
interpretation audit, is preserved intact below
`results/native_recheck_runs/superseded/validation-pre-report-audit-20260905T174058Z/`;
the second audited run described above is in its own sibling namespace. Neither
will be included in the replacement aggregate.

Once the replacement artifacts are populated, a third party without the
ignored raw/run directory can inspect the decision, correctness identity, all
shard checksums and row counts, aggregate statistics, gate candidates, and the
deterministic representative rows from the tracked files. They cannot
independently reaggregate all per-query latency samples, re-read every
LB/Base-cache/build audit, or reconstruct raw observations from the summary.
Raw JSONL is intentionally uncompressed; no compressed artifact is claimed.
