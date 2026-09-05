# Issue #3 native recheck evidence

This directory contains the small immutable publication evidence for GitHub
Issue #3. Large per-query rows, profiles, and indexes remain under the ignored
`results/native_recheck_runs/` namespace. Every tracked aggregate binds its
accepted run and checksums; Issue #3 evidence must not be pooled with Issue #1
evidence or with the archived Issue #3 runs.

## Current publication decision

The source-frozen formal validation run is
`native-recheck-validation-20260905T191058023915Z-227a918c57`. It completed
84/84 blocks without a recorded run failure and contains 84 uncompressed JSONL
shards, 42,064 rows, and 288,076,713 bytes at:

```text
results/native_recheck_runs/validation/
  native-recheck-validation-20260905T191058023915Z-227a918c57/raw/
```

Its canonical raw-inventory SHA-256 is
`d31218eb63a96136c1e553430f5d10f5ec8b8f7b5aa8a07e3e45762ed2049240`;
`COMPLETED.json` has SHA-256
`10665158cdad322c0cf37926d6e21f8225e61ca705972ddfda97c4491bce5366`.
The run is bound to config hash
`227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925`,
implementation-tree hash
`bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f`,
and native shared-object hash
`eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.

The validation gate is `NOT_PASSED` because no non-degenerate candidate beats
both certified native F and optimized A under the preregistered paired-CI
criterion. The terminal stopping-rule verdict is
`NOT_SUPPORTED_IN_TESTED_REGIME`; the engineering decision is `NO_GO`, with
`final_status=NOT_RUN_GATE_NOT_PASSED`. This is a completed negative result,
not an unfinished final run. In accordance with the fail-closed protocol, no
candidate lock, final-run config, fresh SIFT holdout timing, final summary/gate,
pre-HNSW authorization, or final HNSW directory was created.

A separate read-only audit rehashed and re-counted all raw and ancillary
files, reconstructed native call accounting, exact ordering and beta chains,
rechecked every strict LB decision, and regenerated all operating-point and
comparison aggregates. It reported no findings. The 54-file ancillary
inventory contains 30,000,445 bytes and has canonical SHA-256
`7d0675536cc1e307204d0691c40ee588a7a9c28510ef1c1bf45531f6ab84c400`;
the run manifest, checkpoint, and build-manifest SHA-256 values are
`4894711dff55701dcec1548c442ee636dbba0869cefa0c6af6a8d4f339cb933c`,
`ac138ce365f65be68b3904c72dbcdf0f81479ad624f8eb7e670f6df1add57957`,
and `4d9b90706e31c72a94d6ec842c180ec3528ee75fc990907da5701d77a7ee65b1`.

## Publication files

- `development_profile_old.json`: hash-bound development-only profile summary
  for the old Python path; its timings are not pooled with validation.
- `holdout_manifest.json`: preregistration record for the untouched SIFT query
  interval `[1200,2200)` and the historical query-ID audit.
- `native_correctness.json` and `native_correctness_junit.xml`: the actual
  native 10,000-case and regression gate (255 passed, zero failures/errors or
  skips in the saved JUnit).
- `validation_summary.json`: deterministic aggregate, accepted-run identity,
  comparison statistics, integrity results, and the complete raw-shard
  path/row/byte/checksum inventory.
- `validation_gate.json`: all 28 candidate decisions and the durable
  `NOT_PASSED` reason.
- `validation_representative_raw.json`: deterministic 96-row inspection sample
  bound to the exact summary, gate, and accepted run.
- `final_decision.json`: the validation stopping-rule outcome and proof that
  the fresh-final path was not authorized.
- `../../reports/NATIVE_KERNEL_RECHECK_ja.md`: generated Japanese synthesis of
  the correctness, validation, ablation, geometry, build/memory, and stopping
  evidence.

Current publication SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `native_correctness.json` | `dd2cd9b1e50b4202e2d39e39acc36b48a317fdc8908acff6630bc3a56c0e22d5` |
| `native_correctness_junit.xml` | `e066c863c950b62824b32fab8c80d3783cc86015d196abf8951023b2c06c4dba` |
| `validation_summary.json` | `1735992cd55f13bbe5281e354bba8af8bccaf10f073e04f7941e6569e788f41c` |
| `validation_gate.json` | `17e04e5bfbd592dcfee4b43604fdd6bb7860aafa6e29501dfaf2c191a02f5b92` |
| `validation_representative_raw.json` | `bd831a851d52564cad074073076c3adcd4c72da2412a9cc568558989a5a993a7` |
| `final_decision.json` | `7abd3aee91670bfaf0dffcf344597536812072b78151828aa28678af04c91029` |
| `reports/NATIVE_KERNEL_RECHECK_ja.md` | `9807d1da5629855edb772db33f24d8785434225b1ca8ab38ca7729072827b235` |

## Regeneration versus re-execution

With the ignored raw/run directory present, this command revalidates every raw
and ancillary checksum and regenerates the Japanese report without rerunning
timed search:

```bash
make native-report \
  NATIVE_VALIDATION_INPUT=results/native_recheck_runs/validation
```

To independently rebuild the three analysis files from raw before generating
the report, use the frozen bootstrap settings:

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

`make native-validate` is different: it reruns the one-thread benchmark and
creates a new run identity. Never combine rows from a new run with an archived
run, and never copy values from a partial run. A third party without the ignored
raw/run directory can inspect the decision, correctness identity, all shard
checksums and row counts, aggregate statistics, gate candidates, and the
deterministic representative rows. They cannot independently reaggregate all
per-query latency samples, re-read every LB/Base-cache/build audit, or
reconstruct raw observations from the summary.

## Preserved superseded runs

Two earlier complete source-frozen formal runs remain under separate ignored
`results/native_recheck_runs/superseded/` namespaces. They were superseded
after independent audits required changes to the hashed report generator, not
because their frozen raw rows were silently altered or pooled. The later of
them is
`native-recheck-validation-20260905T175114361812Z-227a918c57` under
`validation-pre-complete-report-audit-20260905T175114Z/`; it contains 84
shards and 42,064 rows. The first is under
`validation-pre-report-audit-20260905T174058Z/`. Both complete runs, their
analysis copies, and the separately preserved failed/interim evidence remain
available for audit but are excluded from the current publication aggregate.
