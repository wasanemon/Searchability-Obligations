# Research state

Last updated: 2026-09-06 05:10 (Asia/Tokyo)

## Issue #3 native recheck start

- GitHub Issue #3 was read in full through the connected GitHub application at
  `2026-09-05 23:46 JST`; it is open, was updated at
  `2026-09-05T14:44:29Z`, and has zero comments.
- PR #2 was checked at the same start point. It is open, unmerged, and has head
  branch `codex/issue-1` at local commit
  `2ddada6f162c4ac0759cc238b32483c37cd5358e` with base `main`.
- The working tree was clean on `codex/issue-1`. A separate stacked branch,
  `codex/issue-3-native`, was created from exactly that SHA; PR #2's branch was
  not modified and will not be merged automatically.
- Required pre-work reading completed: `AGENTS.md`, this state file,
  `reports/REPORT_ja.md`, `docs/correctness.md`, and `docs/numerics.md`. Existing
  negative results, immutable evidence, and the preserved incomplete run remain
  untouched.

Exact start commands:

```bash
git status --short --branch
git rev-parse HEAD
git remote -v
git log --oneline --decorate -8
git switch -c codex/issue-3-native
date '+%Y-%m-%d %H:%M:%S %Z'
git status --short --branch
git rev-parse HEAD
```

Outcome: start-state and branching gate passed. No experiment has yet been run
for Issue #3. The next resumable phase is a development-query profile of the
old Python pruning path, followed by the compiled F/N/P kernel design and
correctness-first tests.

## Issue #3 old-path development profile

- Before changing the search implementation, the old Python path was profiled
  on SIFT Base 100,000 / Delta 10,000, dimension 128, `k=10`, `C=64`, 128
  groups, 100 raw-pending records, five validation queries and five development
  queries. Both runs used one CPU thread and the new ignored namespace
  `results/native_recheck_runs/development/`; they are calibration only.
- cProfile run
  `native-recheck-old-python-profile-20260905T145406.329456Z-10eec0e992`
  completed 1/1 block. It recorded 16,289,513 calls in 69.931 s. Across all
  harness calls, `search_pruned` was called 24 times for 3.037 cumulative s,
  `distance_intervals` 6,074 times for 1.393 s, `stable_topk` 63 times for
  1.793 s, exact `Fraction` squared L2 630 times for 1.769 s, `np.stack` 232
  times for 0.524 s, and kth partition 3,198 times. The `.prof` SHA-256 is
  `9a5574fd8dfa48f7933598938eec58a30dd4d46e2db8b46cae0dbcbe90874043`.
- The unprofiled matched run
  `native-recheck-old-python-profile-20260905T145629.346102Z-10eec0e992`
  also completed 1/1 block. Descriptive five-query medians were 23.566 ms E2E
  for optimized Delta Flat versus 84.839 ms for old beta=0 pruning; pruning
  read 7,714.6 Delta vectors and skipped 23.2/128 groups on average. Its
  beta=0 component medians were LB 10.470 ms, group scan 12.045 ms, merge
  19.460 ms, ordering 1.101 ms, visibility 1.399 ms, and Receipt 0.798 ms.
- The tracked, immutable small summary is
  `results/native_recheck_evidence/development_profile_old.json`; it binds the
  config/implementation/run/raw/completion/profile hashes. cProfile-inflated
  timings are not used as benchmark estimates.
- The observations justify the required packed vectors/keys/offsets, one native
  call for batched LB and group scanning, incremental kth-upper maintenance,
  adaptive exact boundary work, and caching of immutable Base visibility maps.
  HNSW build time is excluded from this query-path diagnosis.

Exact profile commands:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m cProfile \
  -o results/native_recheck_runs/development/old_python.prof \
  scripts/run_experiment.py --config configs/native_recheck_profile.json
.venv/bin/python -c "import pstats; ..."
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_experiment.py \
  --config configs/native_recheck_profile.json
jq -s '<five-query descriptive aggregation>' \
  results/native_recheck_runs/development/<run-id>/raw/*.jsonl
sha256sum <profile/completion/raw paths>
```

Outcome: development profiling phase passed. The next resumable phase is to
implement and build the compiled F/N/P kernel without changing centers, group
membership, radii, or Base search parameters, then run native-specific
correctness tests before any validation timing.

## Issue #3 native implementation and pre-registration freeze

- The implementation phase now provides a pybind11/C++17 packed native kernel
  for F/N/P. It keeps contiguous vector/key/offset buffers, batched center/LB
  work, an incremental heap threshold, adaptive exact boundary ordering, exact
  certificate fallback, raw-pending scans, and the existing safe duplicate
  visibility fallback. The build explicitly uses `-fno-fast-math` and
  `-ffp-contract=off`, requires `FE_TONEAREST`, and records compiler, source,
  binary, CPU, and numeric-mode evidence. The loaded binary's source hash and
  current `_native_kernel.cpp` hash are both
  `bf32578fc0e025531c2ab961aa164bd62c7859b93b96a26328176089f8fa6215`;
  its shared-object SHA-256 is
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.
- The optimized A path shares adaptive exact shortlist ranking and small merge
  improvements but remains explicitly distinct from both the certified native
  F path and the non-certified A-reference/Faiss-order path. O remains the old
  same-C reference, and P-old remains development-profile-only. Immutable Base
  visibility/candidate metadata is cached only under the bound
  snapshot/generation/Delta-revision identity; cold/warm cache evidence is
  emitted separately.
- The native runner records separately randomized micro and real API-wall
  measurements, verifies the same frozen C identity, performs truth/oracle and
  LB audits outside timing, preserves per-query raw rows/checkpoints/completion
  receipts, and requires exact method/query/repetition coverage before a run is
  complete. Analysis accepts exactly one completed formal validation run and
  rechecks raw, ancillary, LB, Base-cache, config, implementation, test,
  holdout, native-build, and completion identities fail-closed. Large
  per-query ratio arrays were removed from tracked summaries; deterministic
  folding hashes plus aggregates remain reproducible from raw rows.
- The fixed validation matrix has 13 cells: the SIFT/GIST main anchors, SIFT
  Delta 0/1k/10k/100k, groups 64/128/512, k 1/10/100, beta 0/.01/.05/.10 at
  the main cell, and four required synthetic geometries. The final policy fixes
  the SIFT `[1200,2200)` fresh holdout, three process sessions, the primary and
  anchors, the 1.10 geometric-mean and 1.05 p95 engineering thresholds, and the
  condition that HNSW references run only after a fresh-final performance
  pass. Conditional HNSW raw/checkpoint/completion evidence is additionally
  bound to process build segments and serialized Base/Delta/full Faiss index
  hashes; crash-orphan raw is preserved and revalidated on resume.
- Before any native validation timing, all 44 then-existing split manifests
  were inspected without loading query vectors. Historical SIFT query IDs were
  exactly `[0,1200)` and the pre-registered `[1200,2200)` holdout is disjoint.
  GIST IDs `[0,1000)` were all historically used and are labelled
  non-independent. The frozen file/object SHA-256 pairs are validation config
  `939d940781a215c021a979a734c6121370fb17fb8cb0b177489d90a80200a9c2` /
  `76a8685ae6e8d0d2e736841201459d6a18ce00cf7405f9090167cee9a58bffc4`,
  final policy
  `2cc4c63556707cdd0bfd9ec459c1202d430b3f9d87da78f48dd1dee918bfd811` /
  `44604d74f5e3feec590e5db41c216a223fc745015dfa2cdc5ff74de362509f6a`,
  and holdout manifest file SHA-256
  `c88807e8c140a71c5a5c7bea5e64550d7fd26ad81210d9b90b46dd26d2071886`.
- Three independent code audits hardened test-tree hashing, validation-session
  count, resume/completion coverage, LB audit identity, policy typing and
  hashes, current-runtime checks, final/HNSW ordering, report run binding,
  measured ablations, and raw-evidence disclosure. The final consolidated
  lightweight run selected 244 tests and reported `244 passed, 2 deselected,
  19 warnings in 14.03s`; the warnings are NumPy's optional-PyYAML notice.
  `compileall` and `git diff --check` also succeeded. The executable-tree hash
  at this freeze point is
  `3f11bc88369f879a560b5251e883d5d641e48b874843ba2fb3630028ec8c75c0` and
  the complete `tests/**/*.py` tree hash is
  `ce49f6ad04225beb5727380a228f115e0ee72bf0bf57d3e88b186026b965c839`.

Exact successful freeze checks:

```bash
.venv/bin/python -m compileall -q src scripts tests
git diff --check
make test
.venv/bin/python - <<'PY'
# file/object hashes, implementation_tree_sha256, native_test_tree_sha256,
# native_build_info, and all config/policy/holdout registration assertions
PY
```

### Preserved pre-freeze failures

- An interim `make native-test` was accidentally started while final-policy
  fixtures were still being integrated. The native fixed-seed test itself ran,
  but the suite correctly failed with `1 failed, 219 passed in 64.08s` at
  `test_final_gate_uses_fresh_query_level_three_session_speedup_and_p95`
  (`NativeFinalError: locked final config has no experiment set`). It was
  written as `status=failed`, `fixed_seed_cases=0`, never used to authorize a
  benchmark, and copied before overwrite to
  `results/native_recheck_runs/correctness_failures/native-test-20260905T161341Z/`.
  The preserved JSON/JUnit SHA-256 values are respectively
  `09533c1acea95e8b154a90babc3c98caf1f4f42211d5299d4cc519ede0cf21d6` and
  `1af3dbc484324879c3c5a7a57f65c8bbfba4fadfe6ba5ab73e98fc898b7ac5e8`.
  Later targeted suites and the consolidated lightweight suite passed after
  the fixture/runtime fix; this does not substitute for the formal rerun.
- During report-test development, one targeted invocation reported `1 failed,
  22 passed` because a newly inserted test accidentally enclosed the tail of a
  neighboring fixture (`NameError: decision is not defined`). The placement
  was corrected and the exact target then reported `24 passed in 3.99s`; no
  scientific raw was produced. A requested `ruff` check could not run because
  ruff is not installed and is not claimed as passed.

Outcome: implementation, tests, configs, and pre-registration are ready for a
phase commit. No validation timing has run. The next resumable commands, in
order, are `make setup` and `make native-test`; validation must not start unless
the regenerated correctness evidence is `passed` and matches the frozen source,
test tree, and loaded native binary.

## Issue #3 source-frozen native correctness gate

- The implementation/pre-registration phase was committed as `a399143` with
  message `feat: add native recheck pipeline (refs #1, #3)`. The implementation
  tree remained
  `3f11bc88369f879a560b5251e883d5d641e48b874843ba2fb3630028ec8c75c0`.
- `make setup` completed successfully at that commit. Its PyPI probes emitted
  sandbox DNS retry warnings, but every pinned package was already installed;
  the editable native wheel built and installed, environment/Faiss HNSW smoke
  passed, all five thread variables and Faiss were 1, and the rebuilt native
  shared object reproduced SHA-256
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.
  Thus the warnings are preserved but are not relabelled as a setup failure.
- The formal `make native-test` then collected 246 tests and selected 245 under
  the declared `not evaluation or native_fixed_seed` expression. It completed
  `245 passed, 1 deselected, 20 warnings in 67.26s`. The native fixed-seed case
  property is exactly 10,000; the other full evaluation test was the sole
  deselection. There were zero failures, errors, or skips. Warnings comprise
  the optional-PyYAML messages plus pytest's xunit2 `record_property` notice;
  they remain visible.
- `results/native_recheck_evidence/native_correctness.json` has
  `status=passed`, binds commit `a3991432d06d54aaf3a45d1d21ac49b954a70a74`,
  the implementation hash above, test-tree hash
  `ce49f6ad04225beb5727380a228f115e0ee72bf0bf57d3e88b186026b965c839`,
  and the loaded native binary. Its SHA-256 is
  `1c380b14e29cd13ab31fe7ccd0f533364ab77a20d4c1a78a70932f3790c2b2ea`.
  The JUnit SHA-256 is
  `db450f8e4cb70cb89d9f2ede66049c968ebff8dbc170e70e5d2363f9445a3a1d`.

Exact commands:

```bash
make setup
make native-test
.venv/bin/python -m json.tool \
  results/native_recheck_evidence/native_correctness.json
sha256sum results/native_recheck_evidence/native_correctness.json \
  results/native_recheck_evidence/native_correctness_junit.xml
```

Outcome: the source-frozen correctness prerequisite passed with actual native
execution and may authorize offline smoke. No validation timing has run. The
next resumable command is `make native-smoke`; any later executable or test
change invalidates this evidence and requires `make native-test` again.

## Issue #3 source-frozen offline native smoke

- `make native-smoke` completed the new, isolated `smoke-v3` namespace as run
  `native-recheck-offline-smoke-20260905T165226725173Z-f4ee103dc9`. It executed
  all four synthetic families, F/A/A-reference/N/P, beta 0 and positive beta,
  O correctness rows, one real native backend, eight query blocks, 32 unique
  measurement queries, and 224 raw rows. All 8/8 blocks completed; 128 native
  rows had zero correctness or backend/call failures, and all four 4-query LB
  audits passed with no failure. Raw comprises eight shards and 1,209,570
  bytes under the ignored run directory.
- The effective config hash is
  `f4ee103dc91fca96e6d6593a2ed640eb25a651a6a834702bdcb1ea203bb95b9e`;
  run manifest SHA-256 is
  `426597a2c08099721460743f9f9ed501da25fe8fbd848a8aaa018040c3664e43`,
  checkpoint SHA-256 is
  `d9947c2708f2c207242a5cae0a282ff6d6d23703700969296e4c28f4cc97d0cb`,
  and completion SHA-256 is
  `12154af45644b454fb2a5f71c238e3bb40d5e2a361926d328b269f3a86d270df`.
  The analyzed summary/gate SHA-256 values are respectively
  `d3d7bbbf684cc24bf58b8bf323c9f82e6b298028b6bcf0200e48ab1c7f0e97e2` and
  `c3e769b0931268bf475d4878ece769553cb0d096cf3e1196cb8694bfb62476a5`.
- The smoke analyzer correctly wrote `NOT_PASSED` because smoke evidence is
  synthetic/non-validation and no real Delta>=1,000 candidate can authorize a
  fresh final lock. This is a structural smoke outcome, not the formal
  validation conclusion. No final lock was created.
- This post-registration smoke adds only four separately generated synthetic
  split manifests. It neither reads TEXMEX vectors nor changes the already
  frozen historical SIFT/GIST ID union or the `[1200,2200)` holdout decision;
  the 44-manifest pre-registration audit remains immutable.

Exact command:

```bash
make native-smoke
```

Outcome: offline compiled-path smoke passed and did not authorize final work.
The next resumable command is `make native-validate`, using the already frozen
validation config and correctness evidence. No source, test, config, policy, or
holdout file may change during that run.

## Issue #3 first formal validation and independent report audit

- The first source-frozen `make native-validate` completed run
  `native-recheck-validation-20260905T165354658861Z-227a918c57` in about
  34 minutes. It completed all 84/84 blocks and preserved 84 raw shards,
  42,064 rows, and 288,078,673 bytes. Thirteen experiment cells and 2,056
  unique queries were present; 54 ancillary artifacts and every raw checksum
  verified. Native rows totalled 27,672, with distinct positive native call
  indices and no backend, correctness, beta-chain, lower-bound soundness, or
  strict-predicate failures. The 325 audited query/group sets contained 35,125
  decisions (25,248 scans and 9,877 skips).
- The run's implementation/config identities were respectively
  `3f11bc88369f879a560b5251e883d5d641e48b874843ba2fb3630028ec8c75c0`
  and `227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925`.
  Its completion SHA-256 was
  `83f998af38606ccd0f896a9b60ddbcb86bba6a35f8f1dcc95a01f8b42f6f8176`
  and raw-inventory SHA-256 was
  `2a24974c12311c92ac834a7271f2300773cb03635f79e371d274ab594c5f91f8`.
  Independently regenerated summary/gate/representative SHA-256 values were
  `46bec044213bea0d90c9679163cbd1b7c7087f4e6112d729e93b2d7a688055ce`,
  `2e14434ef016ca9a58d05398562f4b1fe34918725aa42c29a6dbb417618a5652`,
  and `0c7f7af108b508018ba902ee59dc88ccb08db4561015541217a16a8609d7813e`.
- The validation gate was `NOT_PASSED`: all 16 non-degenerate SIFT candidates
  beat certified F at the paired-CI criterion, but none beat optimized A.
  For the pre-specified `sift-initial` family, F/P API-wall geomeans ranged
  1.097--1.131 while A/P ranged 0.582--0.600. Same-C quality mismatch counts
  were zero. `make native-evaluate` therefore wrote decision SHA-256
  `40512a3e821dc100b5be050f110f56cc8c35705e45567c0b1accf2901e43ef15`
  with verdict `NOT_SUPPORTED_IN_TESTED_REGIME`, engineering decision `NO_GO`,
  and `final_status=NOT_RUN_GATE_NOT_PASSED`; no final lock/config/summary/gate,
  fresh-holdout load, final HNSW authorization, or final HNSW directory was
  created.
- Two independent read-only audits found no numerical, raw-inventory,
  ablation, geometry, break-even, gate, or final-decision error. One audit did
  find four publication-level interpretation defects in the generated report:
  it called all 18 real/non-degenerate points gate-eligible although only 16
  SIFT points were eligible; led with post-hoc secondary maxima instead of the
  pre-specified main family; reversed the Base-cache improvement arrow; and
  pooled coordinate-dependent L2 gaps across SIFT, GIST, and synthetic data.
  A fifth editorial issue said summary/gate/representative evidence alone
  exposed the final verdict even though `final_decision.json` is also needed.
- The original run and its five generated analysis artifacts were not deleted
  or relabelled as failures. They were moved intact to the ignored audit path
  `results/native_recheck_runs/superseded/validation-pre-report-audit-20260905T174058Z/`.
  The copied summary, gate, representative, final decision, and report retain
  SHA-256 values `46bec044...`, `2e14434e...`, `0c7f7af1...`, `40512a3e...`,
  and `5a969e93...`. The raw/completion identities above remain verifiable.
- The report generator now separates the 16 SIFT gate candidates from two GIST
  anchors, leads with `sift-initial`, labels secondary maxima as post-hoc
  descriptive values while retaining them in the pre-registered gate candidate
  set, reports RQ3 within that one condition, orients Base
  rebuild-to-cache improvement correctly using the paired geometric mean, and
  states explicitly that the negative verdict is a validation stopping-rule
  result rather than a fresh-final estimate. Regression expectations were
  updated with those distinctions. A follow-up read-only check caught one
  overcorrection: secondary axes are part of the pre-registered gate candidate
  set even though their observed maximum must not replace the main-family
  confirmatory summary. The generator and test now state both facts instead of
  claiming those candidates were excluded from selection.
- The first targeted report-test invocation after this edit reported `1 failed,
  11 passed`: the new assertion expected a scalar skipped-group display while
  the generator intentionally renders an operating-point range. No experiment
  or scientific raw was produced. After correcting only that expected string
  (and the fixture's 600/20=30-query break-even expectation), the same command
  reported `12 passed in 0.68s`. `compileall`, `git diff --check`, and the full
  lightweight `make test` then passed. After the follow-up wording correction,
  the targeted test again reported `12 passed in 0.68s` and the final
  lightweight rerun reported 244 passed, 2 deselected, 19 warnings in 14.00s.
- Because every executable script and every test file is deliberately hashed,
  the reporting corrections changed the implementation-tree/test-tree hashes
  to `d6a9b505d0a3fe75a93ee1810ec69d69c7ea2e8607ecb7181d84b23388593003`
  and `d1e7e181d1684afa60ff156283fcd80f7949c3c01f7b07557b1a29ca07e8ab59`.
  Reusing the first validation as current evidence would therefore fail closed.
  The native binary itself remains byte-identical at
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.

Exact commands for this phase:

```bash
make native-validate
make native-evaluate
make native-report
sha256sum <validation/raw/completion/analysis/final/report artifacts>
mkdir -p results/native_recheck_runs/superseded/\
  validation-pre-report-audit-20260905T174058Z/analysis
cp <five generated analysis artifacts> <superseded analysis directory>
mv results/native_recheck_runs/validation/\
  native-recheck-validation-20260905T165354658861Z-227a918c57 \
  results/native_recheck_runs/superseded/\
  validation-pre-report-audit-20260905T174058Z/
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_native_final_report.py
.venv/bin/python -m compileall -q src scripts tests
git diff --check
make test
```

Outcome: the first validation remains valid for its frozen source but is
superseded for final publication after the reporting-source correction. The
next resumable step is to commit this correction phase, rerun `make setup` and
`make native-test` for the new identities, and then start a completely new
`make native-validate`. The new run must not reuse or pool first-run timing.

## Issue #3 post-audit source-frozen correctness gate

- The report-audit corrections were committed as `601cb9c` with message
  `docs: harden native result interpretation (refs #1, #3)`. A final independent
  read-only diff audit found no remaining P0/P1 issue and verified that
  secondary axes remain in the pre-registered gate set while the report keeps
  their maxima separate from the main-family confirmatory summary.
- Before correctness execution, the stale top-level generated report/analysis
  files were moved, not deleted, beneath the ignored
  `results/native_recheck_runs/superseded/validation-pre-report-audit-20260905T174058Z/top-level-stale/`
  directory. `git status --short --branch` was then clean.
- `make setup` completed at commit
  `601cb9c45eeeb5c1bf474eef30ff139d265aaee5`. Its index probes again emitted
  PyPI DNS retry warnings, but all exact pinned packages were already present;
  the editable wheel built and installed, the environment/Faiss HNSW smoke
  passed, and all five thread variables plus Faiss were fixed to one. The
  rebuilt native shared object was still byte-identical at
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.
- Formal `make native-test` collected 246 tests and selected 245. It completed
  `245 passed, 1 deselected, 20 warnings in 67.90s`, including exactly 10,000
  fixed-seed native cases, required tie/beta/underfill/duplicate tests, and the
  existing lifecycle regressions. There were zero failures, errors, or skips.
- The regenerated correctness JSON is `status=passed` and binds implementation
  hash `d6a9b505d0a3fe75a93ee1810ec69d69c7ea2e8607ecb7181d84b23388593003`,
  test-tree hash
  `d1e7e181d1684afa60ff156283fcd80f7949c3c01f7b07557b1a29ca07e8ab59`,
  and the unchanged native binary above. Its SHA-256 is
  `5bddea79c03b763a8bf498b367e3a81b8b7a001b331d4601dd5a1c99c64601c4`;
  the JUnit SHA-256 is
  `247f0f4cbc691b6cd40e22efb9423eaf0eba7dc8c836a56e24c9c654e0dd5cee`.

Exact commands:

```bash
make setup
make native-test
.venv/bin/python -m json.tool \
  results/native_recheck_evidence/native_correctness.json
sha256sum results/native_recheck_evidence/native_correctness.json \
  results/native_recheck_evidence/native_correctness_junit.xml
```

Outcome: the post-audit source/test/native correctness prerequisite passed.
The next resumable command is `make native-validate`. No executable, test,
config, policy, holdout, or correctness-evidence file may change during that
run.

## Issue #3 post-audit formal validation and stopping decision

- `make native-validate` started from a clean Git state at commit `2a0aff1`
  and created only the new run
  `native-recheck-validation-20260905T175114361812Z-227a918c57`. It started at
  `2026-09-05T17:51:14.400963Z`, completed at
  `2026-09-05T18:25:28.938645Z`, and did not reuse or pool any superseded
  timing. The formal runner completed 84/84 blocks, 84 raw shards, 42,064 rows,
  13 experiments, and 2,056 unique queries in about 34 minutes 15 seconds.
  Raw JSONL is uncompressed and totals 288,072,929 bytes.
- The run binds implementation hash
  `d6a9b505d0a3fe75a93ee1810ec69d69c7ea2e8607ecb7181d84b23388593003`,
  config hash
  `227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925`,
  correctness SHA-256
  `5bddea79c03b763a8bf498b367e3a81b8b7a001b331d4601dd5a1c99c64601c4`,
  test-tree hash
  `d1e7e181d1684afa60ff156283fcd80f7949c3c01f7b07557b1a29ca07e8ab59`,
  and native binary
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.
  The process segment recorded an empty Git porcelain status and all requested
  thread limits equal to one.
- Evidence identities are: run manifest
  `d05f6a8ab17c729a5bf25c3f68414c871d333534450b43d2549ffa073854ef65`,
  checkpoint
  `bcae6c35e619899aaf7f1ff1149fea820f5552b9cb4da43c9d7e4bae3513928a`,
  completion
  `da0daff64d90878b5ef24d48dde6c1d0e6adfbf7b87c62946c5aea135a19b35d`,
  raw inventory
  `10cf04e63b8a699253d84462463b6e213a69cbd590505625732dca1ab0c563de`,
  effective-config file
  `f52e35f8640d1814f94d84bc47c7da2f75cbddcf7d6ad22959f79a7f05eed4e6`,
  and build manifest
  `371090ce6baf3b2242e65048b3f736954d71baae4699cef0676e77167bb9e238`.
  Completion binds 54 ancillary artifacts as well as the frozen validation
  config, final policy, and holdout registration.
- There are 27,672 actual `pybind11_cpp17` F/N/P/ablation rows. Every one
  records exactly two native calls, all 27,672 stored call indices are unique,
  and the process manifest ends at 58,737 native calls including untimed
  setup/audit work. The formal F/N/P integrity aggregate checked 25,872 rows
  with zero backend, fallback, contract, same-C, or beta-chain failure.
  Thirteen LB audit files cover 325 query/group sets and 35,125 decisions:
  25,248 scans and 9,877 strict skips, with lower-bound soundness and strict
  predicate failure counts both zero.
- The deterministic summary, gate, and 96-row representative sample have
  SHA-256 values
  `4511edaaa2f056623cbda7c13f4ceea1ecde6a8908467c6383ba3c9eaff20415`,
  `8c109ac4a844d60493d337e2bcdb6616fd0ba416d447eb4d9ad87d89c766ce45`,
  and `6c01fd2b9a5fed9424e4d96ebaaad81b16a7a0543bfab2a56e3b19c0b2f15c81`.
  A separate analyzer invocation wrote them under
  `/tmp/native-final-analysis-audit-eGUfJg`; all three were byte-for-byte equal
  to the tracked candidates.
- Validation gate status is `NOT_PASSED`, selected candidate is `null`, and the
  single reason is `no_non_degenerate_candidate_beats_both_F_and_A`. Among 16
  gate-eligible SIFT candidates, F/P passed the paired-CI rule in 16/16 while
  A/P passed in 0/16. Both GIST anchor candidates passed neither comparison.
  Same-C quality mismatches were zero; empirical A agreement is still not
  described as the same mathematical guarantee.
- For the pre-specified `sift-initial` family, F/P API-wall geomeans ranged
  1.097--1.131 and all four CI lower bounds exceeded one; A/P ranged
  0.584--0.602 and all four CI upper bounds were below one. At beta=0, P
  API-wall p50/p95/p99 was 8.542/9.355/9.782 ms, F/P was 1.096682
  `[1.085929, 1.108052]`, A/P was 0.583748
  `[0.577843, 0.589578]`, N/P was 1.108, median skipped groups were 20/128,
  and median Delta vectors scanned were 7,892.5/10,000. Observed beta loss was
  zero in every measured positive-beta condition; the report does not call it
  quality-for-speed tradeoff.
- The secondary-sweep maxima are retained as descriptive values, not promoted
  over the main family: F/P 1.163725 `[1.149816, 1.177442]` at SIFT `k=1`,
  beta factor .05; A/P 0.838747 `[0.836900, 0.840617]` at SIFT Delta=1,000,
  beta factor .05. Those conditions remain in the pre-registered gate set.
- Main-family RQ3 diagnostics are N/P 1.108--1.143, operating-point median
  skips 20--30/128, scan exact-min-minus-LB median/p95 325.541/381.721 L2,
  skip gap 304.316/360.617 L2, group build 2,752.491 ms, and finite F-side
  break-even 2,603.7--3,528.9 queries for all four points. Cross-dataset L2
  gaps are no longer pooled. Measured ablations are: rescan-to-heap 1.514x,
  all-exact-to-adaptive F 2.729x, Base rebuild-to-cache 13.860x, old-to-native
  P 13.332x (nonpaired development comparison), and A-reference-to-A 0.643x
  with the reference's non-certified ordering kept explicit.
- `make native-evaluate` consumed the negative gate and wrote only
  `final_decision.json`, SHA-256
  `b0cf8944a2d3c9f2dd4650d6424f426a5f4e1cab1163eb99818f56693a679180`.
  It records `NOT_SUPPORTED_IN_TESTED_REGIME`, `NO_GO`, and
  `final_status=NOT_RUN_GATE_NOT_PASSED`. It created no final lock/config,
  final summary/gate, pre-HNSW authorization, fresh-holdout load, large-final
  timing, or final-HNSW directory. `run_native_final` and the final analyzer
  both printed the expected not-started messages rather than fabricating a
  final result.
- `make native-report` reverified current implementation, correctness, raw,
  ancillary, completion, policy, and holdout identities and generated
  `reports/NATIVE_KERNEL_RECHECK_ja.md` (274 lines, 30,114 bytes), SHA-256
  `95488e87f14f51cac94e2f7c0451d3491937b49e8b7a3fbb5d774fffb32466a7`.
  It leads with the validation stopping-rule nature of the verdict and reports
  RQ1/RQ2/RQ3, all operating points, quality, geometry, build/memory,
  break-even, ablations, evidence identity, and third-party limitations.
- No optional post-validation performance-tuning round was run (0 of the
  permitted maximum two). The compiled/search implementation, centers, group
  counts/membership, beta policy, Base HNSW parameters, validation queries, and
  fresh holdout were not changed after timing. The only pre-rerun change was
  the already documented report interpretation correction, which forced a new
  full source identity and validation rather than reusing old measurements.

Exact commands:

```bash
make native-validate
run_dir=results/native_recheck_runs/validation/\
native-recheck-validation-20260905T175114361812Z-227a918c57
sha256sum "$run_dir/run_manifest.json" "$run_dir/checkpoint.json" \
  "$run_dir/COMPLETED.json" \
  results/native_recheck_evidence/validation_summary.json \
  results/native_recheck_evidence/validation_gate.json \
  results/native_recheck_evidence/validation_representative_raw.json
make native-evaluate
make native-report
for path in results/native_recheck_evidence/final_lock.json \
  results/native_recheck_evidence/final_config.locked.json \
  results/native_recheck_evidence/final_summary.json \
  results/native_recheck_evidence/final_gate.json \
  results/native_recheck_evidence/pre_hnsw_authorization.json \
  results/native_recheck_runs/final_hnsw; do
  test ! -e "$path"
done
audit_dir=$(mktemp -d /tmp/native-final-analysis-audit-XXXXXX)
.venv/bin/python scripts/analyze_native_recheck.py \
  --input results/native_recheck_runs/validation \
  --output "$audit_dir/validation_summary.json" \
  --gate-output "$audit_dir/validation_gate.json" \
  --bootstrap-resamples 5000 --bootstrap-seed 6202052 \
  --representative-raw-output "$audit_dir/validation_representative_raw.json"
cmp "$audit_dir/validation_summary.json" \
  results/native_recheck_evidence/validation_summary.json
cmp "$audit_dir/validation_gate.json" \
  results/native_recheck_evidence/validation_gate.json
cmp "$audit_dir/validation_representative_raw.json" \
  results/native_recheck_evidence/validation_representative_raw.json
```

Outcome: the full post-audit validation and its negative stopping decision are
complete for implementation identity
`d6a9b505d0a3fe75a93ee1810ec69d69c7ea2e8607ecb7181d84b23388593003`.
The independent publication audit below supersedes the previously stated next
step.

## Issue #3 exhaustive evidence and publication audit

- A second independent, read-only integrity audit reconstructed the current
  formal validation directly from raw. It verified 84/84 shards, 42,064 rows,
  288,072,929 raw bytes, 54 ancillary artifacts, and every recorded path,
  row-count, byte-count, and SHA-256. Native call accounting reconciled exactly
  as 55,344 timed + 1,928 beta-calibration + 1,140 warmup + 325 LB-audit calls
  = 58,737. All 27,672 recorded native call indices and all 1,928 calibration
  indices were unique and disjoint.
- The same audit independently recomputed all 35,125 LB decisions for 325
  audit queries (25,248 scan, 9,877 skip) from the stored rational bounds. The
  strict `LB_lower > tau_upper - beta` action, lower-bound soundness, and sqrt
  bracketing all matched. It also rechecked every same-C binding, beta=0 order,
  positive-beta chain, native backend/binary, fallback flag, and A quality
  result with zero failure. All 168 comparisons and 2,688 point/CI/digest
  fields were reconstructed from raw; the official summary, gate, and
  representative export regenerated byte-for-byte. Thus the numerical gate
  and negative verdict have no P0/P1 integrity defect.
- A separate scientific-publication audit found that the conclusion remains
  correct, but the generated report did not numerically expose several items
  explicitly required by Issue #3: the Delta-influence subset and its
  uncertainty; micro/composed/API-wall results side by side; the single-session
  limitation and denominators; warm/cold packed-view construction and memory;
  the positive-beta pruning/latency contribution; and the exact 25-query-per-
  condition geometry-audit population. Merely hand-editing the Markdown would
  violate the clean raw-to-report regeneration requirement, so the generator
  and its tests must be corrected.
- The audit also required clearer scope for the group-only break-even numerator,
  explicit fallback and optional-tuning counts, and removal of the malformed
  headerless duplicate of the main result rows. It confirmed that the report
  contains only one CPU/immutable-snapshot limitation bullet; an apparent
  duplicate came from overlapping inspection ranges and is not a defect.
- Because `scripts/report_native_recheck.py` is deliberately included in the
  implementation-tree identity, a generator correction invalidates the
  current run as publication evidence even though its measured kernel rows are
  internally valid. To avoid selecting or mixing results, the entire run and
  all current analysis outputs were preserved under
  `results/native_recheck_runs/superseded/validation-pre-complete-report-audit-20260905T175114Z/`.
  The formal run directory was moved intact; copies of summary, gate,
  representative raw, final decision, correctness JSON/JUnit, and report are
  under `analysis/`, and the former top-level generated outputs are under
  `top-level-stale/`. Nothing was deleted or relabelled as a failed experiment.

Exact audit/archive commands:

```bash
# Independent auditors recomputed raw/checksum/call/LB/comparison/gate state
# using read-only Python against the one completed validation input.
mkdir -p results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/analysis
cp -a results/native_recheck_evidence/{validation_summary.json,\
validation_gate.json,validation_representative_raw.json,final_decision.json,\
native_correctness.json,native_correctness_junit.xml} \
  results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/analysis/
cp -a reports/NATIVE_KERNEL_RECHECK_ja.md \
  results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/analysis/
mv results/native_recheck_runs/validation/\
native-recheck-validation-20260905T175114361812Z-227a918c57 \
  results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/
mkdir -p results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/top-level-stale
mv reports/NATIVE_KERNEL_RECHECK_ja.md \
  results/native_recheck_evidence/{validation_summary.json,validation_gate.json,\
validation_representative_raw.json,final_decision.json} \
  results/native_recheck_runs/superseded/\
  validation-pre-complete-report-audit-20260905T175114Z/top-level-stale/
```

Outcome: the audited second validation remains immutable and recoverable but is
superseded solely because its hashed report generator is incomplete. The next
resumable phase is to complete all report fixes and focused/full tests, record
new implementation/test identities in a phase commit, then run `make setup`,
`make native-test`, and a new non-pooled `make native-validate`. No performance
tuning, configuration change, holdout access, or result-dependent method
selection is authorized.

## Issue #3 complete-report and stopping-rule hardening

- The Japanese report generator now derives and prints every gate-eligible
  SIFT Delta-influence subset with its query count, F/P and A/P paired
  geomean/CI, plus the two non-independent GIST anchors separately. It also
  reports the dynamic subset-CI counts rather than requiring a reader to infer
  them from the table.
- For the pre-specified main condition it now displays micro, composed E2E, and
  API wall p50/p95/p99 and paired CIs side by side. Denominators and the fact
  that validation contains one process session are explicit, so the
  query-bootstrap CI is not presented as measuring process/session variance.
  A component table folds repetition within session-query before reporting
  Base preparation, native preparation, kernel/Delta search, LB, ordering,
  group/raw scan, adaptive/merge, Receipt, and non-overlapping residual p50s.
- The generator now reports cold group build, cold/warm packed-view handling,
  Base visible-map cache measurements, group/packed retained bytes, and a
  clearly non-attributable process RSS delta. Break-even separates the primary
  group-only numerator (packed view treated as common to F/P) from a
  group-plus-packed sensitivity analysis; both include paired denominators.
- Geometry output now verifies and prints 25 audit validation/calibration
  queries per condition, 325 condition-query sets and 35,125 decisions for the
  archived audit input, explicitly distinct from the 200-query timing test
  partition. Every result/quality/ablation/cache table now carries its relevant
  denominator. The malformed headerless repetition of main result rows was
  removed.
- Positive-beta reporting now derives skip/vector and p50/p95/p99 changes from
  beta=0, checks every positive-P raw `observed <= certified <= requested`
  chain, and reports nonzero observed-gap count across all positive-P raw. Its
  max L2 chain is restricted to `sift-initial` so coordinate-dependent values
  are not pooled across SIFT/GIST/synthetic data. It explicitly distinguishes
  reduced scanning with zero observed quality gap from a quality-for-speed
  trade-off and leaves the A/gate result unchanged.
- A further independent acceptance audit found two fail-closed gaps. Final
  authorization previously checked the external correctness prerequisite and
  native-backend aggregate but not the raw validation correctness aggregate;
  a broken row-level contract could therefore be misclassified as a negative
  performance result. `_validate_source_identity` now requires a well-formed,
  passing, nonempty aggregate with zero failures/examples. Separately, report
  generation now requires the negative/final decision's `correctness_sha256`
  to match the current correctness file, preventing a later native-test from
  silently pairing stale decision evidence with a replacement prerequisite.
- Correctness JUnit does not preserve pytest's deselected count. The generator
  does not invent it: the report says the value is unavailable in the bound
  JSON/JUnit, while exact observed console counts remain in this state file.
  The public README now shows alternate correctness/JUnit output paths for
  post-validation acceptance so formal evidence is not overwritten.
- Focused report and authorization tests completed as `37 passed in 4.13s`.
  After the final cross-dataset beta-max guard, the same focused suites again
  completed `37 passed in 4.13s`. `py_compile`, `compileall`, and
  `git diff --check` passed. The final lightweight command collected 256 tests
  and reported `254 passed, 2 deselected, 19 warnings in 14.05s`; warnings are
  the retained optional-PyYAML notice.
- The new implementation-tree hash is
  `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f`
  and the test-tree hash is
  `9e67be783618aa1ee1d9a53fcb93d2c0fda727fc29f85fc748cc98d27efbb325`.
  Native source and shared object remain byte-identical at
  `bf32578fc0e025531c2ab961aa164bd62c7859b93b96a26328176089f8fa6215`
  and `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`.

Exact verification commands:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  tests/test_native_final_report.py tests/test_native_final.py
.venv/bin/python -m compileall -q src scripts tests
make test
git diff --check
.venv/bin/python -c '<print implementation/test/native identities>'
# Read-only helper invocation against the second archived run verified:
# SIFT subsets=16, GIST anchors=2, timing n=200/session=1,
# geometry=325 sets/35,125 decisions, fallback=0/25,872 (P 0/13,536).
```

Outcome: the complete publication/report and terminal-decision logic are ready
for a source phase commit. The next resumable commands are `make setup` and
`make native-test`; correctness must bind the hashes above before a new empty-
namespace `make native-validate` begins. Any additional executable or test edit
requires repeating this freeze.

## Issue #3 final-report source-frozen native correctness gate

- The complete-report and stopping-rule phase was committed as `b01c883` with
  message `fix: complete native evidence reporting (refs #1, #3)`. The tracked
  tree was clean at full commit
  `b01c883e92998ea4d8f69b7756e61e6131265177`, with implementation/test hashes
  `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f` /
  `9e67be783618aa1ee1d9a53fcb93d2c0fda727fc29f85fc748cc98d27efbb325`.
- `make setup` completed at that commit. Sandbox DNS probes for pinned
  pip/setuptools/wheel again emitted five retry warnings per package, but every
  pinned dependency was already installed. The editable native wheel rebuilt
  and installed successfully; environment/Faiss HNSW smoke passed, Git
  porcelain was empty, and all five thread variables plus Faiss were 1. The
  rebuilt native shared object remained byte-identical at
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`,
  matched source
  `bf32578fc0e025531c2ab961aa164bd62c7859b93b96a26328176089f8fa6215`,
  and reported enforced rounding mode.
- Formal `make native-test` collected 256 tests and selected 255 under the
  declared `not evaluation or native_fixed_seed` expression. It completed
  `255 passed, 1 deselected, 20 warnings in 67.51s`, including exactly 10,000
  fixed-seed native cases and all new report/final fail-closed regressions.
  Failures, errors, and skips were zero; warnings remain the optional-PyYAML
  and xunit2 `record_property` notices.
- The regenerated correctness JSON is `status=passed`, has no failure reason,
  and binds commit `b01c883e92998ea4d8f69b7756e61e6131265177`, the implementation
  and test hashes above, and the native shared object. Its SHA-256 is
  `dd2cd9b1e50b4202e2d39e39acc36b48a317fdc8908acff6630bc3a56c0e22d5`;
  the JUnit SHA-256 is
  `e066c863c950b62824b32fab8c80d3783cc86015d196abf8951023b2c06c4dba`.

Exact commands:

```bash
make setup
.venv/bin/python -c '<assert native source/SO/rounding identity>'
make native-test
jq '{status,fixed_seed_cases,pytest_testcases,pytest_failures,pytest_errors,\
pytest_skipped,implementation_tree_sha256,test_tree_sha256,\
native_shared_object_sha256,git_commit,failure_reasons}' \
  results/native_recheck_evidence/native_correctness.json
sha256sum results/native_recheck_evidence/native_correctness.json \
  results/native_recheck_evidence/native_correctness_junit.xml
```

Outcome: source-frozen correctness passed with actual native execution. The
next resumable command is `make native-validate`; the validation namespace is
empty and superseded runs remain outside it. No executable, test, config,
policy, holdout, or correctness file may change during the formal run.

## Issue #3 final formal validation and terminal decision

- The correctness evidence phase was committed as `2000b6b` before timing.
  The working tree and the designated validation namespace were empty at run
  start. Formal `make native-validate` ran from
  `2026-09-05T19:10:58.068740+00:00` through
  `2026-09-05T19:45:08.442398+00:00` with all five configured thread variables
  and Faiss fixed to one. Run
  `native-recheck-validation-20260905T191058023915Z-227a918c57` completed
  84/84 blocks, recorded 58,737 native calls and no run failures, and produced
  84 uncompressed raw JSONL shards containing 42,064 rows and 288,076,713
  bytes. The run binds config hash
  `227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925`,
  implementation-tree hash
  `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f`,
  native shared-object hash
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`,
  and Git commit `2000b6bcb9d4359abd193338d975a5fa703f5f77`.
- The analyzer accepted exactly this one completed validation run, 13
  experiments, 168 comparisons, one process session, and 2,056 unique queries.
  Row-level correctness and native-backend checks each passed over 25,872 rows
  with zero failures. The canonical raw inventory SHA-256 is
  `d31218eb63a96136c1e553430f5d10f5ec8b8f7b5aa8a07e3e45762ed2049240`;
  `COMPLETED.json` SHA-256 is
  `10665158cdad322c0cf37926d6e21f8225e61ca705972ddfda97c4491bce5366`.
- The preregistered validation gate is `NOT_PASSED`, with durable reason
  `no_non_degenerate_candidate_beats_both_F_and_A`. For the designated main
  `sift-initial` family, F/P API-wall geometric means ranged 1.097--1.131 and
  all four paired-bootstrap lower bounds exceeded 1, while A/P ranged
  0.579--0.597 and all four confidence intervals remained below 1. Across all
  16 gate-eligible SIFT candidates, F/P passed 16/16 and A/P passed 0/16; the
  two non-independent GIST anchor candidates passed neither comparison. Same-C
  quality mismatches were zero. Thus native pruning improved on certified F in
  the tested SIFT grid but did not beat optimized A, so no candidate satisfied
  both required comparators.
- `make native-evaluate` emitted the terminal stopping decision
  `NOT_SUPPORTED_IN_TESTED_REGIME` and engineering decision `NO_GO`, with
  `final_status=NOT_RUN_GATE_NOT_PASSED`, `selected_candidate=null`, and all
  of `lock_created`, `large_final_started`, and `fresh_sift_holdout_loaded`
  false. As required by the fail-closed policy, no final lock/config,
  final summary/gate, pre-HNSW authorization, or HNSW directory exists. The
  untouched SIFT `[1200,2200)` fresh holdout was not loaded.
- `make native-report` generated
  `reports/NATIVE_KERNEL_RECHECK_ja.md`. It reports RQ1--RQ3; main, sweep,
  Delta-influence, and three-scope timing; component ablations; the 325-set /
  35,125-decision geometry audit; build/memory and group-only versus
  group-plus-packed break-even; beta-chain/fallback evidence; and limitations.
  It explicitly states that the one-session bootstrap CI does not estimate
  process/session variation and that the negative stopping result is not a
  fresh-final estimate.
- A second analyzer invocation wrote summary/gate/representative outputs to
  `/tmp/issue3-analyze.G8ZyMC` from the saved raw with the same 5,000-resample,
  seed-6202052 settings. All three files were byte-identical to the publication
  outputs. A first diagnostic hash command incorrectly requested lowercase
  `completion.json` and reported that it did not exist; the actual protocol
  marker is uppercase `COMPLETED.json`, which was then located, parsed, and
  hashed. No result or tracked evidence was changed by the diagnostic typo.
- A separate read-only integrity audit reported no findings at any priority. It
  independently checked every path, row count, byte count, SHA-256, block
  boundary, method/repetition coverage, and frozen-C identity in all 84 raw
  shards; all 54 ancillary files (30,000,445 bytes); and the completion,
  manifest, checkpoint, and build receipts. It reconciled 58,737 calls as
  55,344 timed + 1,928 calibration + 1,140 warmup + 325 LB calls. Across
  27,672 compiled-native rows it found no backend, fallback, contract, same-C,
  API, or Receipt violation. All 20,304 beta-zero rows matched exact ordering;
  all 7,368 positive-beta rows satisfied
  `0 <= observed_lower <= observed_upper <= certified <= requested`.
  Recalculation of 35,125 LB decisions found 25,248 scans, 9,877 strict skips,
  zero equality/predicate/action mismatch, and 35,125/35,125 sound bounds.
  Independent regeneration of all 28 operating points, 168 comparisons, and
  96 representative rows matched the publication artifacts exactly.
- A separate final scientific audit reported no P0/P1 findings. It cross-read
  Issue #3, raw/summary/gate/decision/report and confirmed the ordinary-L2
  versus Faiss-squared-L2 distinction, strict
  `LB_lower > tau_upper - beta` rule, same frozen C, preregistration and
  untouched holdout, RQ1--RQ3 negative-result interpretation, one-session CI
  limitation, build/memory/break-even scope, and hash provenance. It also
  regenerated the report to `/tmp` and obtained the identical
  `9807d1da...` SHA-256. A final lightweight `make test` then reported
  `254 passed, 2 deselected, 19 warnings in 14.03s`; no executable or test file
  had changed since the source-frozen native gate.

Current publication SHA-256 values:

```text
dd2cd9b1e50b4202e2d39e39acc36b48a317fdc8908acff6630bc3a56c0e22d5  results/native_recheck_evidence/native_correctness.json
e066c863c950b62824b32fab8c80d3783cc86015d196abf8951023b2c06c4dba  results/native_recheck_evidence/native_correctness_junit.xml
1735992cd55f13bbe5281e354bba8af8bccaf10f073e04f7941e6569e788f41c  results/native_recheck_evidence/validation_summary.json
17e04e5bfbd592dcfee4b43604fdd6bb7860aafa6e29501dfaf2c191a02f5b92  results/native_recheck_evidence/validation_gate.json
bd831a851d52564cad074073076c3adcd4c72da2412a9cc568558989a5a993a7  results/native_recheck_evidence/validation_representative_raw.json
7abd3aee91670bfaf0dffcf344597536812072b78151828aa28678af04c91029  results/native_recheck_evidence/final_decision.json
9807d1da5629855edb772db33f24d8785434225b1ca8ab38ca7729072827b235  reports/NATIVE_KERNEL_RECHECK_ja.md
```

Exact phase commands:

```bash
make native-validate
make native-evaluate
make native-report
.venv/bin/python -m pytest -m "not evaluation"
.venv/bin/python scripts/analyze_native_recheck.py \
  --input results/native_recheck_runs/validation \
  --output /tmp/issue3-analyze.G8ZyMC/summary.json \
  --gate-output /tmp/issue3-analyze.G8ZyMC/gate.json \
  --bootstrap-resamples 5000 --bootstrap-seed 6202052 \
  --representative-raw-output /tmp/issue3-analyze.G8ZyMC/representative.json
cmp results/native_recheck_evidence/validation_summary.json \
  /tmp/issue3-analyze.G8ZyMC/summary.json
cmp results/native_recheck_evidence/validation_gate.json \
  /tmp/issue3-analyze.G8ZyMC/gate.json
cmp results/native_recheck_evidence/validation_representative_raw.json \
  /tmp/issue3-analyze.G8ZyMC/representative.json
```

Outcome: the formal validation and preregistered negative stopping decision are
complete. The next resumable phase is to commit these immutable small evidence
files and report, then reproduce setup, native correctness, offline smoke, and
report generation from a clean/new environment without overwriting the formal
correctness or validation evidence.

## Issue #3 clean/new-environment acceptance

- The formal publication artifacts were committed as `eaed074` with message
  `research: record final native validation NO-GO (refs #1, #3)`. The tree was
  clean at full commit `eaed0740db9ef18e9ab61aa094fc1b206a002440`.
  Existing `.venv`, `build`, native shared object, pytest/Hypothesis caches,
  and Python bytecode caches were moved intact to ignored
  `.cache/issue3-clean-acceptance-eaed074/preexisting/`; none was deleted.
  `PYTHONPYCACHEPREFIX` pointed at a fresh ignored cache during acceptance.
- The first sandboxed fresh `make setup` created the new `.venv` but failed
  while resolving pinned `pip==26.2.1`: five DNS retries ended with no matching
  distribution visible and Make exited nonzero. This is retained as an
  external-network failure, not called a passed setup. The identical command
  was rerun with network approval and succeeded: all pinned dependencies were
  installed, the editable C++ extension was rebuilt, and the environment check
  recorded clean commit `eaed074`, CPython 3.10.12, NumPy 2.2.6, Faiss 1.15.0,
  Faiss squared-L2 HNSW smoke success, and all five thread variables plus Faiss
  threads equal to 1.
- The fresh native shared object reproduced SHA-256
  `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`
  exactly. `native_build_info()` also reproduced source/compiled-source hash
  `bf32578fc0e025531c2ab961aa164bd62c7859b93b96a26328176089f8fa6215`,
  the four required compile flags, and enforced rounding mode. An initial
  diagnostic imported nonexistent `searchability.native_kernel` and raised
  `ModuleNotFoundError`; the correct module is `searchability.native`, which
  produced the values above. No test or experiment was affected.
- Fresh `make native-test` wrote to alternate ignored paths, leaving the formal
  correctness prerequisite unchanged. It collected 256 tests, selected 255,
  and reported `255 passed, 1 deselected, 20 warnings in 69.50s`, including
  exactly 10,000 fixed-seed native cases, with failures/errors/skips all zero.
  It bound commit `eaed074`, the formal implementation/test hashes
  `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f` /
  `9e67be783618aa1ee1d9a53fcb93d2c0fda727fc29f85fc748cc98d27efbb325`,
  and the identical native shared object. The
  acceptance correctness/JUnit SHA-256 values are
  `a278f049bbe8a5ad110facd08bf82ad3eef1427fee22a01c6be3c126e2411193` /
  `5b0ca917f4fc3f1782ee3a170c41865cc9f78cdf3ee3682c4aab086777eb9613`.
- The preexisting smoke directory was moved intact to ignored
  `results/native_recheck_runs/superseded/smoke-pre-final-clean-eaed074/`.
  Fresh `make native-smoke` then completed run
  `native-recheck-offline-smoke-20260905T200116715380Z-f4ee103dc9`: 8/8
  blocks, 8 raw shards, 224 rows, 1,209,598 bytes, 32 unique queries, 320
  native calls, and no run failure. Row-level correctness/native-backend checks
  passed 128/128 rows. Its expected calibration gate was `NOT_PASSED` and
  created no lock. Completion/manifest/summary/gate SHA-256 values are
  `46760f1ce842a4b9e7c3f825ba7d6a9547ef98e5e8876ddf3e226c8f325c00cf`,
  `541f1b8553a1630d0eff0abe7c76cff3e0313d210817b938252f4e8bf8956f78`,
  `27399a10fc811cd287601f95be81bf4b9ff8edf20c55474df77c78e3973e88c3`,
  and `2f7cfe7b86d6cbed3d26361f83d430755985557db9c2b43cbdf678e13ac81698`,
  respectively; the full files remain in the ignored smoke directory.
- From the new environment, `make native-report` revalidated the formal saved
  raw/ancillary receipts and reproduced the committed Japanese report exactly
  at SHA-256
  `9807d1da5629855edb772db33f24d8785434225b1ca8ab38ca7729072827b235`.
  `git diff --exit-code` confirmed no change to the report, correctness,
  summary, gate, representative sample, or terminal decision; their seven
  hashes all remained identical to the formal publication set.
- A direct post-acceptance environment capture was first invoked without the
  Makefile's exported thread variables; it truthfully recorded five null
  values and was retained as
  `.cache/issue3-clean-acceptance-eaed074/environment-without-make-env.json`
  (SHA-256
  `5e88796f2c41effe79fe8f091751d838b08d90bd525f4522d1ea0c535b6b0f96`).
  Repeating it with the five explicit one-thread variables produced
  `environment.json` (SHA-256
  `be69ee4abec32c989bffa782856737c101d07b274a937b9d8b4571b182aa3484`)
  with a clean tree and all controls equal to 1. This diagnostic distinction
  does not alter the already captured one-thread setup, test, smoke, or formal
  run.
- A final independent read-only acceptance audit reported no P0/P1 findings.
  It confirmed that old and fresh environments/shared objects use distinct
  inodes while the fresh binary bytes match formal evidence; independently
  parsed the 255-test/10,000-case alternate receipts; and reconciled fresh
  smoke calls as 256 timed + 16 calibration + 32 warmup + 16 LB = 320. Its
  smoke LB audit found 48 decisions (32 scans, 16 strict skips), no violation,
  exactly one completed smoke-role run, and no formal-validation mixing. It
  also compared each of the seven formal publication files individually to
  commit `eaed074` and found byte-for-byte identity.

Exact acceptance commands:

```bash
# After moving the preexisting environment/build/cache files intact:
PYTHONPYCACHEPREFIX="$PWD/.cache/issue3-clean-acceptance-eaed074/fresh-pycache" \
  make setup
# The first sandboxed invocation failed DNS; the identical approved invocation passed.
.venv/bin/python -c \
  'from searchability.native import native_build_info; print(native_build_info())'
PYTHONPYCACHEPREFIX="$PWD/.cache/issue3-clean-acceptance-eaed074/fresh-pycache" \
  make \
    NATIVE_CORRECTNESS=.cache/issue3-clean-acceptance-eaed074/native-test/native_correctness.json \
    NATIVE_JUNIT=.cache/issue3-clean-acceptance-eaed074/native-test/native_correctness_junit.xml \
    native-test
mv results/native_recheck_runs/smoke-v3 \
  results/native_recheck_runs/superseded/smoke-pre-final-clean-eaed074
PYTHONPYCACHEPREFIX="$PWD/.cache/issue3-clean-acceptance-eaed074/fresh-pycache" \
  make native-smoke
PYTHONPYCACHEPREFIX="$PWD/.cache/issue3-clean-acceptance-eaed074/fresh-pycache" \
  make native-report
git diff --exit-code -- reports/NATIVE_KERNEL_RECHECK_ja.md \
  results/native_recheck_evidence
```

Outcome: the clean/new-environment acceptance chain passed while keeping both
the formal correctness/validation evidence and every earlier run separate and
recoverable. The next resumable phase is to commit this state-only record,
verify a clean branch, publish `codex/issue-3-native`, open the stacked PR onto
`codex/issue-1`, and wait for CI without merging.

## Scope and source status

- GitHub Issue #1 was read in full through the GitHub REST API. It is open,
  created/updated at `2026-09-05T04:19:23Z`, and has zero comments (the comments
  endpoint returned `[]`).
- The full body was fetched again at this resume point both through the REST
  endpoint and the connected GitHub application. The repository was empty at
  that point. After final evidence commits, the initial scaffold was published
  as `main`, the full work as `codex/issue-1`, and non-merged PR
  `https://github.com/wasanemon/Searchability-Obligations/pull/2` was opened.
- The local and remote repository had no commits or ordinary files at start.
  There were no user changes to preserve.
- Work is proceeding in the required order: contract/oracle, insert-only kernel,
  tests and experiments, durable lifecycle reference, then synthesis.

## Environment observations

- OS: Ubuntu 22.04.5 LTS, Linux 5.15.0-186, x86_64, glibc 2.35.
- CPU: 2 x Intel Xeon Gold 5416S, 32 physical / 64 logical CPUs. Benchmarks use
  one thread; this is a shared host and no claim of exclusive CPU access is made.
- RAM observed: 29 GiB total, approximately 26 GiB available; swap 4 GiB.
- Workspace free space observed: approximately 236 GiB.
- GPU is unavailable (`nvidia-smi` could not communicate with a driver); the
  issue requires the CPU implementation anyway.
- CPython 3.10.12. A fresh `.venv` was created.
- Verified combination: `faiss-cpu==1.15.0`, `numpy==2.2.6`.
- Faiss smoke: `IndexHNSWFlat(2,16)` accepted 3 vectors and returned expected
  squared-L2 values; `faiss.omp_get_max_threads()` was 1. Compile options report
  AVX2/AVX512/AVX512_SPR.

## Commands already executed

```bash
git status --short --branch
git remote -v
python3 -m venv .venv
.venv/bin/python -m pip index versions faiss-cpu
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install faiss-cpu==1.15.0 numpy==2.2.6 \
  pytest==9.1.1 hypothesis==6.167.1 matplotlib==3.10.8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python <Faiss add/search smoke>
.venv/bin/python -m pip freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_numerics.py tests/test_search.py \
  tests/test_randomized.py -m 'not evaluation' -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_randomized.py::test_fixed_seed_ten_thousand_random_cases -q \
  --junitxml=results/test_evidence/randomized_10000.xml
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only sift
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only gist
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_store.py tests/test_crash.py
.venv/bin/python scripts/run_experiment.py --config configs/smoke.json
.venv/bin/python scripts/analyze_results.py --input results/smoke
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_store.py tests/test_crash.py
make test
make test-full
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_randomized.py::test_fixed_seed_ten_thousand_random_cases -q \
  --junitxml=results/test_evidence/randomized_10000_final.xml
make smoke
make data
git add README.md reports/REPORT_ja.md
git commit -m "docs: scaffold Issue #1 research report"
curl -L --fail-with-body \
  https://api.github.com/repos/wasanemon/Searchability-Obligations/issues/1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_search.py::test_strict_lower_bound_boundaries -q
.venv/bin/python -m compileall -q src scripts tests
git diff --check
make test
make test-full
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_randomized.py::test_fixed_seed_ten_thousand_random_cases -q \
  --junitxml=results/test_evidence/randomized_10000_post_audit.xml
.venv/bin/python scripts/run_experiment.py --config configs/evaluate.json \
  --only-experiment sift-initial-group-build-seed-0 \
  --max-base 2000 --max-delta 200 --max-validation-queries 5 \
  --max-test-queries 5 --max-repetitions 1
.venv/bin/python scripts/analyze_results.py \
  --input results/runs/texmex-evaluation-20260905T061412.978191Z-500834d59d \
  --evidence-role calibration
make test
```

## Executed correctness results

- Independent-oracle/core lightweight suite: `60 passed, 1 deselected in
  1.35s` (agent rerun on the current core).
- Required fixed-seed randomized evaluation: exactly 10,000 cases, `1 passed in
  35.30s` in the recorded run. JUnit:
  `results/test_evidence/randomized_10000.xml`; SHA-256
  `ac4640c4af331bb8460ed3372108dde8145ee2dac3f37729e6d93c2f71c4848e`.
- Explicit counterexample/boundary/core subset after the radius-integrity fix:
  `63 passed in 0.64s`.
- No contract violation has been observed. This is evidence for the stated
  finite-input implementation, not a proof of novelty or universal usefulness.

## Executed lifecycle results

- SQLite/MVCC/generation/fault-injection suite: `21 passed in 1.98s` after the
  final manifest-publication ordering change. Tests include actual subprocess
  `os._exit(86)` at transaction, generation file/fsync/rename/manifest/catalog,
  and group-catalog publication boundaries.
- An earlier integrated `make test` run, before that final ordering-only change,
  reported `84 passed, 1 deselected in 3.07s`. It is not substituted for the
  required final rerun.
- A subsequent `make test-full` was interrupted while the 10,000-case evaluation
  test was still running. It is explicitly incomplete and must be rerun after
  the current independent audit fixes are integrated.
- Lifecycle hardening then added explicit integrity initialization on generation
  load, runtime-thread selection, nonduplicating raw fallback with Receipt
  details, group-catalog revision and deterministic Delta-view bindings, real
  insert/update/delete pre-obligation process crash points, and temporary
  generation-ID rejection. The targeted lifecycle/crash rerun reported
  `29 passed in 2.58s`.
- The post-hardening lightweight integrated `make test` was independently
  rerun by the primary agent and reported `115 passed, 1 deselected, 3 warnings
  in 5.33s`. This is an executed intermediate result; another final rerun is
  still required after benchmark hardening.
- After the immutable-payload and benchmark measurement fixes present at
  15:03 JST, `make test` reported `126 passed, 1 deselected, 5 warnings in
  5.60s`, and `make test-full` reported `127 passed, 5 warnings in 52.57s`.
  A dedicated 10,000-case rerun then passed in `46.76s`; its new JUnit SHA-256
  is `d417bd3faa174db506a5d46a84889961faa17b05e4513ad84c0b351d96f7728b`.
  This is preserved as post-audit evidence, but the benchmark agent announced
  one subsequent cross-experiment truth-cache edit; therefore a source-frozen
  final suite/JUnit rerun is still required and will not be inferred from this
  result.
- After committing the complete implementation and pipeline, the clean tree at
  Git HEAD `29772a0` and implementation-tree SHA-256 `d2e55be3b4f19ddf...`
  passed the source-frozen final `make test-full`: `129 passed, 7 warnings in
  53.34s`. The dedicated fixed-seed 10,000-case rerun passed in `46.88s`.
  Final JUnit: `results/test_evidence/randomized_10000_final.xml`, SHA-256
  `187f865e49683a4845ae0ffe67061561057228f863bfd3786972635e01a7dee9`.
  The warnings are NumPy's optional-PyYAML configuration-display warning and do
  not change test results; they remain visible rather than being suppressed.

## Independent audit state

- Core-integrity fixes now bind certified groups to exact vector payloads and
  keys, bind externally supplied candidate sets to query/snapshot/generation and
  the Base universe, supplement ANN underfill safely, reject complex inputs,
  retain exact audit rationals/keys, and avoid double-counted component timers.
  The post-fix lightweight suite reported `100 passed, 1 deselected` before the
  lifecycle hardening; the integrated result above supersedes it.
- Acceptance and performance audits found measurement blockers that must be
  fixed before final timing: the optimized Delta Flat baseline currently scans
  Delta twice; the reported QPS is a reciprocal-latency estimate rather than
  measured batch throughput; Base `efSearch` is implicitly 512 instead of the
  required initial 128; certificate validation is not fail-closed; the exact
  path needs chunking and an independent sample check; repeated set/key scans,
  group materialization, beta0-only audit serialization, and recomputation of
  final distance intervals contaminate latency. The pre-audit smoke is therefore
  explicitly excluded from the final comparison.
- The benchmark hardening phase resolved those blockers. Performance Delta
  Flat and full-population Flat each perform one Faiss scan plus exact shortlist
  boundary reranking, while separate non-timed certified truth is generated
  once per query. Base `efSearch` defaults to 128 and is a separate sweep axis.
  Actual sequential batch QPS, per-query repetition medians, deterministic
  paired bootstrap intervals, fail-closed Receipt/baseline validation, a
  sampled independent Fraction oracle, source-tree hashes, per-experiment
  repetitions, and final/calibration evidence roles are saved. A final-role run
  without `COMPLETED.json` is listed with its failure state but excluded from
  every scientific aggregate.
- Profiling identified repeated final interval evaluation as one avoidable
  cost. The optimized path carries already computed intervals; every condition
  retains a certified `group_pruning_beta0_recompute_intervals_ablation` with
  identical decisions and result keys. A reduced SIFT calibration (not final
  evidence) completed with 80 raw rows, zero contract violations, zero baseline
  validation failures, and 1/1 independent-oracle match. On only five queries,
  Delta Flat p50 was 13.600 ms, interval-carry beta=0 was 25.365 ms, recompute
  ablation was 26.342 ms, and no-pruning was 27.188 ms. This is engineering
  evidence for the small optimization and a warning that this small-Delta
  condition loses, not the H2 conclusion.
- A later independent scientific audit reproduced a new **correctness P0**:
  NumPy owning arrays marked only with `write=False` can be made writable again
  by a holder of the public reference. After a group was validated, changing a
  member vector from 100 to 0 left its packed matrix/center at 100; the full
  reference then selected that member while pruning skipped the group and
  returned another ID with a false `certified_beta=0`. This is preserved as an
  observed pre-fix counterexample, not called a pass. Final testing/timing is
  blocked until canonical vectors/matrices use non-writeable immutable backing
  and the regression test passes.
- That P0 is now fixed by canonicalizing public arrays onto immutable `bytes`
  backing, rather than relying only on NumPy's reversible `WRITEABLE` flag.
  The fix covers records, candidate sets, groups and their visible matrices,
  trained centers, Base indexes (including generation reload), dataset splits,
  and tuple-normalized Delta raw records. The exact pre-fix exploit is now a
  regression test. Targeted immutable/core/dataset/search checks reported
  `53 passed in 0.39s`; a second store/crash/benchmark subset reported
  `38 passed in 4.41s`. These are interim targeted runs, not the final suite.
- The strict pruning boundary now explicitly tests `nextafter` immediately
  below the equality boundary as a scan, equality as a scan, and immediately
  above as a skip. Its targeted rerun reported `4 passed in 0.28s`.
- The independent scientific review found no further unresolved correctness
  P0. It did require a new post-fix 10,000-case artifact, corrected an overly
  broad checksum statement, and separated actual subprocess crash coverage
  from the in-process generation-pin/GC interleaving claim. Those documentation
  corrections are in the working tree; the post-fix full suite remains next.

## Executed smoke state

- The first offline smoke run completed all 8 query blocks across four required
  synthetic distributions and ten named methods. It saved 320 query-method rows
  under
  `results/smoke/offline-smoke-20260905T050837.464036Z-c36b660d90` and recorded
  zero contract-violation rows.
- Analyzer checksum verification of those shards completed with 1 run, 40
  method/condition groups, and 320 rows. Running the analyzer directly (outside
  the Makefile) emitted a harmless unwritable default Matplotlib-cache warning
  and used `/tmp`; the Makefile sets `MPLCONFIGDIR` to the writable repository
  cache. A fresh smoke will be retained after audit fixes rather than treating
  this pre-audit run as final evidence.
- The source-frozen post-audit smoke completed as
  `offline-smoke-20260905T062705.177285Z-43d3397eca`, with all 8/8 blocks,
  384 raw rows, zero contract violations, zero optimized-baseline validation
  failures, and 4/4 independent Fraction-oracle matches. Its implementation
  tree hash is `d2e55be3b4f19ddff046d4cdedbfeafe07ad12ccdf5a9603726eab612a269095`.
  The analyzer selected only the calibration-role post-audit run, verified the
  checkpoint hashes, and emitted 48 method/condition groups.
- In this deliberately tiny smoke, beta=0 pruning remained slower than Delta
  Flat in all four distributions. It skipped 56.25% of groups for clustered,
  75% for outlier-radius, and 0% for isotropic and Delta-near-query data. All
  five displayed certified/full-scan methods had exact recall 1.0 here. These
  observations validate negative-condition reporting but do not replace the
  real-data evaluation.

## Reporting state

- `reports/REPORT_ja.md` began as a manually reviewed Japanese scaffold distinct
  from the generated aggregate. Its interim `TBD（未実行）` values and overall
  `INCONCLUSIVE` decision were subsequently replaced from completed evidence;
  its current overall decision is `NOT_SUPPORTED_IN_TESTED_REGIME`. Novelty and
  FTO alone remain explicitly `INCONCLUSIVE`, because this experiment cannot
  establish either claim.
- `README.md` now distinguishes CI vs. the 10,000-case suite, raw/checkpoint vs.
  completion markers, generated vs. human reports, reduced diagnostics vs.
  acceptance runs, and the clean reproduction order.
- The report scaffold and README update were fixed as phase commit `1f0c74e`
  (`docs: scaffold Issue #1 research report`).

## Dataset acquisition state

- TEXMEX's historical HTTP archive paths returned HTTP 404. The official page
  currently links `ftp://ftp.irisa.fr/local/texmex/corpus/`; its MD5SUM file and
  CC0 notice were inspected instead of silently changing provenance.
- SIFT1M download/extraction completed. Archive is 168,280,445 bytes, official
  MD5 `b23d1b3b2ee8469d819b61ca900ef0ed`, observed SHA-256
  `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`.
  Per-file hashes are in `data/manifests/sift.json`; large files are ignored.
- GIST1M download/extraction completed. The archive is 2,740,172,684 bytes;
  official and observed MD5 are both
  `31185e0f00854f74d27e8ad8d52628a9`, and observed SHA-256 is
  `01469a7f1c3768853525e543d537e2dfa1adece927616405e360952e3f67df73`.
  Per-file hashes are in `data/manifests/gist.json`; large files are ignored.
- Small reader checks successfully loaded Base/Delta/validation/test shapes for
  both archives. The extracted SIFT and GIST files remain outside Git.
- `make data` was rerun idempotently after the hardened downloader landed. It
  reused both existing archives, rechecked their official MD5 and pinned
  SHA-256 values, rehashed the extracted files, completed successfully, and
  rewrote both manifests with an explicit statement that TEXMEX publishes MD5
  while this artifact pins the observed SHA-256. `last_attempt.json` contains
  `"failures":[]`.

## Final evaluation allocation

- `configs/evaluate.json` contains two full main conditions (SIFT: 100,000
  Base / 10,000 Delta / 1,000 test; GIST: 50,000 / 5,000 / 800), each with
  three timing repetitions, three beta factors, and three comparator
  `efSearch` values. The remaining group-build-seed and one-axis conditions use
  200 test queries, one repetition, beta factor 0.05, and comparator
  `efSearch` 128/512. This is 15 experiments and an expected 110,400 raw rows.
- The SIFT Delta=100,000, 200-query condition is isolated in
  `configs/evaluate_delta100k.json` (expected 2,400 rows). Its independent
  Fraction-oracle ceiling is 120,000, which admits its same-C population rather
  than silently skipping the sample. `make evaluate` executes both resumable
  configurations. This reduced sweep allocation is documented and will not be
  described as 1,000 queries for every condition.
- TEXMEX uses recorded immutable prefix row ranges; `split_seed` is `null`.
  Seeds 0/1/2 are accurately labelled `group_build_seed` and change k-means
  group construction, not the data split.
- After phase commit `5293311`, the post-fix implementation-tree hash was
  `d56668c2585afac1cbc15e99720d41bedb5967ead6ff25288321ee7c26aa1c76`
  and the worktree was clean. Because the preserved failed run had the old
  implementation hash, the fresh main run was started explicitly with
  `.venv/bin/python scripts/run_experiment.py --config configs/evaluate.json`
  rather than mixing checkpoints. Run
  `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` completed
  176/176 blocks from `2026-09-05T08:50:44.778738+00:00` to
  `2026-09-05T11:39:17.413780+00:00`. It saved exactly 110,400 raw rows in
  176 checksum-addressed shards (598,994,358 bytes), reported no run failures,
  executed 15/15 Fraction-oracle samples with zero skips/mismatches, and
  recorded peak RSS 6,193,922,048 bytes. Its completion checkpoint SHA-256 is
  `8b7fa835f2f492dc021499100072feeef91a57ddcbe1eeb706d85f3884a97075`.
- The isolated heavy condition was then started explicitly with
  `.venv/bin/python scripts/run_experiment.py --config
  configs/evaluate_delta100k.json`. Run
  `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085`
  completed 8/8 blocks from `2026-09-05T11:39:47.109100+00:00` to
  `2026-09-05T11:57:43.092488+00:00`. It saved exactly 2,400 raw rows in 8
  checksum-addressed shards (24,101,234 bytes), reported no run failures,
  executed its 1/1 Fraction-oracle sample with no skip/mismatch, and recorded
  peak RSS 1,611,010,048 bytes. Config hash is
  `e46edaa0851a1719b6894bc5c6783db31e5e64996b50e130391bc7149d8cf353`;
  completion checkpoint SHA-256 is
  `0a04714768e603ece8374b75164ae313633c7effe655c8fd1adc5d0a40818a07`.
- Both final runs share the same post-fix implementation hash and have durable
  `COMPLETED.json` markers. The checksum-verifying analyzer has now accepted
  both and produced the final totals recorded below.

## Final analysis, evidence, and decision

- `make report` completed against `results/runs` after both final runs. It
  discovered three final-role runs, fail-closed excluded the incomplete one,
  and accepted two completed runs. It reverified 184 raw shards and aggregated
  112,800 query-method rows into 200 method-condition groups. Totals were:
  contract violations 0, optimized-baseline validation failures 0, independent
  Fraction oracle 16 executed / 16 matched / 0 skipped.
- The analyzer was then hardened before final synthesis. It now validates the
  `COMPLETED.json` run/config/implementation identities, checkpoint and
  run-manifest hashes, raw-shard count, manifest status/block counts, effective
  config object hash, and every raw row/hash/count. Tau, certificate, Delta
  exact-neighbor capture, fallback reasons, scan amounts, and Delta-influence
  aggregates are deduplicated at `(query_id, query_position)` before treating
  repetitions as observations. A synthetic repeated-query unit test and a
  deliberately tampered completion-receipt rejection test cover these rules.
- `make report` was rerun after those changes. It generated the human-readable
  aggregate, three figures, and deterministic tracked snapshot
  `results/final_evidence/summary.json` (1,002,420 bytes, SHA-256
  `ac6eb0cf49371588690aec15ea416dd7421e665525bb036f4949081e08d1e72c`).
  The snapshot omits the analysis clock/local path but retains both completion
  receipts and all 184 run-relative raw paths, byte counts, row counts, and
  SHA-256 values. `manifest.json` binds that snapshot hash.
- The proposal's pruning implementation was slower than optimized Delta Flat
  in every one of 36 primary pruning method-conditions. No pruning or interval
  recomputation summary had a paired-bootstrap 95% CI lower bound above 1.
  Best was the degenerate SIFT Delta=0 beta=0 condition at 0.9727
  `[0.9711, 0.9735]`; SIFT initial was 0.2780 `[0.2762, 0.2803]`, GIST reduced
  initial 0.6294 `[0.6290, 0.6298]`, and SIFT Delta=100,000 0.05023
  `[0.04913, 0.05073]`. There is no query-side or maintenance-amortized
  break-even in the tested grid.
- H1 is supported only for the tested same-frozen-`C` contract, H2 is not
  supported, and H3 is supported only for the tested reference lifecycle.
  The single overall decision in `reports/REPORT_ja.md` is therefore
  `NOT_SUPPORTED_IN_TESTED_REGIME`.
- A final independent report audit identified one presentation gap rather than
  a result change: Issue #1 explicitly asks for latency on queries where Delta
  affects the answer and for update-skew effects. The report now gives the
  stored influence-subset p50 pairs for SIFT initial (22.414/81.740 ms), GIST
  reduced (111.276/176.932 ms), and SIFT Delta=100,000 (27.619/567.678 ms), in
  each case Delta Flat / beta=0 pruning. It also records the clean synthetic
  Delta-near stress: 0% group skip, 24/24 Delta vectors read, and 1.668/2.437 ms.
  These values were independently read from the final and clean-smoke summaries;
  they do not alter the H1/H2/H3 decisions.

## Current-source verification and smoke

- After adding the analysis-integrity regressions, `make test` collected 133
  tests and reported `132 passed, 1 deselected, 8 warnings in 7.44s`.
  `make test-full` reported `133 passed, 8 warnings in 54.54s`.
- The dedicated fixed-seed 10,000-case command then reported `1 passed in
  46.94s`. `results/test_evidence/randomized_10000_final.xml` has SHA-256
  `4701592b238fc7dff6bd30ae568dee41049723efcecf2ee1f26c3674b9e9bbd9`;
  its suite time is 46.815s and the executable-tree hash is
  `8c4581cf733bf25b3e27c115be1dbdad1285564f03846c0d0e93b822c2c80f31`.
- `make smoke` on that same executable tree completed current calibration run
  `offline-smoke-20260905T121406.178089Z-43d3397eca`: 8/8 blocks, 384 rows,
  no run failure, no contract/baseline failure, and 4/4 Fraction-oracle
  matches. A run-specific analysis is saved under
  `results/smoke/final_analysis`. Beta=0 group skips were 56.25%, 0%, 0%, and
  75% for clustered, isotropic, Delta-near-query, and outlier-radius; exact
  recall was 1.0, but pruning was slower than Delta Flat in all four.
- A direct run-specific analyzer command emitted a Matplotlib unwritable-cache
  warning and used `/tmp`; this is preserved as a non-result-affecting warning.
  The Makefile supplies the repository-local `MPLCONFIGDIR` and did not emit it.

## Clean-environment acceptance verification

- Analysis/evidence was fixed as phase commit `ebdeb68`; the Japanese decision,
  current-source smoke, and JUnit were fixed as phase commit `7bfebe9`. The
  primary worktree was clean at `7bfebe935257da8d6c427e52430f35c156e3c50b`.
- A new directory was created with
  `mktemp -d /tmp/searchability-clean-XXXXXX`, yielding
  `/tmp/searchability-clean-n1Ehz2`, followed by
  `git worktree add --detach /tmp/searchability-clean-n1Ehz2 7bfebe9`.
- The first sandboxed `make setup` created `.venv` but could not resolve PyPI;
  pip exhausted five connection attempts and Make exited 2 at the pinned-pip
  install. This external DNS failure is preserved. The exact same `make setup`
  was rerun with network approval and succeeded. `scripts/check_environment.py`
  recorded commit `7bfebe935257...`, empty Git porcelain status, CPython
  3.10.12, NumPy 2.2.6, Faiss 1.15.0, successful squared-L2 HNSW smoke, and
  all five thread environment variables plus Faiss threads equal to 1.
- Clean `make test` reported `132 passed, 1 deselected, 8 warnings in 8.49s`.
- Clean `make smoke` completed run
  `offline-smoke-20260905T122801.383599Z-43d3397eca`, 8/8 blocks, config hash
  `43d3397eca1f74b736d7e0b94184e900ca879f55d4d2c1f5ab206d8492f5f4bc`,
  implementation hash
  `8c4581cf733bf25b3e27c115be1dbdad1285564f03846c0d0e93b822c2c80f31`,
  checkpoint SHA-256
  `a3450087326a30b32901db792c594f4cf9bb9fc4ff430823a0b1dd16fb469982`,
  and peak RSS 57,339,904 bytes. Its `COMPLETED.json` SHA-256 is
  `6f0205c5801df4bff241b32b8127e96914942c481d7a6f266cd65eb8f63a6b95`.
  Its raw directory was copied into the primary tracked `results/smoke/` tree
  before clean-worktree cleanup; it remains separate from the designated smoke
  rather than being pooled as extra scientific observations.
- Clean report command was exactly:

  ```bash
  make report \
    REPORT_INPUT=/home/wasanemon/project/Searchability-Obligations/results/runs \
    REPORT_OUTPUT=.cache/final-report-analysis
  ```

  It revalidated two completed final runs and excluded the one incomplete run,
  aggregated 112,800 rows with contract/baseline failures 0 and Fraction oracle
  16/16, and regenerated `results/final_evidence/summary.json` with exactly the
  committed SHA-256
  `ac6eb0cf49371588690aec15ea416dd7421e665525bb036f4949081e08d1e72c`.
  `git diff` showed no change for either immutable evidence file.
- As an additional clean check, `make test-full` reported
  `133 passed, 8 warnings in 54.55s`.
- After copying the clean smoke, it was independently re-read with
  `scripts/analyze_results.py --input
  results/smoke/offline-smoke-20260905T122801.383599Z-43d3397eca --output
  /tmp/searchability-smoke-audit-nohnRP --evidence-role calibration`. Checksum
  validation accepted one completed run, 384 rows and 48 groups, with contract
  and optimized-baseline failures 0 and Fraction oracle 4/4. Because that
  analyzer also refreshes the shared generated report/figures, `make report`
  was immediately rerun against `results/runs`; it completed with the final
  2-run/112,800-row totals above and reproduced immutable summary SHA-256
  `ac6eb0cf49371588690aec15ea416dd7421e665525bb036f4949081e08d1e72c`.

## External constraints and observed failures

- Sandbox DNS blocked the first GitHub API, `git ls-remote`, and PyPI attempts.
  Approved network execution succeeded for the Issue API and PyPI. No failure
  is being reclassified as a successful experiment.
- The GitHub CLI is not installed. The first publication attempt,
  `git push origin b5dc576:refs/heads/main`, failed with exit 128 because the
  HTTPS remote had no interactive credential (`could not read Username`). The
  failure was not relabelled as success. Existing SSH authentication was then
  checked with `ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T
  git@github.com`; GitHub identified `wasanemon` (the expected no-shell exit 1).
  Explicit SSH pushes succeeded without changing the configured remote URL.
- GIST can require several GiB and multiple indexes exceed naive in-memory
  estimates. Evaluation must stream/chunk exact work and build baselines
  sequentially. Any reduced sweep will be recorded as reduced, not complete.
- The first source-frozen final real-data attempt was run with exactly
  `make evaluate`. Run
  `results/runs/texmex-main-and-lean-sweeps-20260905T062813.056342Z-98bb0214e0`
  used config hash
  `98bb0214e0869353364c1b5560616c9ccb065b5b1f0c65e7243ab9eace1a3909`
  and implementation-tree hash
  `d2e55be3b4f19ddff046d4cdedbfeafe07ad12ccdf5a9603726eab612a269095`.
  It preserved 104/176 checksum-addressed raw blocks (SIFT/GIST main and all
  seed conditions), approximately 460.6 MB of raw JSONL, zero recorded
  benchmark failures through those blocks, and 6/6 independent Fraction-oracle
  matches. At `2026-09-05T08:43:24.520290+00:00` it correctly ended as
  `failed_incomplete`: the first `delta_count=0` sweep exposed
  `TypeError: memoryview: cannot cast view with zeros in shape or strides` in
  `DatasetSplit.sha256()`. This failure and its raw data remain in place and
  must stay excluded from final aggregates. The cause is the zero-byte ndarray
  hash path, not a measured search-contract violation.
- Because fixing the empty-array hash changes the implementation-tree hash, the
  104 completed blocks above will not be mixed with post-fix blocks. After a
  regression test and full verification, `make evaluate` must start a new final
  run; the failed run remains immutable audit evidence.
- The zero-byte hash fix now skips only the absent ndarray payload after already
  hashing the field name, dtype, and full shape; non-empty arrays retain the
  original 8 MiB chunked hashing path. The new regression covers empty Delta,
  validation/test queries, and ID arrays. The exact one-thread targeted command
  reported `8 passed in 0.36s`. A deliberately reduced end-to-end calibration
  of the formerly failing `sift-delta-0` condition then completed 1/1 block with
  no failures and 1/1 Fraction-oracle match as
  `results/runs/texmex-main-and-lean-sweeps-20260905T084538.695374Z-4db427566a`.
  Its config hash is
  `4db427566abbb96e00022968cc162e31ac60810462284fa5fa7b753582ba0a59`;
  it is calibration evidence only and will not enter final aggregates.
- Post-fix `make test-full` collected 130 tests and completed with
  `130 passed, 7 warnings in 53.50s`. The dedicated fixed-seed 10,000-case
  command then completed with `1 passed in 47.46s`; updated JUnit
  `results/test_evidence/randomized_10000_final.xml` has SHA-256
  `f441aa7324ea66ab09e9d346aa9d71c00782c9d9f11ab27cd7f2054a7462fd67`.
  The executable implementation-tree hash for both is
  `d56668c2585afac1cbc15e99720d41bedb5967ead6ff25288321ee7c26aa1c76`.
  This run was hash-bound but performed before committing the fix, so its
  manifest truthfully records a dirty worktree; the required clean-environment
  verification remains a later phase.

## Publication state

- Phase-sized local history was preserved. Commit `b5dc576` was pushed as the
  empty repository's `main` baseline with:

  ```bash
  git push git@github.com:wasanemon/Searchability-Obligations.git \
    b5dc576:refs/heads/main
  ```

- The completed branch through report-audit commit `1eacd1e` was pushed with:

  ```bash
  git push git@github.com:wasanemon/Searchability-Obligations.git \
    codex/issue-1:refs/heads/codex/issue-1
  ```

- The connected GitHub application opened PR #2 from `codex/issue-1` to
  `main`: `https://github.com/wasanemon/Searchability-Obligations/pull/2`.
  It was created open, non-draft, and unmerged. Its Japanese body records the
  negative overall decision, hypothesis-specific decisions, main timings,
  contract scope, final and failed run identities, clean verification, raw-data
  policy, and limitations; it says explicitly that it will not be auto-merged.
- GitHub Actions `ci` run `33966873199` (run number 5) for publication-state
  HEAD `8f32181a36e61f85ea4bafe94030d1375d02c622` completed successfully. The
  workflow installs the pinned dependencies on Ubuntu 22.04 / Python 3.10 with
  one-thread environment variables and executes the non-evaluation pytest
  suite. The connected application then reported PR #2 as open, mergeable, and
  unmerged.
- The detached temporary worktree `/tmp/searchability-clean-n1Ehz2` and its
  dedicated `.venv` were removed with `git worktree remove --force` only after
  its completed smoke raw directory had been copied and committed. The final
  real-data directories remain untouched.

## Phase checklist

- [x] Inspect empty history, resources, network constraints, and Faiss compatibility.
- [x] Define/prove the contract, numerical mode, and independent oracle.
- [x] Implement insert-only HNSW + Delta kernel and required baseline index types.
- [x] Run boundary and fixed-seed 10,000-case tests; lifecycle-specific
      counterexamples remain tied to the lifecycle phase.
- [x] Run an initial offline smoke and preserve per-query raw evidence.
- [x] Re-run smoke after independent-audit fixes and designate final evidence.
- [x] Acquire and checksum-verify SIFT and GIST from the official TEXMEX source.
- [x] Evaluate both real datasets and preserve query-level evidence.
- [x] Implement and test SQLite obligations, MVCC snapshots, actual-process
      crash recovery, generation publication/pinning, and conservative GC.
- [x] Regenerate figures/aggregates and the Japanese report from saved raw data.
- [x] Re-run setup -> test -> smoke -> report from a clean environment.
- [x] Create phase commits and an open, non-merged Issue #1 PR.

## Current resume point

Dataset acquisition, implementation, tests, final real-data runs,
checksum-verifying analysis, tracked evidence export, Japanese decision,
detached clean-environment acceptance, phase commits, branch publication, and
PR creation are complete. The preserved failed attempt remains excluded and PR
#2 remains intentionally unmerged. CI run 33966873199 passed on the complete
source and evidence state; this final state-only note does not alter executable
code or experimental evidence. The exact future resume checks are:

```bash
git status --short --branch
git diff --check
git rev-parse HEAD
```

Then inspect PR #2 for any new review or GitHub Actions result on the current
HEAD. If a later check fails, preserve the log before changing anything and
rerun only the failing verification locally. Do not merge the PR, and do not
rerun or modify either completed final real-data run merely to refresh
timestamps.

Both dataset manifests report no acquisition failures. To re-verify or resume a
future partial GIST acquisition, run:

```bash
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only gist
```

The `.part` file is retained and the downloader requests a byte range when the
server supports it. Do not start final real-data timing if any new correctness
test reports a contract violation.
