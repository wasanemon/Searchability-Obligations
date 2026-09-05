# Research state

Last updated: 2026-09-05 15:27 (Asia/Tokyo)

## Scope and source status

- GitHub Issue #1 was read in full through the GitHub REST API. It is open,
  created/updated at `2026-09-05T04:19:23Z`, and has zero comments (the comments
  endpoint returned `[]`).
- The full body was fetched again at this resume point both through the REST
  endpoint and the connected GitHub application. The repository remains empty
  remotely with default branch `main`; the connected application reports
  repository push/admin permission, so branch publication and a non-merged PR
  will be attempted only after the final local evidence commit.
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

## Reporting state

- `reports/REPORT_ja.md` now exists as a manually reviewed Japanese report
  scaffold distinct from the generated aggregate. It covers every Issue #1
  reporting dimension, uses explicit `TBD（未実行）` markers instead of invented
  numbers, and currently states the only defensible interim decision:
  `INCONCLUSIVE`. It is not a final report until those markers are replaced or
  explicitly resolved from completed evidence.
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

## External constraints and observed failures

- Sandbox DNS blocked the first GitHub API, `git ls-remote`, and PyPI attempts.
  Approved network execution succeeded for the Issue API and PyPI. No failure
  is being reclassified as a successful experiment.
- The GitHub CLI is not installed. Remote branch/PR creation has not been
  attempted. If it remains unavailable, local commits, diff, and exact commands
  are the issue-authorized equivalent handoff.
- GIST can require several GiB and multiple indexes exceed naive in-memory
  estimates. Evaluation must stream/chunk exact work and build baselines
  sequentially. Any reduced sweep will be recorded as reduced, not complete.

## Phase checklist

- [x] Inspect empty history, resources, network constraints, and Faiss compatibility.
- [x] Define/prove the contract, numerical mode, and independent oracle.
- [x] Implement insert-only HNSW + Delta kernel and required baseline index types.
- [x] Run boundary and fixed-seed 10,000-case tests; lifecycle-specific
      counterexamples remain tied to the lifecycle phase.
- [x] Run an initial offline smoke and preserve per-query raw evidence.
- [ ] Re-run smoke after independent-audit fixes and designate final evidence.
- [x] Acquire and checksum-verify SIFT and GIST from the official TEXMEX source.
- [ ] Evaluate both real datasets and preserve query-level evidence.
- [x] Implement and test SQLite obligations, MVCC snapshots, actual-process
      crash recovery, generation publication/pinning, and conservative GC.
- [ ] Regenerate figures/aggregates and the Japanese report from saved raw data.
- [ ] Re-run setup -> test -> smoke -> report from a clean environment.
- [ ] Create phase commits and, if remote tooling permits, an Issue #1 PR.

## Current resume point

Dataset acquisition, lifecycle hardening, benchmark hardening, and the
immutable-payload correctness fix are implemented and source-frozen. The final
full suite and 10,000-case evidence passed as recorded above. Commit the new
test evidence, then resume with:

```bash
make smoke
make evaluate
make report
```

Both dataset manifests report no acquisition failures. To re-verify or resume a
future partial GIST acquisition, run:

```bash
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only gist
```

The `.part` file is retained and the downloader requests a byte range when the
server supports it. Do not start final real-data timing if any new correctness
test reports a contract violation.
