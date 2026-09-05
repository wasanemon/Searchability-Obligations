# Research state

Last updated: 2026-09-05 21:39 (Asia/Tokyo)

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
- The GitHub CLI is not installed. Remote branch/PR creation has not been
  attempted. If it remains unavailable, local commits, diff, and exact commands
  are the issue-authorized equivalent handoff.
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
- [ ] Create phase commits and, if remote tooling permits, an Issue #1 PR.

## Current resume point

Dataset acquisition, final real-data runs, checksum-verifying analysis,
tracked evidence export, Japanese decision, current-source verification, and
the detached clean-environment acceptance sequence are complete. The preserved
failed attempt remains excluded. The exact next resumable phase is:

```bash
git status --short --branch
git diff --check
```

Then commit this clean-verification record and copied smoke artifact, publish
the empty remote's initial commit lineage as `main` plus `codex/issue-1`, and
open (but do not merge) the Issue #1 PR. Do not rerun or modify either completed
final real-data run.

Both dataset manifests report no acquisition failures. To re-verify or resume a
future partial GIST acquisition, run:

```bash
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only gist
```

The `.part` file is retained and the downloader requests a byte range when the
server supports it. Do not start final real-data timing if any new correctness
test reports a contract violation.
