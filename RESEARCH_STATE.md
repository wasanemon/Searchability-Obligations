# Research state

Last updated: 2026-09-05 13:58 (Asia/Tokyo)

## Scope and source status

- GitHub Issue #1 was read in full through the GitHub REST API. It is open,
  created/updated at `2026-09-05T04:19:23Z`, and has zero comments (the comments
  endpoint returned `[]`).
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

## Dataset acquisition state

- TEXMEX's historical HTTP archive paths returned HTTP 404. The official page
  currently links `ftp://ftp.irisa.fr/local/texmex/corpus/`; its MD5SUM file and
  CC0 notice were inspected instead of silently changing provenance.
- SIFT1M download/extraction completed. Archive is 168,280,445 bytes, official
  MD5 `b23d1b3b2ee8469d819b61ca900ef0ed`, observed SHA-256
  `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`.
  Per-file hashes are in `data/manifests/sift.json`; large files are ignored.
- GIST1M (official size 2,740,172,684 bytes) is currently downloading. Do not
  mark it acquired until official MD5 `31185e0f00854f74d27e8ad8d52628a9`
  and extraction manifests have passed.

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
- [ ] Run offline smoke and preserve per-query raw evidence.
- [ ] Acquire and evaluate SIFT plus GIST or a documented public replacement.
- [ ] Implement and test SQLite obligations, MVCC snapshots, crash recovery,
      generation publication/pinning, and conservative GC.
- [ ] Regenerate figures/aggregates and the Japanese report from saved raw data.
- [ ] Re-run setup -> test -> smoke -> report from a clean environment.
- [ ] Create phase commits and, if remote tooling permits, an Issue #1 PR.

## Current resume point

Contract/oracle/kernel correctness is complete at the current revision. Finish
the in-progress GIST download, benchmark harness, and lifecycle reference, then
run:

```bash
make test
make smoke
```

If interrupted during GIST acquisition, resume with:

```bash
.venv/bin/python scripts/download_datasets.py --config configs/data.json --only gist
```

The `.part` file is retained and the downloader requests a byte range when the
server supports it. Do not start final real-data timing if any new correctness
test reports a contract violation.
