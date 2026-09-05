# Research state

Last updated: 2026-09-05 (Asia/Tokyo)

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
```

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
- [ ] Define/prove the contract, numerical mode, and independent oracle.
- [ ] Implement insert-only HNSW + Delta methods and all required baselines.
- [ ] Run boundary, counterexample, and fixed-seed 10,000-case tests.
- [ ] Run offline smoke and preserve per-query raw evidence.
- [ ] Acquire and evaluate SIFT plus GIST or a documented public replacement.
- [ ] Implement and test SQLite obligations, MVCC snapshots, crash recovery,
      generation publication/pinning, and conservative GC.
- [ ] Regenerate figures/aggregates and the Japanese report from saved raw data.
- [ ] Re-run setup -> test -> smoke -> report from a clean environment.
- [ ] Create phase commits and, if remote tooling permits, an Issue #1 PR.

## Current resume point

The dependency compatibility check is complete. Continue with the contract,
numerics, oracle, and insert-only kernel, then run:

```bash
make setup
make test
```

Do not start final real-data timing until correctness tests report zero contract
violations.

