# Agent and contributor rules

This repository is a research artifact governed by GitHub Issue #1. Scientific
integrity and reproducibility take priority over producing a positive result.

## Required workflow

1. Read `RESEARCH_STATE.md` before changing code or running an experiment.
2. Use the pinned `.venv` and one CPU thread unless a config explicitly says
   otherwise. Run commands through the `Makefile` where possible.
3. Never label an unexecuted test or experiment as passed. Preserve failures,
   raw per-query rows, manifests, config hashes, and dataset checksums.
4. Correctness comparisons must reuse one frozen base candidate set `C`.
5. Distinguish ordinary L2 from Faiss squared-L2. Never weaken `LB > tau-beta`
   to a non-strict comparison.
6. Do not call an epsilon-only or plain-float path certified. A certified search
   uses the documented interval/error-bound path and exact boundary ordering.
7. Do not tune centers, group count, or beta on final test queries.
8. Do not commit downloaded datasets, generated indexes, environments, or large
   run directories. Small immutable evidence belongs under `results/`.
9. Update `RESEARCH_STATE.md` after each phase with exact commands, outcomes,
   failures, artifacts, and the next resumable command.
10. Keep commits phase-sized and mention Issue #1. Do not auto-merge a PR.

## Verification commands

```bash
make setup
make test
make smoke
make data
make evaluate
make report
```

`make test-full` includes the fixed-seed 10,000-case evaluation suite. The
default `make test` remains small enough for CI.

