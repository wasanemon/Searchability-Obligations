# Searchability Obligations

This repository is a reproducible CPU research prototype for the hypothesis in
[Issue #1](https://github.com/wasanemon/Searchability-Obligations/issues/1):
committed vectors not yet incorporated into an immutable HNSW generation can be
searched selectively while bounding the extra *k*-th ordinary-L2 distance
relative to the same frozen base candidate set plus a complete Delta scan.

The promised comparison is deliberately narrow. It is not exact database
top-k, fresh-rebuilt-HNSW equivalence, an ID/recall guarantee, elapsed-time
freshness, production MVCC, or a cryptographic proof. See
`docs/correctness.md` and `docs/numerics.md` before interpreting a Receipt.

## Quick start

CPython 3.10 on Linux x86_64 is the verified environment.

```bash
make setup
make test
make smoke
make report
```

The smoke run is offline and emits full-Delta, pruning with beta=0, pruning
with beta>0, and no-pruning measurements plus machine-checked guarantee fields.
Small tracked evidence is written below `results/smoke`; resumable evaluation
runs are placed below the gitignored `results/runs` directory.

Real-data acquisition and bounded evaluation are separate:

```bash
make data
make evaluate
```

Dataset URLs, checksums, extraction IDs, licenses/terms links, failures, and
resume commands are recorded rather than silently substituting data. All
benchmarks force Faiss and common BLAS implementations to one thread by
default. Full commands and current completion state are in
`RESEARCH_STATE.md`.

## Core contract

For a query, snapshot, immutable base generation, and a base ANN candidate set
`C` frozen *after* visibility filtering and candidate supplementation:

```text
R_ref  = TopK(C union visible Delta)
R_prop = TopK(C union scanned Delta subset)
0 <= tau_prop - tau_ref <= requested_beta
```

Groups use fixed centers, conservative upper radii, lower distance bounds, and
the strict rule `skip iff LB > tau - beta`. Raw (not-yet-grouped) committed
vectors are always scanned first. A Receipt reports the certified upper bound,
scope, fallback, numerical mode, and measured component costs.

## Repository map

- `src/searchability/`: kernel, independent oracle, baselines, SQLite store, and generations
- `tests/`: unit/property, counterexample, randomized, snapshot, and crash tests
- `configs/`: immutable JSON experiment inputs
- `scripts/`: environment capture, data acquisition, experiment, aggregation, plotting
- `results/`: small raw evidence, manifests, summaries, and failure logs
- `reports/REPORT_ja.md`: Japanese research result and decision
- `RESEARCH_STATE.md`: authoritative progress and exact resume instructions

