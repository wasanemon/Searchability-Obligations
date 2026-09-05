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

`make test` is the CI-sized suite. The fixed-seed 10,000-case evaluation test is
deliberately separate and must be run before a final research claim:

```bash
make test-full
```

The smoke run needs no network. It emits the optimized full-Delta reference,
pruning with beta=0 and beta>0, the required practical baselines, and the
no-pruning plus final-interval-recomputation ablations. It also saves
machine-checkable guarantee fields and raw
query-method rows. A completed smoke is a pipeline/correctness check, not
evidence that the method is useful on real data.

Real-data acquisition and bounded evaluation are separate:

```bash
make data
make evaluate
```

Dataset URLs, checksums, extraction IDs, licenses/terms links, failures, and
resume commands are recorded rather than silently substituting data. All
benchmarks force Faiss and common BLAS implementations to one thread by
default. SIFT and especially GIST require several GiB of disk and the configured
evaluation can run for a long time on one CPU thread.

`make evaluate` executes the main/lean SIFT+GIST matrix and then the isolated
Delta=100,000 SIFT condition. Both are resumable: the runner selects an
incomplete run with the same effective config hash, checkpoints every query
block atomically, and verifies a completed shard before skipping it. The main
SIFT/GIST conditions use 1,000/800 test queries and three timing repetitions;
the one-axis and group-build-seed sweeps explicitly use 200 queries and one
repetition. The exact matrix and the reason the heavy condition is isolated are
in [`configs/README.md`](configs/README.md).

The command prints an exact resume command; retain its `--run-id`. For a
resource-bounded diagnostic (not a substitute for the configured final runs),
use the same runner options, for example:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/evaluate.json \
  --max-experiments 1 --max-test-queries 25 --max-repetitions 1
```

Limits become part of the effective config, produce a different config hash,
and automatically change `evidence_role` from `final` to `calibration`. Never
report such a reduced run as the configured final sweep.
Full commands, observed failures, and the current exact resume point are in
[`RESEARCH_STATE.md`](RESEARCH_STATE.md).

## Issue #3 native kernel recheck

[Issue #3](https://github.com/wasanemon/Searchability-Obligations/issues/3)
adds a packed pybind11/C++17 F/N/P kernel, an independently optimized Faiss A
path, required ablations, a validation gate, and a separate Japanese report.
The reproducible command chain is:

```bash
make setup
make native-test
make native-smoke
make native-validate
make native-evaluate
make native-report
```

`native-test` executes the native fixed-seed 10,000-case gate. `native-smoke`
is offline calibration. `native-validate` is the heavy one-thread timing run;
`native-evaluate` may authorize a fresh final only if a non-degenerate real-data
candidate beats both the certified F and practical A baselines under the
pre-registered paired-CI rule. If that gate does not pass, it deliberately
creates no final lock and reads no fresh holdout.

The completed study reached `NOT_SUPPORTED_IN_TESTED_REGIME` / `NO_GO`: P beat
F in the eligible SIFT validation cells but beat A in none, so the fresh final
was not authorized. See
[`reports/NATIVE_KERNEL_RECHECK_ja.md`](reports/NATIVE_KERNEL_RECHECK_ja.md)
and [`results/native_recheck_evidence/README.md`](results/native_recheck_evidence/README.md).
Saved raw can be rechecked by `make native-report` without rerunning the timed
experiment; `make native-validate` starts a new experimental identity.

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
- `data/manifests/`: source URLs, terms notes, archive and extracted-file checksums
- `results/`: small raw evidence, manifests, summaries, and failure logs
- `reports/REPORT_ja.generated.md`: regenerable raw-data aggregate; it makes no research decision
- `reports/REPORT_ja.md`: manually reviewed Japanese synthesis and decision
- `RESEARCH_STATE.md`: authoritative progress and exact resume instructions

## Artifact and integrity model

Every benchmark run has an immutable identity consisting of its run ID, config
hash, dataset hash, and split ID. Its directory contains:

```text
effective_config.json        exact configuration actually executed
run_manifest.json            environment, identities, status, failures, peak RSS
build_manifest.json          index/group construction and memory observations
splits/*.json                Base/Delta/validation/test identity and provenance
validation/*.json            choices made without looking at final test queries
raw/*.jsonl                  one row per query, method, and repetition
checkpoint.json              raw-shard path, row count, and SHA-256
COMPLETED.json               present only after every shard was reverified
```

Small offline evidence is written below `results/smoke`. Real-data raw runs are
written below the gitignored `results/runs` because they can be large. Do not
copy a number out of an incomplete run merely because some shards exist.
Failures and pre-audit runs are retained and explicitly excluded when
appropriate; they are not rewritten as passes.

`make report` searches the saved runs under `results`, verifies each checkpoint
entry's row count and SHA-256, then regenerates machine-readable summaries,
figures, and `reports/REPORT_ja.generated.md`. It does not rerun search and does
not choose the final hypothesis outcome. The human-reviewed decision lives in
`reports/REPORT_ja.md`, which includes slots for paired confidence intervals,
break-even, maintenance cost, negative conditions, limitations, and the exact
run identities used.

## Full reproduction order

From a fresh checkout on the documented CPython/Linux platform:

```bash
make setup
make test
make test-full
make smoke
make data
make evaluate
make report
```

The scientific dependency order matters: do not use performance results after
a correctness violation, do not tune centers/groups/beta on final test queries,
and do not mix ordinary L2 with Faiss squared-L2. Correctness comparisons reuse
one frozen Base candidate set `C`, and certified pruning uses the strict rule
`LB > tau - beta`.

For a clean-environment verification that does not destroy an existing virtual
environment, use a fresh checkout/worktree and run `make setup` there. Record
the commit, dirty state, exact commands, output, failures, and artifact hashes
in `RESEARCH_STATE.md`. Downloaded corpora, generated indexes, virtual
environments, and large run directories remain outside Git; their manifests and
regeneration instructions are the durable record.
