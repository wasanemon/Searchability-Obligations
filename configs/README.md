# Experiment configurations

All configurations use one CPU thread. TEXMEX row ranges are deterministic
prefixes: Base rows precede Delta rows, validation query IDs are disjoint from
test query IDs, and `split_seed` is therefore `null`. `group_seed` changes only
the k-means/group construction; it is not described as a data-split seed.

## Final real-data matrix

`make evaluate` runs the following two independently resumable final-evidence
configurations in order.

### `evaluate.json`

- Main SIFT condition: Base 100,000, Delta 10,000, validation 200, test 1,000,
  `k=10`, 128 groups, Base `efSearch=128`, three timing repetitions.
- Main GIST condition: Base 50,000, Delta 5,000, validation 200, test 800, the
  same remaining initial settings, three timing repetitions.
- Group-build seeds 0/1/2 for both datasets. Seed 0 is the main condition;
  seeds 1/2 use 200 test queries and one repetition.
- One-axis SIFT sweeps use 200 test queries and one repetition: Delta
  0/1,000/10,000; groups 16/64/128/512; `k` 1/10/100; and Base HNSW
  `efSearch` 32/128/512. Anchor values are supplied by the main condition.
- The main conditions include positive beta factors 0.01/0.05/0.10 and Delta
  and full-population HNSW `efSearch` 32/128/512. Lean sweep conditions retain
  beta factor 0.05 and comparator `efSearch` 128/512 to bound runtime.
- Expected raw size with the current method set is 110,400 query-method rows.

### `evaluate_delta100k.json`

- The heavy SIFT Delta=100,000 axis is isolated so a resource failure cannot
  invalidate the completed main/sweep run.
- Base 100,000, validation/test 200/200, 128 groups, `k=10`, one repetition.
- Its independent `Fraction` oracle limit is 120,000, so the single sampled
  same-C population (at most 100,064 rows before de-duplication) is executed
  rather than silently skipped.
- Expected raw size is 2,400 query-method rows.

The reduced sweep query count is an explicit resource allocation, not a claim
that every one-axis condition used 1,000 queries. The main p99 claims use the
1,000-query SIFT condition; all sub-1,000 conditions carry an analyzer warning.

## Calibration and diagnostics

`smoke.json` is offline calibration evidence only. Runner options such as
`--only-experiment` and `--max-*` change the effective config and automatically
downgrade a source marked `final` to `calibration`; such output is never used by
`make report` as final evidence.

The benchmark retains two engineering ablations in every condition:
grouped scanning with pruning disabled, and the pre-optimization final interval
recomputation path. They separate the center/radius pruning effect from the
small interval-carry optimization selected after the initial profile.

## Issue #3 native recheck matrix

`native_recheck_validation.json` is a separately namespaced validation study.
It fixes SIFT Base 100k with Delta 0/1k/10k/100k, groups 64/128/512, and
`k=1/10/100`; GIST Base 50k / Delta 5k is a high-dimensional anchor. The main
family is SIFT Base 100k / Delta 10k / `k=10` / `C=64` / groups 128 with beta
factors 0/.01/.05/.10. Four offline synthetic geometries are also included.
Axes are changed one at a time rather than as a factorial search.

Validation uses SIFT query IDs 0..199 only to calibrate absolute beta and
200..399 for timing. The final policy and `holdout_manifest.json` pre-register
the untouched SIFT interval 1200..2199. Centers, group membership/count, beta,
and Base HNSW settings are not tuned on that holdout.

The gate requires a real SIFT condition with Delta at least 1,000, a nonempty
Delta-influence subset, passed native correctness, and paired 95% CI lower
bound strictly above one against both F and A. A `NOT_PASSED` gate creates no
final lock. `native_recheck_final_policy.json` fixes the three-session fresh
evaluation and engineering thresholds; the decision step verifies it even on a
negative gate, but no fresh-run field can authorize execution without a lock.
Conditional HNSW references likewise cannot start before a fresh-final
performance pass.
