# Prototype and lifecycle design

This document describes the complete research-prototype architecture: the
first insert-only search kernel and the later durable update/delete,
snapshot, grouping, and generation lifecycle.  Correctness takes precedence
over availability or speed; any state that cannot justify pruning falls back
to scanning.

## 1. Components and responsibilities

| Component | Responsibility |
|---|---|
| `BaseIndex` | Immutable Faiss `IndexHNSWFlat`, plus a persisted ordinal to `(logical_id, version_id)` map.  Faiss squared-L2 is candidate-generation data only. |
| `VersionStore` | Canonical float32 vectors and MVCC begin/end commit sequences. |
| `ObligationStore` | Durable insert/update/delete effects not covered by each eligible base generation. |
| `DeltaStore` | A query view of visible raw and published-grouped obligations; never the source of durability by itself. |
| `GroupDirectory` | Immutable in-memory centers, conservative radius upper endpoints, and contiguous member arrays reconstructed from one SQLite catalog revision. |
| `SearchEngine` | Pins a view, freezes base candidates, scans raw data, certifies group skips, exactly resolves the boundary, and merges results. |
| `Oracle` | Separate `Fraction`-based exhaustive enumeration over one explicit frozen candidate universe. |
| `Receipt` | Scope, numeric mode, certificate, fallback, counts, hashes, and component timings. |
| `GenerationManager` | Builds, validates, loads, and inventories immutable generation directories; the SQLite lifecycle store publishes and pins them. |
| `Benchmark` | Supplies common frozen inputs to methods, separates micro/end-to-end timing, and saves per-query raw records. |

The first performance kernel is insert-only and operates on immutable in-memory
snapshots with one CPU thread.  The durable layer is a deliberately small
SQLite-backed reference implementation; it is not presented as a production
DBMS or a high-concurrency Faiss wrapper.

## 2. Identity, MVCC, and commit records

`logical_id` names an application object.  Every change creates a distinct,
monotonically unique `version_id`; neither is a Faiss ordinal.  A vector version
is visible at snapshot sequence `s` exactly when

```text
begin_seq <= s and (end_seq is NULL or s < end_seq)
```

The store enforces at most one visible version per logical ID at any sequence.
An update closes the prior version and creates a new vector version.  A delete
closes the prior version and records a tombstone/obligation but creates no
searchable vector.

A single SQLite transaction performs all of the following:

1. increments a singleton commit clock;
2. writes the new vector version, if any;
3. sets the old version's `end_seq`, if any;
4. writes the insert/update/delete obligation;
5. commits the assigned sequence.

The clock increment rolls back with the transaction, so committed sequences
form a gap-free prefix.  Only under this invariant may a generation publish
`covered_commit_seq = p`, meaning that every committed effect through `p` is
incorporated.  If an alternative allocator can leave holes, the implementation
must persist an explicit coverage set/range manifest and leave the prefix field
unset; taking `max(seq)` is forbidden.

Vector bytes and their obligation are therefore both durable or both absent.
Current-latest metadata is an optimization only.  Visibility always uses the
version intervals appropriate to the requested snapshot, including old
snapshots.

## 3. Immutable base generations

A generation manifest contains at least:

```text
generation_id, covered_commit_seq, build_snapshot,
metric, dimension, dtype, numeric_contract_version,
hnsw_parameters, vector_count, ordinal_map_hash,
index_hash, metadata_hash, build_config_hash, created_at
```

Generation `g` materializes all versions visible at its gap-free coverage
sequence `p_g`.  It is suitable for a query snapshot `s >= p_g`: base versions
made obsolete after `p_g` are rejected by MVCC, while visible versions begun in
`(p_g, s]` are supplied by obligations.  For an older snapshot, select an older
retained generation with coverage no later than that snapshot.  If none exists,
directly enumerate retained versions; if policy has expired those versions,
reject the snapshot rather than returning a partial answer.

Generations are never modified after publication.  Searches and Faiss `add`
do not run concurrently on the same index.  Building a replacement occurs in a
separate directory and process/object.

The manifest records the thread count used to build the generation as
provenance.  Loading does not turn that historical value into a runtime policy:
the current store's configured thread count is installed on the loaded
`BaseIndex` and applied before Faiss search.

## 4. Search-view acquisition

At query start, one consistent SQLite read view obtains and pins:

```text
snapshot_id and snapshot_seq s
generation_id g and its gap-free coverage p_g
group_catalog_revision h
the obligation/visibility view for (p_g, s]
```

The in-process pin protects the generation and catalog files from garbage
collection until the query ends.  A process crash releases its in-memory pins
because no query survives the process; durable snapshot leases, if exposed to
clients, are separate records with explicit expiry.

The query view divides visible Delta versions into two disjoint logical sets:

- `raw_pending`: committed versions not represented by a group in catalog
  revision `h`;
- `grouped_pending`: committed versions represented by exactly one immutable
  group in `h`.

The underlying version rows remain durable in both cases.  Deduplication by
version key is a defensive invariant.  If a transitional bug exposes a member
in both sets, search scans/deduplicates it rather than omitting it, records a
catalog-integrity failure, and does not claim the normal certified lifecycle
status.

## 5. Base candidate set and query pipeline

For the pinned view, the pipeline is:

1. Search the immutable HNSW with configured `efSearch` and overfetch.
2. Translate ordinals to version keys, apply visibility at `s`, and deduplicate.
3. Replenish candidates before freezing if filtering caused underfill.  Record
   the request count, attempts, rejections, complete visible-base count,
   supplementation count, fallback reason, and final candidate-set hash.  If
   ANN remains underfilled, enumerate the complete visible base universe and
   exactly rank missing versions; if the visible population is smaller than the
   request, include every visible version and report that explicitly.
4. Freeze this exact `C`; both proposed and full-Delta correctness paths receive
   it.  Its hash binds ordered keys/vector bits to the query, snapshot,
   generation, complete base-universe hash, and request count.  Search rejects
   an external `C` whose binding or records do not match the pinned base.
   End-to-end benchmarks include steps 1--3, while coupled Delta microbenchmarks
   begin from the saved `C`.
5. Scan every raw-pending vector with the certified bulk float64 kernel.
6. Compute conservative group lower bounds, order groups by
   `(LB_lower, group_id)`, and scan or skip using the strict Fraction rule in
   `correctness.md`.
7. Exactly re-evaluate all boundary contenders and order by
   `(exact_squared_distance, logical_id, version_id)`.
8. Return results and construct a receipt.  Receipt construction is timed.

All ordinary-L2 bounds are produced by `numerics.md`.  Group scans are bulk
float64 operations over contiguous arrays rather than one Python call per
vector.  The full-scan baseline receives equivalent vector layout and
optimization.  A non-certified fast path, if added, has a different
`numeric_mode` and cannot emit `CERTIFIED`.

## 6. Group construction and publication

Centers are learned only from Base or a designated training split.  Validation
may select group count and related parameters; final test queries, future
Delta, and final-test outcomes may not.  Centers are fixed canonical float32
vectors.  Nearest-center assignment is useful but not required for correctness;
complete membership coverage by the radius is required.

Grouping follows a copy-then-publish protocol:

1. Read a stable set of committed raw obligation/version keys.  They remain
   eligible for raw scanning.
2. Build immutable, contiguous member arrays and compute every member's
   certified center-distance upper endpoint.  Before using a radius, verify
   that every row bit-matches its member version, every version key has one
   group owner, and group IDs are unique.  Radius is their maximum.
3. In one SQLite catalog transaction, insert the complete center/radius and
   membership rows under a new revision, then advance the singleton catalog
   revision.  Until that transaction commits, readers reconstruct members as
   raw from retained version/obligation rows; after commit, new readers
   reconstruct the immutable groups.  There is no separate group-directory
   file or rename protocol in this reference implementation.
4. Retain underlying version/obligation data according to the generation and
   snapshot GC rules; publication is not permission to erase it.

A query that pinned the old revision continues to see the raw members.  A new
query sees the published group.  There is no revision in which a committed
member is in neither location.  Center movement, member removal, outlier
splitting, or radius shrinking creates another immutable group and uses the
same atomic catalog switch.

## 7. Generation build and crash-safe publication

For a chosen committed cutoff `p`, generation construction is:

1. Open a stable database snapshot at `p` and build the Faiss index plus
   ordinal/version metadata in a unique `.tmp-...` generation directory.
2. Flush and fsync the index and metadata files, then fsync the temporary
   directory.  No manifest or SQLite reference exists yet.
3. Atomically rename the directory to the final generation name and fsync its
   parent directory.  User generation IDs beginning with `.tmp-` are reserved
   and rejected, so recovery cannot confuse a final directory with a temporary.
4. Atomically install and fsync the self-validating manifest *inside the renamed
   final directory*.  A crash before this step leaves an unreferenced invalid
   orphan; a crash after it leaves an unreferenced complete orphan.
5. Revalidate lengths and cryptographic hashes, then in a durable SQLite
   transaction insert the generation record and publish it as eligible.
   Keep prior generation records and files.

The database never points to a generation before its files are durable.  On
startup, recovery verifies manifests, lengths, hashes, numeric contract, and
the ordinal map before use.  An unreferenced complete directory left before
step 4 is an orphan and may be quarantined/reused; a partial temporary
directory is ignored.  If a newly published generation fails validation after
an assumed filesystem fault, recovery falls back to the newest older valid
generation plus its obligations and records the failure.  It never silently
advances coverage.

The guarantee assumes SQLite's documented transaction durability and the
filesystem/fsync/atomic-rename contract.  Process-kill fault tests validate the
implemented boundaries.  They are not evidence of survival under every power
loss, controller-cache, or filesystem-corruption scenario.

## 8. Obligation retention and conservative GC

Publishing generation `g_new` does not by itself make an obligation or old
version collectible.  A pinned query may still use `g_old`, and an old
snapshot may need a version absent from the new generation.

An obligation/vector/tombstone can be removed only after proving all of the
following:

- no active query or durable snapshot lease can select a generation/view that
  needs it;
- every snapshot still admitted by retention has an eligible generation that
  covers the effect, or exact-fallback data that represents it;
- no retained snapshot needs the version's begin/end interval or tombstone to
  reject a stale base candidate;
- all referenced group and generation manifests have moved beyond it and their
  checksums/coverage are valid.

An old generation file is removable only when it is unpublished for all
admissible snapshots, has no pin, and a valid replacement/fallback exists.
Deletion is staged (mark, rename/quarantine, then unlink after another pin
check) so recovery can distinguish live data from garbage.

The minimal reference implementation is permitted to implement only generation
pinning plus conservative file cleanup and to retain all obligations, vectors,
and tombstones indefinitely.  That behavior is safe but leaks storage; it must
be reported as `GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION`, not as completed
garbage collection.

## 9. Recovery and required interleavings

The implementation exposes fault points and verifies these outcomes:

| Fault/interleaving | Required recovery/result |
|---|---|
| Exit after an insert version row but before its obligation row | SQLite rolls the transaction back: snapshot remains `0`, with no visible version and no obligation in the fixed test fixture. |
| Exit after an update closes/inserts versions but before its obligation row | SQLite rolls the transaction back: snapshot remains `1`, old key `(7, 1)` remains visible, and only its original insert obligation remains. |
| Exit after a delete closes the old interval but before its obligation row | SQLite rolls the transaction back: snapshot remains `1`, key `(7, 1)` remains visible, and only its original insert obligation remains. |
| Exit after commit and before grouping | The version is reconstructed as raw pending and searchable. |
| Query overlaps group catalog switch | Its pinned revision contains the member in raw or group form; never neither. |
| Exit after index/metadata writes or their directory fsync | A `.tmp-...` directory is ignored; recovery uses the old generation plus obligations. |
| Exit after directory rename but before/after manifest install or before SQLite publication | The final directory is an unreferenced orphan; recovery uses the old generation plus obligations. |
| Exit after SQLite generation publication | Recovery validates and uses the new generation; if validation fails, it marks it invalid and uses the prior valid generation plus obligations. |
| Old query remains pinned while new generation publishes and GC runs | Old generation, obligations, versions, and tombstones it needs remain. |
| Update/delete followed by old and new snapshot searches | Each result uses its snapshot's version intervals; current-latest state cannot overwrite history. |

Fault injection uses `os._exit(86)` at the named boundaries, reopens the
database and files, and checks the fixed snapshot, visible version-key,
obligation-kind, raw/group-count, and selected-generation expectations listed
above.  These lifecycle tests complement, but do not themselves invoke, the
separate mathematical search oracle.  A clean shutdown/restart alone is not
labeled a crash test.

## 10. Operations and explicit fallbacks

The reference store offers operations equivalent to:

```text
commit(operations)
search(q, k, beta, snapshot=None)
group_pending()
build_generation()
recover()
```

`commit` acknowledges only after the SQLite transaction is durable under the
configured mode.  `search` defaults to a newly captured snapshot, but an
explicit retained snapshot is immutable.  `group_pending` and
`build_generation` use the publication protocols above.  `recover` validates
artifacts and reconstructs raw/group/generation eligibility from durable state,
not from a best-effort memory cache.

Safe fallbacks are part of the design:

- unknown/invalid group bound: scan that group's visible members;
- missing compatible generation: exact scan retained visible versions;
- fewer than `k` total visible candidates: return all and mark the kth-distance
  certificate not applicable;
- failed checksum or numeric precondition: do not use the artifact for a
  certified response;
- expired snapshot: explicit error, never reinterpret it as the latest
  snapshot.

## 11. Receipt and audit surface

Every search receipt contains at least:

```text
snapshot_id, snapshot_seq, base_generation, covered_commit_seq,
group_catalog_revision, delta_view_hash, candidate_set_hash, metric, k, requested_beta,
certified_beta_fraction, certified_beta_display, certificate_status,
tau_returned, tau_exact_upper, min_skipped_lb,
raw_pending_scanned, groups_scanned, groups_skipped, vectors_scanned,
visibility_rejections, fallback_reason, numeric_mode,
component_timings, config_hash, dataset_or_store_hash
```

`delta_view_hash` is a deterministic SHA-256 over the pinned snapshot,
generation coverage, catalog revision, raw/group ownership, vector-version
intervals and bits, centers, and radius bits.  A group payload that fails
validation is scanned raw exactly once per version; the receipt's fallback
reason and certificate details record affected group/vector counts and stable
failure categories.  Audit mode additionally records group IDs, exact version
keys, rational bound endpoints, and skip decisions.  A receipt is
reproducibility evidence tied to checksummed local data; it is not a
self-authenticating or cryptographic proof.

Component timers are disjoint wall-clock intervals inside one search call.  In
the reference, `candidate_delta_distance_ns` covers materialization and one
bulk distance pass over frozen `C` plus visible Delta; that value is not copied
into raw and group timer fields.  In pruning, `candidate_raw_distance_ns`,
lower-bound calculation, group ordering, each scanned-group bulk pass, final
merge/refinement, and receipt serialization are non-overlapping phases.
`base_search_ns` is provenance copied from creation of frozen `C`; it occurred
before a coupled Delta call and must not be added to that call's micro latency.
Final merge carries the already computed estimate/lower/upper interval row for
each surviving logical ID into exact boundary ranking; it does not perform a
second bulk distance pass.  Matrix packing and exact boundary refinement belong
only to `merge_ns`.  The benchmark-only
`group_pruning_beta0_recompute_intervals_ablation` deliberately restores that
second pass, with identical pruning decisions and certified ranking, so the
interval-carry optimization can be isolated from sphere pruning.  Its repeated
pass is recorded in the disjoint `final_interval_recompute_ns` component and is
contained in the method wall, but not `merge_ns`.  `receipt_ns` includes receipt allocation and one complete
`to_dict` serialization preview, excluding only the final immutable
timing-field replacement.  Benchmark JSON serialization after the call is not
included in query latency, and LB audit collection is a separate non-timed run.

## 12. Non-goals and claims boundary

The prototype does not implement distributed transactions, SQL query support,
GPU search, Faiss internals, production concurrent MVCC, a custom WAL, or
unbounded historical snapshots.  Search-kernel timing is reported separately
from the durable store.  HNSW add/build timing does not demonstrate ACID commit
latency.  Unless a sustained mixed workload is actually run with equal resource
budgets, no steady-state write-throughput or backlog-stability claim is made.

The experimental contract compares pruning with a full Delta scan over the
same frozen `C`.  Base-HNSW quality and exact-all-visible recall are reported as
additional measurements, not folded into the omission certificate.
