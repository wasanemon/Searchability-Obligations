# Correctness contract

This document is normative for the certified search path.  It states the
mathematical contract independently of Faiss's search quality.  Numerical
construction of the intervals used below is specified in
[`numerics.md`](numerics.md), and snapshot/generation ownership is specified in
[`design.md`](design.md).

## 1. Scope and accepted inputs

The certified metric is ordinary Euclidean distance

\[
d(q,x)=\lVert q-x\rVert_2,
\]

not the squared-L2 number returned by a Faiss L2 index.  Squared distance may
be used internally for exact ordering, but `radius`, `LB`, `tau`, and `beta`
are all in ordinary-L2 units.

The public certified path accepts vectors and group centers whose canonical
representation is IEEE-754 binary32 (`float32`), with all of the following
conditions:

- dimension `d` is an integer in `1..4096` and is identical for every vector;
- every component is finite and has absolute value at most `1e15` after
  canonicalization to binary32;
- `k` is an integer greater than zero;
- `beta` is a finite, non-negative binary64 value.  Its contractual value is
  the exact rational `Fraction.from_float(beta)`, and its bit pattern/hex form
  is recorded in the receipt;
- logical and version identifiers have a stable total order and a version is
  associated with exactly one logical identifier.

NaN, infinity, a complex-valued vector/matrix, a dimension mismatch, an
out-of-range component, negative `beta`, and an invalid `k` are rejected before
any certified result is returned.  In particular, a complex-to-real cast that
would discard the imaginary component is forbidden.  Silently clipping or
replacing such values is forbidden.  A
non-certified adapter may convert other input formats, but its conversion is
outside this contract and the resulting canonical binary32 values must be
recorded.

## 2. Snapshot-relative reference result

Fix one query `q`, MVCC snapshot `s`, and one pinned, immutable base generation
`g`.  The snapshot and generation do not change during a search.  Define:

- `B_s,g`: versions stored in `g` and visible at `s`;
- `Delta_s,g`: versions visible at `s` whose effects are not included in `g`;
  this includes raw, not-yet-grouped versions;
- `C(q,s,g)`: the set of base ANN candidates after snapshot visibility
  filtering, `(logical_id, version_id)` deduplication, and any candidate
  replenishment;
- `TopK(A)`: the first `k` members of `A` ordered by
  `(exact_distance, logical_id, version_id)`;
- `R_ref = TopK(C union Delta_s,g)`;
- `R_prop = TopK(C union S)`, where `S` is precisely the Delta subset actually
  distance-evaluated by the proposed search.

`C` is materialized once and frozen.  A correctness comparison must pass that
same collection, including the same versions, to the full-Delta reference and
the proposed path.  Running HNSW twice is not an acceptable way to reconstruct
`C`.  Faiss ordinals are translated to version keys before freezing it.  Its
content hash binds the ordered version keys and vector bits to the canonical
query hash, snapshot ID, base generation ID, full base-universe hash, and
requested candidate count.  A production search validates all those fields and
verifies every supplied version against its immutable base before using an
externally supplied `C`; a caller-provided hash string is not trusted alone.

When the relevant set has at least `k` elements, let `tau_ref` and `tau_prop`
be the exact ordinary-L2 distance of the kth member of `R_ref` and `R_prop`.
The promised result is

\[
0 \leq \tau_{prop}-\tau_{ref}\leq \beta.
\]

For `beta = 0`, the stronger result is exact equality of the ordered `TopK`
lists, including the fixed identifier tie-break.  For `beta > 0`, only the kth
distance is guaranteed.  The IDs, recall, all ranks' distances, and elapsed
freshness are measured properties, not consequences of this theorem.

## 3. Certified group rule

Each immutable non-empty group `G` has a fixed canonical binary32 center `c`,
member version keys, and a stored radius upper endpoint `r_hi` satisfying

\[
\forall x\in G,\quad d(c,x)\leq r_{hi}.
\]

At construction/load time, every contiguous matrix row must bit-match the
corresponding member record (including the sign bit of zero), version keys may
not repeat within or across groups, and group IDs must be unique.  Radius
validation happens only after those ownership checks, so a radius cannot
certify one matrix while search returns records from another.

For a query-center distance interval `[dc_lo, dc_hi]`, define the rational
lower bound

\[
L(G)=\max(0,
  \operatorname{Fraction}(dc_{lo})-
  \operatorname{Fraction}(r_{hi})).
\]

Here `Fraction(z)` means the exact rational represented by the binary64 value
`z`; the subtraction itself is not rounded.  The triangle inequality gives,
for every member `x`,

\[
d(q,x)\geq d(q,c)-d(c,x)\geq L(G).
\]

For every evaluated candidate the distance kernel supplies a certified upper
endpoint.  If at least `k` candidates have been evaluated, `T` is the kth
smallest of those upper endpoints.  Its comparison value is likewise converted
with `Fraction.from_float`.  The only certified skip rule is

```text
skip G iff Fraction(LB_lower(G)) > Fraction(tau_upper) - Fraction(beta)
```

where `LB_lower(G)` is already the exact rational `L(G)` above.  The inequality
is deliberately strict.  Equality at `LB = tau - beta` scans the group.  An
epsilon-only comparison, a plain-float subtraction, a squared-L2/ordinary-L2
mix, or `>=` is not certified.

If fewer than `k` candidates have been evaluated, `T` is absent rather than
floating-point infinity and no group may be skipped.  Groups are considered in
nondecreasing `(L(G), group_id)` order; this ordering is useful for pruning but
is not needed by the theorem.  Empty catalog entries are ignored and cannot
produce a skip certificate.

## 4. Search procedure and invariants

The certified procedure has this order:

1. Validate the numeric contract and pin `(s, g, catalog_revision)`.
2. Resolve visibility, remove stale/duplicate versions, replenish base ANN
   candidates as configured, and freeze `C` and its hash.
3. Evaluate all visible raw-pending Delta versions.  They are never pruned.
4. Compute every published group's certified `L(G)` and establish a stable
   order.  A group with invalid metadata or an uncertifiable bound is scanned.
5. For each group, compute the current kth upper endpoint `T`.  Apply the exact
   rational, strict rule above; otherwise evaluate every visible member.
6. Resolve the final exact boundary as described in Section 5, then form the
   receipt and certificate.

Only candidates are added after `C` is frozen.  Snapshot filtering, version
deduplication, and delete handling occur before a value can affect `T`; a
candidate that helped establish `T` is never invalidated later.

**Lemma (upper-threshold monotonicity).**  Let `U_k(E)` be the kth order
statistic of candidate distance upper endpoints.  If `E` is extended only by
insertion, then `U_k(E)` cannot increase.

**Proof.**  The previous `k` endpoints remain in the extended multiset.  Its
kth smallest endpoint is therefore no greater than the previous kth smallest
endpoint.  This argument also covers equal endpoints.  QED.

Consequently, a skip made using an earlier `T` remains valid when later scans
lower `T`.  This is why delete/visibility post-processing is forbidden: removal
can increase the kth threshold and invalidates this monotonicity argument.

## 5. Exact boundary ordering

Interval endpoints alone are not used to decide an ambiguous final rank.  Let
`E = C union S` be all evaluated candidates, and let `U` be the kth smallest
distance upper endpoint in `E`.  Form

```text
A = {x in E : distance_lower(x) <= U}
```

with exact rational endpoint comparison.  For every member of `A`, a separate
exact routine converts every canonical binary32 component with
`Fraction.from_float`, sums `(q_i - x_i)^2` as `Fraction`, and obtains an exact
squared distance.  Sort `A` by

```text
(exact_squared_distance, logical_id, version_id)
```

and take the first `k`.

This produces `TopK(E)`: at least `k` candidates have an upper endpoint at most
`U`, while any excluded candidate has exact distance strictly greater than
`U`.  Squared-distance ordering is identical to ordinary-distance ordering for
non-negative distances.  Including `distance_lower == U` is necessary to
preserve identifier tie-breaking.  If interval overlap is widespread this
step may evaluate all of `E`; that is a safe performance fallback, not a
correctness failure.

For certification, compute an outward upper rational `T_exact_hi` for the
ordinary distance of the exact kth result.  The exact-square-root procedure is
defined in `numerics.md`.  It is also capped by the final pre-refinement `U`,
which is independently a valid upper bound.  Thus `T_exact_hi` is no larger
than any threshold used to skip a group.

## 6. Main theorem and returned certificate

Let `L_min` be the minimum rational `L(G)` among skipped groups.  If no group
was skipped, define `certified_beta = 0`.  Otherwise return the exact rational

\[
b_{cert}=\max(0,T_{exact\_hi}-L_{min}).
\]

The receipt may additionally contain a binary64 display value, but it is
rounded upward and the rational numerator/denominator remain authoritative.

**Theorem (impact-bounded omission).**  Under the input, snapshot, interval,
radius, and ordering assumptions in this document, the certified search with
at least `k` reference candidates satisfies

\[
0\leq\tau_{prop}-\tau_{ref}\leq b_{cert}\leq\beta.
\]

With no skipped group the difference is zero.  If `beta > 0` and at least one
group is skipped, the final inequality is strict; if `beta = 0`, both
`b_cert` and the distance difference are zero.  Hence the public non-strict
contract `tau_prop - tau_ref <= beta` always holds.

**Proof.**  `C union S` is a subset of `C union Delta_s,g`, so adding omitted
candidates cannot increase a kth order statistic and
`tau_prop >= tau_ref`.  Every omitted candidate belongs to a skipped group and
has distance at least `L_min`.  If `L_min > tau_prop`, no omitted candidate can
alter `TopK(C union S)`, so the difference is zero.  Otherwise, the kth result
after adding omitted candidates cannot be below `L_min`: if `k` evaluated
candidates were below `L_min`, then `tau_prop < L_min`; if fewer were, a kth
result using an omitted candidate is at least `L_min`.  Therefore

\[
\tau_{prop}-\tau_{ref}
\leq \max(0,\tau_{prop}-L_{min})
\leq \max(0,T_{exact\_hi}-L_{min})=b_{cert}.
\]

For each skipped group `i`, strict skipping gave
`L_i > T_i - beta`.  Threshold monotonicity and the cap in Section 5 give
`T_exact_hi <= T_i`, hence
`T_exact_hi - L_i < beta`.  This remains true for the minimum `L_i`, proving
`b_cert < beta` when positive.  QED.

For `beta = 0`, every omitted candidate is strictly farther than a valid upper
bound on the evaluated kth distance.  It cannot change membership or a tie at
the boundary.  Section 5 then applies the same exact distance and identifier
ordering as the reference, so the complete ordered result is identical.

The certificate is a safe upper bound, not the observed error.  Experiments
compute `beta_observed = tau_prop - tau_ref` independently and must check a
safe comparison equivalent to

```text
0 <= beta_observed <= certified_beta <= requested_beta
```

without rounding an irrational distance difference downward.

## 7. Independent oracle

The small-case oracle is a separate direct implementation, not a call into the
production distance, grouping, pruning, merge, or exact-boundary routines.  It
receives the frozen `C` and the full visible `Delta_s,g`, converts every
binary32 component via `Fraction.from_float`, exhaustively sums exact squared
L2 distances, and sorts every candidate by
`(exact_squared_distance, logical_id, version_id)`.

The oracle exposes exact squared distances and rigorously bracketed ordinary
distances.  It validates the production result and observed gap.  Snapshot-set
construction is checked separately by comparing explicit version keys; doing
so prevents a correct distance enumeration from hiding a lifecycle omission.
Faiss float32 distances and Faiss's unspecified equal-distance ordering are
never the oracle.

## 8. Fallback and exceptional cases

- If `|C union Delta_s,g| < k`, scan and exactly order all available candidates,
  return all of them, and set the certificate status to
  `NOT_APPLICABLE_INSUFFICIENT_POPULATION`; no kth-distance statement is made.
- If HNSW visibility filtering underfills `C`, replenish before freezing it.
  Enumerate the complete visible base universe, exactly rank all missing
  versions, and supplement to the requested count.  If fewer visible base
  versions exist, `C` contains all of them.  Both cases record the fallback
  reason, visible count, and supplemented count; an unexplained underfilled ANN
  result is never treated as the configured `C`.
- If a distance interval, group radius, or catalog record cannot be certified,
  scan the affected group/version.  If uncertainty is global, use exact
  enumeration.  Never skip on an invalid bound.
- If a requested retained snapshot has no compatible generation, use the
  retained-version exact fallback.  If the snapshot has expired under the
  declared retention policy, reject it explicitly.
- Empty Delta is valid.  A singleton, zero radius, `LB = 0`, `LB = T`,
  `LB = T - beta`, and `beta > T` all use the same strict rule; no special case
  may weaken it.
- Exact duplicate version keys are deduplicated before ranking.  Distinct
  versions are filtered by snapshot visibility; a valid store exposes at most
  one visible version per logical ID at a snapshot.

## 9. What is not guaranteed

This contract does not guarantee global exact nearest neighbours, base-HNSW
recall, equivalence to rebuilding HNSW through the query snapshot, any returned
ID overlap for positive `beta`, wall-clock staleness, latency, throughput, or
cryptographic verifiability of a receipt.  It also does not cover a mutable
generation searched concurrently with writes, centers or vectors outside the
numeric contract, corrupted persistent data that fails integrity checks,
hardware faults, non-Euclidean metrics, or a fast path labeled
`non_certified`.  These cases must be rejected, safely enumerated, or reported
without the `CERTIFIED` status.
