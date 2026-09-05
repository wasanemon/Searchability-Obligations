# Native kernel numerical and selection contract

This document is normative for the Issue #3 compiled F/N/P paths. It extends,
and does not weaken, [`correctness.md`](correctness.md) and
[`numerics.md`](numerics.md). The independent oracle remains
`searchability.oracle`; it does not import this kernel or its ranking helpers.

## 1. Backend and arithmetic assumptions

The certified native backend accepts the same canonical finite binary32 domain:
dimension `1..4096` and component magnitude at most `1e15`. Input arrays are
copied once into private, C-contiguous, immutable backing when the prepared view
is built. Query-time `forcecast`, per-group copies, and mutable borrowed buffers
are not part of the certified path. Packed ordinals and `(logical_id,
version_id)` arrays are bound to the packed vector bytes and group offsets.

The extension is compiled as portable C++17 with:

```text
-O3 -fno-fast-math -ffp-contract=off
```

It does not use `-ffast-math` or architecture-specific code generation. At
entry it requires the process rounding mode to be `FE_TONEAREST`. Every stored
binary32 component converts exactly to binary64. The kernel accumulates each
squared distance as the specified scalar sequence:

```text
sum = 0
for i in 0..d-1:
    difference = double(q[i]) - double(x[i])
    square = difference * difference
    sum = sum + square
```

Contraction is disabled. Reassociation, approximate reciprocal/square root,
unchecked fast math, and a float32 accumulator are forbidden. The input bounds
ensure every nonzero binary64 product is normal, so the proof does not depend
on subnormal multiplication or summation. A non-default rounding mode,
non-finite intermediate, or failed self-test prevents a certified result.

The existing conservative `gamma_(d+2)` enclosure covers two appearances of
the rounded subtraction, one multiplication, and at most `d-1` sequential
non-negative additions. Thus the native squared and ordinary-distance interval
construction uses exactly the outward formulas in `numerics.md` Sections 2–3.
Changing the compiler flags, reduction, SIMD, FMA, or scalar type requires a
new derivation, build identity, and test run; a faster unproved variant must be
labelled non-certified.

## 2. Packed immutable view

One snapshot-specific prepared view stores:

```text
Delta vectors = [all visible raw | group 0 | ... | group n-1]
logical IDs, version IDs, source/group ordinals
group offsets, stable group IDs, centers, radius upper endpoints
```

Groups are normalized by `group_id`; visible raw rows and visible members
within each group are normalized by `(logical_id, version_id)`. Groups with no
visible member are omitted from the native packed directory, because they have
no distance or pruning decision to contribute. Remaining group offsets are
absolute packed-row offsets and may repeat only if a low-level caller directly
constructs an empty native group. The packed hash binds snapshot and exact
vector/key/source/center/radius/offset bytes. The prepared-query binding
separately binds Base generation/universe and frozen C; a new immutable
`SearchEngine` is constructed for a new Delta/catalog revision. Source mutation
cannot change the bytes-backed view.

Raw pending rows precede every group row and are always scanned. Before P may
skip, the wrapper proves that the frozen C plus visible Delta contains one
version per logical ID. A duplicate or handoff overlap is resolved before any
threshold is formed and dispatched to native F/no-skip; replacement after a
candidate has affected the threshold is forbidden because it would invalidate
monotonicity.

## 3. Shared F/N/P distance and incremental threshold

- **F** scans frozen C and the entire contiguous packed Delta. It does not pay
  for group lower bounds or ordering.
- **N** computes and orders group lower bounds exactly as P does but disables
  skipping and scans every group.
- **P** applies the certified strict rule and may omit whole groups.

All three use the same native distance interval and final ranking contract.
The kernel first scans C and raw Delta. A size-`k` max heap contains the `k`
smallest distance upper endpoints seen so far. Each insertion is `O(log k)`;
the heap root is the current `tau_upper`. No partition of a growing all-candidate
array is performed. With fewer than `k` evaluated candidates, tau is absent and
P scans the next group.

Adding candidates cannot increase the kth upper endpoint. This is the same
monotonicity lemma proved in `correctness.md`; no candidate is removed after
heap insertion.

## 4. Conservative native skip comparison

For a certified query-center interval and persisted radius upper endpoint, the
native lower display value is

```text
lb_lo = max(0, nextafter(center_distance_lo - radius_hi, -infinity))
```

The extra step toward negative infinity makes `lb_lo` no greater than the exact
difference of the two binary64 operands, including when the hardware
subtraction rounded upward. For a present threshold,

```text
threshold_hi = nextafter(tau_upper - beta, +infinity)
skip iff lb_lo > threshold_hi
```

The extra upward step makes `threshold_hi` no smaller than the exact dyadic
subtraction. Therefore

```text
lb_lo > threshold_hi
implies exact_LB >= lb_lo > exact(tau_upper - beta),
```

which implies the normative rational strict rule. These widenings can cause an
extra scan but cannot cause a false skip. Equality, either before or after
widening, scans. `tau_upper`, radius, LB and beta are ordinary L2; Faiss's
squared-L2 output is never used here.

Groups are processed in stable `(lb_lo, group_id)` order. For audit mode the
kernel records the decision-time tau, conservative threshold, action, and
logical member count. Audit construction is outside timing; the minimum
production Receipt remains inside it.

## 5. Survivor retention and adaptive exact order

The heap alone is not a result set. After scanning, let `U` be the final kth
upper endpoint. The kernel retains every evaluated candidate satisfying

```text
distance_lower <= U
```

including equality. A candidate with a larger upper endpoint may still have a
smaller exact distance, so retaining only heap members, estimates, or a fixed
top-k shortlist is forbidden. The kernel returns only this necessary boundary
set and its intervals; widespread overlap is an explicit safe large fallback.

The production wrapper sorts boundary rows by `(lower, upper, logical_id,
version_id)` and forms transitive overlap components. A component continues
while

```text
next.lower <= maximum upper endpoint already in the component.
```

Equality joins the component because exact equality may require identifier
tie-breaking. Strictly separated components have a proven global order.
Singleton components can therefore be emitted without Fraction work. Every
non-singleton component intersecting the first `k` ranks is exactly evaluated
from binary32 components and sorted by

```text
(exact squared distance, logical_id, version_id).
```

Components wholly after the selected prefix need no exact work. The kth row is
exactly evaluated when needed for ordering or a tighter certificate; otherwise
its native upper endpoint remains a proven tau upper bound. Counters separately
record boundary candidates and exact/Fraction evaluations. The independent
oracle is still exhaustive and shares none of this selection logic.

If total unique population is below `k`, all available candidates are returned
in exact order, `tau` and certified beta are absent, and status is
`NOT_APPLICABLE_INSUFFICIENT_POPULATION`. The last returned element is not
misreported as a kth result.

## 6. Certificate

If P skips no group, certified beta is zero. Otherwise let `L_min` be the
minimum conservative `lb_lo` of skipped groups and let `T_final` be the final
proven kth upper endpoint. Python converts both binary64 values to their exact
dyadic `Fraction`s and computes

```text
gap = max(0, Fraction(T_final) - Fraction(L_min)).
```

The skip implication above and threshold monotonicity prove `gap < beta` when
positive. The wrapper checks this relation fail-closed. Its authoritative
numerator/denominator are stored; an outward float is display only. If display
rounding would exceed the requested binary64 beta, the requested beta itself is
returned as the safe public upper bound only after the exact Fraction check.
For beta zero, every omitted row is strictly beyond a valid kth upper endpoint,
so the adaptive exact ordering equals F/O including ID/version ties.

## 7. A and O distinctions

**O** is the old independent same-C reference plus exhaustive Fraction oracle;
it is correctness evidence, not the primary performance baseline. **A** uses a
single Faiss Delta Flat scan and a finite returned shortlist plus frozen C. It
uses the same overlap-component adaptive ordering rule for that small merge:
strictly separated singleton intervals avoid Fraction work, overlapping
components intersecting the prefix are exact-ranked, and the kth row is exact
when a kth result exists. The raw evidence records A's exact-recheck count.
The absent Delta rows were not certifiably excluded, so A remains a practical
performance/quality comparator and is never labelled the same guarantee as
F/N/P. Any A mismatch remains in latency and quality aggregates; it is not
dropped when selecting a winner.

## 8. Identity and reported counters

Every run records the implementation-tree hash, loaded extension SHA-256,
module path, and the current native-source SHA-256. `setup.py` hashes the C++
source before compilation and embeds that digest; `native_build_info()` rejects
a loaded extension whose embedded digest differs from the current source.
The build also embeds the compiler driver and flag prefix selected by
setuptools: `compiler_cxx[0]`, `compiler_so[1:]`, and the complete
`Extension.extra_compile_args`. It labels this as a driver/default/explicit
flag prefix, not as the later per-source command containing generated `-I`,
`-D`, `-c`, and `-o` arguments. The compiler's own `__VERSION__` fingerprint,
exact explicit flags, Python/NumPy versions, CPU, thread environment,
rounding-mode check, backend name, and native call counter are also recorded.
A required-native command fails if import/build/source identity fails or if its
call counter does not increase; it never silently passes on a Python fallback.

Reported vector/byte counts are logical rows and payload bytes presented to the
kernel. They are not hardware cache-line, DRAM-traffic, or performance-counter
measurements. Allocation/copy fields include only quantities the implementation
can account for directly; unavailable hardware measures are `N/A`, not zero.
