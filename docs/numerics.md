# Numerical specification for certified L2

This document defines the interval/error-bound path used by the certified
search.  A final `nextafter` around an otherwise ordinary floating-point
calculation is **not** sufficient: analytic error from all subtraction,
multiplication, and accumulation is bounded first, and every subsequent
division and square root is rounded in its own outward direction.

## 1. Arithmetic model and range

Vectors, queries, and centers are canonical IEEE-754 binary32 values.  The
certified kernel converts them exactly to IEEE-754 binary64 and requires
round-to-nearest, ties-to-even arithmetic.  Dimension is in `1..4096`, every
component is finite, and `abs(component) <= 1e15`.

These limits rule out exceptional intermediate values:

- `abs(q_i - x_i) <= 2e15`;
- each square is at most `4e30`;
- the exact squared norm is at most `4096 * 4e30 = 1.6384e34`;
- the ordinary distance is at most `1.28e17`, far below binary64 overflow.

The smallest non-zero difference of two binary32 numbers is at least
`2^-149`, so its square is at least `2^-298`, still much larger than the
smallest normal binary64 value `2^-1022`.  Thus the proof does not rely on
subnormal binary64 multiplication or summation.  Zero is handled separately.

Conversion from binary32 to binary64 is exact.  Subtraction is *not* assumed
exact: two binary32 operands with widely separated exponents can have an exact
difference needing more than 53 significant binary digits.  The non-zero
exact difference is, however, in the normal binary64 range, so the usual
relative rounding model applies.  The Higham envelope below covers this
subtraction error as well as squaring and accumulation.

## 2. Squared-distance enclosure

Let `u = 2^-53` be binary64 unit roundoff and

\[
\gamma_n=\frac{nu}{1-nu},\qquad n=d+2.
\]

At the largest supported dimension, `n = 4098` and
`gamma_n` is approximately `4.549693954915961e-13`, safely below one.

For exact

\[
S=\sum_{i=1}^d(q_i-x_i)^2
\]

and the binary64 bulk-kernel result `s_hat`, the certified implementation must
use an operation order covered by the standard non-negative dot-product
analysis and establish

\[
|s_{hat}-S|\leq\gamma_{d+2}S.
\]

For one coordinate, write the rounded subtraction as
`y_hat = (q_i - x_i)(1 + delta_i)`, with `abs(delta_i) <= u`.  Its rounded
square contains two occurrences of the subtraction factor and one
multiplication factor.  A term that takes the longest path through a
deterministic non-negative reduction then encounters at most `d - 1` rounded
additions.  The standard product lemma therefore gives at most
`2 + 1 + (d - 1) = d + 2` factors, and summing non-negative exact squares gives
the stated `gamma_(d+2)` relative enclosure.  Pairwise reduction has a shorter
path and is still covered.  A backend using unchecked fast-math, flushing
relevant values, an opaque reduction with a weaker error contract, or
operations outside this envelope is not the certified kernel.  FMA may reduce
the actual error but does not shrink the documented envelope.

`gamma` itself is constructed conservatively.  Its mathematical value is the
exact rational

```text
g_exact = Fraction(d + 2, 2**53 - (d + 2))
```

and `g_hi` is the smallest readily obtained binary64 value verified with
`Fraction.from_float(g_hi) >= g_exact`.  A practical construction converts the
rational to nearest binary64 and, if the exact comparison shows that it rounded
downward, applies `nextafter(..., +inf)`.  This adjustment covers only that
conversion rounding, not the distance calculation.

From the error inequality,

\[
\frac{s_{hat}}{1+\gamma}\leq S\leq
\frac{s_{hat}}{1-\gamma}.
\]

The implementation obtains binary64 squared-distance endpoints as follows:

```text
plus_hi  = nextafter(1.0 + g_hi, +inf)
minus_lo = nextafter(1.0 - g_hi, -inf)

sq_lo = max(0.0, nextafter(s_hat / plus_hi,  -inf))
sq_hi =          nextafter(s_hat / minus_lo, +inf)
```

The directions matter: `plus_hi` is an upper denominator for the lower bound,
and `minus_lo` is a lower positive denominator for the upper bound.  Each
division's own rounding is then expanded outward.  Implementations may make
the endpoints tighter only after an exact-rational containment check.

Because no non-zero term can underflow, `s_hat == 0` implies `S == 0`; return
`[0, 0]` exactly in that case.  Negative `s_hat`, a non-finite result, or a
failed containment/self-test disables pruning and selects an exact fallback.

## 3. Ordinary-distance enclosure

The squared interval is converted to ordinary L2 with monotonic square roots:

```text
dist_lo = 0.0 if sq_lo == 0.0 else nextafter(sqrt(sq_lo), -inf)
dist_hi =          nextafter(sqrt(sq_hi), +inf)
```

This requires an IEEE-754 correctly-rounded binary64 square root (or a backend
whose error is no larger than one ulp and is explicitly tested).  Each square
root rounding is expanded separately.  The lower endpoint is clamped at the
mathematical lower limit zero.  The interval postcondition is

```text
Fraction.from_float(dist_lo) <= exact L2 <= Fraction.from_float(dist_hi)
```

where the middle value is understood through its exact squared distance.  Test
code verifies containment using independent `Fraction` squared-distance
comparisons, including adjacent-float boundary cases.

Faiss `METRIC_L2` output is squared L2.  It may be used to generate base
candidates, but is never substituted for the ordinary-distance interval in a
skip decision or receipt.

## 4. Radius and lower-bound construction

For every group member `x`, compute the center-member distance interval by the
same certified primitive.  The immutable group radius is

```text
radius_hi = max(member_dist_hi)
```

and is persisted by exact binary64 bit pattern (hex text or eight bytes), not
by a lossy decimal round-trip.  Adding a member can only replace the radius by
the maximum of the old value and the new upper endpoint.  Moving a center,
shrinking a radius, or changing a member list requires a newly built immutable
group and complete radius recomputation.

For query-center interval `[center_dist_lo, center_dist_hi]`, no floating
subtraction is needed:

```text
lb_lower = max(
    Fraction(0),
    Fraction.from_float(center_dist_lo)
      - Fraction.from_float(radius_hi),
)
```

This is a lower bound because the center distance is rounded down and the
radius is rounded up.  A display float for `lb_lower` must be rounded downward;
the rational value remains authoritative for decisions and receipts.

## 5. Threshold and strict skip comparison

Every evaluated candidate has `[dist_lo, dist_hi]`.  With at least `k`
candidates, `tau_upper` is the kth smallest `dist_hi` (ties may use the stable
IDs, although the numeric endpoint is the same).  The exact decision is

```python
Fraction(lb_lower) > (
    Fraction.from_float(tau_upper) - Fraction.from_float(beta)
)
```

`lb_lower` is already a `Fraction`; the notation above emphasizes that all
three operands are rational at comparison time.  No tolerance is added.  In
particular, exact equality scans.  While candidates are only added,
`tau_upper` is non-increasing; recomputing it after each scanned group can only
make subsequent pruning more permissive.

Using upper endpoints for `tau` and a lower endpoint for `LB` is conservative.
A candidate whose interval is wide causes extra scanning, never an unjustified
skip.

## 6. Exact boundary refinement

Let `U` be the final kth upper endpoint.  Every evaluated candidate with
`dist_lo <= U` is re-evaluated independently as

```text
q_i_exact = Fraction.from_float(float(q_i))
x_i_exact = Fraction.from_float(float(x_i))
S_exact   = sum((q_i_exact - x_i_exact) ** 2)
```

where converting a binary32 to Python `float` is exact.  Exact squared
fractions, followed by `(logical_id, version_id)`, decide final order.  This
resolves ties without assuming that interval width is an epsilon.

To construct ordinary-distance endpoints from an exact squared fraction
`S_exact`:

1. Use `sqrt(float(S_exact))` only as a binary64 seed.  It is not accepted as a
   bound merely because the host square root is expected to be correctly
   rounded.
2. Convert the seed back to an exact rational square.  Move the lower candidate
   with `nextafter(..., -inf)` until `lower**2 <= S_exact`, and move the upper
   candidate with `nextafter(..., +inf)` until `upper**2 >= S_exact`.  Every
   comparison is between `Fraction` values, so the endpoints are proven rather
   than epsilon-expanded guesses.
3. Use the lower endpoint for result validation and convert the upper endpoint
   with `Fraction.from_float` for certificate arithmetic.
4. Take the minimum of the upper endpoint with the final kth interval upper
   endpoint, another proven upper bound.  This cap preserves both safety and
   the monotonic-threshold argument used by the correctness proof.

This quantity is `tau_exact_hi`; the word `exact` refers to the exact
Fraction-based squared-distance/ranking path, while `_hi` makes clear that an
irrational square root is represented by a one-sided rational endpoint.

The certified beta is computed entirely as rational arithmetic:

```text
0                                      if no group was skipped
max(Fraction(0), tau_exact_hi - min_skipped_lb) otherwise
```

If serialized as a float, it is rounded toward `+inf`; numerator and
denominator are saved so that experiments can audit `certified_beta <= beta`
without decimal ambiguity.

## 7. Limits and safe degradation

- The derivation applies only to canonical binary32 vector/center components
  in the stated range and dimension.  Other real inputs are first canonicalized
  as the documented stored binary32 object.  Complex inputs are rejected before
  conversion; their imaginary components are never silently discarded.
- The bound describes the specified bulk kernel, not arbitrary BLAS/GPU/Faiss
  distance code.  Candidate generation may use Faiss, but certified pruning
  recomputes the relevant distances here.
- A single final `nextafter`, an empirical epsilon, `numpy.isclose`, or wider
  tolerances inferred from passing tests cannot replace the analytic bound.
- Interval overlap can make exact refinement or scanning expensive.  The
  correct response is more work or an explicit non-certified mode, never a
  narrower unproved interval.
- The analysis assumes functioning IEEE arithmetic and uncorrupted inputs.
  The lifecycle layer verifies the published Base-generation manifest and the
  SHA-256/length bindings for its Faiss index, metadata, and ordinal map before
  use.  For Delta it reconstructs a pinned SQLite view, validates each loaded
  group's dimensions, bitwise member payloads, and radius coverage, and binds
  that reconstructed view (including vectors, centers, radii, and membership)
  into the Receipt with a SHA-256 digest.  This is not an independent checksum
  of every SQLite row/page and does not detect arbitrary storage or hardware
  corruption.  A supported validation failure forces the documented exact/raw
  fallback or prevents use of the affected accelerator.
