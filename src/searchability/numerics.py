"""Conservative distance intervals and exact boundary ordering.

Bulk work stays vectorized.  Certification does not trust the point estimate:
it first encloses all floating operations with a derived gamma bound and
outward-rounded elementary operations, then re-evaluates every candidate that
can affect the top-k boundary using exact rational squared L2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

import numpy as np

from .models import SearchHit, VectorRecord, as_float32_matrix, as_float32_vector


UNIT_ROUNDOFF = 2.0 ** -53
NUMERIC_MODE = "float64_gamma_interval+fraction_boundary"


@dataclass(frozen=True, slots=True)
class DistanceIntervals:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True)
class RankedTopK:
    hits: tuple[SearchHit, ...]
    tau_lower: float | None
    tau_upper: float | None
    exact_rechecks: int


def validate_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("beta must be finite and non-negative")
    return value


def validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    value = int(k)
    if value <= 0:
        raise ValueError("k must be positive")
    return value


def _gamma(operation_count: int) -> float:
    if operation_count >= 2**53:
        raise ValueError("dimension is too large for the documented error bound")
    # gamma_n = n*u/(1-n*u) with u=2^-53.  Form it as an exact
    # fraction first; a binary64 evaluation could otherwise round downward.
    exact = Fraction(operation_count, 2**53 - operation_count)
    result = float(exact)
    if Fraction.from_float(result) < exact:
        result = math.nextafter(result, math.inf)
    return result


def distance_intervals(
    query: np.ndarray,
    vectors: np.ndarray,
    *,
    chunk_rows: int = 8192,
) -> DistanceIntervals:
    """Bound ordinary L2 distances from one stored float32 query to a matrix.

    `gamma_(d+2)` conservatively covers the subtraction error twice when its
    result is squared, one multiplication, and at most d-1 non-negative
    additions.  The divisions and square roots used to transform that analytic
    bound are then rounded outwards individually.
    """

    q = as_float32_vector(query, name="query")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)):
        raise TypeError("chunk_rows must be an integer")
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    # Validate and convert one block at a time.  Calling ``as_float32_matrix``
    # on the complete GIST matrix would otherwise materialize complete
    # float64 and float32 copies before the distance temporaries are allocated.
    # Keeping only the three O(n) interval outputs and O(chunk_rows * d)
    # temporaries makes the exhaustive exact comparator usable without changing
    # any numerical decision.
    original = np.asarray(vectors)
    if original.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if not 1 <= original.shape[1] <= 4096:
        raise ValueError("vectors dimension must be in [1, 4096]")
    if not np.issubdtype(original.dtype, np.number) or np.issubdtype(
        original.dtype, np.complexfloating
    ):
        raise TypeError("vectors must contain real numeric values")
    if original.shape[1] != q.shape[0]:
        raise ValueError("query/vector dimension mismatch")
    if original.shape[0] == 0:
        # Exercise the canonical validator even for the empty case so its
        # dtype and certified component-domain contract remains authoritative.
        as_float32_matrix(original, name="vectors")
        empty = np.empty(0, dtype=np.float64)
        return DistanceIntervals(empty, empty.copy(), empty.copy())

    gamma = _gamma(q.shape[0] + 2)
    denominator_high = math.nextafter(1.0 + gamma, math.inf)
    denominator_low = math.nextafter(1.0 - gamma, -math.inf)
    count = int(original.shape[0])
    estimate = np.empty(count, dtype=np.float64)
    lower = np.empty(count, dtype=np.float64)
    upper = np.empty(count, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    for start in range(0, count, chunk_rows):
        stop = min(count, start + chunk_rows)
        canonical = as_float32_matrix(original[start:stop], name="vectors")
        x64 = np.asarray(canonical, dtype=np.float64, order="C")
        differences = x64 - q64
        squared = differences * differences
        sums = np.sum(squared, axis=1, dtype=np.float64)
        if not np.all(np.isfinite(sums)):
            raise ValueError("distance accumulation overflowed")

        lower_sq = np.nextafter(sums / denominator_high, -np.inf)
        upper_sq = np.nextafter(sums / denominator_low, np.inf)
        lower_sq = np.maximum(lower_sq, 0.0)
        upper_sq = np.maximum(upper_sq, 0.0)
        estimate_block = np.sqrt(sums)
        lower_block = np.maximum(
            np.nextafter(np.sqrt(lower_sq), -np.inf), 0.0
        )
        upper_block = np.maximum(
            np.nextafter(np.sqrt(upper_sq), np.inf), 0.0
        )
        zeros = sums == 0.0
        lower_block[zeros] = 0.0
        upper_block[zeros] = 0.0
        estimate[start:stop] = estimate_block
        lower[start:stop] = lower_block
        upper[start:stop] = upper_block
    return DistanceIntervals(estimate=estimate, lower=lower, upper=upper)


def exact_squared_l2(left: np.ndarray, right: np.ndarray) -> Fraction:
    """Squared L2 of stored binary floats, with no rounding at all."""

    a = as_float32_vector(left, name="left")
    b = as_float32_vector(right, name="right")
    if a.shape != b.shape:
        raise ValueError("vector dimension mismatch")
    total = Fraction(0)
    for av, bv in zip(a, b):
        delta = Fraction.from_float(float(av)) - Fraction.from_float(float(bv))
        total += delta * delta
    return total


def _square_of_float(value: float) -> Fraction:
    exact = Fraction.from_float(value)
    return exact * exact


def sqrt_fraction_bracket(value: Fraction) -> tuple[float, float]:
    """Return adjacent-or-equal binary64 values proven to bracket sqrt(value)."""

    if value < 0:
        raise ValueError("cannot take square root of a negative fraction")
    if value == 0:
        return (0.0, 0.0)
    seed = math.sqrt(float(value))
    if not math.isfinite(seed):
        raise ValueError("distance is outside finite binary64 range")

    lower = seed
    for _ in range(64):
        if _square_of_float(lower) <= value:
            break
        lower = math.nextafter(lower, -math.inf)
    else:  # pragma: no cover - defensive guard against a non-conforming runtime
        raise ArithmeticError("failed to establish lower sqrt bracket")

    upper = seed
    for _ in range(64):
        if _square_of_float(upper) >= value:
            break
        upper = math.nextafter(upper, math.inf)
    else:  # pragma: no cover
        raise ArithmeticError("failed to establish upper sqrt bracket")
    return (lower, upper)


def fraction_to_lower_float(value: Fraction) -> float:
    result = float(value)
    for _ in range(8):
        if Fraction.from_float(result) <= value:
            return result
        result = math.nextafter(result, -math.inf)
    raise ArithmeticError("failed to round fraction downward")


def fraction_to_upper_float(value: Fraction) -> float:
    result = float(value)
    for _ in range(8):
        if Fraction.from_float(result) >= value:
            return result
        result = math.nextafter(result, math.inf)
    raise ArithmeticError("failed to round fraction upward")


def lower_bound_fraction(
    query: np.ndarray, center: np.ndarray, radius_upper: float
) -> tuple[Fraction, float, float]:
    """Return a proven rational lower bound plus display distance bounds."""

    if not math.isfinite(radius_upper) or radius_upper < 0.0:
        raise ValueError("radius_upper must be finite and non-negative")
    interval = distance_intervals(query, np.asarray(center, dtype=np.float32)[None, :])
    distance_lower = float(interval.lower[0])
    distance_upper = float(interval.upper[0])
    raw = Fraction.from_float(distance_lower) - Fraction.from_float(radius_upper)
    return (max(Fraction(0), raw), distance_lower, distance_upper)


def kth_upper_bound(uppers: Sequence[float] | np.ndarray, k: int) -> float:
    k = validate_k(k)
    values = np.asarray(uppers, dtype=np.float64)
    if values.size < k:
        return math.inf
    return float(np.partition(values, k - 1)[k - 1])


def stable_topk(
    query: np.ndarray,
    records: Sequence[VectorRecord],
    vectors: np.ndarray,
    sources: Sequence[str],
    k: int,
    intervals: DistanceIntervals | None = None,
) -> RankedTopK:
    """Resolve exact top-k while only exact-rechecking boundary competitors."""

    k = validate_k(k)
    if len(records) != len(sources):
        raise ValueError("records/sources length mismatch")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("records/vectors length mismatch")
    if not records:
        return RankedTopK((), None, None, 0)
    bounds = intervals or distance_intervals(query, matrix)
    take = min(k, len(records))
    cutoff = kth_upper_bound(bounds.upper, take)
    relevant = np.flatnonzero(bounds.lower <= cutoff)

    ranked: list[tuple[Fraction, int, int, int]] = []
    for index in relevant.tolist():
        record = records[index]
        ranked.append(
            (
                exact_squared_l2(query, matrix[index]),
                record.logical_id,
                record.version_id,
                index,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = ranked[:take]
    hits: list[SearchHit] = []
    for squared_distance, _, _, index in selected:
        record = records[index]
        hits.append(
            SearchHit(
                logical_id=record.logical_id,
                version_id=record.version_id,
                distance=math.sqrt(float(squared_distance)),
                source=sources[index],
            )
        )
    if len(records) < k:
        # Return every available row in exact order, but do not call the last
        # row a kth result when the contractual population is underfilled.
        tau_low = None
        tau_high = None
    else:
        tau_low, tau_high = sqrt_fraction_bracket(selected[-1][0])
        # Both are proven upper bounds.  Keeping their minimum strengthens the
        # certificate and links the final exact boundary to the monotone interval
        # statistic used by every earlier skip decision.
        tau_high = min(tau_high, kth_upper_bound(bounds.upper, take))
    return RankedTopK(tuple(hits), tau_low, tau_high, len(relevant))


def adaptive_stable_topk(
    query: np.ndarray,
    records: Sequence[VectorRecord],
    vectors: np.ndarray,
    sources: Sequence[str],
    k: int,
    intervals: DistanceIntervals | None = None,
) -> RankedTopK:
    """Resolve exact top-k order by exact-ranking only overlapping intervals.

    Exact distances lie inside their supplied ordinary-L2 intervals. Sorting
    by lower endpoint and joining transitively overlapping intervals therefore
    partitions candidates into components with a proven strict order between
    components. Singleton components need no exact arithmetic; non-singleton
    components that intersect the returned prefix use the full exact key.
    """

    k = validate_k(k)
    if len(records) != len(sources):
        raise ValueError("records/sources length mismatch")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("records/vectors length mismatch")
    if not records:
        return RankedTopK((), None, None, 0)
    bounds = intervals or distance_intervals(query, matrix)
    estimate = np.asarray(bounds.estimate, dtype=np.float64)
    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    for name, values in (
        ("estimate", estimate),
        ("lower", lower),
        ("upper", upper),
    ):
        if values.shape != (len(records),):
            raise ValueError(f"distance interval {name} shape mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"distance interval {name} must be finite")
    if (
        np.any(lower < 0.0)
        or np.any(lower > estimate)
        or np.any(estimate > upper)
    ):
        raise ValueError("distance intervals are invalid")

    take = min(k, len(records))
    cutoff = kth_upper_bound(upper, take)
    relevant = np.flatnonzero(lower <= cutoff).tolist()
    ordered = sorted(
        relevant,
        key=lambda index: (
            float(lower[index]),
            float(upper[index]),
            records[index].logical_id,
            records[index].version_id,
            index,
        ),
    )

    exact: dict[int, Fraction] = {}

    def exact_value(index: int) -> Fraction:
        value = exact.get(index)
        if value is None:
            value = exact_squared_l2(query, matrix[index])
            exact[index] = value
        return value

    components: list[list[int]] = []
    running_upper = -math.inf
    for index in ordered:
        lower_endpoint = float(lower[index])
        upper_endpoint = float(upper[index])
        if not components or lower_endpoint > running_upper:
            components.append([index])
            running_upper = upper_endpoint
        else:
            components[-1].append(index)
            running_upper = max(running_upper, upper_endpoint)

    prefix: list[int] = []
    for component in components:
        if len(prefix) >= take:
            break
        if len(component) > 1:
            component = sorted(
                component,
                key=lambda index: (
                    exact_value(index),
                    records[index].logical_id,
                    records[index].version_id,
                ),
            )
        prefix.extend(component)
    selected = prefix[:take]
    if len(selected) != take:
        raise RuntimeError("adaptive ranking underfilled top-k")

    tau_low: float | None = None
    tau_high: float | None = None
    if len(records) >= k:
        kth_exact = exact_value(selected[k - 1])
        tau_low, exact_tau_high = sqrt_fraction_bracket(kth_exact)
        tau_high = min(exact_tau_high, kth_upper_bound(upper, take))

    hits: list[SearchHit] = []
    for index in selected:
        squared = exact.get(index)
        distance = (
            float(estimate[index])
            if squared is None
            else math.sqrt(float(squared))
        )
        record = records[index]
        hits.append(
            SearchHit(
                logical_id=record.logical_id,
                version_id=record.version_id,
                distance=distance,
                source=sources[index],
            )
        )
    return RankedTopK(tuple(hits), tau_low, tau_high, len(exact))
