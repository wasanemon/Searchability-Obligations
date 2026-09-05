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

from .models import SearchHit, VectorRecord, as_float32_vector


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


def distance_intervals(query: np.ndarray, vectors: np.ndarray) -> DistanceIntervals:
    """Bound ordinary L2 distances from one stored float32 query to a matrix.

    `gamma_(d+2)` conservatively covers the subtraction error twice when its
    result is squared, one multiplication, and at most d-1 non-negative
    additions.  The divisions and square roots used to transform that analytic
    bound are then rounded outwards individually.
    """

    q = as_float32_vector(query, name="query")
    original = np.asarray(vectors)
    if original.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if original.shape[1] != q.shape[0]:
        raise ValueError("query/vector dimension mismatch")
    if original.shape[0] == 0:
        empty = np.empty(0, dtype=np.float64)
        return DistanceIntervals(empty, empty.copy(), empty.copy())
    original64 = np.asarray(original, dtype=np.float64, order="C")
    if not np.all(np.isfinite(original64)):
        raise ValueError("vectors contain NaN or infinity")
    if np.any(np.abs(original64) > 1.0e15):
        raise ValueError("vector component exceeds certified magnitude limit")
    canonical = np.asarray(original64, dtype=np.float32, order="C")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("vectors overflow float32 storage")
    x64 = np.asarray(canonical, dtype=np.float64, order="C")

    q64 = np.asarray(q, dtype=np.float64)
    differences = x64 - q64
    squared = differences * differences
    sums = np.sum(squared, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(sums)):
        raise ValueError("distance accumulation overflowed")

    gamma = _gamma(q.shape[0] + 2)
    denominator_high = math.nextafter(1.0 + gamma, math.inf)
    denominator_low = math.nextafter(1.0 - gamma, -math.inf)
    lower_sq = np.nextafter(sums / denominator_high, -np.inf)
    upper_sq = np.nextafter(sums / denominator_low, np.inf)
    lower_sq = np.maximum(lower_sq, 0.0)
    upper_sq = np.maximum(upper_sq, 0.0)

    estimate = np.sqrt(sums)
    lower = np.nextafter(np.sqrt(lower_sq), -np.inf)
    upper = np.nextafter(np.sqrt(upper_sq), np.inf)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    zeros = sums == 0.0
    lower[zeros] = 0.0
    upper[zeros] = 0.0
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
    tau_low, tau_high = sqrt_fraction_bracket(selected[-1][0])
    # Both are proven upper bounds.  Keeping their minimum strengthens the
    # certificate and links the final exact boundary to the monotone interval
    # statistic used by every earlier skip decision.
    tau_high = min(tau_high, kth_upper_bound(bounds.upper, take))
    return RankedTopK(tuple(hits), tau_low, tau_high, len(relevant))
