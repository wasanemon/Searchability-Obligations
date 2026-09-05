"""Independent exact oracle for small correctness experiments.

The production search and numerical routines are deliberately not imported in
this module.  Every distance is exhaustively accumulated as a
``fractions.Fraction`` from the *stored float32 values*.  This makes the oracle
slow, but keeps it a useful independent check of interval and pruning code.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Sequence

import numpy as np

from .models import CandidateSet, VectorRecord


_MAX_DIMENSION = 4096
_MAX_ABS_COMPONENT = 1.0e15


@dataclass(frozen=True, slots=True)
class OracleHit:
    """One exactly ranked result.

    ``squared_distance`` is exact.  The ordinary-L2 distance generally is not
    rational, so it is represented by the proven enclosing binary64 pair.
    """

    logical_id: int
    version_id: int
    squared_distance: Fraction
    distance_lower: float
    distance_upper: float
    source: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.logical_id, self.version_id)


@dataclass(frozen=True, slots=True)
class OracleResult:
    """Exact ordering and a proven bracket for its k-th ordinary-L2 distance."""

    hits: tuple[OracleHit, ...]
    population: int
    requested_k: int

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(hit.key for hit in self.hits)

    @property
    def tau_squared(self) -> Fraction | None:
        return None if not self.hits else self.hits[-1].squared_distance

    @property
    def tau_lower(self) -> float | None:
        return None if not self.hits else self.hits[-1].distance_lower

    @property
    def tau_upper(self) -> float | None:
        return None if not self.hits else self.hits[-1].distance_upper


@dataclass(frozen=True, slots=True)
class GapBracket:
    """Outward-rounded bounds on ``tau_proposed - tau_reference``."""

    lower: float
    upper: float


def _stored_float32_vector(value: object, *, name: str) -> np.ndarray:
    """Independently apply the repository's stored-vector input contract."""

    original = np.asarray(value)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 1 <= original.shape[0] <= _MAX_DIMENSION:
        raise ValueError(f"{name} dimension is outside the oracle range")
    if not np.issubdtype(original.dtype, np.number) or np.issubdtype(
        original.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain numeric values")
    as_float64 = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(as_float64)):
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(np.abs(as_float64) > _MAX_ABS_COMPONENT):
        raise ValueError(f"{name} component magnitude exceeds the oracle range")
    stored = np.array(as_float64, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(stored)):
        raise ValueError(f"{name} overflows float32 storage")
    stored.setflags(write=False)
    return stored


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    value = int(k)
    if value <= 0:
        raise ValueError("k must be positive")
    return value


def exact_squared_l2(left: object, right: object) -> Fraction:
    """Exhaustively compute squared L2 from independently stored float32 values."""

    a = _stored_float32_vector(left, name="left")
    b = _stored_float32_vector(right, name="right")
    if a.shape != b.shape:
        raise ValueError("vector dimension mismatch")
    total = Fraction(0)
    for left_value, right_value in zip(a, b):
        left_exact = Fraction.from_float(float(left_value))
        right_exact = Fraction.from_float(float(right_value))
        difference = left_exact - right_exact
        total += difference * difference
    return total


def _square(value: float) -> Fraction:
    as_fraction = Fraction.from_float(value)
    return as_fraction * as_fraction


def sqrt_bracket(value: Fraction) -> tuple[float, float]:
    """Return binary64 values proven to enclose the non-negative square root."""

    if not isinstance(value, Fraction):
        value = Fraction(value)
    if value < 0:
        raise ValueError("cannot take the square root of a negative value")
    if value == 0:
        return (0.0, 0.0)
    approximate = math.sqrt(float(value))
    if not math.isfinite(approximate):
        raise ValueError("square root is outside the finite binary64 range")

    lower = approximate
    while _square(lower) > value:
        lower = math.nextafter(lower, -math.inf)

    upper = approximate
    while _square(upper) < value:
        upper = math.nextafter(upper, math.inf)
    return (lower, upper)


def _fraction_to_lower_float(value: Fraction) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("value is outside the finite binary64 range")
    while Fraction.from_float(converted) > value:
        converted = math.nextafter(converted, -math.inf)
    return converted


def _fraction_to_upper_float(value: Fraction) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("value is outside the finite binary64 range")
    while Fraction.from_float(converted) < value:
        converted = math.nextafter(converted, math.inf)
    return converted


def exact_topk(
    query: object,
    records: Sequence[VectorRecord],
    *,
    k: int,
    sources: Sequence[str] | None = None,
) -> OracleResult:
    """Rank every supplied record by ``(squared distance, logical ID, version ID)``."""

    stored_query = _stored_float32_vector(query, name="query")
    requested_k = _validate_k(k)
    source_values = tuple("oracle" for _ in records) if sources is None else tuple(sources)
    if len(source_values) != len(records):
        raise ValueError("records/sources length mismatch")

    ranked: list[tuple[Fraction, int, int, int, str]] = []
    for position, (record, source) in enumerate(zip(records, source_values)):
        squared = exact_squared_l2(stored_query, record.vector)
        ranked.append(
            (squared, record.logical_id, record.version_id, position, str(source))
        )
    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    selected: list[OracleHit] = []
    for squared, logical_id, version_id, _, source in ranked[:requested_k]:
        lower, upper = sqrt_bracket(squared)
        selected.append(
            OracleHit(
                logical_id=logical_id,
                version_id=version_id,
                squared_distance=squared,
                distance_lower=lower,
                distance_upper=upper,
                source=source,
            )
        )
    return OracleResult(tuple(selected), len(records), requested_k)


def _visible_at(record: VectorRecord, snapshot_id: int) -> bool:
    if snapshot_id < 0:
        raise ValueError("snapshot_id must be non-negative")
    return record.begin_seq <= snapshot_id and (
        record.end_seq is None or snapshot_id < record.end_seq
    )


def _deduplicate_newest(
    rows: Iterable[tuple[VectorRecord, str]],
) -> tuple[tuple[VectorRecord, ...], tuple[str, ...]]:
    """Apply logical-ID set semantics without calling the search implementation."""

    selected: dict[int, tuple[VectorRecord, str, int]] = {}
    for position, (record, source) in enumerate(rows):
        current = selected.get(record.logical_id)
        rank = (record.begin_seq, record.version_id, position)
        if current is None or rank > (
            current[0].begin_seq,
            current[0].version_id,
            current[2],
        ):
            selected[record.logical_id] = (record, source, position)
    ordered = sorted(selected.values(), key=lambda item: item[2])
    return (
        tuple(item[0] for item in ordered),
        tuple(item[1] for item in ordered),
    )


def reference_topk(
    query: object,
    candidates: CandidateSet | Sequence[VectorRecord],
    *,
    raw_records: Sequence[VectorRecord] = (),
    grouped_records: Sequence[VectorRecord] = (),
    snapshot_id: int = 0,
    k: int,
) -> OracleResult:
    """Compute ``TopK(C union visible Delta)`` by independent exact enumeration.

    ``candidates`` is the already frozen, visibility-filtered set ``C``.  Delta
    visibility and logical-ID de-duplication are evaluated locally.  Callers
    should pass the exact same ``CandidateSet`` object used by both search
    methods when checking the same-C guarantee.
    """

    if snapshot_id < 0:
        raise ValueError("snapshot_id must be non-negative")
    candidate_records = (
        tuple(candidates.records)
        if isinstance(candidates, CandidateSet)
        else tuple(candidates)
    )
    rows: list[tuple[VectorRecord, str]] = [
        (record, "base") for record in candidate_records
    ]
    rows.extend(
        (record, "raw")
        for record in raw_records
        if _visible_at(record, snapshot_id)
    )
    rows.extend(
        (record, "group")
        for record in grouped_records
        if _visible_at(record, snapshot_id)
    )
    records, sources = _deduplicate_newest(rows)
    return exact_topk(query, records, k=k, sources=sources)


def observed_gap_bracket(
    proposed_tau_squared: Fraction, reference_tau_squared: Fraction
) -> GapBracket:
    """Independently enclose the observed ordinary-L2 k-th-distance gap."""

    if proposed_tau_squared < 0 or reference_tau_squared < 0:
        raise ValueError("squared distances must be non-negative")
    if proposed_tau_squared == reference_tau_squared:
        return GapBracket(0.0, 0.0)
    proposed_lower, proposed_upper = sqrt_bracket(proposed_tau_squared)
    reference_lower, reference_upper = sqrt_bracket(reference_tau_squared)
    lower_exact = Fraction.from_float(proposed_lower) - Fraction.from_float(
        reference_upper
    )
    upper_exact = Fraction.from_float(proposed_upper) - Fraction.from_float(
        reference_lower
    )
    return GapBracket(
        _fraction_to_lower_float(lower_exact),
        _fraction_to_upper_float(upper_exact),
    )


def exact_group_radius_squared(
    center: object, member_vectors: Iterable[object]
) -> Fraction:
    """Return the exact squared radius required to cover all members."""

    stored_center = _stored_float32_vector(center, name="center")
    maximum = Fraction(0)
    for member in member_vectors:
        maximum = max(maximum, exact_squared_l2(stored_center, member))
    return maximum


def radius_covers(
    center: object, member_vectors: Iterable[object], radius_upper: float
) -> bool:
    """Check a supplied ordinary-L2 radius using only exact rational squares."""

    radius = float(radius_upper)
    if not math.isfinite(radius) or radius < 0.0:
        return False
    radius_squared = Fraction.from_float(radius) ** 2
    required_squared = exact_group_radius_squared(center, member_vectors)
    return radius_squared >= required_squared


def exact_distance_for_key(
    query: object,
    records: Iterable[VectorRecord],
    key: tuple[int, int],
) -> Fraction:
    """Find one version by key and return its exact squared query distance."""

    matches = [record for record in records if record.key == key]
    if not matches:
        raise KeyError(key)
    first = exact_squared_l2(query, matches[0].vector)
    if any(exact_squared_l2(query, record.vector) != first for record in matches[1:]):
        raise ValueError(f"key {key!r} maps to inconsistent stored vectors")
    return first
