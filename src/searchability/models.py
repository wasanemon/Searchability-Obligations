"""Immutable value objects shared by the search prototype.

The mathematical object searched by this project is the stored float32 vector,
not an unrecorded higher-precision value supplied by a caller.  Conversion to
float32 therefore happens once, at record/query construction, and all certified
reasoning refers to those exact binary values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


MAX_DIMENSION = 4096
MAX_ABS_COMPONENT = 1.0e15


def as_float32_vector(value: Any, *, name: str = "vector") -> np.ndarray:
    """Return a validated, immutable, contiguous one-dimensional float32 vector."""

    original = np.asarray(value)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 1 <= original.shape[0] <= MAX_DIMENSION:
        raise ValueError(f"{name} dimension must be in [1, {MAX_DIMENSION}]")
    if not np.issubdtype(original.dtype, np.number) or np.issubdtype(
        original.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain numeric values")
    original64 = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(original64)):
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(np.abs(original64) > MAX_ABS_COMPONENT):
        raise ValueError(
            f"{name} component magnitude exceeds certified limit {MAX_ABS_COMPONENT:g}"
        )
    converted = np.array(original64, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} overflows float32 storage")
    converted.setflags(write=False)
    return converted


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One logical object's immutable vector version and MVCC interval."""

    logical_id: int
    version_id: int
    vector: np.ndarray = field(compare=False, repr=False)
    begin_seq: int = 0
    end_seq: Optional[int] = None
    commit_seq: Optional[int] = None

    def __post_init__(self) -> None:
        if self.logical_id < 0 or self.version_id < 0:
            raise ValueError("logical_id and version_id must be non-negative")
        if self.begin_seq < 0:
            raise ValueError("begin_seq must be non-negative")
        if self.end_seq is not None and self.end_seq <= self.begin_seq:
            raise ValueError("end_seq must be greater than begin_seq")
        if self.commit_seq is not None and self.commit_seq < 0:
            raise ValueError("commit_seq must be non-negative")
        object.__setattr__(self, "vector", as_float32_vector(self.vector))

    def visible_at(self, snapshot_id: int) -> bool:
        if snapshot_id < 0:
            raise ValueError("snapshot_id must be non-negative")
        return self.begin_seq <= snapshot_id and (
            self.end_seq is None or snapshot_id < self.end_seq
        )

    @property
    def key(self) -> tuple[int, int]:
        return (self.logical_id, self.version_id)


@dataclass(frozen=True, slots=True)
class SearchHit:
    logical_id: int
    version_id: int
    distance: float
    source: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.logical_id, self.version_id)


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Frozen visibility-filtered base ANN candidate set C."""

    records: tuple[VectorRecord, ...]
    vectors: np.ndarray = field(compare=False, repr=False)
    candidate_hash: str
    visibility_rejections: int
    search_ns: int
    requested_count: int

    def __post_init__(self) -> None:
        matrix = np.asarray(self.vectors, dtype=np.float32, order="C")
        if matrix.ndim != 2:
            raise ValueError("candidate vectors must be a matrix")
        if matrix.shape[0] != len(self.records):
            raise ValueError("candidate records/vectors length mismatch")
        matrix = np.array(matrix, dtype=np.float32, order="C", copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "vectors", matrix)


@dataclass(frozen=True, slots=True)
class Receipt:
    snapshot_id: int
    base_generation: str
    covered_commit_seq: Optional[int]
    candidate_set_id_or_hash: str
    metric: str
    k: int
    requested_beta: float
    certified_beta: Optional[float]
    certificate_status: str
    tau_returned: Optional[float]
    min_skipped_lb: Optional[float]
    raw_pending_scanned: int
    groups_scanned: int
    groups_skipped: int
    vectors_scanned: int
    visibility_rejections: int
    fallback_reason: Optional[str]
    numeric_mode: str
    component_timings: Mapping[str, int]
    distance_evaluations: int = 0
    exact_boundary_rechecks: int = 0
    certificate_details: Optional[Mapping[str, Any]] = None
    audit: Optional[Sequence[Mapping[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    receipt: Receipt

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(hit.logical_id for hit in self.hits)
