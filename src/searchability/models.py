"""Immutable value objects shared by the search prototype.

The mathematical object searched by this project is the stored float32 vector,
not an unrecorded higher-precision value supplied by a caller.  Conversion to
float32 therefore happens once, at record/query construction, and all certified
reasoning refers to those exact binary values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import struct
from typing import Any, Mapping, Optional, Sequence

import numpy as np


MAX_DIMENSION = 4096
MAX_ABS_COMPONENT = 1.0e15


def _immutable_c_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Copy ``value`` onto an immutable ``bytes`` buffer.

    Clearing NumPy's ``WRITEABLE`` flag alone is not an immutability boundary:
    an owning array can set that flag back to true. Arrays returned here are
    views of immutable Python ``bytes``, so NumPy itself rejects that change.
    """

    converted = np.asarray(value, dtype=dtype, order="C")
    owner: object = converted
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if converted.flags.c_contiguous and isinstance(owner, bytes):
        return converted
    payload = converted.tobytes(order="C")
    return np.frombuffer(payload, dtype=dtype).reshape(converted.shape)


def _validate_real_components(original: np.ndarray, *, name: str) -> None:
    """Validate before conversion without allocating a full float64 copy."""

    if not np.issubdtype(original.dtype, np.number) or np.issubdtype(
        original.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(original)):
        raise ValueError(f"{name} contains NaN or infinity")
    # Direct two-sided comparisons avoid signed-integer ``abs(min)`` overflow.
    if np.any(original > MAX_ABS_COMPONENT) or np.any(
        original < -MAX_ABS_COMPONENT
    ):
        raise ValueError(
            f"{name} component magnitude exceeds certified limit "
            f"{MAX_ABS_COMPONENT:g}"
        )


def as_float32_vector(value: Any, *, name: str = "vector") -> np.ndarray:
    """Return a validated, deeply immutable contiguous float32 vector."""

    original = np.asarray(value)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 1 <= original.shape[0] <= MAX_DIMENSION:
        raise ValueError(f"{name} dimension must be in [1, {MAX_DIMENSION}]")
    _validate_real_components(original, name=name)
    converted = _immutable_c_array(original, dtype=np.dtype(np.float32))
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} overflows float32 storage")
    return converted


def as_float32_matrix(value: Any, *, name: str = "vectors") -> np.ndarray:
    """Return a validated, immutable, contiguous two-dimensional float32 array.

    Checking the original dtype before conversion is important: NumPy otherwise
    permits a complex-to-real cast that silently discards the imaginary part.
    Certified code must reject that change of mathematical input.
    """

    original = np.asarray(value)
    if original.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not 1 <= original.shape[1] <= MAX_DIMENSION:
        raise ValueError(f"{name} dimension must be in [1, {MAX_DIMENSION}]")
    _validate_real_components(original, name=name)
    converted = _immutable_c_array(original, dtype=np.dtype(np.float32))
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} overflows float32 storage")
    return converted


def vector_query_hash(query: np.ndarray) -> str:
    """Hash one canonical query, including its dimension and exact float32 bits."""

    canonical = as_float32_vector(query, name="query")
    digest = hashlib.sha256()
    digest.update(b"searchability-query-v1\0")
    digest.update(struct.pack("<q", canonical.shape[0]))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def records_universe_hash(
    records: Sequence["VectorRecord"], *, dimension: int, generation_id: str
) -> str:
    """Hash the immutable base universe to prevent generation-name aliasing."""

    digest = hashlib.sha256()
    digest.update(b"searchability-base-universe-v1\0")
    generation = generation_id.encode("utf-8")
    digest.update(struct.pack("<q", len(generation)))
    digest.update(generation)
    digest.update(struct.pack("<qq", int(dimension), len(records)))
    for record in records:
        digest.update(struct.pack("<qqq", record.logical_id, record.version_id, record.begin_seq))
        digest.update(
            struct.pack("<q", -1 if record.end_seq is None else int(record.end_seq))
        )
        digest.update(
            struct.pack("<q", -1 if record.commit_seq is None else int(record.commit_seq))
        )
        digest.update(record.vector.tobytes(order="C"))
    return digest.hexdigest()


def frozen_candidate_hash(
    records: Sequence["VectorRecord"],
    vectors: np.ndarray,
    *,
    base_generation: str,
    base_universe_hash: str,
    snapshot_id: int,
    query_hash: str,
    requested_count: int,
) -> str:
    """Hash all scope and content that define one frozen candidate set ``C``."""

    matrix = as_float32_matrix(vectors, name="candidate vectors")
    if matrix.shape[0] != len(records):
        raise ValueError("candidate records/vectors length mismatch")
    digest = hashlib.sha256()
    digest.update(b"searchability-frozen-c-v1\0")
    for value in (base_generation, base_universe_hash, query_hash):
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<q", len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack("<qq", int(snapshot_id), int(requested_count)))
    for record, row in zip(records, matrix):
        digest.update(struct.pack("<qq", record.logical_id, record.version_id))
        digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


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
    snapshot_id: int | None = None
    base_generation: str | None = None
    base_universe_hash: str | None = None
    query_hash: str | None = None
    supplemented_count: int = 0
    available_visible_count: int | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, VectorRecord) for record in records):
            raise TypeError("candidate records must be VectorRecord instances")
        matrix = as_float32_matrix(self.vectors, name="candidate vectors")
        if matrix.shape[0] != len(self.records):
            raise ValueError("candidate records/vectors length mismatch")
        if len({record.key for record in records}) != len(records):
            raise ValueError("candidate set contains duplicate version keys")
        if len({record.logical_id for record in records}) != len(records):
            raise ValueError("candidate set contains duplicate logical IDs")
        expected = (
            np.ascontiguousarray(
                np.stack([record.vector for record in records]), dtype=np.float32
            )
            if records
            else np.empty(matrix.shape, dtype=np.float32)
        )
        if matrix.tobytes(order="C") != expected.tobytes(order="C"):
            raise ValueError("candidate vectors do not bit-match candidate records")
        if self.visibility_rejections < 0 or self.search_ns < 0:
            raise ValueError("candidate counts and timings must be non-negative")
        if self.requested_count < 0 or self.supplemented_count < 0:
            raise ValueError("candidate counts must be non-negative")
        if self.available_visible_count is not None and self.available_visible_count < 0:
            raise ValueError("available_visible_count must be non-negative")
        bindings = (
            self.snapshot_id,
            self.base_generation,
            self.base_universe_hash,
            self.query_hash,
        )
        if any(value is not None for value in bindings) and not all(
            value is not None for value in bindings
        ):
            raise ValueError("candidate binding fields must be supplied together")
        if self.snapshot_id is not None:
            if self.snapshot_id < 0:
                raise ValueError("candidate snapshot_id must be non-negative")
            assert self.base_generation is not None
            assert self.base_universe_hash is not None
            assert self.query_hash is not None
            if not self.base_generation or not self.base_universe_hash or not self.query_hash:
                raise ValueError("candidate binding hashes/generation must be non-empty")
            if any(not record.visible_at(self.snapshot_id) for record in records):
                raise ValueError("candidate set contains a record invisible at its snapshot")
            expected_hash = frozen_candidate_hash(
                records,
                matrix,
                base_generation=self.base_generation,
                base_universe_hash=self.base_universe_hash,
                snapshot_id=self.snapshot_id,
                query_hash=self.query_hash,
                requested_count=self.requested_count,
            )
            if self.candidate_hash != expected_hash:
                raise ValueError("candidate hash does not match its bound scope/content")
        object.__setattr__(self, "records", records)
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
    group_catalog_revision: Optional[int] = None
    delta_view_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    receipt: Receipt

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(hit.logical_id for hit in self.hits)
