"""Packed compiled F/N/P search paths for the Issue #3 recheck.

The C++ extension owns the bulk distance, group-ordering, pruning, and
incremental-kth work.  This module keeps scope validation, immutable Python
ownership, the deliberately small adaptive exact boundary, and Receipt
construction in auditable Python.  The independent oracle never imports this
module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from .models import (
    CandidateSet,
    Receipt,
    SearchHit,
    SearchResult,
    VectorRecord,
    _immutable_c_array,
    as_float32_matrix,
    as_float32_vector,
)
from .numerics import (
    RankedTopK,
    exact_squared_l2,
    fraction_to_lower_float,
    fraction_to_upper_float,
    sqrt_fraction_bracket,
    validate_beta,
    validate_k,
)

if TYPE_CHECKING:  # pragma: no cover
    from .search import SearchEngine


NativeMode = Literal["F", "N", "P"]
ThresholdMode = Literal["heap", "rescan"]
RankStrategy = Literal["adaptive", "all_boundary_exact"]
NATIVE_NUMERIC_MODE = "native_float64_gamma_interval+adaptive_fraction_boundary_v1"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


try:
    from . import _native_kernel as _kernel
except ImportError as _native_import_error:  # pragma: no cover - exercised pre-build
    _kernel = None
else:
    _native_import_error = None


def require_native() -> Any:
    """Return the compiled module or fail instead of silently using Python."""

    if _kernel is None:
        raise RuntimeError(
            "the compiled searchability._native_kernel backend is required"
        ) from _native_import_error
    return _kernel


def native_available() -> bool:
    return _kernel is not None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def native_build_info() -> Mapping[str, Any]:
    backend = require_native()
    info = dict(backend.build_info())
    location = Path(backend.__file__).resolve()
    source = Path(__file__).with_name("_native_kernel.cpp").resolve()
    source_sha256 = _file_sha256(source)
    compiled_source_sha256 = info.get("compiled_source_sha256")
    source_matches_binary = compiled_source_sha256 == source_sha256
    if not source_matches_binary:
        raise RuntimeError(
            "loaded native extension was compiled from a different _native_kernel.cpp"
        )
    info.update(
        {
            "shared_object_path": str(location),
            "shared_object_sha256": _file_sha256(location),
            "native_source_path": str(source),
            "native_source_sha256": source_sha256,
            "native_source_matches_loaded_binary": source_matches_binary,
        }
    )
    return MappingProxyType(info)


def native_call_counter() -> int:
    return int(require_native().call_counter())


def _checked_int64(values: Sequence[int], *, name: str) -> np.ndarray:
    checked: list[int] = []
    for value in values:
        integer = int(value)
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise OverflowError(f"{name} value is outside the native signed-int64 domain")
        checked.append(integer)
    return _immutable_c_array(checked, dtype=np.dtype(np.int64))


def _immutable_float64(values: Any) -> np.ndarray:
    return _immutable_c_array(values, dtype=np.dtype(np.float64))


def _hash_array(digest: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(8, "little"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(8, "little"))
    for extent in array.shape:
        digest.update(int(extent).to_bytes(8, "little", signed=False))
    digest.update(array.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class NativePackedView:
    """One deterministic, bytes-backed visible Delta layout.

    Rows are ``[raw | group-0 | ...]``.  ``group_offsets`` are absolute row
    offsets into that matrix.  A duplicate visible logical ID makes grouped
    pruning inapplicable; construction then resolves the newest version once,
    packs the unique population as raw, and records a forced-full-scan reason.
    """

    snapshot_id: int
    records: tuple[VectorRecord, ...]
    vectors: np.ndarray
    logical_ids: np.ndarray
    version_ids: np.ndarray
    begin_seqs: np.ndarray
    ordinals: np.ndarray
    source_group_ids: np.ndarray
    raw_count: int
    original_raw_count: int
    group_offsets: np.ndarray
    group_ids: np.ndarray
    centers: np.ndarray
    radius_uppers: np.ndarray
    packed_view_hash: str
    force_full_reason: str | None
    build_ns: int
    python_owned_bytes: int
    _logical_to_ordinal: Mapping[int, int]
    _native: Any

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.group_ids.size)

    @property
    def native_owned_bytes(self) -> int:
        return int(self._native.owned_bytes)

    def record_for_ordinal(self, ordinal: int) -> VectorRecord:
        if ordinal < 0 or ordinal >= len(self.records):
            raise RuntimeError("native kernel returned an invalid Delta ordinal")
        return self.records[ordinal]


@dataclass(frozen=True, slots=True)
class PreparedNativeQuery:
    """One validated frozen C plus one pinned packed Delta view."""

    query: np.ndarray
    candidates: CandidateSet
    view: NativePackedView
    candidate_logical_ids: np.ndarray
    candidate_version_ids: np.ndarray
    candidate_begin_seqs: np.ndarray
    duplicate_with_delta: bool
    prepare_ns: int
    _engine_token: int


@dataclass(frozen=True, slots=True)
class _BoundaryRow:
    origin: int
    ordinal: int
    record: VectorRecord
    vector: np.ndarray
    source: str
    estimate: float
    lower: float
    upper: float

    @property
    def token(self) -> tuple[int, int]:
        return (self.origin, self.ordinal)


def _matrix(records: Sequence[VectorRecord], dimension: int, *, name: str) -> np.ndarray:
    return as_float32_matrix(
        np.stack([record.vector for record in records])
        if records
        else np.empty((0, dimension), dtype=np.float32),
        name=name,
    )


def build_packed_view(
    delta: Any, *, snapshot_id: int, dimension: int
) -> NativePackedView:
    """Resolve visibility and build one deterministic immutable native view."""

    backend = require_native()
    start_ns = time.perf_counter_ns()
    if snapshot_id < 0:
        raise ValueError("snapshot_id must be non-negative")
    if snapshot_id > _INT64_MAX:
        raise OverflowError("snapshot_id is outside the native signed-int64 domain")

    # Entries are (record, vector, source_group_id, stable_position).  Sort raw
    # keys, groups, and member keys so the packed ordinal map is part of a
    # deterministic identity rather than depending on catalog iteration order.
    entries: list[tuple[VectorRecord, np.ndarray, int, int]] = []
    raw_records = sorted(
        (record for record in delta.raw_records if record.visible_at(snapshot_id)),
        key=lambda record: record.key,
    )
    for record in raw_records:
        entries.append((record, record.vector, -1, len(entries)))

    group_payloads: list[
        tuple[int, np.ndarray, float, list[tuple[VectorRecord, np.ndarray, int, int]]]
    ] = []
    if delta.grouped is not None:
        for group in sorted(delta.grouped.groups, key=lambda item: item.group_id):
            visible = [
                (record, group.vectors[index])
                for index, record in enumerate(group.members)
                if record.visible_at(snapshot_id)
            ]
            visible.sort(key=lambda item: item[0].key)
            packed_members: list[tuple[VectorRecord, np.ndarray, int, int]] = []
            for record, vector in visible:
                item = (record, vector, int(group.group_id), len(entries))
                entries.append(item)
                packed_members.append(item)
            if packed_members:
                group_payloads.append(
                    (
                        int(group.group_id),
                        group.center,
                        float(group.radius_upper),
                        packed_members,
                    )
                )

    by_logical: dict[int, tuple[VectorRecord, np.ndarray, int, int]] = {}
    duplicate_visible = False
    for entry in entries:
        record, _, _, position = entry
        previous = by_logical.get(record.logical_id)
        if previous is not None:
            duplicate_visible = True
        if previous is None or (record.begin_seq, record.version_id, position) >= (
            previous[0].begin_seq,
            previous[0].version_id,
            previous[3],
        ):
            by_logical[record.logical_id] = entry

    force_full_reason: str | None = None
    if duplicate_visible:
        force_full_reason = "duplicate_visible_logical_id_forced_native_full_scan"
        entries = sorted(by_logical.values(), key=lambda item: item[3])
        entries = [
            (record, vector, -1, position)
            for position, (record, vector, _, _) in enumerate(entries)
        ]
        raw_count = len(entries)
        group_payloads = []
    else:
        raw_count = len(raw_records)

    records = tuple(item[0] for item in entries)
    vectors = _matrix(records, dimension, name="native packed Delta vectors")
    # The deterministic matrix is rebuilt from canonical records, not from a
    # mutable caller view; Group construction already checked bit identity.
    logical_ids = _checked_int64(
        [record.logical_id for record in records], name="logical_id"
    )
    version_ids = _checked_int64(
        [record.version_id for record in records], name="version_id"
    )
    begin_seqs = _checked_int64(
        [record.begin_seq for record in records], name="begin_seq"
    )
    ordinals = _checked_int64(list(range(len(records))), name="ordinal")

    if force_full_reason is not None:
        source_group_ids = _checked_int64([-1] * len(records), name="source_group_id")
        offsets_values = [raw_count]
        group_ids_values: list[int] = []
        centers = as_float32_matrix(
            np.empty((0, dimension), dtype=np.float32), name="native group centers"
        )
        radius_uppers = _immutable_float64(np.empty(0, dtype=np.float64))
    else:
        sources = [-1] * raw_count
        offsets_values = [raw_count]
        group_ids_values = []
        center_values: list[np.ndarray] = []
        radius_values: list[float] = []
        cursor = raw_count
        for group_id, center, radius, members in group_payloads:
            group_ids_values.append(group_id)
            center_values.append(center)
            radius_values.append(radius)
            sources.extend([group_id] * len(members))
            cursor += len(members)
            offsets_values.append(cursor)
        source_group_ids = _checked_int64(sources, name="source_group_id")
        centers = as_float32_matrix(
            np.stack(center_values)
            if center_values
            else np.empty((0, dimension), dtype=np.float32),
            name="native group centers",
        )
        radius_uppers = _immutable_float64(radius_values)

    group_offsets = _checked_int64(offsets_values, name="group_offset")
    group_ids = _checked_int64(group_ids_values, name="group_id")
    digest = hashlib.sha256()
    digest.update(b"searchability-native-packed-view-v1\0")
    digest.update(int(snapshot_id).to_bytes(8, "little", signed=True))
    for name, array in (
        ("vectors", vectors),
        ("logical_ids", logical_ids),
        ("version_ids", version_ids),
        ("begin_seqs", begin_seqs),
        ("ordinals", ordinals),
        ("source_group_ids", source_group_ids),
        ("group_offsets", group_offsets),
        ("group_ids", group_ids),
        ("centers", centers),
        ("radius_uppers", radius_uppers),
    ):
        _hash_array(digest, name, array)
    packed_hash = digest.hexdigest()

    native_view = backend.PackedDeltaView(
        vectors,
        logical_ids,
        version_ids,
        begin_seqs,
        ordinals,
        raw_count,
        group_offsets,
        group_ids,
        centers,
        radius_uppers,
    )
    arrays = (
        vectors,
        logical_ids,
        version_ids,
        begin_seqs,
        ordinals,
        source_group_ids,
        group_offsets,
        group_ids,
        centers,
        radius_uppers,
    )
    logical_to_ordinal = MappingProxyType(
        {record.logical_id: ordinal for ordinal, record in enumerate(records)}
    )
    elapsed = time.perf_counter_ns() - start_ns
    return NativePackedView(
        snapshot_id=int(snapshot_id),
        records=records,
        vectors=vectors,
        logical_ids=logical_ids,
        version_ids=version_ids,
        begin_seqs=begin_seqs,
        ordinals=ordinals,
        source_group_ids=source_group_ids,
        raw_count=raw_count,
        original_raw_count=len(raw_records),
        group_offsets=group_offsets,
        group_ids=group_ids,
        centers=centers,
        radius_uppers=radius_uppers,
        packed_view_hash=packed_hash,
        force_full_reason=force_full_reason,
        build_ns=elapsed,
        python_owned_bytes=sum(int(array.nbytes) for array in arrays),
        _logical_to_ordinal=logical_to_ordinal,
        _native=native_view,
    )


def prepare_native_query(
    engine: "SearchEngine",
    query: np.ndarray,
    *,
    snapshot_id: int,
    k: int,
    candidate_count: int,
    candidate_set: CandidateSet | None = None,
) -> PreparedNativeQuery:
    """Validate C once and bind it to the engine's cached native Delta view."""

    start_ns = time.perf_counter_ns()
    q = as_float32_vector(query, name="query")
    validate_k(k)
    candidates = candidate_set or engine.prepare_candidates(
        q, snapshot_id=snapshot_id, k=k, candidate_count=candidate_count
    )
    engine._validate_candidate_binding(candidates, q, snapshot_id)
    view = engine._native_packed_view(snapshot_id)
    candidate_logical_ids = _checked_int64(
        [record.logical_id for record in candidates.records], name="candidate logical_id"
    )
    candidate_version_ids = _checked_int64(
        [record.version_id for record in candidates.records], name="candidate version_id"
    )
    candidate_begin_seqs = _checked_int64(
        [record.begin_seq for record in candidates.records], name="candidate begin_seq"
    )
    duplicate_with_delta = False
    for record in candidates.records:
        delta_ordinal = view._logical_to_ordinal.get(record.logical_id)
        if delta_ordinal is None:
            continue
        duplicate_with_delta = True
        delta_record = view.records[delta_ordinal]
        if record.key == delta_record.key and (
            record.begin_seq != delta_record.begin_seq
            or record.end_seq != delta_record.end_seq
            or record.commit_seq != delta_record.commit_seq
            or record.vector.tobytes(order="C")
            != delta_record.vector.tobytes(order="C")
        ):
            raise ValueError("one version key maps to inconsistent Base/Delta content")
    return PreparedNativeQuery(
        query=q,
        candidates=candidates,
        view=view,
        candidate_logical_ids=candidate_logical_ids,
        candidate_version_ids=candidate_version_ids,
        candidate_begin_seqs=candidate_begin_seqs,
        duplicate_with_delta=duplicate_with_delta,
        prepare_ns=time.perf_counter_ns() - start_ns,
        _engine_token=id(engine),
    )


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _source(view: NativePackedView, ordinal: int, source_group_id: int) -> str:
    if source_group_id == -1:
        return "raw"
    if source_group_id >= 0:
        return f"group:{source_group_id}"
    raise RuntimeError("native kernel returned an invalid Delta source")


def _rows_from_kernel(
    prepared: PreparedNativeQuery, payload: Mapping[str, Any]
) -> list[_BoundaryRow]:
    names = (
        "origin",
        "ordinal",
        "logical_ids",
        "version_ids",
        "source_group_ids",
        "estimate",
        "lower",
        "upper",
    )
    arrays = {name: np.asarray(payload[name]) for name in names}
    sizes = {int(value.size) for value in arrays.values()}
    if len(sizes) != 1:
        raise RuntimeError("native retained arrays have inconsistent lengths")
    rows: list[_BoundaryRow] = []
    count = sizes.pop()
    for index in range(count):
        origin = int(arrays["origin"][index])
        ordinal = int(arrays["ordinal"][index])
        if origin == 0:
            if ordinal < 0 or ordinal >= len(prepared.candidates.records):
                raise RuntimeError("native kernel returned an invalid C ordinal")
            record = prepared.candidates.records[ordinal]
            vector = prepared.candidates.vectors[ordinal]
            source = "base"
            expected_group = -2
        elif origin == 1:
            record = prepared.view.record_for_ordinal(ordinal)
            vector = prepared.view.vectors[ordinal]
            expected_group = int(prepared.view.source_group_ids[ordinal])
            source = _source(prepared.view, ordinal, expected_group)
        else:
            raise RuntimeError("native kernel returned an invalid candidate origin")
        if (
            int(arrays["logical_ids"][index]) != record.logical_id
            or int(arrays["version_ids"][index]) != record.version_id
            or int(arrays["source_group_ids"][index]) != expected_group
        ):
            raise RuntimeError("native ordinal/key/source binding mismatch")
        estimate = float(arrays["estimate"][index])
        lower = float(arrays["lower"][index])
        upper = float(arrays["upper"][index])
        if not all(math.isfinite(value) for value in (estimate, lower, upper)):
            raise RuntimeError("native kernel returned a non-finite distance interval")
        if lower < 0.0 or not lower <= estimate <= upper:
            raise RuntimeError("native kernel returned an invalid distance interval")
        rows.append(
            _BoundaryRow(
                origin=origin,
                ordinal=ordinal,
                record=record,
                vector=vector,
                source=source,
                estimate=estimate,
                lower=lower,
                upper=upper,
            )
        )
    if int(payload["retained_count"]) != len(rows):
        raise RuntimeError("native retained_count disagrees with returned arrays")
    return rows


def _adaptive_exact_topk(
    query: np.ndarray,
    rows: Sequence[_BoundaryRow],
    *,
    population: int,
    k: int,
    final_kth_upper: float | None,
    rank_strategy: RankStrategy,
) -> RankedTopK:
    """Exactly resolve only overlap components that can enter the prefix."""

    if rank_strategy not in {"adaptive", "all_boundary_exact"}:
        raise ValueError("rank_strategy must be 'adaptive' or 'all_boundary_exact'")
    if population == 0:
        if rows:
            raise RuntimeError("empty population returned boundary candidates")
        return RankedTopK((), None, None, 0)
    take = min(k, population)
    if len(rows) < take:
        raise RuntimeError("native kernel discarded a required boundary candidate")

    exact: dict[tuple[int, int], Fraction] = {}

    def exact_value(row: _BoundaryRow) -> Fraction:
        value = exact.get(row.token)
        if value is None:
            value = exact_squared_l2(query, row.vector)
            exact[row.token] = value
        return value

    ordered = sorted(
        rows,
        key=lambda row: (
            row.lower,
            row.upper,
            row.record.logical_id,
            row.record.version_id,
            row.origin,
            row.ordinal,
        ),
    )
    if rank_strategy == "all_boundary_exact":
        selected = sorted(
            ordered,
            key=lambda row: (
                exact_value(row),
                row.record.logical_id,
                row.record.version_id,
            ),
        )[:take]
    else:
        components: list[list[_BoundaryRow]] = []
        running_upper = -math.inf
        for row in ordered:
            if not components or row.lower > running_upper:
                components.append([row])
                running_upper = row.upper
            else:
                components[-1].append(row)
                running_upper = max(running_upper, row.upper)

        prefix: list[_BoundaryRow] = []
        for component in components:
            if len(prefix) >= take:
                break
            if len(component) > 1:
                component = sorted(
                    component,
                    key=lambda row: (
                        exact_value(row),
                        row.record.logical_id,
                        row.record.version_id,
                    ),
                )
            prefix.extend(component)
        selected = prefix[:take]
    if len(selected) != take:
        raise RuntimeError("adaptive ranking underfilled top-k")

    tau_low: float | None = None
    tau_high: float | None = None
    if population >= k:
        if final_kth_upper is None or not math.isfinite(final_kth_upper):
            raise RuntimeError("native kernel omitted the final kth upper bound")
        kth_exact = exact_value(selected[k - 1])
        tau_low, exact_tau_high = sqrt_fraction_bracket(kth_exact)
        tau_high = min(exact_tau_high, float(final_kth_upper))

    hits: list[SearchHit] = []
    for row in selected:
        squared = exact.get(row.token)
        distance = row.estimate if squared is None else math.sqrt(float(squared))
        hits.append(
            SearchHit(
                logical_id=row.record.logical_id,
                version_id=row.record.version_id,
                distance=distance,
                source=row.source,
            )
        )
    return RankedTopK(tuple(hits), tau_low, tau_high, len(exact))


def _rank_kernel_rows(
    prepared: PreparedNativeQuery,
    rows: Sequence[_BoundaryRow],
    *,
    population: int,
    k: int,
    final_kth_upper: float | None,
    rank_strategy: RankStrategy,
) -> RankedTopK:
    return _adaptive_exact_topk(
        prepared.query,
        rows,
        population=population,
        k=k,
        final_kth_upper=final_kth_upper,
        rank_strategy=rank_strategy,
    )


def _merge_reason(*reasons: str | None) -> str | None:
    values = [value for value in reasons if value]
    return None if not values else "|".join(dict.fromkeys(values))


def run_prepared_native(
    engine: "SearchEngine",
    prepared: PreparedNativeQuery,
    *,
    k: int,
    beta: float,
    mode: NativeMode,
    threshold_mode: ThresholdMode = "heap",
    rank_strategy: RankStrategy = "adaptive",
    audit: bool = False,
) -> SearchResult:
    """Run F, N, or P with one compiled boundary crossing."""

    backend = require_native()
    k = validate_k(k)
    beta = validate_beta(beta)
    if mode not in {"F", "N", "P"}:
        raise ValueError("mode must be F, N, or P")
    if threshold_mode not in {"heap", "rescan"}:
        raise ValueError("threshold_mode must be 'heap' or 'rescan'")
    if rank_strategy not in {"adaptive", "all_boundary_exact"}:
        raise ValueError("invalid rank_strategy")
    if prepared._engine_token != id(engine) or (
        engine._native_packed_view(prepared.view.snapshot_id) is not prepared.view
    ):
        raise ValueError("prepared native query belongs to a different engine/view")
    if prepared.view.snapshot_id != prepared.candidates.snapshot_id and (
        prepared.candidates.snapshot_id is not None
    ):
        raise ValueError("prepared C and Delta view have different snapshots")

    duplicate_reason = prepared.view.force_full_reason
    if prepared.duplicate_with_delta:
        duplicate_reason = "duplicate_visible_logical_id_forced_native_full_scan"
    executed_mode: NativeMode = "F" if duplicate_reason is not None else mode
    before_calls = native_call_counter()
    kernel_payload = dict(
        backend.run_kernel(
            prepared.view._native,
            prepared.query,
            prepared.candidates.vectors,
            prepared.candidate_logical_ids,
            prepared.candidate_version_ids,
            prepared.candidate_begin_seqs,
            k,
            beta,
            executed_mode,
            threshold_mode,
            bool(audit),
            True,
        )
    )
    after_calls = native_call_counter()
    if after_calls != before_calls + 1 or not kernel_payload.get("native_executed"):
        raise RuntimeError("compiled kernel invocation evidence is missing")

    rows = _rows_from_kernel(prepared, kernel_payload)
    population = int(kernel_payload["population"])
    final_kth_upper_value = kernel_payload.get("final_kth_upper")
    final_kth_upper = (
        None if final_kth_upper_value is None else float(final_kth_upper_value)
    )
    exact_start = time.perf_counter_ns()
    ranked = _rank_kernel_rows(
        prepared,
        rows,
        population=population,
        k=k,
        final_kth_upper=final_kth_upper,
        rank_strategy=rank_strategy,
    )
    adaptive_exact_ns = time.perf_counter_ns() - exact_start

    groups_skipped = int(kernel_payload["groups_skipped"])
    population_small = population < k
    min_skipped_value = kernel_payload.get("min_skipped_lb")
    minimum = (
        None if min_skipped_value is None else Fraction.from_float(float(min_skipped_value))
    )
    beta_fraction = Fraction.from_float(beta)
    gap: Fraction | None
    if population_small:
        gap = None
        certified_beta = None
        min_skipped_lb = None
        status = "not_applicable_population_below_k"
    elif groups_skipped == 0:
        gap = Fraction(0)
        certified_beta = 0.0
        min_skipped_lb = None
        status = (
            "full_scan_reference"
            if executed_mode == "F" and duplicate_reason is None
            else (
                "fallback_full_scan_duplicate_visible_logical_id"
                if duplicate_reason is not None
                else "certified_no_skip"
            )
        )
    else:
        if ranked.tau_upper is None or minimum is None:
            raise AssertionError("skipped groups require tau and min skipped LB")
        gap = max(Fraction(0), Fraction.from_float(ranked.tau_upper) - minimum)
        if gap > beta_fraction:
            raise AssertionError("native certificate exceeds requested beta")
        outward_gap = fraction_to_upper_float(gap)
        certified_beta = beta if outward_gap > beta else outward_gap
        min_skipped_lb = fraction_to_lower_float(minimum)
        status = "certified"

    details: dict[str, Any] = {
        "requested_beta_fraction": _fraction_text(beta_fraction),
        "requested_beta_hex": beta.hex(),
        "candidate_snapshot_id": prepared.candidates.snapshot_id,
        "candidate_base_generation": prepared.candidates.base_generation,
        "candidate_base_universe_hash": prepared.candidates.base_universe_hash,
        "candidate_query_hash": prepared.candidates.query_hash,
        "candidate_version_keys": [list(record.key) for record in prepared.candidates.records],
        "returned_version_keys": [list(hit.key) for hit in ranked.hits],
        "candidate_supplemented_count": prepared.candidates.supplemented_count,
        "candidate_available_visible_count": prepared.candidates.available_visible_count,
        "candidate_fallback_reason": prepared.candidates.fallback_reason,
        "packed_delta_view_hash": prepared.view.packed_view_hash,
        "native_backend": kernel_payload.get("backend"),
        "native_call_index": int(kernel_payload["call_index"]),
        "native_trusted_prevalidated_inputs": bool(
            kernel_payload.get("trusted_prevalidated_inputs", False)
        ),
        "native_mode_requested": mode,
        "native_mode_executed": kernel_payload.get("mode_executed", executed_mode),
        "threshold_mode": threshold_mode,
        "rank_strategy": rank_strategy,
        "native_build": dict(native_build_info()),
        "native_feature_flags": dict(kernel_payload.get("feature_flags", {})),
        "packed_view_build_ns": prepared.view.build_ns,
        "packed_view_python_owned_bytes": prepared.view.python_owned_bytes,
        "packed_view_native_owned_bytes": prepared.view.native_owned_bytes,
        "prepared_query_ns": prepared.prepare_ns,
        "retained_boundary_candidates": len(rows),
        "kernel_raw_rows_evaluated": int(kernel_payload["raw_pending_scanned"]),
        "original_visible_raw_rows": prepared.view.original_raw_count,
        "base_visible_view_cache": (
            engine.base.visible_view_cache_stats()
            if hasattr(engine.base, "visible_view_cache_stats")
            else {}
        ),
    }
    if gap is not None:
        details.update(
            {
                "authoritative_gap": _fraction_text(gap),
                "certified_beta_fraction": _fraction_text(gap),
                "tau_upper_fraction": _fraction_text(
                    None
                    if ranked.tau_upper is None
                    else Fraction.from_float(ranked.tau_upper)
                ),
                "min_skipped_lb_lower_fraction": _fraction_text(minimum),
            }
        )
    else:
        details["authoritative_gap"] = None

    audit_rows = kernel_payload.get("audit")
    if audit_rows is not None:
        enriched: list[dict[str, Any]] = []
        for original in audit_rows:
            row = dict(original)
            for key in (
                "radius_upper",
                "center_distance_lower",
                "center_distance_upper",
                "lb_lower",
                "tau_upper_at_decision",
                "threshold_tau_minus_beta",
            ):
                value = row.get(key)
                row[f"{key}_fraction"] = (
                    None if value is None else _fraction_text(Fraction.from_float(float(value)))
                )
            row["requested_beta_fraction"] = _fraction_text(beta_fraction)
            enriched.append(row)
        audit_rows = enriched

    receipt_start = time.perf_counter_ns()
    component_timings = {
        str(name): int(value)
        for name, value in dict(kernel_payload["component_timings_ns"]).items()
    }
    component_timings["adaptive_exact_ns"] = adaptive_exact_ns
    component_timings["receipt_ns"] = 0
    fallback_reason = _merge_reason(
        duplicate_reason,
        kernel_payload.get("fallback_reason"),
        prepared.candidates.fallback_reason,
        "population_below_k" if population_small else None,
    )
    receipt = Receipt(
        snapshot_id=prepared.view.snapshot_id,
        base_generation=engine.base.generation_id,
        covered_commit_seq=engine.base.covered_commit_seq,
        candidate_set_id_or_hash=prepared.candidates.candidate_hash,
        metric="L2",
        k=k,
        requested_beta=beta,
        certified_beta=certified_beta,
        certificate_status=status,
        tau_returned=(
            None
            if population_small or not ranked.hits
            else float(ranked.hits[-1].distance)
        ),
        min_skipped_lb=min_skipped_lb,
        raw_pending_scanned=int(kernel_payload["raw_pending_scanned"]),
        groups_scanned=int(kernel_payload["groups_scanned"]),
        groups_skipped=groups_skipped,
        vectors_scanned=int(kernel_payload["vectors_scanned"]),
        visibility_rejections=prepared.candidates.visibility_rejections,
        fallback_reason=fallback_reason,
        numeric_mode=NATIVE_NUMERIC_MODE,
        component_timings=component_timings,
        distance_evaluations=int(kernel_payload["distance_evaluations"]),
        exact_boundary_rechecks=ranked.exact_rechecks,
        certificate_details=details,
        audit=audit_rows if audit else None,
    )
    receipt.to_dict()
    receipt_ns = time.perf_counter_ns() - receipt_start
    timings = dict(receipt.component_timings)
    timings["receipt_ns"] = receipt_ns
    return SearchResult(ranked.hits, replace(receipt, component_timings=timings))
