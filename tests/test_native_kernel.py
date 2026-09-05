from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import pytest

import searchability.native as native_module
from searchability.base import BaseIndex
from searchability.groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from searchability.models import CandidateSet, VectorRecord
from searchability.native import native_build_info, native_call_counter
from searchability.oracle import (
    exact_distance_for_key,
    exact_group_radius_squared,
    observed_gap_bracket,
    reference_topk,
    sqrt_bracket,
)
from searchability.search import SearchEngine


NATIVE_RANDOM_SEED = 0x1_553_2026
NATIVE_CI_CASES = 200
NATIVE_EVALUATION_CASES = 10_000


@dataclass
class _FrozenBaseStub:
    dimension: int
    generation_id: str = "native-test-generation"
    covered_commit_seq: int | None = 0

    def prepare_candidates(self, *args: object, **kwargs: object) -> CandidateSet:
        raise AssertionError("test must pass the frozen candidate set")


def _record(
    logical_id: int,
    values: tuple[float, ...],
    *,
    version_id: int = 0,
    begin_seq: int = 0,
    end_seq: int | None = None,
) -> VectorRecord:
    return VectorRecord(
        logical_id=logical_id,
        version_id=version_id,
        vector=np.asarray(values, dtype=np.float32),
        begin_seq=begin_seq,
        end_seq=end_seq,
    )


def _candidate_set(records: tuple[VectorRecord, ...], dimension: int) -> CandidateSet:
    vectors = (
        np.ascontiguousarray(
            np.stack([record.vector for record in records]), dtype=np.float32
        )
        if records
        else np.empty((0, dimension), dtype=np.float32)
    )
    return CandidateSet(
        records=records,
        vectors=vectors,
        candidate_hash="frozen-native-C",
        visibility_rejections=0,
        search_ns=0,
        requested_count=len(records),
    )


def _group(
    group_id: int,
    center: tuple[float, ...],
    records: tuple[VectorRecord, ...],
) -> Group:
    center_array = np.asarray(center, dtype=np.float32)
    vectors = (
        np.ascontiguousarray(
            np.stack([record.vector for record in records]), dtype=np.float32
        )
        if records
        else np.empty((0, center_array.size), dtype=np.float32)
    )
    return Group(
        group_id=group_id,
        center=center_array,
        radius_upper=0.0,
        members=records,
        vectors=vectors,
    )


def _bounded_group(
    group_id: int,
    center: np.ndarray,
    records: tuple[VectorRecord, ...],
    dimension: int,
) -> Group:
    vectors = (
        np.ascontiguousarray(
            np.stack([record.vector for record in records]), dtype=np.float32
        )
        if records
        else np.empty((0, dimension), dtype=np.float32)
    )
    required = exact_group_radius_squared(
        center, (record.vector for record in records)
    )
    return Group(
        group_id=group_id,
        center=center,
        radius_upper=sqrt_bracket(required)[1],
        members=records,
        vectors=vectors,
    )


def _engine(
    dimension: int,
    *,
    raw: tuple[VectorRecord, ...] = (),
    groups: tuple[Group, ...] = (),
) -> SearchEngine:
    directory = (
        GroupDirectory(
            groups=groups,
            dimension=dimension,
            build_stats=GroupBuildStats(0, 0, 0, 0, 0),
        )
        if groups
        else None
    )
    return SearchEngine(  # type: ignore[arg-type]
        _FrozenBaseStub(dimension), DeltaStore(raw_records=raw, grouped=directory)
    )


def _keys(result: object) -> tuple[tuple[int, int], ...]:
    return tuple(hit.key for hit in result.hits)  # type: ignore[attr-defined]


def _has_bytes_owner(array: np.ndarray) -> bool:
    owner: object = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    return isinstance(owner, bytes)


def test_native_f_n_p_share_results_and_expose_ablation_evidence() -> None:
    candidates = _candidate_set(
        (_record(10, (5.0, 0.0)), _record(11, (6.0, 0.0))), 2
    )
    raw = (_record(12, (4.0, 0.0)),)
    near = _group(3, (1.0, 0.0), (_record(1, (1.0, 0.0)),))
    far = _group(9, (100.0, 0.0), (_record(2, (100.0, 0.0)),))
    engine = _engine(2, raw=raw, groups=(near, far))
    query = np.zeros(2, dtype=np.float32)
    prepared = engine.prepare_native_query(
        query, k=2, candidate_count=2, candidate_set=candidates
    )

    before = native_call_counter()
    full = engine.search_native_prepared(prepared, k=2, mode="F")
    no_skip = engine.search_native_prepared(prepared, k=2, mode="N")
    heap = engine.search_native_prepared(
        prepared, k=2, beta=0.0, mode="P", audit=True
    )
    rescan = engine.search_native_prepared(
        prepared, k=2, beta=0.0, mode="P", threshold_mode="rescan"
    )
    exact_all = engine.search_native_prepared(
        prepared, k=2, beta=0.0, mode="P", rank_strategy="all_boundary_exact"
    )
    assert native_call_counter() == before + 5

    expected = ((1, 0), (12, 0))
    assert _keys(full) == _keys(no_skip) == _keys(heap) == expected
    assert _keys(rescan) == _keys(exact_all) == expected
    assert heap.receipt.groups_skipped == rescan.receipt.groups_skipped == 1
    assert no_skip.receipt.groups_skipped == 0
    assert heap.receipt.certificate_status == "certified"
    assert heap.receipt.certified_beta == 0.0
    assert heap.receipt.certificate_details["threshold_mode"] == "heap"
    assert rescan.receipt.certificate_details["threshold_mode"] == "rescan"
    assert exact_all.receipt.certificate_details["rank_strategy"] == (
        "all_boundary_exact"
    )
    assert heap.receipt.certificate_details["native_feature_flags"][
        "incremental_kth_heap"
    ]
    assert heap.receipt.audit is not None
    assert {row["action"] for row in heap.receipt.audit} == {"scan", "skip"}
    assert all("lb_lower_fraction" in row for row in heap.receipt.audit)

    view = prepared.view
    assert view.raw_count == 1
    assert view.group_offsets.tolist() == [1, 2, 3]
    assert view.group_ids.tolist() == [3, 9]
    for array in (
        view.vectors,
        view.logical_ids,
        view.group_offsets,
        view.centers,
        view.radius_uppers,
    ):
        assert not array.flags.writeable
        assert _has_bytes_owner(array)

    build = dict(native_build_info())
    assert build["backend"] == "pybind11_cpp17"
    assert build["compile_macros"]["__FAST_MATH__"] is False
    assert build["rounding_mode_enforced"] is True
    assert tuple(build["compile_flags"]) == (
        "-O3",
        "-std=c++17",
        "-fno-fast-math",
        "-ffp-contract=off",
    )
    assert isinstance(build["compiler_executable"], str)
    assert build["compiler_executable"]
    assert isinstance(build["compile_driver_and_flags"], str)
    assert build["compiler_executable"] in build["compile_driver_and_flags"]
    assert all(
        flag in build["compile_driver_and_flags"]
        for flag in build["compile_flags"]
    )
    assert "setuptools build_ext" in build["compile_command_provenance"]
    assert "per-source" in build["compile_command_scope"]
    assert build["native_source_matches_loaded_binary"] is True
    assert build["compiled_source_sha256"] == build["native_source_sha256"]
    assert len(build["native_source_sha256"]) == 64
    assert len(build["shared_object_sha256"]) == 64


def test_native_packed_view_sorts_keys_and_omits_empty_groups() -> None:
    raw = (_record(40, (40.0,)), _record(10, (10.0,)))
    group_nine_records = (_record(90, (90.0,)), _record(80, (80.0,)))
    group_three_records = (_record(31, (31.0,)), _record(30, (30.0,)))
    groups = (
        _bounded_group(9, np.asarray([85.0], dtype=np.float32), group_nine_records, 1),
        _group(1, (0.0,), ()),
        _bounded_group(3, np.asarray([30.5], dtype=np.float32), group_three_records, 1),
    )
    engine = _engine(1, raw=raw, groups=groups)
    candidates = _candidate_set((_record(1, (1.0,)),), 1)

    prepared = engine.prepare_native_query(
        np.zeros(1, dtype=np.float32),
        k=1,
        candidate_count=1,
        candidate_set=candidates,
    )
    view = prepared.view

    assert [record.logical_id for record in view.records] == [10, 40, 30, 31, 80, 90]
    assert view.raw_count == 2
    assert view.group_ids.tolist() == [3, 9]
    assert view.group_offsets.tolist() == [2, 4, 6]


def test_native_build_info_rejects_stale_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = native_module.require_native()
    original_build_info = backend.build_info
    stale = dict(original_build_info())
    stale["compiled_source_sha256"] = "0" * 64
    monkeypatch.setattr(backend, "build_info", lambda: stale)
    native_module.native_build_info.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="compiled from a different"):
            native_module.native_build_info()
    finally:
        native_module.native_build_info.cache_clear()


def test_native_adaptive_ranking_resolves_equal_distance_identifier_ties() -> None:
    candidates = _candidate_set(
        (_record(9, (1.0, 0.0)), _record(3, (-1.0, 0.0))), 2
    )
    raw = (_record(1, (0.0, 1.0), version_id=4),)
    group = _group(0, (0.0, -1.0), (_record(5, (0.0, -1.0), version_id=2),))
    engine = _engine(2, raw=raw, groups=(group,))
    query = np.zeros(2, dtype=np.float32)
    prepared = engine.prepare_native_query(
        query, k=3, candidate_count=2, candidate_set=candidates
    )

    adaptive = engine.search_native_prepared(
        prepared, k=3, beta=0.0, mode="P", rank_strategy="adaptive"
    )
    exact_all = engine.search_native_prepared(
        prepared, k=3, beta=0.0, mode="P", rank_strategy="all_boundary_exact"
    )
    assert _keys(adaptive) == _keys(exact_all) == ((1, 4), (3, 0), (5, 2))
    assert adaptive.receipt.exact_boundary_rechecks == 4


def test_duplicate_visibility_forces_full_scan_before_any_skip() -> None:
    base_version = _record(7, (50.0,), version_id=0, begin_seq=0)
    delta_version = _record(7, (1.0,), version_id=1, begin_seq=1)
    candidates = _candidate_set((base_version,), 1)
    engine = _engine(1, groups=(_group(4, (1.0,), (delta_version,)),))
    query = np.zeros(1, dtype=np.float32)
    prepared = engine.prepare_native_query(
        query,
        snapshot_id=1,
        k=1,
        candidate_count=1,
        candidate_set=candidates,
    )
    result = engine.search_native_prepared(prepared, k=1, beta=0.0, mode="P")
    assert _keys(result) == ((7, 1),)
    assert result.receipt.groups_skipped == 0
    assert result.receipt.certificate_status == (
        "fallback_full_scan_duplicate_visible_logical_id"
    )
    assert "duplicate_visible_logical_id" in (result.receipt.fallback_reason or "")
    assert result.receipt.certificate_details["native_mode_requested"] == "P"
    assert result.receipt.certificate_details["native_mode_executed"] == "F"

    raw_old = _record(8, (20.0,), version_id=0, begin_seq=0)
    grouped_new = _record(8, (2.0,), version_id=1, begin_seq=1)
    internal_engine = _engine(
        1, raw=(raw_old,), groups=(_group(5, (2.0,), (grouped_new,)),)
    )
    internal_candidates = _candidate_set((_record(1, (10.0,)),), 1)
    internal_prepared = internal_engine.prepare_native_query(
        query,
        snapshot_id=1,
        k=1,
        candidate_count=1,
        candidate_set=internal_candidates,
    )
    internal = internal_engine.search_native_prepared(
        internal_prepared, k=1, beta=0.0, mode="P"
    )
    assert _keys(internal) == ((8, 1),)
    assert internal_prepared.view.force_full_reason is not None
    assert internal_prepared.view.group_count == 0
    assert internal.receipt.groups_skipped == 0


def test_base_visible_cache_and_legacy_rebuild_freeze_identical_c() -> None:
    base = BaseIndex.from_arrays(
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        logical_ids=[10, 11, 12],
        threads=1,
    )
    query = np.asarray([0.5], dtype=np.float32)
    cached_first = base.prepare_candidates(
        query, snapshot_id=0, k=2, candidate_count=3, use_cached_view=True
    )
    cached_second = base.prepare_candidates(
        query, snapshot_id=0, k=2, candidate_count=3, use_cached_view=True
    )
    legacy = base.prepare_candidates(
        query, snapshot_id=0, k=2, candidate_count=3, use_cached_view=False
    )
    assert cached_first.candidate_hash == cached_second.candidate_hash == legacy.candidate_hash
    assert tuple(record.key for record in cached_first.records) == tuple(
        record.key for record in legacy.records
    )
    stats = base.visible_view_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] >= 1
    assert stats["legacy_rebuilds"] == 1
    assert stats["cached_snapshots"] == 1
    assert stats["cache_lookup_ns"] >= 0
    assert stats["cache_build_ns"] >= 0
    assert stats["legacy_rebuild_ns"] >= 0

    base.initialize_integrity_metadata()
    assert base.visible_view_cache_stats() == {
        "hits": 0,
        "misses": 0,
        "legacy_rebuilds": 0,
        "cached_snapshots": 0,
        "cache_lookup_ns": 0,
        "cache_build_ns": 0,
        "legacy_rebuild_ns": 0,
    }


@pytest.mark.parametrize("dimension", [1, 3, 7, 15, 31, 127, 129, 4096])
def test_native_dimension_tails_through_maximum_match_fraction_oracle(
    dimension: int,
) -> None:
    indices = np.arange(dimension, dtype=np.float32)
    query = np.where(indices % 2 == 0, np.float32(1.0), np.float32(-1.0))
    candidate_vector = np.nextafter(
        query, np.full(dimension, np.float32(math.inf)), dtype=np.float32
    )
    raw_vector = query.copy()
    group_vector = np.where(indices % 3 == 0, query, np.float32(0.0)).astype(
        np.float32
    )
    candidate_record = VectorRecord(10, 0, candidate_vector)
    raw_record = VectorRecord(11, 0, raw_vector)
    grouped_record = VectorRecord(12, 0, group_vector)
    candidates = _candidate_set((candidate_record,), dimension)
    group = _group(4, tuple(float(value) for value in group_vector), (grouped_record,))
    engine = _engine(dimension, raw=(raw_record,), groups=(group,))
    prepared = engine.prepare_native_query(
        query, k=2, candidate_count=1, candidate_set=candidates
    )
    full = engine.search_native_prepared(prepared, k=2, mode="F")
    proposed = engine.search_native_prepared(
        prepared, k=2, beta=0.0, mode="P"
    )
    oracle = reference_topk(
        query,
        candidates,
        raw_records=(raw_record,),
        grouped_records=(grouped_record,),
        k=2,
    )
    assert _keys(full) == _keys(proposed) == oracle.keys


def _unaligned_float32(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype=np.float32, order="C")
    owner = b"x" + canonical.tobytes(order="C")
    return np.frombuffer(
        owner, dtype=np.float32, count=canonical.size, offset=1
    ).reshape(canonical.shape)


def test_native_accepts_immutable_unaligned_query_and_frozen_c() -> None:
    query = _unaligned_float32(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    candidate_vector = _unaligned_float32(
        np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
    )
    candidate = VectorRecord(1, 0, candidate_vector)
    candidate_matrix = _unaligned_float32(candidate_vector.reshape(1, 3))
    candidates = CandidateSet(
        records=(candidate,),
        vectors=candidate_matrix,
        candidate_hash="unaligned-frozen-C",
        visibility_rejections=0,
        search_ns=0,
        requested_count=1,
    )
    assert query.ctypes.data % np.dtype(np.float32).alignment != 0
    assert candidates.vectors.ctypes.data % np.dtype(np.float32).alignment != 0
    engine = _engine(3, raw=(_record(2, (0.0, 0.0, 1.0)),))
    prepared = engine.prepare_native_query(
        query, k=1, candidate_count=1, candidate_set=candidates
    )
    result = engine.search_native_prepared(
        prepared, k=1, beta=0.0, mode="P"
    )
    assert _keys(result) == ((2, 0),)
    assert result.receipt.certificate_details[
        "native_trusted_prevalidated_inputs"
    ]


def test_native_empty_and_population_below_k_have_no_false_kth_threshold() -> None:
    query = np.zeros(2, dtype=np.float32)
    empty_candidates = _candidate_set((), 2)
    empty_engine = _engine(2)
    empty_prepared = empty_engine.prepare_native_query(
        query, k=3, candidate_count=1, candidate_set=empty_candidates
    )
    empty = empty_engine.search_native_prepared(
        empty_prepared, k=3, beta=0.0, mode="P"
    )
    assert empty.hits == ()
    assert empty.receipt.tau_returned is None
    assert empty.receipt.certified_beta is None
    assert empty.receipt.certificate_status == "not_applicable_population_below_k"

    candidate = _record(1, (3.0, 0.0))
    member = _record(2, (1.0, 0.0))
    candidates = _candidate_set((candidate,), 2)
    engine = _engine(2, groups=(_group(0, (1.0, 0.0), (member,)),))
    prepared = engine.prepare_native_query(
        query, k=5, candidate_count=1, candidate_set=candidates
    )
    underfilled = engine.search_native_prepared(
        prepared, k=5, beta=100.0, mode="P"
    )
    assert _keys(underfilled) == ((2, 0), (1, 0))
    assert underfilled.receipt.groups_skipped == 0
    assert underfilled.receipt.tau_returned is None
    assert underfilled.receipt.certified_beta is None


@pytest.mark.parametrize(
    ("group_distance", "expected_action"),
    [
        (float(np.nextafter(np.float32(2.0), np.float32(-math.inf))), "scan"),
        (2.0, "scan"),
        (float(np.nextafter(np.float32(2.0), np.float32(math.inf))), "skip"),
    ],
)
def test_native_nextafter_strict_skip_boundary(
    group_distance: float, expected_action: str
) -> None:
    candidates = _candidate_set((_record(1, (3.0,)),), 1)
    member = _record(2, (group_distance,))
    engine = _engine(1, groups=(_group(0, (group_distance,), (member,)),))
    query = np.zeros(1, dtype=np.float32)
    prepared = engine.prepare_native_query(
        query, k=1, candidate_count=1, candidate_set=candidates
    )
    result = engine.search_native_prepared(
        prepared, k=1, beta=1.0, mode="P", audit=True
    )
    assert result.receipt.audit is not None
    assert result.receipt.audit[0]["action"] == expected_action


def test_native_positive_beta_reports_nonzero_gap_bounded_by_request() -> None:
    candidate = _record(10, (10.0,))
    omitted = _record(20, (9.0,))
    candidates = _candidate_set((candidate,), 1)
    engine = _engine(1, groups=(_group(0, (9.0,), (omitted,)),))
    query = np.zeros(1, dtype=np.float32)
    prepared = engine.prepare_native_query(
        query, k=1, candidate_count=1, candidate_set=candidates
    )
    proposed = engine.search_native_prepared(
        prepared, k=1, beta=2.0, mode="P", audit=True
    )
    oracle = reference_topk(
        query, candidates, grouped_records=(omitted,), k=1
    )
    assert _keys(proposed) == ((10, 0),)
    assert proposed.receipt.groups_skipped == 1
    assert oracle.tau_squared is not None
    proposed_tau = exact_distance_for_key(
        query, (candidate, omitted), proposed.hits[-1].key
    )
    observed = observed_gap_bracket(proposed_tau, oracle.tau_squared)
    assert observed.lower <= 1.0 <= observed.upper
    assert proposed.receipt.certified_beta is not None
    assert 0.0 < proposed.receipt.certified_beta <= 2.0
    assert observed.upper <= proposed.receipt.certified_beta
    assert proposed.receipt.certificate_details["authoritative_gap"] not in {
        None,
        "0/1",
    }


def test_native_snapshot_lifecycle_and_prepared_binding_are_fail_closed() -> None:
    old = _record(7, (1.0,), version_id=0, begin_seq=0, end_seq=1)
    new = _record(7, (2.0,), version_id=1, begin_seq=1)
    candidate = _record(20, (10.0,))
    candidates = _candidate_set((candidate,), 1)
    engine = _engine(1, raw=(old, new))
    query = np.zeros(1, dtype=np.float32)
    prepared_zero = engine.prepare_native_query(
        query,
        snapshot_id=0,
        k=1,
        candidate_count=1,
        candidate_set=candidates,
    )
    prepared_one = engine.prepare_native_query(
        query,
        snapshot_id=1,
        k=1,
        candidate_count=1,
        candidate_set=candidates,
    )
    assert prepared_zero.view is engine._native_packed_view(0)
    assert prepared_one.view is engine._native_packed_view(1)
    assert prepared_zero.view is not prepared_one.view
    assert _keys(
        engine.search_native_prepared(prepared_zero, k=1, mode="F")
    ) == ((7, 0),)
    assert _keys(
        engine.search_native_prepared(prepared_one, k=1, mode="F")
    ) == ((7, 1),)

    other_engine = _engine(1, raw=(old, new))
    with pytest.raises(ValueError, match="different engine/view"):
        other_engine.search_native_prepared(prepared_zero, k=1, mode="F")

    real_base = BaseIndex.from_arrays(
        np.asarray([[1.0], [2.0]], dtype=np.float32), threads=1
    )
    bound_engine = SearchEngine(real_base, DeltaStore(raw_records=()))
    bound_c = bound_engine.prepare_candidates(
        query, snapshot_id=0, k=1, candidate_count=1
    )
    with pytest.raises(ValueError, match="different query"):
        bound_engine.prepare_native_query(
            np.asarray([1.0], dtype=np.float32),
            snapshot_id=0,
            k=1,
            candidate_count=1,
            candidate_set=bound_c,
        )
    with pytest.raises(ValueError, match="different snapshot"):
        bound_engine.prepare_native_query(
            query,
            snapshot_id=1,
            k=1,
            candidate_count=1,
            candidate_set=bound_c,
        )


def _exact_result_tau_squared(
    query: np.ndarray,
    result: object,
    records: tuple[VectorRecord, ...],
) -> Fraction:
    hits = result.hits  # type: ignore[attr-defined]
    assert hits
    return exact_distance_for_key(query, records, hits[-1].key)


def _run_native_fixed_seed_cases(count: int) -> None:
    rng = np.random.default_rng(NATIVE_RANDOM_SEED)
    scales = np.asarray([2.0**-30, 1.0, 2.0**20, 1.0e8], dtype=np.float64)
    beta_factors = np.asarray([0.0, 0.0, 0.01, 0.25, 1.5], dtype=np.float64)

    for case_id in range(count):
        dimension = int(rng.integers(1, 9))
        scale = float(scales[int(rng.integers(0, len(scales)))])
        query = (rng.normal(size=dimension) * scale).astype(np.float32)
        if case_id % 17 == 0:
            query[:] = 0.0

        next_id = 0
        candidate_records: list[VectorRecord] = []
        for _ in range(int(rng.integers(1, 6))):
            vector = (rng.normal(size=dimension) * scale).astype(np.float32)
            if case_id % 19 == 0:
                vector[:] = np.float32(scale)
            candidate_records.append(VectorRecord(next_id, 0, vector))
            next_id += 1

        raw_records: list[VectorRecord] = []
        for _ in range(int(rng.integers(0, 4))):
            vector = (rng.normal(size=dimension) * scale).astype(np.float32)
            if case_id % 23 == 0:
                vector[:] = query
            raw_records.append(VectorRecord(next_id, 0, vector))
            next_id += 1

        groups: list[Group] = []
        grouped_records: list[VectorRecord] = []
        for group_id in range(int(rng.integers(0, 4))):
            center = (rng.normal(size=dimension) * scale).astype(np.float32)
            members: list[VectorRecord] = []
            for _ in range(int(rng.integers(0, 4))):
                spread = scale * float(rng.choice([0.0, 0.01, 0.5, 3.0]))
                vector = (center + rng.normal(size=dimension) * spread).astype(
                    np.float32
                )
                members.append(VectorRecord(next_id, 0, vector))
                next_id += 1
            members_tuple = tuple(members)
            grouped_records.extend(members_tuple)
            groups.append(
                _bounded_group(group_id, center, members_tuple, dimension)
            )

        candidates = _candidate_set(tuple(candidate_records), dimension)
        raw = tuple(raw_records)
        grouped = tuple(grouped_records)
        engine = _engine(dimension, raw=raw, groups=tuple(groups))
        population = len(candidate_records) + len(raw) + len(grouped)
        k = int(rng.integers(1, population + 1))
        beta = float(
            beta_factors[int(rng.integers(0, len(beta_factors)))]
            * max(scale, 1.0)
        )
        oracle = reference_topk(
            query,
            candidates,
            raw_records=raw,
            grouped_records=grouped,
            k=k,
        )
        prepared = engine.prepare_native_query(
            query,
            k=k,
            candidate_count=len(candidate_records),
            candidate_set=candidates,
        )
        before_calls = native_call_counter()
        full = engine.search_native_prepared(
            prepared, k=k, beta=beta, mode="F"
        )
        no_skip = engine.search_native_prepared(
            prepared, k=k, beta=beta, mode="N"
        )
        proposed = engine.search_native_prepared(
            prepared, k=k, beta=beta, mode="P", audit=True
        )
        rescan = engine.search_native_prepared(
            prepared,
            k=k,
            beta=beta,
            mode="P",
            threshold_mode="rescan",
            audit=True,
        )
        exact_boundary = engine.search_native_prepared(
            prepared,
            k=k,
            beta=beta,
            mode="P",
            rank_strategy="all_boundary_exact",
        )
        context = (
            f"native fixed-seed case {case_id}, beta={beta}, "
            f"k={k}, dim={dimension}"
        )
        assert native_call_counter() == before_calls + 5, context
        assert _keys(full) == oracle.keys, context
        assert _keys(no_skip) == oracle.keys, context
        assert _keys(proposed) == _keys(rescan) == _keys(exact_boundary), context
        assert proposed.receipt.groups_scanned == rescan.receipt.groups_scanned, context
        assert proposed.receipt.groups_skipped == rescan.receipt.groups_skipped, context
        assert proposed.receipt.audit is not None, context
        assert rescan.receipt.audit is not None, context
        assert [
            (row["group_id"], row["action"]) for row in proposed.receipt.audit
        ] == [
            (row["group_id"], row["action"]) for row in rescan.receipt.audit
        ], context
        assert proposed.receipt.certificate_details[
            "native_trusted_prevalidated_inputs"
        ], context
        if beta == 0.0:
            assert _keys(proposed) == oracle.keys, context

        assert oracle.tau_squared is not None
        all_records = tuple(candidate_records) + raw + grouped
        proposed_tau = _exact_result_tau_squared(query, proposed, all_records)
        assert proposed_tau >= oracle.tau_squared, context
        observed = observed_gap_bracket(proposed_tau, oracle.tau_squared)
        assert observed.lower >= 0.0, context
        assert proposed.receipt.certified_beta is not None, context
        assert observed.upper <= proposed.receipt.certified_beta, context
        assert proposed.receipt.certified_beta <= beta, context
        assert proposed.receipt.candidate_set_id_or_hash == "frozen-native-C", context


@pytest.mark.native_fixed_seed(
    seed=NATIVE_RANDOM_SEED, case_count=NATIVE_CI_CASES
)
def test_native_fixed_seed_ci_prefix_matches_fraction_oracle() -> None:
    _run_native_fixed_seed_cases(NATIVE_CI_CASES)


@pytest.mark.evaluation
@pytest.mark.native_fixed_seed(
    seed=NATIVE_RANDOM_SEED, case_count=NATIVE_EVALUATION_CASES
)
def test_native_fixed_seed_10000_matches_fraction_oracle(
    record_property: Callable[[str, object], None],
) -> None:
    """Exactly 10,000 deterministic native F/N/P/toggle comparisons."""

    record_property("native_fixed_seed", NATIVE_RANDOM_SEED)
    record_property("native_case_count", NATIVE_EVALUATION_CASES)
    _run_native_fixed_seed_cases(NATIVE_EVALUATION_CASES)
