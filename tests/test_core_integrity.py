from __future__ import annotations

import math

import numpy as np
import pytest

from searchability.base import BaseIndex
from searchability.datasets import DatasetSplit, synthetic_dataset
from searchability.groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from searchability.models import (
    CandidateSet,
    VectorRecord,
    frozen_candidate_hash,
    vector_query_hash,
)
from searchability.numerics import distance_intervals
from searchability.search import SearchEngine


def _record(identifier: int, value: float, **kwargs: object) -> VectorRecord:
    return VectorRecord(
        logical_id=identifier,
        version_id=int(kwargs.pop("version_id", 0)),
        vector=np.asarray([value], dtype=np.float32),
        **kwargs,
    )


def _group(group_id: int, record: VectorRecord) -> Group:
    return Group(
        group_id=group_id,
        center=record.vector,
        radius_upper=0.0,
        members=(record,),
        vectors=record.vector[None, :],
    )


def _assert_deeply_readonly(array: np.ndarray) -> None:
    assert array.flags.c_contiguous
    assert not array.flags.owndata
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)


def test_all_public_vector_arrays_use_immutable_bytes_backing() -> None:
    record = _record(1, -0.0)
    base = BaseIndex((record,), generation_id="immutable", threads=1)
    candidate = base.prepare_candidates(
        np.asarray([0.0], dtype=np.float32),
        snapshot_id=0,
        k=1,
        candidate_count=1,
    )
    group = _group(0, record)
    _, visible_vectors = group.visible(0)
    future = VectorRecord(2, 0, np.asarray([1.0], dtype=np.float32), begin_seq=2)
    empty_group = _group(1, future)
    _, empty_visible_vectors = empty_group.visible(0)
    centers, _ = GroupDirectory.train_centers(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        1,
        seed=0,
        iterations=1,
        threads=1,
    )
    dataset = synthetic_dataset(
        kind="isotropic",
        n_base=2,
        n_delta=1,
        n_validation=1,
        n_test=1,
        dimension=2,
        seed=7,
    )

    float_arrays = (
        record.vector,
        base.vectors,
        candidate.vectors,
        group.center,
        group.vectors,
        visible_vectors,
        empty_visible_vectors,
        centers,
        dataset.base,
        dataset.delta,
        dataset.validation_queries,
        dataset.test_queries,
    )
    for array in float_arrays:
        assert array.dtype == np.float32
        _assert_deeply_readonly(array)
    for array in (
        dataset.base_ids,
        dataset.delta_ids,
        dataset.validation_query_ids,
        dataset.test_query_ids,
    ):
        assert array.dtype == np.int64
        _assert_deeply_readonly(array)

    # Canonicalization retains exact IEEE-754 payload bits, including -0.0.
    assert record.vector.tobytes() == np.asarray([-0.0], dtype=np.float32).tobytes()


def test_dataset_split_direct_constructor_copies_mutable_inputs() -> None:
    source = np.asarray([[3.0]], dtype=np.float32)
    empty = np.empty((0, 1), dtype=np.float32)
    empty_ids = np.empty(0, dtype=np.int64)
    split = DatasetSplit(
        name="direct",
        base=source,
        delta=empty,
        validation_queries=empty,
        test_queries=empty,
        base_ids=np.asarray([4], dtype=np.int64),
        delta_ids=empty_ids,
        validation_query_ids=empty_ids,
        test_query_ids=empty_ids,
        metadata={},
    )
    source[0, 0] = 99.0
    assert split.base[0, 0] == np.float32(3.0)
    _assert_deeply_readonly(split.base)


def test_group_member_cannot_be_mutated_after_radius_was_certified() -> None:
    """Regression for a mutation that could make a skipped group relevant."""

    query = np.asarray([0.0], dtype=np.float32)
    base = BaseIndex((_record(1, 50.0),), generation_id="p0-regression", threads=1)
    frozen = base.prepare_candidates(query, snapshot_id=0, k=1, candidate_count=1)
    member = _record(2, 100.0)
    group = _group(0, member)

    # Before the bytes-backed boundary an owning, write-disabled vector could
    # re-enable writes here, diverging from Group.vectors/radius and yielding a
    # false beta=0 certificate after changing 100 to 0.
    with pytest.raises(ValueError, match="WRITEABLE"):
        member.vector.setflags(write=True)
    assert member.vector[0] == np.float32(100.0)

    engine = SearchEngine(
        base,
        DeltaStore(
            raw_records=(),
            grouped=GroupDirectory(
                (group,), 1, GroupBuildStats(0, 0, 0, 0, 0)
            ),
        ),
    )
    result = engine.search_pruned(
        query, k=1, beta=0.0, candidate_set=frozen
    )
    assert result.ids == (1,)
    assert result.receipt.groups_skipped == 1
    assert result.receipt.certificate_status == "certified"


def test_delta_store_freezes_raw_sequence_and_validates_duplicate_content() -> None:
    record = _record(1, 1.0)
    supplied = [record]
    store = DeltaStore(raw_records=supplied)  # type: ignore[arg-type]
    supplied.clear()
    assert store.raw_records == (record,)

    with pytest.raises(TypeError, match="VectorRecord"):
        DeltaStore(raw_records=[object()])  # type: ignore[list-item,arg-type]

    conflicting = _record(1, 2.0)
    grouped = GroupDirectory(
        (_group(0, conflicting),), 1, GroupBuildStats(0, 0, 0, 0, 0)
    )
    with pytest.raises(ValueError, match="conflicting stored content"):
        DeltaStore(raw_records=(record,), grouped=grouped)

    # Identical raw/group overlap remains valid: this is the documented atomic
    # handoff state and SearchEngine forces its safe full-Delta path.
    identical_grouped = GroupDirectory(
        (_group(0, record),), 1, GroupBuildStats(0, 0, 0, 0, 0)
    )
    overlap = DeltaStore(raw_records=(record,), grouped=identical_grouped)
    assert overlap.raw_records == (record,)


def test_candidate_and_group_matrices_must_bit_match_their_records() -> None:
    positive_zero = _record(1, 0.0)
    negative_zero_matrix = np.asarray([[-0.0]], dtype=np.float32)
    assert positive_zero.vector[0] == negative_zero_matrix[0, 0]
    assert positive_zero.vector.tobytes() != negative_zero_matrix[0].tobytes()

    with pytest.raises(ValueError, match="bit-match"):
        CandidateSet(
            records=(positive_zero,),
            vectors=negative_zero_matrix,
            candidate_hash="unbound-test-C",
            visibility_rejections=0,
            search_ns=0,
            requested_count=1,
        )
    with pytest.raises(ValueError, match="bit-match"):
        Group(
            group_id=0,
            center=np.asarray([0.0], dtype=np.float32),
            radius_upper=0.0,
            members=(positive_zero,),
            vectors=negative_zero_matrix,
        )


def test_group_directory_rejects_duplicate_group_ids_and_version_keys() -> None:
    first = _record(1, 1.0)
    second = _record(2, 2.0)
    stats = GroupBuildStats(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="duplicate group IDs"):
        GroupDirectory((_group(7, first), _group(7, second)), 1, stats)
    with pytest.raises(ValueError, match="more than one group"):
        GroupDirectory((_group(7, first), _group(8, first)), 1, stats)


def test_complex_matrix_inputs_are_rejected_before_any_real_cast() -> None:
    complex_matrix = np.asarray([[1.0 + 2.0j]], dtype=np.complex64)
    with pytest.raises(TypeError, match="real numeric"):
        distance_intervals(np.asarray([0.0], dtype=np.float32), complex_matrix)
    with pytest.raises(TypeError, match="real numeric"):
        BaseIndex.from_arrays(complex_matrix, threads=1)
    with pytest.raises(TypeError, match="real numeric"):
        GroupDirectory.from_centers(complex_matrix, (_record(1, 1.0),), threads=1)


def test_production_search_rejects_candidate_scope_substitution() -> None:
    query = np.asarray([0.0], dtype=np.float32)
    records = (_record(1, 1.0), _record(2, 2.0))
    base = BaseIndex(records, generation_id="bound-A", m=4, ef_search=8, threads=1)
    frozen = base.prepare_candidates(query, snapshot_id=0, k=1, candidate_count=2)
    assert SearchEngine(base, DeltaStore(())).search_full_scan(
        query, k=1, candidate_set=frozen
    ).ids == (1,)

    with pytest.raises(ValueError, match="different query"):
        SearchEngine(base, DeltaStore(())).search_full_scan(
            np.asarray([3.0], dtype=np.float32), k=1, candidate_set=frozen
        )
    with pytest.raises(ValueError, match="different snapshot"):
        SearchEngine(base, DeltaStore(())).search_full_scan(
            query, k=1, snapshot_id=1, candidate_set=frozen
        )

    other_generation = BaseIndex(
        records, generation_id="bound-B", m=4, ef_search=8, threads=1
    )
    with pytest.raises(ValueError, match="different base generation"):
        SearchEngine(other_generation, DeltaStore(())).search_full_scan(
            query, k=1, candidate_set=frozen
        )

    changed_universe = BaseIndex(
        (_record(1, 9.0), _record(2, 2.0)),
        generation_id="bound-A",
        m=4,
        ef_search=8,
        threads=1,
    )
    with pytest.raises(ValueError, match="different base universe"):
        SearchEngine(changed_universe, DeltaStore(())).search_full_scan(
            query, k=1, candidate_set=frozen
        )


def test_forged_candidate_content_is_rejected_against_bound_base() -> None:
    query = np.asarray([0.0], dtype=np.float32)
    base = BaseIndex((_record(1, 1.0),), generation_id="bound", threads=1)
    outsider = _record(99, 0.0)
    matrix = outsider.vector[None, :]
    query_hash = vector_query_hash(query)
    digest = frozen_candidate_hash(
        (outsider,),
        matrix,
        base_generation=base.generation_id,
        base_universe_hash=base.universe_hash,
        snapshot_id=0,
        query_hash=query_hash,
        requested_count=1,
    )
    forged = CandidateSet(
        records=(outsider,),
        vectors=matrix,
        candidate_hash=digest,
        visibility_rejections=0,
        search_ns=0,
        requested_count=1,
        snapshot_id=0,
        base_generation=base.generation_id,
        base_universe_hash=base.universe_hash,
        query_hash=query_hash,
    )
    with pytest.raises(ValueError, match="outside its bound base"):
        SearchEngine(base, DeltaStore(())).search_full_scan(
            query, k=1, candidate_set=forged
        )


def test_ann_underfill_scans_visible_base_and_records_fallback() -> None:
    records = (_record(1, 30.0), _record(2, 2.0), _record(3, 1.0))
    base = BaseIndex(records, generation_id="underfill", m=4, ef_search=1, threads=1)

    class _UnderfilledIndex:
        def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
            distances = np.full((1, count), math.inf, dtype=np.float32)
            labels = np.full((1, count), -1, dtype=np.int64)
            labels[0, 0] = 0
            return distances, labels

    base.index = _UnderfilledIndex()  # type: ignore[assignment]
    frozen = base.prepare_candidates(
        np.asarray([0.0], dtype=np.float32), snapshot_id=0, k=3, candidate_count=3
    )
    assert len(frozen.records) == 3
    assert frozen.supplemented_count == 2
    assert frozen.available_visible_count == 3
    assert frozen.fallback_reason == "ann_underfill_full_visible_supplement"
    assert {record.key for record in frozen.records} == {record.key for record in records}


def test_visible_base_population_underfill_is_explicit() -> None:
    base = BaseIndex((_record(1, 1.0), _record(2, 2.0)), threads=1)
    frozen = base.prepare_candidates(
        np.asarray([0.0], dtype=np.float32), snapshot_id=0, k=5, candidate_count=5
    )
    assert len(frozen.records) == frozen.available_visible_count == 2
    assert frozen.fallback_reason == "base_visible_population_below_requested"


def test_audit_preserves_rational_decision_values_and_version_keys() -> None:
    query = np.asarray([0.0], dtype=np.float32)
    base = BaseIndex((_record(1, 10.0),), generation_id="audit", threads=1)
    frozen = base.prepare_candidates(query, snapshot_id=0, k=1, candidate_count=1)
    member = _record(2, 9.0, version_id=4)
    directory = GroupDirectory(
        (_group(3, member),), 1, GroupBuildStats(0, 0, 0, 0, 0)
    )
    result = SearchEngine(base, DeltaStore((), directory)).search_pruned(
        query, k=1, beta=2.0, candidate_set=frozen, audit=True
    )
    assert result.receipt.audit is not None
    decision = result.receipt.audit[0]
    assert decision["member_version_keys"] == [[2, 4]]
    for key in (
        "radius_upper_fraction",
        "center_distance_lower_fraction",
        "center_distance_upper_fraction",
        "lb_lower_fraction",
        "tau_upper_at_decision_fraction",
        "threshold_tau_minus_beta_fraction",
        "requested_beta_fraction",
    ):
        assert isinstance(decision[key], str) and "/" in decision[key]
    details = result.receipt.certificate_details
    assert details is not None
    assert details["candidate_version_keys"] == [[1, 0]]
    assert details["returned_version_keys"] == [[1, 0]]
    assert details["requested_beta_fraction"] == "2/1"


def test_component_timing_names_do_not_double_book_full_scan_work() -> None:
    query = np.asarray([0.0], dtype=np.float32)
    base = BaseIndex((_record(1, 1.0),), threads=1)
    frozen = base.prepare_candidates(query, snapshot_id=0, k=1, candidate_count=1)
    member = _record(2, 2.0)
    directory = GroupDirectory(
        (_group(0, member),), 1, GroupBuildStats(0, 0, 0, 0, 0)
    )
    engine = SearchEngine(base, DeltaStore((_record(3, 3.0),), directory))
    full = engine.search_full_scan(query, k=1, candidate_set=frozen)
    pruned = engine.search_pruned(query, k=1, beta=0.0, candidate_set=frozen)

    assert "candidate_delta_distance_ns" in full.receipt.component_timings
    assert "raw_scan_ns" not in full.receipt.component_timings
    assert full.receipt.component_timings["group_scan_ns"] == 0
    assert "candidate_raw_distance_ns" in pruned.receipt.component_timings
    assert "raw_scan_ns" not in pruned.receipt.component_timings
