from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import pytest

from searchability.groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from searchability.models import CandidateSet, VectorRecord
from searchability.oracle import (
    OracleResult,
    exact_distance_for_key,
    exact_group_radius_squared,
    observed_gap_bracket,
    reference_topk,
    sqrt_bracket,
)
from searchability.search import SearchEngine


@dataclass
class _FrozenBaseStub:
    dimension: int
    generation_id: str = "frozen-test-generation"
    covered_commit_seq: int | None = 0

    def prepare_candidates(self, *args: object, **kwargs: object) -> CandidateSet:
        raise AssertionError("tests must pass one frozen candidate set C")


def _record(
    logical_id: int,
    values: list[float] | tuple[float, ...],
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


def _candidate_set(
    records: tuple[VectorRecord, ...], dimension: int, *, identity: str = "frozen-C"
) -> CandidateSet:
    vectors = (
        np.ascontiguousarray(np.stack([record.vector for record in records]), dtype=np.float32)
        if records
        else np.empty((0, dimension), dtype=np.float32)
    )
    return CandidateSet(
        records=records,
        vectors=vectors,
        candidate_hash=identity,
        visibility_rejections=0,
        search_ns=0,
        requested_count=len(records),
    )


def _safe_radius(center: np.ndarray, records: tuple[VectorRecord, ...]) -> float:
    squared = exact_group_radius_squared(
        center, (record.vector for record in records)
    )
    return sqrt_bracket(squared)[1]


def _group(
    group_id: int,
    center: list[float] | tuple[float, ...],
    records: tuple[VectorRecord, ...],
    *,
    radius_upper: float | None = None,
) -> Group:
    center_vector = np.asarray(center, dtype=np.float32)
    vectors = (
        np.ascontiguousarray(np.stack([record.vector for record in records]), dtype=np.float32)
        if records
        else np.empty((0, center_vector.shape[0]), dtype=np.float32)
    )
    radius = _safe_radius(center_vector, records) if radius_upper is None else radius_upper
    return Group(
        group_id=group_id,
        center=center_vector,
        radius_upper=radius,
        members=records,
        vectors=vectors,
    )


def _directory(dimension: int, *groups: Group) -> GroupDirectory:
    return GroupDirectory(
        groups=tuple(groups),
        dimension=dimension,
        build_stats=GroupBuildStats(0, 0, 0, 0, 0),
    )


def _engine(
    dimension: int,
    *,
    raw: tuple[VectorRecord, ...] = (),
    groups: tuple[Group, ...] = (),
) -> SearchEngine:
    grouped = _directory(dimension, *groups) if groups else None
    return SearchEngine(  # type: ignore[arg-type]
        _FrozenBaseStub(dimension), DeltaStore(raw_records=raw, grouped=grouped)
    )


def _flatten(groups: tuple[Group, ...]) -> tuple[VectorRecord, ...]:
    return tuple(record for group in groups for record in group.members)


def _result_keys(result: object) -> tuple[tuple[int, int], ...]:
    return tuple(hit.key for hit in result.hits)  # type: ignore[attr-defined]


def _assert_result_matches_oracle(result: object, oracle: OracleResult) -> None:
    assert _result_keys(result) == oracle.keys
    for actual, expected in zip(result.hits, oracle.hits):  # type: ignore[attr-defined]
        assert expected.distance_lower <= actual.distance <= expected.distance_upper


def test_empty_delta_and_empty_total_population() -> None:
    query = np.array([0.0, 0.0], dtype=np.float32)
    empty_candidates = _candidate_set((), 2)
    engine = _engine(2)
    full = engine.search_full_scan(query, k=3, candidate_set=empty_candidates)
    proposed = engine.search_pruned(
        query, k=3, beta=0.0, candidate_set=empty_candidates
    )
    assert full.hits == proposed.hits == ()
    for result in (full, proposed):
        assert result.receipt.certificate_status == "not_applicable_population_below_k"
        assert result.receipt.certified_beta is None
        assert result.receipt.tau_returned is None


def test_empty_group_is_ignored_without_changing_the_answer() -> None:
    candidate = (_record(1, (1.0,)),)
    candidates = _candidate_set(candidate, 1)
    empty_group = _group(17, (4.0,), ())
    engine = _engine(1, groups=(empty_group,))
    result = engine.search_pruned(
        np.array([0.0], dtype=np.float32),
        k=1,
        beta=0.0,
        candidate_set=candidates,
    )
    assert result.ids == (1,)
    assert result.receipt.groups_scanned == 0
    assert result.receipt.groups_skipped == 0


def test_singleton_radius_zero_and_k_larger_than_population() -> None:
    candidate = (_record(10, (8.0,)),)
    singleton = _record(2, (1.0,))
    group = _group(0, (1.0,), (singleton,), radius_upper=0.0)
    candidates = _candidate_set(candidate, 1)
    engine = _engine(1, groups=(group,))
    query = np.array([0.0], dtype=np.float32)

    result = engine.search_pruned(
        query, k=5, beta=0.0, candidate_set=candidates
    )
    oracle = reference_topk(
        query, candidates, grouped_records=(singleton,), snapshot_id=0, k=5
    )
    _assert_result_matches_oracle(result, oracle)
    assert result.receipt.certificate_status == "not_applicable_population_below_k"
    assert result.receipt.fallback_reason == "population_below_k"


def test_equal_distances_use_logical_then_version_id_tie_break() -> None:
    candidates = _candidate_set(
        (_record(9, (1.0, 0.0)), _record(3, (-1.0, 0.0))), 2
    )
    raw = (_record(1, (0.0, 1.0), version_id=4),)
    grouped_record = _record(5, (0.0, -1.0), version_id=2)
    group = _group(0, (0.0, -1.0), (grouped_record,), radius_upper=0.0)
    engine = _engine(2, raw=raw, groups=(group,))
    query = np.zeros(2, dtype=np.float32)

    full = engine.search_full_scan(query, k=3, candidate_set=candidates)
    proposed = engine.search_pruned(
        query, k=3, beta=0.0, candidate_set=candidates
    )
    oracle = reference_topk(
        query,
        candidates,
        raw_records=raw,
        grouped_records=(grouped_record,),
        k=3,
    )
    assert oracle.keys == ((1, 4), (3, 0), (5, 2))
    _assert_result_matches_oracle(full, oracle)
    _assert_result_matches_oracle(proposed, oracle)


def test_all_identical_vectors_still_have_deterministic_topk() -> None:
    value = (4.0, -2.0)
    candidates = _candidate_set((_record(8, value), _record(4, value)), 2)
    raw = (_record(6, value),)
    member = _record(2, value)
    group = _group(0, value, (member,), radius_upper=0.0)
    engine = _engine(2, raw=raw, groups=(group,))
    result = engine.search_pruned(
        np.asarray(value, dtype=np.float32),
        k=3,
        beta=0.0,
        candidate_set=candidates,
    )
    assert _result_keys(result) == ((2, 0), (4, 0), (6, 0))


def test_beta_zero_matches_same_frozen_c_full_scan_and_exact_oracle() -> None:
    candidates = _candidate_set(
        (_record(0, (10.0, 0.0)), _record(1, (0.0, 9.0))), 2,
        identity="one-and-only-C",
    )
    raw = (_record(2, (3.0, 4.0)), _record(3, (50.0, 50.0)))
    grouped_records = (
        _record(4, (1.0, 1.0)),
        _record(5, (-4.0, 2.0)),
        _record(6, (100.0, -100.0)),
    )
    groups = (
        _group(0, (0.0, 0.0), grouped_records[:2]),
        _group(1, (100.0, -100.0), grouped_records[2:]),
    )
    query = np.zeros(2, dtype=np.float32)
    engine = _engine(2, raw=raw, groups=groups)

    full = engine.search_full_scan(query, k=3, candidate_set=candidates)
    proposed = engine.search_pruned(
        query, k=3, beta=0.0, candidate_set=candidates, audit=True
    )
    oracle = reference_topk(
        query,
        candidates,
        raw_records=raw,
        grouped_records=grouped_records,
        k=3,
    )
    _assert_result_matches_oracle(full, oracle)
    _assert_result_matches_oracle(proposed, oracle)
    assert proposed.receipt.candidate_set_id_or_hash == "one-and-only-C"
    assert full.receipt.candidate_set_id_or_hash == "one-and-only-C"
    assert proposed.receipt.certified_beta == 0.0


def test_positive_beta_observed_gap_is_bounded_by_certificate_and_request() -> None:
    candidate_records = (_record(10, (10.0,)),)
    candidates = _candidate_set(candidate_records, 1)
    omitted = _record(20, (9.0,))
    group = _group(0, (9.0,), (omitted,), radius_upper=0.0)
    query = np.array([0.0], dtype=np.float32)
    engine = _engine(1, groups=(group,))

    full = engine.search_full_scan(query, k=1, candidate_set=candidates)
    proposed = engine.search_pruned(
        query, k=1, beta=2.0, candidate_set=candidates, audit=True
    )
    reference = reference_topk(
        query, candidates, grouped_records=(omitted,), k=1
    )
    _assert_result_matches_oracle(full, reference)
    assert proposed.ids == (10,)
    proposed_squared = exact_distance_for_key(
        query, candidate_records + (omitted,), proposed.hits[-1].key
    )
    assert reference.tau_squared is not None
    observed = observed_gap_bracket(proposed_squared, reference.tau_squared)
    assert observed.lower <= 1.0 <= observed.upper
    assert proposed.receipt.certified_beta is not None
    assert observed.upper <= proposed.receipt.certified_beta
    assert proposed.receipt.certified_beta <= proposed.receipt.requested_beta == 2.0
    assert proposed.receipt.groups_skipped == 1


def test_positive_beta_without_skip_has_zero_observed_and_certified_gap() -> None:
    candidates = _candidate_set((_record(1, (10.0,)),), 1)
    member = _record(2, (1.0,))
    group = _group(0, (1.0,), (member,), radius_upper=0.0)
    query = np.array([0.0], dtype=np.float32)
    result = _engine(1, groups=(group,)).search_pruned(
        query, k=1, beta=0.5, candidate_set=candidates
    )
    reference = reference_topk(query, candidates, grouped_records=(member,), k=1)
    _assert_result_matches_oracle(result, reference)
    assert result.receipt.certified_beta == 0.0
    assert result.receipt.certificate_status == "certified_no_skip"


@pytest.mark.parametrize(
    ("base_distance", "group_distance", "beta", "expected_action"),
    [
        (3.0, 3.0, 0.0, "scan"),  # exact LB = tau
        (3.0, 2.0, 1.0, "scan"),  # exact LB = tau - beta
        (3.0, float(np.nextafter(np.float32(2.0), np.float32(math.inf))), 1.0, "skip"),
    ],
)
def test_strict_lower_bound_boundaries(
    base_distance: float,
    group_distance: float,
    beta: float,
    expected_action: str,
) -> None:
    candidates = _candidate_set((_record(1, (base_distance,)),), 1)
    member = _record(2, (group_distance,))
    group = _group(0, (group_distance,), (member,), radius_upper=0.0)
    result = _engine(1, groups=(group,)).search_pruned(
        np.array([0.0], dtype=np.float32),
        k=1,
        beta=beta,
        candidate_set=candidates,
        audit=True,
    )
    assert result.receipt.audit is not None
    assert result.receipt.audit[0]["action"] == expected_action


def test_lb_zero_group_is_scanned() -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    member = _record(2, (0.0,))
    group = _group(0, (5.0,), (member,), radius_upper=5.0)
    result = _engine(1, groups=(group,)).search_pruned(
        np.array([0.0], dtype=np.float32),
        k=1,
        beta=0.0,
        candidate_set=candidates,
        audit=True,
    )
    assert result.ids == (2,)
    assert result.receipt.audit is not None
    assert result.receipt.audit[0]["lb_lower"] == 0.0
    assert result.receipt.audit[0]["action"] == "scan"


def test_beta_larger_than_tau_can_skip_a_better_group_but_certifies_gap() -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    member = _record(2, (0.0,))
    group = _group(0, (0.0,), (member,), radius_upper=0.0)
    query = np.array([0.0], dtype=np.float32)
    result = _engine(1, groups=(group,)).search_pruned(
        query, k=1, beta=2.0, candidate_set=candidates
    )
    reference = reference_topk(query, candidates, grouped_records=(member,), k=1)
    assert result.ids == (1,)
    assert reference.keys == ((2, 0),)
    assert result.receipt.certified_beta is not None
    assert 1.0 <= result.receipt.certified_beta <= 2.0


def test_raw_and_grouped_pending_visibility_and_versions() -> None:
    candidates = _candidate_set((_record(100, (20.0,)),), 1)
    raw = (
        _record(7, (2.0,), version_id=0, begin_seq=0, end_seq=5),
        _record(7, (8.0,), version_id=1, begin_seq=5),
        _record(8, (0.5,), begin_seq=10),
    )
    old_group_record = _record(9, (1.0,), begin_seq=0, end_seq=5)
    new_group_record = _record(9, (3.0,), version_id=1, begin_seq=5)
    deleted = _record(11, (0.0,), begin_seq=0, end_seq=1)
    grouped_records = (old_group_record, new_group_record, deleted)
    group = _group(0, (0.0,), grouped_records)
    engine = _engine(1, raw=raw, groups=(group,))
    query = np.array([0.0], dtype=np.float32)

    for snapshot in (0, 5, 10):
        full = engine.search_full_scan(
            query, k=3, snapshot_id=snapshot, candidate_set=candidates
        )
        proposed = engine.search_pruned(
            query,
            k=3,
            beta=0.0,
            snapshot_id=snapshot,
            candidate_set=candidates,
        )
        oracle = reference_topk(
            query,
            candidates,
            raw_records=raw,
            grouped_records=grouped_records,
            snapshot_id=snapshot,
            k=3,
        )
        _assert_result_matches_oracle(full, oracle)
        _assert_result_matches_oracle(proposed, oracle)


def test_all_delta_versions_deleted_returns_only_frozen_c() -> None:
    candidates = _candidate_set((_record(1, (4.0,)),), 1)
    deleted_raw = (_record(2, (0.0,), end_seq=1),)
    deleted_member = _record(3, (1.0,), end_seq=1)
    group = _group(0, (1.0,), (deleted_member,), radius_upper=0.0)
    result = _engine(1, raw=deleted_raw, groups=(group,)).search_pruned(
        np.array([0.0], dtype=np.float32),
        k=1,
        beta=0.0,
        snapshot_id=1,
        candidate_set=candidates,
    )
    assert result.ids == (1,)
    assert result.receipt.raw_pending_scanned == 0
    assert result.receipt.groups_scanned == 0


def test_no_pruning_ablation_scans_every_visible_group() -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    far = _record(2, (100.0,))
    group = _group(0, (100.0,), (far,), radius_upper=0.0)
    result = _engine(1, groups=(group,)).search_pruned(
        np.array([0.0], dtype=np.float32),
        k=1,
        beta=1000.0,
        candidate_set=candidates,
        no_pruning=True,
    )
    assert result.ids == (1,)
    assert result.receipt.groups_scanned == 1
    assert result.receipt.groups_skipped == 0
    assert result.receipt.certified_beta == 0.0


def test_duplicate_during_raw_to_group_handoff_does_not_corrupt_pruning_tau() -> None:
    """A duplicate version is one set member and must not occupy two top-k slots."""

    candidates_records = (_record(2, (11.0,)), _record(4, (12.0,)))
    candidates = _candidate_set(candidates_records, 1)
    overlapping = _record(1, (1.0,))
    relevant = _record(3, (2.0,))
    groups = (
        _group(0, (1.0,), (overlapping,), radius_upper=0.0),
        _group(1, (2.0,), (relevant,), radius_upper=0.0),
    )
    query = np.array([0.0], dtype=np.float32)
    engine = _engine(1, raw=(overlapping,), groups=groups)

    full = engine.search_full_scan(query, k=2, candidate_set=candidates)
    proposed = engine.search_pruned(
        query, k=2, beta=0.0, candidate_set=candidates, audit=True
    )
    oracle = reference_topk(
        query,
        candidates,
        raw_records=(overlapping,),
        grouped_records=_flatten(groups),
        k=2,
    )
    assert oracle.keys == ((1, 0), (3, 0))
    _assert_result_matches_oracle(full, oracle)
    _assert_result_matches_oracle(proposed, oracle)


@pytest.mark.parametrize("beta", [-1.0, math.nan, math.inf, -math.inf])
def test_search_rejects_invalid_beta(beta: float) -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    with pytest.raises(ValueError):
        _engine(1).search_pruned(
            np.array([0.0], dtype=np.float32),
            k=1,
            beta=beta,
            candidate_set=candidates,
        )


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_search_rejects_invalid_k(k: object) -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    with pytest.raises((TypeError, ValueError)):
        _engine(1).search_pruned(
            np.array([0.0], dtype=np.float32),
            k=k,  # type: ignore[arg-type]
            beta=0.0,
            candidate_set=candidates,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1.0e16])
def test_search_rejects_invalid_query_numeric(value: float) -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    with pytest.raises(ValueError):
        _engine(1).search_pruned(
            np.array([value], dtype=np.float64),
            k=1,
            beta=0.0,
            candidate_set=candidates,
        )


def test_search_rejects_query_dimension_mismatch() -> None:
    candidates = _candidate_set((_record(1, (1.0,)),), 1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        _engine(1).search_pruned(
            np.array([0.0, 1.0], dtype=np.float32),
            k=1,
            beta=0.0,
            candidate_set=candidates,
        )
