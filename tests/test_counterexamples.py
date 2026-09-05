"""Executable counterexamples required by Issue #1.

Each test first makes the tempting broken rule concrete, then checks the safe
prototype behavior or the invariant that rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from searchability.groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from searchability.models import CandidateSet, VectorRecord
from searchability.search import SearchEngine


@dataclass
class _FrozenBase:
    dimension: int = 1
    generation_id: str = "counterexample-generation"
    covered_commit_seq: int | None = 0

    def prepare_candidates(self, *args: object, **kwargs: object) -> CandidateSet:
        raise AssertionError("counterexamples pass an already frozen C")


def _record(
    logical_id: int,
    value: float,
    *,
    version_id: int = 0,
    begin_seq: int = 0,
    end_seq: int | None = None,
) -> VectorRecord:
    return VectorRecord(
        logical_id=logical_id,
        version_id=version_id,
        vector=np.asarray([value], dtype=np.float32),
        begin_seq=begin_seq,
        end_seq=end_seq,
    )


def _candidates(*records: VectorRecord) -> CandidateSet:
    vectors = (
        np.stack([record.vector for record in records]).astype(np.float32)
        if records
        else np.empty((0, 1), dtype=np.float32)
    )
    return CandidateSet(tuple(records), vectors, "counterexample-C", 0, 0, len(records))


def _directory(group: Group) -> GroupDirectory:
    return GroupDirectory((group,), 1, GroupBuildStats(0, 0, 0, 0, 0))


def test_delete_after_pruning_has_a_minimal_false_skip_counterexample() -> None:
    """A deleted near item must not establish tau before group decisions."""

    deleted_near = _record(1, 1.0, end_seq=1)
    live_far = _record(2, 100.0)
    member = _record(3, 20.0)
    group = Group(0, np.asarray([20.0], np.float32), 0.0, (member,), member.vector[None, :])

    # Broken post-filtering: tau=1 from the deleted item, so LB=20 is skipped;
    # deletion then leaves distance 100 although the reference answer is 20.
    broken_tau_before_delete = 1.0
    assert 20.0 > broken_tau_before_delete
    broken_answer_after_delete = live_far.logical_id
    assert broken_answer_after_delete != member.logical_id

    # Safe pipeline freezes visibility first.  C contains only the live item,
    # hence the group is scanned and the nearer member is returned.
    engine = SearchEngine(_FrozenBase(), DeltaStore((), _directory(group)))  # type: ignore[arg-type]
    safe = engine.search_pruned(
        np.asarray([0.0], np.float32),
        k=1,
        beta=0.0,
        snapshot_id=1,
        candidate_set=_candidates(live_far),
        audit=True,
    )
    assert safe.ids == (member.logical_id,)
    assert safe.receipt.audit is not None
    assert safe.receipt.audit[0]["action"] == "scan"
    assert not deleted_near.visible_at(1)


def test_current_latest_table_cannot_answer_an_old_snapshot() -> None:
    old = _record(7, 0.0, version_id=1, begin_seq=1, end_seq=3)
    current = _record(7, 100.0, version_id=2, begin_seq=3)
    snapshot = 2
    naive_current_latest = current
    assert not naive_current_latest.visible_at(snapshot)
    visible_by_intervals = [row for row in (old, current) if row.visible_at(snapshot)]
    assert [row.version_id for row in visible_by_intervals] == [1]


def test_moving_center_without_recomputing_radius_is_rejected() -> None:
    member = _record(1, 1.0)
    original = Group(
        0,
        np.asarray([0.0], np.float32),
        1.0,
        (member,),
        member.vector[None, :],
    )
    assert original.radius_upper == 1.0

    # If accepted, the broken metadata would claim LB=99 for a member at
    # distance 1 from query zero and could falsely skip it against tau=50.
    with pytest.raises(ValueError, match="does not cover"):
        Group(
            1,
            np.asarray([100.0], np.float32),
            1.0,
            (member,),
            member.vector[None, :],
        )


def test_small_beta_distance_contract_does_not_imply_id_recall() -> None:
    base = tuple(_record(index, 10.0) for index in range(10))
    delta = tuple(_record(100 + index, 9.9) for index in range(10))
    matrix = np.stack([record.vector for record in delta]).astype(np.float32)
    group = Group(0, np.asarray([9.9], np.float32), 0.0, delta, matrix)
    engine = SearchEngine(_FrozenBase(), DeltaStore((), _directory(group)))  # type: ignore[arg-type]
    frozen = _candidates(*base)
    query = np.asarray([0.0], np.float32)
    reference = engine.search_full_scan(query, k=10, candidate_set=frozen)
    proposed = engine.search_pruned(
        query, k=10, beta=0.2, candidate_set=frozen, audit=True
    )

    assert set(reference.ids).isdisjoint(proposed.ids)
    assert proposed.receipt.groups_skipped == 1
    assert 0.0 < proposed.hits[-1].distance - reference.hits[-1].distance < 0.2
    assert proposed.receipt.certified_beta is not None
    assert proposed.receipt.certified_beta <= 0.2

