from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from searchability.groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from searchability.models import CandidateSet, VectorRecord
from searchability.oracle import (
    exact_distance_for_key,
    exact_group_radius_squared,
    observed_gap_bracket,
    reference_topk,
    sqrt_bracket,
)
from searchability.search import SearchEngine


RANDOM_SEED = 0x5E_A2_C4
CI_CASES = 200
EVALUATION_CASES = 10_000


@dataclass
class _BaseStub:
    dimension: int
    generation_id: str = "randomized-frozen-generation"
    covered_commit_seq: int | None = 0

    def prepare_candidates(self, *args: object, **kwargs: object) -> CandidateSet:
        raise AssertionError("randomized checks must reuse their frozen C")


def _candidate_set(records: tuple[VectorRecord, ...], dimension: int) -> CandidateSet:
    matrix = np.ascontiguousarray(
        np.stack([record.vector for record in records]), dtype=np.float32
    )
    return CandidateSet(
        records=records,
        vectors=matrix,
        candidate_hash="randomized-frozen-C",
        visibility_rejections=0,
        search_ns=0,
        requested_count=len(records),
    )


def _make_group(
    group_id: int,
    center: np.ndarray,
    records: tuple[VectorRecord, ...],
    dimension: int,
) -> Group:
    matrix = (
        np.ascontiguousarray(
            np.stack([record.vector for record in records]), dtype=np.float32
        )
        if records
        else np.empty((0, dimension), dtype=np.float32)
    )
    required = exact_group_radius_squared(
        center, (record.vector for record in records)
    )
    radius_upper = sqrt_bracket(required)[1]
    return Group(
        group_id=group_id,
        center=center,
        radius_upper=radius_upper,
        members=records,
        vectors=matrix,
    )


def _exact_result_tau(
    query: np.ndarray,
    result: object,
    all_records: tuple[VectorRecord, ...],
) -> Fraction:
    hits = result.hits  # type: ignore[attr-defined]
    assert hits
    return exact_distance_for_key(query, all_records, hits[-1].key)


def _run_fixed_seed_cases(count: int) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    scales = np.asarray([2.0**-12, 1.0, 2.0**12, 1.0e8], dtype=np.float64)
    beta_factors = np.asarray([0.0, 0.0, 0.01, 0.25, 1.5], dtype=np.float64)

    for case_id in range(count):
        dimension = int(rng.integers(1, 9))
        scale = float(scales[int(rng.integers(0, len(scales)))])
        query = (rng.normal(size=dimension) * scale).astype(np.float32)
        if case_id % 17 == 0:
            query[:] = 0.0

        next_id = 0
        candidate_records: list[VectorRecord] = []
        for _ in range(int(rng.integers(1, 7))):
            vector = (rng.normal(size=dimension) * scale).astype(np.float32)
            if case_id % 19 == 0:
                vector[:] = np.float32(scale)
            candidate_records.append(VectorRecord(next_id, 0, vector))
            next_id += 1

        raw_records: list[VectorRecord] = []
        for _ in range(int(rng.integers(0, 5))):
            vector = (rng.normal(size=dimension) * scale).astype(np.float32)
            if case_id % 23 == 0:
                vector[:] = query
            raw_records.append(VectorRecord(next_id, 0, vector))
            next_id += 1

        groups: list[Group] = []
        grouped_records: list[VectorRecord] = []
        for group_id in range(int(rng.integers(0, 5))):
            center = (rng.normal(size=dimension) * scale).astype(np.float32)
            members: list[VectorRecord] = []
            for _ in range(int(rng.integers(0, 5))):
                # Local clouds give both useful and weak lower-bound cases.
                spread = scale * float(rng.choice([0.0, 0.01, 0.5, 3.0]))
                vector = (center + rng.normal(size=dimension) * spread).astype(
                    np.float32
                )
                members.append(VectorRecord(next_id, 0, vector))
                next_id += 1
            members_tuple = tuple(members)
            grouped_records.extend(members_tuple)
            groups.append(_make_group(group_id, center, members_tuple, dimension))

        candidate_tuple = tuple(candidate_records)
        raw_tuple = tuple(raw_records)
        grouped_tuple = tuple(grouped_records)
        candidates = _candidate_set(candidate_tuple, dimension)
        directory = (
            GroupDirectory(
                groups=tuple(groups),
                dimension=dimension,
                build_stats=GroupBuildStats(0, 0, 0, 0, 0),
            )
            if groups
            else None
        )
        engine = SearchEngine(  # type: ignore[arg-type]
            _BaseStub(dimension),
            DeltaStore(raw_records=raw_tuple, grouped=directory),
        )

        total_population = len(candidate_tuple) + len(raw_tuple) + len(grouped_tuple)
        k = int(rng.integers(1, total_population + 1))
        beta = float(
            beta_factors[int(rng.integers(0, len(beta_factors)))] * max(scale, 1.0)
        )
        oracle = reference_topk(
            query,
            candidates,
            raw_records=raw_tuple,
            grouped_records=grouped_tuple,
            k=k,
        )
        full = engine.search_full_scan(
            query, k=k, beta=beta, candidate_set=candidates
        )
        proposed = engine.search_pruned(
            query, k=k, beta=beta, candidate_set=candidates
        )

        context = f"fixed-seed case {case_id}, beta={beta}, k={k}, dim={dimension}"
        assert tuple(hit.key for hit in full.hits) == oracle.keys, context
        if beta == 0.0:
            assert tuple(hit.key for hit in proposed.hits) == oracle.keys, context

        assert oracle.tau_squared is not None
        all_records = candidate_tuple + raw_tuple + grouped_tuple
        proposed_tau = _exact_result_tau(query, proposed, all_records)
        assert proposed_tau >= oracle.tau_squared, context
        observed = observed_gap_bracket(proposed_tau, oracle.tau_squared)
        assert observed.lower >= 0.0, context
        assert proposed.receipt.certified_beta is not None, context
        assert observed.upper <= proposed.receipt.certified_beta, context
        assert proposed.receipt.certified_beta <= beta, context
        assert proposed.receipt.candidate_set_id_or_hash == "randomized-frozen-C"


def test_fixed_seed_randomized_ci_subset() -> None:
    """Fast deterministic prefix run by the default CI test target."""

    _run_fixed_seed_cases(CI_CASES)


@pytest.mark.evaluation
def test_fixed_seed_ten_thousand_random_cases() -> None:
    """The Issue #1 acceptance run: exactly 10,000 deterministic small cases."""

    _run_fixed_seed_cases(EVALUATION_CASES)
