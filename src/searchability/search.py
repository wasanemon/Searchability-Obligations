"""Reference search engine implementing strict, certified group pruning."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import time
from typing import Iterable, Sequence

import numpy as np

from .base import BaseIndex
from .groups import DeltaStore, Group
from .models import (
    CandidateSet,
    Receipt,
    SearchResult,
    VectorRecord,
    as_float32_vector,
    vector_query_hash,
)
from .numerics import (
    NUMERIC_MODE,
    DistanceIntervals,
    distance_intervals,
    fraction_to_lower_float,
    fraction_to_upper_float,
    kth_upper_bound,
    lower_bound_fraction,
    stable_topk,
    validate_beta,
    validate_k,
)


def _empty_matrix(dimension: int) -> np.ndarray:
    return np.empty((0, dimension), dtype=np.float32)


def _deduplicate(
    records: Sequence[VectorRecord], sources: Sequence[str]
) -> tuple[list[VectorRecord], list[str]]:
    """Keep the newest visible version for each logical ID deterministically."""

    if len(records) != len(sources):
        raise ValueError("records/sources length mismatch")
    chosen: dict[int, tuple[VectorRecord, str, int]] = {}
    for position, (record, source) in enumerate(zip(records, sources)):
        current = chosen.get(record.logical_id)
        rank = (record.begin_seq, record.version_id, position)
        if current is None or rank > (
            current[0].begin_seq,
            current[0].version_id,
            current[2],
        ):
            chosen[record.logical_id] = (record, source, position)
    ordered = sorted(chosen.values(), key=lambda item: item[2])
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _version_keys(records: Sequence[VectorRecord]) -> list[list[int]]:
    return [[record.logical_id, record.version_id] for record in records]


def _same_stored_record(left: VectorRecord, right: VectorRecord) -> bool:
    return (
        left.key == right.key
        and left.begin_seq == right.begin_seq
        and left.end_seq == right.end_seq
        and left.commit_seq == right.commit_seq
        and left.vector.tobytes(order="C") == right.vector.tobytes(order="C")
    )


class SearchEngine:
    def __init__(self, base: BaseIndex, delta: DeltaStore) -> None:
        if delta.dimension is not None and delta.dimension != base.dimension:
            raise ValueError("base/delta dimension mismatch")
        self.base = base
        self.delta = delta
        # DeltaStore and GroupDirectory are immutable value objects.  A search
        # engine therefore has one immutable snapshot view for each MVCC
        # snapshot ID.  Caching prevents every benchmark method/repetition from
        # rescanning all group members and copying the same fancy-indexed
        # matrices before query work begins.
        self._visible_raw_cache: dict[int, tuple[VectorRecord, ...]] = {}
        self._visible_group_cache: dict[
            int, tuple[tuple[Group, tuple[VectorRecord, ...], np.ndarray], ...]
        ] = {}
        # SearchEngine owns one immutable DeltaStore.  Native packed views are
        # therefore valid for this engine's lifetime and invalidated by making
        # a new engine for a new Delta/catalog revision; snapshot remains part
        # of the key because visibility changes the packed population.
        self._native_view_cache: dict[int, object] = {}

    def prepare_candidates(
        self,
        query: np.ndarray,
        *,
        snapshot_id: int,
        k: int,
        candidate_count: int,
        use_cached_view: bool = True,
    ) -> CandidateSet:
        return self.base.prepare_candidates(
            query,
            snapshot_id=snapshot_id,
            k=k,
            candidate_count=candidate_count,
            use_cached_view=use_cached_view,
        )

    def _native_packed_view(self, snapshot_id: int) -> object:
        cached = self._native_view_cache.get(snapshot_id)
        if cached is not None:
            return cached
        from .native import build_packed_view

        packed = build_packed_view(
            self.delta, snapshot_id=snapshot_id, dimension=self.base.dimension
        )
        self._native_view_cache[snapshot_id] = packed
        return packed

    def prepare_native_query(
        self,
        query: np.ndarray,
        *,
        snapshot_id: int = 0,
        k: int,
        candidate_count: int = 64,
        candidate_set: CandidateSet | None = None,
    ) -> object:
        """Validate/freeze C once for reuse by native F/N/P comparisons."""

        from .native import prepare_native_query

        return prepare_native_query(
            self,
            query,
            snapshot_id=snapshot_id,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidate_set,
        )

    def search_native_prepared(
        self,
        prepared_query: object,
        *,
        k: int,
        beta: float = 0.0,
        mode: str,
        threshold_mode: str = "heap",
        rank_strategy: str = "adaptive",
        audit: bool = False,
    ) -> SearchResult:
        """Run an already bound native query without repeating C validation."""

        from .native import PreparedNativeQuery, run_prepared_native

        if not isinstance(prepared_query, PreparedNativeQuery):
            raise TypeError("prepared_query must be PreparedNativeQuery")
        return run_prepared_native(
            self,
            prepared_query,
            k=k,
            beta=beta,
            mode=mode,  # type: ignore[arg-type]
            threshold_mode=threshold_mode,  # type: ignore[arg-type]
            rank_strategy=rank_strategy,  # type: ignore[arg-type]
            audit=audit,
        )

    def search_native_full_scan(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float = 0.0,
        snapshot_id: int = 0,
        candidate_count: int = 64,
        candidate_set: CandidateSet | None = None,
        prepared_query: object | None = None,
        threshold_mode: str = "heap",
        rank_strategy: str = "adaptive",
        audit: bool = False,
    ) -> SearchResult:
        prepared = prepared_query or self.prepare_native_query(
            query,
            snapshot_id=snapshot_id,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidate_set,
        )
        return self.search_native_prepared(
            prepared,
            k=k,
            beta=beta,
            mode="F",
            threshold_mode=threshold_mode,
            rank_strategy=rank_strategy,
            audit=audit,
        )

    def search_native_pruned(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float,
        snapshot_id: int = 0,
        candidate_count: int = 64,
        candidate_set: CandidateSet | None = None,
        prepared_query: object | None = None,
        no_pruning: bool = False,
        threshold_mode: str = "heap",
        rank_strategy: str = "adaptive",
        audit: bool = False,
    ) -> SearchResult:
        prepared = prepared_query or self.prepare_native_query(
            query,
            snapshot_id=snapshot_id,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidate_set,
        )
        return self.search_native_prepared(
            prepared,
            k=k,
            beta=beta,
            mode="N" if no_pruning else "P",
            threshold_mode=threshold_mode,
            rank_strategy=rank_strategy,
            audit=audit,
        )

    def _validate_candidate_binding(
        self, candidates: CandidateSet, query: np.ndarray, snapshot_id: int
    ) -> None:
        """Reject an externally supplied ``C`` from another search scope.

        The production BaseIndex exposes a content hash and key map.  Small
        independent-oracle tests intentionally use a structural stub and are
        the sole compatibility path for legacy, unbound CandidateSet fixtures.
        """

        if candidates.vectors.shape[1] != self.base.dimension:
            raise ValueError("candidate/base dimension mismatch")
        if not isinstance(self.base, BaseIndex):
            # If a test double chooses to provide binding fields, still check
            # every field that can be checked without a production universe.
            if (
                candidates.snapshot_id is not None
                and candidates.snapshot_id != snapshot_id
            ):
                raise ValueError("candidate set is bound to a different snapshot")
            if (
                candidates.base_generation is not None
                and candidates.base_generation != self.base.generation_id
            ):
                raise ValueError("candidate set is bound to a different base generation")
            if (
                candidates.query_hash is not None
                and candidates.query_hash != vector_query_hash(query)
            ):
                raise ValueError("candidate set is bound to a different query")
            return

        if not hasattr(self.base, "universe_hash") or not hasattr(
            self.base, "_records_by_key"
        ):
            self.base.initialize_integrity_metadata()
        if candidates.snapshot_id is None:
            raise ValueError("production search rejects an unbound candidate set")
        if candidates.snapshot_id != snapshot_id:
            raise ValueError("candidate set is bound to a different snapshot")
        if candidates.base_generation != self.base.generation_id:
            raise ValueError("candidate set is bound to a different base generation")
        if candidates.base_universe_hash != self.base.universe_hash:
            raise ValueError("candidate set is bound to a different base universe")
        if candidates.query_hash != vector_query_hash(query):
            raise ValueError("candidate set is bound to a different query")
        for candidate in candidates.records:
            base_record = self.base._records_by_key.get(candidate.key)
            if base_record is None or not _same_stored_record(candidate, base_record):
                raise ValueError("candidate set contains a record outside its bound base")
            if not candidate.visible_at(snapshot_id):
                raise ValueError("candidate set contains a snapshot-invisible record")

    @staticmethod
    def _scope_details(
        candidates: CandidateSet,
        raw: Sequence[VectorRecord],
        hits: Sequence[object],
        beta: float,
    ) -> dict[str, object]:
        beta_fraction = Fraction.from_float(beta)
        return {
            "requested_beta_fraction": _fraction_text(beta_fraction),
            "requested_beta_hex": beta.hex(),
            "candidate_snapshot_id": candidates.snapshot_id,
            "candidate_base_generation": candidates.base_generation,
            "candidate_base_universe_hash": candidates.base_universe_hash,
            "candidate_query_hash": candidates.query_hash,
            "candidate_version_keys": _version_keys(candidates.records),
            "raw_pending_version_keys": _version_keys(raw),
            "returned_version_keys": [list(hit.key) for hit in hits],
            "candidate_supplemented_count": candidates.supplemented_count,
            "candidate_available_visible_count": candidates.available_visible_count,
            "candidate_fallback_reason": candidates.fallback_reason,
        }

    def _visible_raw(self, snapshot_id: int) -> tuple[VectorRecord, ...]:
        cached = self._visible_raw_cache.get(snapshot_id)
        if cached is not None:
            return cached
        visible = tuple(
            record for record in self.delta.raw_records if record.visible_at(snapshot_id)
        )
        self._visible_raw_cache[snapshot_id] = visible
        return visible

    def _visible_groups(
        self, snapshot_id: int
    ) -> tuple[tuple[Group, tuple[VectorRecord, ...], np.ndarray], ...]:
        cached = self._visible_group_cache.get(snapshot_id)
        if cached is not None:
            return cached
        if self.delta.grouped is None:
            self._visible_group_cache[snapshot_id] = ()
            return ()
        visible: list[tuple[Group, tuple[VectorRecord, ...], np.ndarray]] = []
        for group in self.delta.grouped.groups:
            records, vectors = group.visible(snapshot_id)
            if records:
                visible.append((group, records, vectors))
        frozen = tuple(visible)
        self._visible_group_cache[snapshot_id] = frozen
        return frozen

    def search_full_scan(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float = 0.0,
        snapshot_id: int = 0,
        candidate_count: int = 64,
        candidate_set: CandidateSet | None = None,
        audit: bool = False,
    ) -> SearchResult:
        """Same-C reference: scan every visible raw and grouped Delta vector."""

        q = as_float32_vector(query, name="query")
        k = validate_k(k)
        beta = validate_beta(beta)
        candidates = (
            candidate_set
            if candidate_set is not None
            else self.prepare_candidates(
                q, snapshot_id=snapshot_id, k=k, candidate_count=candidate_count
            )
        )
        self._validate_candidate_binding(candidates, q, snapshot_id)
        visibility_start = time.perf_counter_ns()
        raw = self._visible_raw(snapshot_id)
        visible_groups = self._visible_groups(snapshot_id)
        visibility_ns = time.perf_counter_ns() - visibility_start

        scan_start = time.perf_counter_ns()
        records = list(candidates.records) + list(raw)
        sources = ["base"] * len(candidates.records) + ["raw"] * len(raw)
        group_count = 0
        for _, group_records, _ in visible_groups:
            records.extend(group_records)
            sources.extend(["group"] * len(group_records))
            group_count += 1
        records, sources = _deduplicate(records, sources)
        matrix = (
            np.ascontiguousarray(np.stack([record.vector for record in records]), dtype=np.float32)
            if records
            else _empty_matrix(self.base.dimension)
        )
        intervals = distance_intervals(q, matrix)
        candidate_delta_distance_ns = time.perf_counter_ns() - scan_start

        merge_start = time.perf_counter_ns()
        ranked = stable_topk(q, records, matrix, sources, k, intervals)
        merge_ns = time.perf_counter_ns() - merge_start
        population_small = len(records) < k
        certificate_details = self._scope_details(candidates, raw, ranked.hits, beta)
        if not population_small:
            certificate_details.update(
                {
                    "authoritative_gap": "0/1",
                    "certified_beta_fraction": "0/1",
                    "tau_upper_fraction": _fraction_text(
                        None
                        if ranked.tau_upper is None
                        else Fraction.from_float(ranked.tau_upper)
                    ),
                }
            )
        receipt_start = time.perf_counter_ns()
        receipt = Receipt(
            snapshot_id=snapshot_id,
            base_generation=self.base.generation_id,
            covered_commit_seq=self.base.covered_commit_seq,
            candidate_set_id_or_hash=candidates.candidate_hash,
            metric="L2",
            k=k,
            requested_beta=beta,
            certified_beta=None if population_small else 0.0,
            certificate_status=(
                "not_applicable_population_below_k" if population_small else "full_scan_reference"
            ),
            tau_returned=None if ranked.tau_upper is None else ranked.hits[-1].distance,
            min_skipped_lb=None,
            raw_pending_scanned=len(raw),
            groups_scanned=group_count,
            groups_skipped=0,
            vectors_scanned=len(raw) + sum(len(items[1]) for items in visible_groups),
            visibility_rejections=candidates.visibility_rejections,
            fallback_reason=(
                "population_below_k"
                if population_small
                else candidates.fallback_reason
            ),
            numeric_mode=NUMERIC_MODE,
            component_timings={
                "base_search_ns": candidates.search_ns,
                "visibility_ns": visibility_ns,
                "candidate_delta_distance_ns": candidate_delta_distance_ns,
                "lb_calculation_ns": 0,
                "group_ordering_ns": 0,
                "group_scan_ns": 0,
                "merge_ns": merge_ns,
                "receipt_ns": 0,
            },
            distance_evaluations=len(records),
            exact_boundary_rechecks=ranked.exact_rechecks,
            certificate_details=certificate_details,
            audit=[] if audit else None,
        )
        # Exercise the same structured conversion used by JSONL writers so the
        # reported Receipt cost is not merely dataclass allocation time.
        receipt.to_dict()
        receipt_ns = time.perf_counter_ns() - receipt_start
        timings = dict(receipt.component_timings)
        timings["receipt_ns"] = receipt_ns
        receipt = replace(receipt, component_timings=timings)
        return SearchResult(ranked.hits, receipt)

    def search_pruned(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float,
        snapshot_id: int = 0,
        candidate_count: int = 64,
        candidate_set: CandidateSet | None = None,
        no_pruning: bool = False,
        recompute_final_intervals: bool = False,
        audit: bool = False,
    ) -> SearchResult:
        q = as_float32_vector(query, name="query")
        k = validate_k(k)
        beta = validate_beta(beta)
        candidates = (
            candidate_set
            if candidate_set is not None
            else self.prepare_candidates(
                q, snapshot_id=snapshot_id, k=k, candidate_count=candidate_count
            )
        )
        self._validate_candidate_binding(candidates, q, snapshot_id)

        visibility_start = time.perf_counter_ns()
        raw = self._visible_raw(snapshot_id)
        visible_groups = self._visible_groups(snapshot_id)
        visible_logical_ids = [record.logical_id for record in candidates.records]
        visible_logical_ids.extend(record.logical_id for record in raw)
        for _, records, _ in visible_groups:
            visible_logical_ids.extend(record.logical_id for record in records)
        duplicate_visible_logical = len(visible_logical_ids) != len(
            set(visible_logical_ids)
        )
        visibility_ns = time.perf_counter_ns() - visibility_start

        raw_start = time.perf_counter_ns()
        scanned_records, scanned_sources = _deduplicate(
            list(candidates.records) + list(raw),
            ["base"] * len(candidates.records) + ["raw"] * len(raw),
        )
        scanned_matrix = (
            np.ascontiguousarray(
                np.stack([record.vector for record in scanned_records]), dtype=np.float32
            )
            if scanned_records
            else _empty_matrix(self.base.dimension)
        )
        scanned_intervals = distance_intervals(q, scanned_matrix)
        candidate_raw_distance_ns = time.perf_counter_ns() - raw_start
        initial_distance_evaluations = len(scanned_records)

        lb_start = time.perf_counter_ns()
        bounded_groups: list[
            tuple[Fraction, Group, tuple[VectorRecord, ...], np.ndarray, float, float]
        ] = []
        for group, records, vectors in visible_groups:
            lower, center_lower, center_upper = lower_bound_fraction(
                q, group.center, group.radius_upper
            )
            bounded_groups.append(
                (lower, group, records, vectors, center_lower, center_upper)
            )
        lb_ns = time.perf_counter_ns() - lb_start
        order_start = time.perf_counter_ns()
        bounded_groups.sort(key=lambda item: (item[0], item[1].group_id))
        ordering_ns = time.perf_counter_ns() - order_start

        group_scan_ns = 0
        groups_scanned = 0
        groups_skipped = 0
        scanned_delta_vectors = len(raw)
        skipped_lbs: list[Fraction] = []
        audit_rows: list[dict[str, object]] = []
        current_estimate = scanned_intervals.estimate.tolist()
        current_lower = scanned_intervals.lower.tolist()
        current_upper = scanned_intervals.upper.tolist()
        logical_positions = {
            record.logical_id: index for index, record in enumerate(scanned_records)
        }

        for lb_lower, group, records, vectors, center_lower, center_upper in bounded_groups:
            tau_upper = kth_upper_bound(current_upper, k)
            threshold = (
                Fraction.from_float(tau_upper) - Fraction.from_float(beta)
                if math.isfinite(tau_upper)
                else None
            )
            should_skip = (
                not no_pruning
                and not duplicate_visible_logical
                and threshold is not None
                and lb_lower > threshold
            )
            if should_skip:
                groups_skipped += 1
                skipped_lbs.append(lb_lower)
                action = "skip"
            else:
                start = time.perf_counter_ns()
                group_intervals = distance_intervals(q, vectors)
                group_scan_ns += time.perf_counter_ns() - start
                for offset, record in enumerate(records):
                    position = logical_positions.get(record.logical_id)
                    source = f"group:{group.group_id}"
                    if position is None:
                        logical_positions[record.logical_id] = len(scanned_records)
                        scanned_records.append(record)
                        scanned_sources.append(source)
                        current_estimate.append(float(group_intervals.estimate[offset]))
                        current_lower.append(float(group_intervals.lower[offset]))
                        current_upper.append(float(group_intervals.upper[offset]))
                    else:
                        prior = scanned_records[position]
                        if (record.begin_seq, record.version_id) >= (
                            prior.begin_seq,
                            prior.version_id,
                        ):
                            scanned_records[position] = record
                            scanned_sources[position] = source
                            current_estimate[position] = float(
                                group_intervals.estimate[offset]
                            )
                            current_lower[position] = float(group_intervals.lower[offset])
                            current_upper[position] = float(group_intervals.upper[offset])
                groups_scanned += 1
                scanned_delta_vectors += len(records)
                action = "scan"
            if audit:
                audit_rows.append(
                    {
                        "group_id": group.group_id,
                        "members_visible": len(records),
                        "radius_upper": group.radius_upper,
                        "radius_upper_fraction": _fraction_text(
                            Fraction.from_float(group.radius_upper)
                        ),
                        "center_distance_lower": center_lower,
                        "center_distance_lower_fraction": _fraction_text(
                            Fraction.from_float(center_lower)
                        ),
                        "center_distance_upper": center_upper,
                        "center_distance_upper_fraction": _fraction_text(
                            Fraction.from_float(center_upper)
                        ),
                        "lb_lower": fraction_to_lower_float(lb_lower),
                        "lb_lower_fraction": _fraction_text(lb_lower),
                        "tau_upper_at_decision": (
                            None if not math.isfinite(tau_upper) else tau_upper
                        ),
                        "tau_upper_at_decision_fraction": (
                            None
                            if not math.isfinite(tau_upper)
                            else _fraction_text(Fraction.from_float(tau_upper))
                        ),
                        "threshold_tau_minus_beta": (
                            None if threshold is None else float(threshold)
                        ),
                        "threshold_tau_minus_beta_fraction": _fraction_text(threshold),
                        "requested_beta_fraction": _fraction_text(
                            Fraction.from_float(beta)
                        ),
                        "member_version_keys": _version_keys(records),
                        "action": action,
                    }
                )

        merge_start = time.perf_counter_ns()
        # ``logical_positions`` kept the arrays de-duplicated while groups
        # were appended/replaced.  Carry the already-computed interval rows
        # into final exact boundary ranking instead of performing a second
        # exhaustive distance scan over all surviving vectors.
        final_matrix = (
            np.ascontiguousarray(
                np.stack([record.vector for record in scanned_records]), dtype=np.float32
            )
            if scanned_records
            else _empty_matrix(self.base.dimension)
        )
        final_interval_recompute_ns = 0
        if recompute_final_intervals:
            # Benchmark-only compatibility ablation for the pre-optimization
            # implementation.  It deliberately repeats the exhaustive interval
            # pass after group scanning while preserving identical certified
            # ranking semantics.
            interval_start = time.perf_counter_ns()
            final_intervals = distance_intervals(q, final_matrix)
            final_interval_recompute_ns = time.perf_counter_ns() - interval_start
        else:
            final_intervals = DistanceIntervals(
                estimate=np.asarray(current_estimate, dtype=np.float64),
                lower=np.asarray(current_lower, dtype=np.float64),
                upper=np.asarray(current_upper, dtype=np.float64),
            )
        ranked = stable_topk(
            q, scanned_records, final_matrix, scanned_sources, k, final_intervals
        )
        merge_ns = time.perf_counter_ns() - merge_start - final_interval_recompute_ns

        population_small = len(scanned_records) < k and groups_skipped == 0
        certified_beta: float | None
        min_skipped_lb: float | None
        certificate_details = self._scope_details(candidates, raw, ranked.hits, beta)
        certificate_details["final_interval_strategy"] = (
            "recomputed_full_survivor_matrix_ablation"
            if recompute_final_intervals
            else "carried_from_initial_and_group_scans"
        )
        if population_small:
            certified_beta = None
            min_skipped_lb = None
            status = "not_applicable_population_below_k"
            certificate_details.update({"authoritative_gap": None})
        elif not skipped_lbs:
            certified_beta = 0.0
            min_skipped_lb = None
            status = (
                "fallback_full_scan_duplicate_visible_logical_id"
                if duplicate_visible_logical
                else "certified_no_skip"
            )
            certificate_details.update(
                {
                    "authoritative_gap": "0/1",
                    "certified_beta_fraction": "0/1",
                    "tau_upper_fraction": _fraction_text(
                        None
                        if ranked.tau_upper is None
                        else Fraction.from_float(ranked.tau_upper)
                    ),
                }
            )
        else:
            minimum = min(skipped_lbs)
            assert ranked.tau_upper is not None
            gap = max(
                Fraction(0), Fraction.from_float(ranked.tau_upper) - minimum
            )
            beta_fraction = Fraction.from_float(beta)
            if gap > beta_fraction:  # Never hide a proof/implementation failure.
                raise AssertionError(
                    "internal certificate exceeds requested beta; preserve this failure"
                )
            converted_gap = fraction_to_upper_float(gap)
            # If outward conversion crosses the representable requested beta,
            # beta itself remains an authoritative upper bound only after the
            # exact Fraction comparison above has succeeded.
            certified_beta = beta if converted_gap > beta else converted_gap
            min_skipped_lb = fraction_to_lower_float(minimum)
            status = "certified"
            tau_fraction = Fraction.from_float(ranked.tau_upper)
            beta_fraction = Fraction.from_float(beta)
            certificate_details.update(
                {
                    "tau_upper_fraction": (
                        f"{tau_fraction.numerator}/{tau_fraction.denominator}"
                    ),
                    "min_skipped_lb_lower_fraction": (
                        f"{minimum.numerator}/{minimum.denominator}"
                    ),
                    "authoritative_gap": f"{gap.numerator}/{gap.denominator}",
                    "certified_beta_fraction": f"{gap.numerator}/{gap.denominator}",
                    "requested_beta_fraction": (
                        f"{beta_fraction.numerator}/{beta_fraction.denominator}"
                    ),
                }
            )

        receipt_start = time.perf_counter_ns()
        receipt = Receipt(
            snapshot_id=snapshot_id,
            base_generation=self.base.generation_id,
            covered_commit_seq=self.base.covered_commit_seq,
            candidate_set_id_or_hash=candidates.candidate_hash,
            metric="L2",
            k=k,
            requested_beta=beta,
            certified_beta=certified_beta,
            certificate_status=status,
            tau_returned=None if not ranked.hits else ranked.hits[-1].distance,
            min_skipped_lb=min_skipped_lb,
            raw_pending_scanned=len(raw),
            groups_scanned=groups_scanned,
            groups_skipped=groups_skipped,
            vectors_scanned=scanned_delta_vectors,
            visibility_rejections=candidates.visibility_rejections,
            fallback_reason=(
                "population_below_k"
                if population_small
                else (
                    "duplicate_visible_logical_id_forced_full_delta_scan"
                    if duplicate_visible_logical
                    else candidates.fallback_reason
                )
            ),
            numeric_mode=NUMERIC_MODE,
            component_timings={
                "base_search_ns": candidates.search_ns,
                "visibility_ns": visibility_ns,
                "candidate_raw_distance_ns": candidate_raw_distance_ns,
                "lb_calculation_ns": lb_ns,
                "group_ordering_ns": ordering_ns,
                "group_scan_ns": group_scan_ns,
                "final_interval_recompute_ns": final_interval_recompute_ns,
                "merge_ns": merge_ns,
                "receipt_ns": 0,
            },
            distance_evaluations=(
                initial_distance_evaluations
                + scanned_delta_vectors
                - len(raw)
                + len(bounded_groups)
                + (len(scanned_records) if recompute_final_intervals else 0)
            ),
            exact_boundary_rechecks=ranked.exact_rechecks,
            certificate_details=certificate_details,
            audit=audit_rows if audit else None,
        )
        receipt.to_dict()
        receipt_ns = time.perf_counter_ns() - receipt_start
        timings = dict(receipt.component_timings)
        timings["receipt_ns"] = receipt_ns
        receipt = replace(receipt, component_timings=timings)
        return SearchResult(ranked.hits, receipt)
