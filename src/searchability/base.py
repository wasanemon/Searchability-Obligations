"""Immutable Faiss HNSW generation and candidate freezing."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import faiss
import numpy as np

from .models import (
    CandidateSet,
    VectorRecord,
    as_float32_matrix,
    as_float32_vector,
    frozen_candidate_hash,
    records_universe_hash,
    vector_query_hash,
)
from .numerics import exact_squared_l2, validate_k


class BaseIndex:
    """An immutable HNSW index with explicit ordinal-to-version metadata."""

    def __init__(
        self,
        records: Sequence[VectorRecord],
        *,
        dimension: int | None = None,
        generation_id: str = "generation-0",
        covered_commit_seq: int | None = None,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        threads: int = 1,
    ) -> None:
        if records:
            inferred_dimension = records[0].vector.shape[0]
            if dimension is not None and dimension != inferred_dimension:
                raise ValueError("explicit dimension disagrees with base vectors")
            dimension = inferred_dimension
        elif dimension is None or dimension <= 0:
            raise ValueError("an empty BaseIndex requires a positive dimension")
        assert dimension is not None
        if any(record.vector.shape[0] != dimension for record in records):
            raise ValueError("all base vectors must have the same dimension")
        if len({record.key for record in records}) != len(records):
            raise ValueError("base contains duplicate version keys")
        if m <= 0 or ef_construction <= 0 or ef_search <= 0 or threads <= 0:
            raise ValueError("HNSW parameters and threads must be positive")
        self.records = tuple(records)
        self.vectors = as_float32_matrix(
            np.stack([record.vector for record in records])
            if records
            else np.empty((0, dimension), dtype=np.float32),
            name="base vectors",
        )
        self.dimension = dimension
        self.generation_id = str(generation_id)
        self.covered_commit_seq = covered_commit_seq
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.threads = int(threads)
        self.initialize_integrity_metadata()
        faiss.omp_set_num_threads(self.threads)
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        if records:
            self.index.add(self.vectors)

    def initialize_integrity_metadata(self) -> None:
        """Initialize candidate-binding metadata, including after index loading.

        Persistence loaders that intentionally construct ``BaseIndex`` through
        ``__new__`` must set ``records``, ``dimension`` and ``generation_id``
        first, then call this method.  Calling it repeatedly is harmless.
        """

        self.records = tuple(self.records)
        if any(not isinstance(record, VectorRecord) for record in self.records):
            raise TypeError("base records must be VectorRecord instances")
        if len({record.key for record in self.records}) != len(self.records):
            raise ValueError("base contains duplicate version keys")
        # Persistence loaders populate fields through ``__new__`` to avoid
        # rebuilding Faiss. Canonicalize that public matrix here as well, so
        # every initialized BaseIndex has the same bytes-backed boundary.
        self.vectors = as_float32_matrix(self.vectors, name="base vectors")
        if self.vectors.shape != (len(self.records), self.dimension):
            raise ValueError("base records/vectors shape mismatch")
        self.universe_hash = records_universe_hash(
            self.records, dimension=self.dimension, generation_id=self.generation_id
        )
        self._records_by_key = {record.key: record for record in self.records}
        # BaseIndex is an immutable generation.  These caches are valid for the
        # lifetime of that generation and are keyed by the complete MVCC
        # snapshot ID; reinitializing/loading a generation clears both caches.
        self._visible_latest_cache: dict[int, tuple[VectorRecord, ...]] = {}
        self._visible_latest_by_logical_cache: dict[int, dict[int, VectorRecord]] = {}
        self._visible_view_cache_hits = 0
        self._visible_view_cache_misses = 0
        self._visible_view_legacy_rebuilds = 0
        self._visible_view_cache_lookup_ns = 0
        self._visible_view_cache_build_ns = 0
        self._visible_view_legacy_rebuild_ns = 0

    @classmethod
    def from_arrays(
        cls,
        vectors: np.ndarray,
        logical_ids: Iterable[int] | None = None,
        **kwargs: object,
    ) -> "BaseIndex":
        matrix = as_float32_matrix(vectors, name="vectors")
        ids = list(range(matrix.shape[0])) if logical_ids is None else list(logical_ids)
        if len(ids) != matrix.shape[0]:
            raise ValueError("logical_ids length mismatch")
        records = [
            VectorRecord(logical_id=int(identifier), version_id=0, vector=matrix[i])
            for i, identifier in enumerate(ids)
        ]
        return cls(records, dimension=matrix.shape[1], **kwargs)

    def visible_records(self, snapshot_id: int) -> tuple[VectorRecord, ...]:
        return tuple(record for record in self.records if record.visible_at(snapshot_id))

    def _visible_latest_records(
        self, snapshot_id: int, *, use_cached_view: bool = True
    ) -> tuple[VectorRecord, ...]:
        """Return one latest visible base version per logical ID, cached by snapshot."""

        if use_cached_view:
            cached = self._visible_latest_cache.get(snapshot_id)
            if cached is not None:
                return cached
        visible_by_logical = self._visible_latest_by_logical(
            snapshot_id, use_cached_view=use_cached_view
        )
        result = tuple(visible_by_logical.values())
        if use_cached_view:
            self._visible_latest_cache[snapshot_id] = result
        return result

    def _visible_latest_by_logical(
        self, snapshot_id: int, *, use_cached_view: bool = True
    ) -> dict[int, VectorRecord]:
        """Return the cached visible-ID table for one immutable generation/view.

        The returned mapping is private and must not be mutated.  Its only
        invalidation event is :meth:`initialize_integrity_metadata`, which is
        called when a generation is constructed or loaded and resets the cache.
        """

        if use_cached_view:
            lookup_start = time.perf_counter_ns()
            cached = self._visible_latest_by_logical_cache.get(snapshot_id)
            self._visible_view_cache_lookup_ns += (
                time.perf_counter_ns() - lookup_start
            )
            if cached is not None:
                self._visible_view_cache_hits += 1
                return cached
            self._visible_view_cache_misses += 1
        else:
            self._visible_view_legacy_rebuilds += 1
        build_start = time.perf_counter_ns()
        visible_by_logical: dict[int, VectorRecord] = {}
        for record in self.records:
            if not record.visible_at(snapshot_id):
                continue
            current = visible_by_logical.get(record.logical_id)
            if current is None or (record.begin_seq, record.version_id) > (
                current.begin_seq,
                current.version_id,
            ):
                visible_by_logical[record.logical_id] = record
        build_ns = time.perf_counter_ns() - build_start
        if use_cached_view:
            self._visible_view_cache_build_ns += build_ns
            self._visible_latest_by_logical_cache[snapshot_id] = visible_by_logical
        else:
            self._visible_view_legacy_rebuild_ns += build_ns
        return visible_by_logical

    def visible_view_cache_stats(self) -> dict[str, int]:
        """Expose cache/legacy-path counters for the Issue #3 ablation."""

        return {
            "hits": int(self._visible_view_cache_hits),
            "misses": int(self._visible_view_cache_misses),
            "legacy_rebuilds": int(self._visible_view_legacy_rebuilds),
            "cached_snapshots": len(self._visible_latest_by_logical_cache),
            "cache_lookup_ns": int(self._visible_view_cache_lookup_ns),
            "cache_build_ns": int(self._visible_view_cache_build_ns),
            "legacy_rebuild_ns": int(self._visible_view_legacy_rebuild_ns),
        }

    def prepare_candidates(
        self,
        query: np.ndarray,
        *,
        snapshot_id: int,
        k: int,
        candidate_count: int,
        use_cached_view: bool = True,
    ) -> CandidateSet:
        # Also supports validated persistence loaders created before integrity
        # metadata became part of BaseIndex's public construction contract.
        if (
            not hasattr(self, "universe_hash")
            or not hasattr(self, "_records_by_key")
            or not hasattr(self, "_visible_latest_cache")
            or not hasattr(self, "_visible_latest_by_logical_cache")
            or not hasattr(self, "_visible_view_cache_hits")
            or not hasattr(self, "_visible_view_cache_lookup_ns")
        ):
            self.initialize_integrity_metadata()
        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/base dimension mismatch")
        k = validate_k(k)
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        requested = max(k, int(candidate_count))
        query_hash = vector_query_hash(q)
        if not self.records:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
            candidate_hash = frozen_candidate_hash(
                (),
                matrix,
                base_generation=self.generation_id,
                base_universe_hash=self.universe_hash,
                snapshot_id=snapshot_id,
                query_hash=query_hash,
                requested_count=requested,
            )
            return CandidateSet(
                records=(),
                vectors=matrix,
                candidate_hash=candidate_hash,
                visibility_rejections=0,
                search_ns=0,
                requested_count=requested,
                snapshot_id=snapshot_id,
                base_generation=self.generation_id,
                base_universe_hash=self.universe_hash,
                query_hash=query_hash,
                available_visible_count=0,
                fallback_reason="base_population_empty",
            )
        visible_by_logical = self._visible_latest_by_logical(
            snapshot_id, use_cached_view=use_cached_view
        )
        visible = tuple(visible_by_logical.values())
        target = min(max(requested, 1), len(self.records))
        rejections = 0
        accepted: list[VectorRecord] = []

        start = time.perf_counter_ns()
        while True:
            faiss.omp_set_num_threads(self.threads)
            _, labels = self.index.search(q[None, :], target)
            accepted = []
            seen: set[int] = set()
            rejections = 0
            for ordinal in labels[0].tolist():
                if ordinal < 0:
                    continue
                record = self.records[ordinal]
                selected_version = visible_by_logical.get(record.logical_id)
                if selected_version is None or selected_version.key != record.key:
                    rejections += 1
                    continue
                if record.logical_id in seen:
                    rejections += 1
                    continue
                seen.add(record.logical_id)
                accepted.append(record)
                if len(accepted) >= requested:
                    break
            if len(accepted) >= requested or target == len(self.records):
                break
            target = min(len(self.records), max(target + 1, target * 2))
        target_count = min(requested, len(visible))
        supplemented_count = 0
        fallback_reason: str | None = None
        if len(accepted) < target_count:
            # HNSW is not required to return every label even when asked for
            # ntotal neighbors.  Enumerate the complete visible base universe,
            # rank missing versions exactly, and supplement deterministically.
            present = {record.logical_id for record in accepted}
            missing = [
                record for record in visible if record.logical_id not in present
            ]
            missing.sort(
                key=lambda record: (
                    exact_squared_l2(q, record.vector),
                    record.logical_id,
                    record.version_id,
                )
            )
            needed = target_count - len(accepted)
            accepted.extend(missing[:needed])
            supplemented_count = min(needed, len(missing))
            fallback_reason = "ann_underfill_full_visible_supplement"
        if len(visible) < requested:
            fallback_reason = "base_visible_population_below_requested"
        elapsed = time.perf_counter_ns() - start

        if accepted:
            matrix = np.ascontiguousarray(
                np.stack([record.vector for record in accepted]), dtype=np.float32
            )
        else:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
        candidate_hash = frozen_candidate_hash(
            accepted,
            matrix,
            base_generation=self.generation_id,
            base_universe_hash=self.universe_hash,
            snapshot_id=snapshot_id,
            query_hash=query_hash,
            requested_count=requested,
        )
        return CandidateSet(
            records=tuple(accepted),
            vectors=matrix,
            candidate_hash=candidate_hash,
            visibility_rejections=rejections,
            search_ns=elapsed,
            requested_count=requested,
            snapshot_id=snapshot_id,
            base_generation=self.generation_id,
            base_universe_hash=self.universe_hash,
            query_hash=query_hash,
            supplemented_count=supplemented_count,
            available_visible_count=len(visible),
            fallback_reason=fallback_reason,
        )
