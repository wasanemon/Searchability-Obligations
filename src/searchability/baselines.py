"""Faiss baselines used by the common benchmark harness.

Prepared indexes are immutable.  Their construction/add/serialized-size costs
are exposed separately from query time; they are not described as ACID commit
costs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import faiss
import numpy as np

from .models import CandidateSet, SearchHit, VectorRecord, as_float32_vector
from .numerics import distance_intervals, stable_topk, validate_k


@dataclass(frozen=True, slots=True)
class IndexBuildStats:
    kind: str
    construct_ns: int
    add_ns: int
    serialize_ns: int
    serialized_bytes: int
    vectors: int
    dimension: int
    m: int | None
    ef_construction: int | None


@dataclass(frozen=True, slots=True)
class BaselineResult:
    hits: tuple[SearchHit, ...]
    search_ns: int
    merge_ns: int
    candidate_count: int

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(hit.logical_id for hit in self.hits)


class PreparedFaissIndex:
    """A fixed-population FlatL2 or HNSW baseline.

    ``IndexFlatL2`` is used as the optimized full-scan kernel, but its first
    ``k`` labels are not, by themselves, a mathematically exact top-k.  Faiss
    returns float32 squared distances and does not apply this artifact's
    ``(distance, logical_id, version_id)`` tie rule.  :meth:`search` therefore
    generates mathematical truth with one certified chunked pass instead.
    Separate ``*_performance`` methods retain a single Faiss scan and only
    rerank returned labels.  HNSW searches intentionally remain approximate.
    """

    def __init__(
        self,
        records: Sequence[VectorRecord],
        *,
        dimension: int | None = None,
        kind: str,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        threads: int = 1,
    ) -> None:
        if records:
            inferred = records[0].vector.shape[0]
            if dimension is not None and dimension != inferred:
                raise ValueError("explicit dimension disagrees with records")
            dimension = inferred
        if dimension is None or dimension <= 0:
            raise ValueError("a positive dimension is required")
        if any(record.vector.shape[0] != dimension for record in records):
            raise ValueError("record dimensions differ")
        if kind not in {"flat", "hnsw"}:
            raise ValueError("kind must be 'flat' or 'hnsw'")
        if threads <= 0:
            raise ValueError("threads must be positive")
        self.records = tuple(records)
        self.dimension = int(dimension)
        self.kind = kind
        self.threads = int(threads)
        self.ef_search = int(ef_search)
        packed_vectors = (
            np.ascontiguousarray(
                np.stack([record.vector for record in records]), dtype=np.float32
            )
            if records
            else np.empty((0, dimension), dtype=np.float32)
        )
        # A NumPy array that owns its storage can be made writable again by an
        # external caller even after ``setflags(write=False)``.  Use immutable
        # bytes as the backing store so the prepared index population cannot be
        # changed through this public diagnostic matrix.
        self._vector_bytes = packed_vectors.tobytes(order="C")
        self.vectors = np.frombuffer(self._vector_bytes, dtype=np.float32).reshape(
            len(records), self.dimension
        )

        faiss.omp_set_num_threads(self.threads)
        construct_start = time.perf_counter_ns()
        if kind == "flat":
            index: faiss.Index = faiss.IndexFlatL2(self.dimension)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, int(m), faiss.METRIC_L2)
            index.hnsw.efConstruction = int(ef_construction)
            index.hnsw.efSearch = self.ef_search
        construct_ns = time.perf_counter_ns() - construct_start
        add_start = time.perf_counter_ns()
        if records:
            index.add(self.vectors)
        add_ns = time.perf_counter_ns() - add_start
        serialize_start = time.perf_counter_ns()
        serialized = faiss.serialize_index(index)
        serialize_ns = time.perf_counter_ns() - serialize_start
        self.index = index
        self.build_stats = IndexBuildStats(
            kind=kind,
            construct_ns=construct_ns,
            add_ns=add_ns,
            serialize_ns=serialize_ns,
            serialized_bytes=int(serialized.nbytes),
            vectors=len(records),
            dimension=self.dimension,
            m=int(m) if kind == "hnsw" else None,
            ef_construction=int(ef_construction) if kind == "hnsw" else None,
        )

    def set_ef_search(self, value: int) -> None:
        if self.kind != "hnsw":
            raise ValueError("efSearch applies only to HNSW")
        if value <= 0:
            raise ValueError("efSearch must be positive")
        self.ef_search = int(value)
        self.index.hnsw.efSearch = int(value)

    def search(
        self,
        query: np.ndarray,
        *,
        k: int,
        candidate_count: int | None = None,
    ) -> BaselineResult:
        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/index dimension mismatch")
        k = validate_k(k)
        if not self.records:
            return BaselineResult((), 0, 0, 0)
        requested = min(
            len(self.records), max(k, int(candidate_count) if candidate_count else k)
        )

        # Mathematical truth is deliberately separate from the one-Faiss-scan
        # performance comparator.  Generate it with one chunked certified
        # interval pass and exact boundary ordering; a preceding Faiss scan
        # would be unused duplicate work.
        if self.kind == "flat":
            exact_start = time.perf_counter_ns()
            bounds = distance_intervals(q, self.vectors)
            ranked = stable_topk(
                q,
                self.records,
                self.vectors,
                ["flat_exact_truth"] * len(self.records),
                k,
                bounds,
            )
            exact_ns = time.perf_counter_ns() - exact_start
            return BaselineResult(ranked.hits, exact_ns, 0, requested)

        faiss.omp_set_num_threads(self.threads)
        start = time.perf_counter_ns()
        squared, labels = self.index.search(q[None, :], requested)
        search_ns = time.perf_counter_ns() - start

        hits = []
        for distance_squared, ordinal in zip(squared[0].tolist(), labels[0].tolist()):
            if ordinal < 0:
                continue
            record = self.records[ordinal]
            hits.append(
                SearchHit(
                    logical_id=record.logical_id,
                    version_id=record.version_id,
                    distance=math.sqrt(max(0.0, float(distance_squared))),
                    source=f"{self.kind}_index",
                )
            )
        hits.sort(key=lambda item: (item.distance, item.logical_id, item.version_id))
        return BaselineResult(tuple(hits[:k]), search_ns, 0, requested)

    def search_performance(
        self,
        query: np.ndarray,
        *,
        k: int,
        candidate_count: int | None = None,
    ) -> BaselineResult:
        """One Faiss query plus exact reranking of only returned labels.

        For HNSW this is identical in scope to :meth:`search`.  For Flat it is
        the optimized exhaustive-scan performance comparator: Faiss scans the
        population once, an over-fetched returned boundary is deterministically
        reranked, and no second all-population interval pass is performed.
        The result is therefore checked against, but never called, the exact
        full-visible truth.
        """

        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/index dimension mismatch")
        k = validate_k(k)
        if not self.records:
            return BaselineResult((), 0, 0, 0)
        requested = min(
            len(self.records), max(k, int(candidate_count) if candidate_count else k)
        )
        faiss.omp_set_num_threads(self.threads)
        search_start = time.perf_counter_ns()
        squared, labels = self.index.search(q[None, :], requested)
        search_ns = time.perf_counter_ns() - search_start
        ordinals = [int(value) for value in labels[0] if value >= 0]
        if self.kind == "hnsw":
            hits = []
            for distance_squared, ordinal in zip(
                squared[0].tolist(), labels[0].tolist()
            ):
                if ordinal < 0:
                    continue
                record = self.records[ordinal]
                hits.append(
                    SearchHit(
                        logical_id=record.logical_id,
                        version_id=record.version_id,
                        distance=math.sqrt(max(0.0, float(distance_squared))),
                        source="hnsw_index",
                    )
                )
            hits.sort(key=lambda item: (item.distance, item.logical_id, item.version_id))
            return BaselineResult(tuple(hits[:k]), search_ns, 0, requested)

        rerank_start = time.perf_counter_ns()
        records = [self.records[index] for index in ordinals]
        matrix = (
            np.ascontiguousarray(self.vectors[ordinals], dtype=np.float32)
            if ordinals
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        ranked = stable_topk(
            q, records, matrix, ["flat_performance"] * len(records), k
        )
        rerank_ns = time.perf_counter_ns() - rerank_start
        return BaselineResult(ranked.hits, search_ns, rerank_ns, requested)

    def search_and_merge_candidates(
        self,
        query: np.ndarray,
        *,
        k: int,
        base_candidates: CandidateSet,
        ann_candidate_count: int | None = None,
        source: str = "delta_hnsw",
    ) -> BaselineResult:
        """Search this Delta index and merge it with a caller-frozen C."""

        k = validate_k(k)
        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/index dimension mismatch")
        requested = min(
            len(self.records),
            max(k, int(ann_candidate_count) if ann_candidate_count else max(k, 64)),
        )
        faiss.omp_set_num_threads(self.threads)
        search_start = time.perf_counter_ns()
        if requested:
            _, labels = self.index.search(q[None, :], requested)
            returned_ordinals = [int(value) for value in labels[0] if value >= 0]
        else:
            returned_ordinals = []
        search_ns = time.perf_counter_ns() - search_start

        # The Flat path is the same-C exact reference.  A k-label result from
        # Faiss is insufficient at a tie boundary (for example +1 and -1 in
        # one dimension), so all Delta ordinals remain eligible for the
        # interval/exact boundary pass.  HNSW is a deliberately approximate
        # comparator and retains only its returned labels.
        selected_ordinals = (
            list(range(len(self.records)))
            if self.kind == "flat"
            else returned_ordinals
        )

        merge_start = time.perf_counter_ns()
        records = list(base_candidates.records) + [
            self.records[index] for index in selected_ordinals
        ]
        sources = ["base"] * len(base_candidates.records) + [source] * len(
            selected_ordinals
        )
        # Valid MVCC yields one visible version per logical ID.  Resolve any
        # malformed/transition duplicate safely in favor of the newest version.
        chosen: dict[int, tuple[VectorRecord, str, int]] = {}
        for position, (record, label) in enumerate(zip(records, sources)):
            current = chosen.get(record.logical_id)
            if current is None or (record.begin_seq, record.version_id, position) > (
                current[0].begin_seq,
                current[0].version_id,
                current[2],
            ):
                chosen[record.logical_id] = (record, label, position)
        ordered = sorted(chosen.values(), key=lambda item: item[2])
        records = [item[0] for item in ordered]
        sources = [item[1] for item in ordered]
        matrix = (
            np.ascontiguousarray(
                np.stack([record.vector for record in records]), dtype=np.float32
            )
            if records
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        bounds = distance_intervals(q, matrix)
        ranked = stable_topk(q, records, matrix, sources, k, bounds)
        merge_ns = time.perf_counter_ns() - merge_start
        return BaselineResult(ranked.hits, search_ns, merge_ns, requested)

    def search_and_merge_candidates_performance(
        self,
        query: np.ndarray,
        *,
        k: int,
        base_candidates: CandidateSet,
        ann_candidate_count: int | None = None,
        source: str = "delta_flat_performance",
    ) -> BaselineResult:
        """Optimized one-index-scan Delta baseline over a frozen ``C``.

        This path deliberately has different semantics from
        :meth:`search_and_merge_candidates` for a Flat index.  Faiss performs
        the one exhaustive Delta scan and returns an over-fetched shortlist;
        only that shortlist and ``C`` are then exactly boundary-ranked.  It
        does *not* rescan every Delta vector with the certified interval
        kernel.  Consequently it is the fair performance baseline, not the
        mathematical same-C reference.  The benchmark compares every result
        against ``certified_full_delta_reference`` and records any mismatch.
        """

        if self.kind != "flat":
            raise ValueError("the optimized full-scan baseline requires a Flat index")
        k = validate_k(k)
        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/index dimension mismatch")
        requested = min(
            len(self.records),
            max(k, int(ann_candidate_count) if ann_candidate_count else max(k, 64)),
        )
        faiss.omp_set_num_threads(self.threads)
        search_start = time.perf_counter_ns()
        if requested:
            _, labels = self.index.search(q[None, :], requested)
            returned_ordinals = [int(value) for value in labels[0] if value >= 0]
        else:
            returned_ordinals = []
        search_ns = time.perf_counter_ns() - search_start

        merge_start = time.perf_counter_ns()
        records = list(base_candidates.records)
        records.extend(self.records[index] for index in returned_ordinals)
        sources = ["base"] * len(base_candidates.records)
        sources.extend([source] * len(returned_ordinals))
        chosen: dict[int, tuple[VectorRecord, str, int]] = {}
        for position, (record, label) in enumerate(zip(records, sources)):
            current = chosen.get(record.logical_id)
            if current is None or (record.begin_seq, record.version_id, position) > (
                current[0].begin_seq,
                current[0].version_id,
                current[2],
            ):
                chosen[record.logical_id] = (record, label, position)
        ordered = sorted(chosen.values(), key=lambda item: item[2])
        records = [item[0] for item in ordered]
        sources = [item[1] for item in ordered]
        matrix = (
            np.ascontiguousarray(
                np.stack([record.vector for record in records]), dtype=np.float32
            )
            if records
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        bounds = distance_intervals(q, matrix)
        ranked = stable_topk(q, records, matrix, sources, k, bounds)
        merge_ns = time.perf_counter_ns() - merge_start
        return BaselineResult(ranked.hits, search_ns, merge_ns, requested)


def visible_unique_records(
    records: Sequence[VectorRecord], snapshot_id: int
) -> tuple[VectorRecord, ...]:
    """Freeze a fixed visible population for full-index baselines."""

    chosen: dict[int, tuple[VectorRecord, int]] = {}
    for position, record in enumerate(records):
        if not record.visible_at(snapshot_id):
            continue
        current = chosen.get(record.logical_id)
        if current is None or (record.begin_seq, record.version_id, position) > (
            current[0].begin_seq,
            current[0].version_id,
            current[1],
        ):
            chosen[record.logical_id] = (record, position)
    return tuple(item[0] for item in sorted(chosen.values(), key=lambda value: value[1]))
