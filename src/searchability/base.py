"""Immutable Faiss HNSW generation and candidate freezing."""

from __future__ import annotations

import hashlib
import struct
import time
from typing import Iterable, Sequence

import faiss
import numpy as np

from .models import CandidateSet, VectorRecord, as_float32_vector
from .numerics import validate_k


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
        if m <= 0 or ef_construction <= 0 or ef_search <= 0 or threads <= 0:
            raise ValueError("HNSW parameters and threads must be positive")
        self.records = tuple(records)
        self.vectors = (
            np.ascontiguousarray(
                np.stack([record.vector for record in records]), dtype=np.float32
            )
            if records
            else np.empty((0, dimension), dtype=np.float32)
        )
        self.dimension = dimension
        self.generation_id = str(generation_id)
        self.covered_commit_seq = covered_commit_seq
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.threads = int(threads)
        faiss.omp_set_num_threads(self.threads)
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        if records:
            self.index.add(self.vectors)

    @classmethod
    def from_arrays(
        cls,
        vectors: np.ndarray,
        logical_ids: Iterable[int] | None = None,
        **kwargs: object,
    ) -> "BaseIndex":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("vectors must be a matrix with positive dimension")
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

    def prepare_candidates(
        self,
        query: np.ndarray,
        *,
        snapshot_id: int,
        k: int,
        candidate_count: int,
    ) -> CandidateSet:
        q = as_float32_vector(query, name="query")
        if q.shape[0] != self.dimension:
            raise ValueError("query/base dimension mismatch")
        k = validate_k(k)
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        requested = max(k, int(candidate_count))
        if not self.records:
            digest = hashlib.sha256()
            digest.update(self.generation_id.encode("utf-8"))
            digest.update(struct.pack("<qq", int(snapshot_id), requested))
            digest.update(q.tobytes(order="C"))
            return CandidateSet(
                records=(),
                vectors=np.empty((0, self.dimension), dtype=np.float32),
                candidate_hash=digest.hexdigest(),
                visibility_rejections=0,
                search_ns=0,
                requested_count=requested,
            )
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
                if not record.visible_at(snapshot_id) or record.logical_id in seen:
                    rejections += 1
                    continue
                seen.add(record.logical_id)
                accepted.append(record)
                if len(accepted) >= requested:
                    break
            if len(accepted) >= requested or target == len(self.records):
                break
            target = min(len(self.records), max(target + 1, target * 2))
        elapsed = time.perf_counter_ns() - start

        if accepted:
            matrix = np.ascontiguousarray(
                np.stack([record.vector for record in accepted]), dtype=np.float32
            )
        else:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(self.generation_id.encode("utf-8"))
        digest.update(struct.pack("<qq", int(snapshot_id), requested))
        digest.update(q.tobytes(order="C"))
        for record in sorted(accepted, key=lambda item: item.key):
            digest.update(struct.pack("<qq", record.logical_id, record.version_id))
        return CandidateSet(
            records=tuple(accepted),
            vectors=matrix,
            candidate_hash=digest.hexdigest(),
            visibility_rejections=rejections,
            search_ns=elapsed,
            requested_count=requested,
        )
