"""Deterministic synthetic generators and TEXMEX vector readers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import as_float32_matrix


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    name: str
    base: np.ndarray
    delta: np.ndarray
    validation_queries: np.ndarray
    test_queries: np.ndarray
    base_ids: np.ndarray
    delta_ids: np.ndarray
    validation_query_ids: np.ndarray
    test_query_ids: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Enforce the boundary for direct construction too; frozen dataclasses
        # do not otherwise freeze mutable ndarray arguments.
        for name in ("base", "delta", "validation_queries", "test_queries"):
            object.__setattr__(
                self,
                name,
                _readonly_float32(getattr(self, name), name=name),
            )
        for name in (
            "base_ids",
            "delta_ids",
            "validation_query_ids",
            "test_query_ids",
        ):
            object.__setattr__(self, name, _readonly_int64(getattr(self, name)))

    @property
    def dimension(self) -> int:
        return int(self.base.shape[1] if self.base.size else self.delta.shape[1])

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, array in (
            ("base", self.base),
            ("delta", self.delta),
            ("validation_queries", self.validation_queries),
            ("test_queries", self.test_queries),
            ("base_ids", self.base_ids),
            ("delta_ids", self.delta_ids),
            ("validation_query_ids", self.validation_query_ids),
            ("test_query_ids", self.test_query_ids),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
            view = memoryview(contiguous).cast("B")
            chunk_bytes = 8 * 1024 * 1024
            for offset in range(0, len(view), chunk_bytes):
                digest.update(view[offset : offset + chunk_bytes])
        return digest.hexdigest()


def _readonly_float32(array: np.ndarray, *, name: str = "dataset") -> np.ndarray:
    candidate = np.asarray(array)
    owner: object = candidate
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if (
        candidate.ndim == 2
        and candidate.dtype == np.float32
        and candidate.flags.c_contiguous
        and isinstance(owner, bytes)
    ):
        return candidate
    return as_float32_matrix(array, name=name)


def _readonly_int64(array: np.ndarray) -> np.ndarray:
    converted = np.asarray(array, dtype=np.int64, order="C")
    owner: object = converted
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if converted.flags.c_contiguous and isinstance(owner, bytes):
        return converted
    payload = converted.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.int64).reshape(converted.shape)


def synthetic_dataset(
    *,
    kind: str,
    n_base: int,
    n_delta: int,
    n_validation: int,
    n_test: int,
    dimension: int,
    seed: int,
    clusters: int = 16,
) -> DatasetSplit:
    if min(n_base, n_delta, n_validation, n_test) < 0:
        raise ValueError("dataset counts must be non-negative")
    if n_base == 0 and n_delta == 0:
        raise ValueError("at least one vector is required")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    total_queries = n_validation + n_test
    correlation_note = "none"

    if kind == "clustered":
        center_count = max(1, min(clusters, n_base))
        centers = rng.normal(0.0, 8.0, size=(center_count, dimension))
        base_labels = rng.integers(0, center_count, size=n_base)
        delta_labels = rng.integers(0, center_count, size=n_delta)
        query_labels = rng.integers(0, center_count, size=total_queries)
        base = centers[base_labels] + rng.normal(0.0, 0.7, size=(n_base, dimension))
        delta = centers[delta_labels] + rng.normal(0.0, 0.7, size=(n_delta, dimension))
        queries = centers[query_labels] + rng.normal(
            0.0, 0.7, size=(total_queries, dimension)
        )
    elif kind == "isotropic":
        base = rng.normal(0.0, 1.0, size=(n_base, dimension))
        delta = rng.normal(0.0, 1.0, size=(n_delta, dimension))
        queries = rng.normal(0.0, 1.0, size=(total_queries, dimension))
    elif kind == "delta_near_queries":
        base = rng.normal(0.0, 1.0, size=(n_base, dimension))
        queries = rng.normal(0.0, 1.0, size=(total_queries, dimension))
        delta = rng.normal(0.0, 1.0, size=(n_delta, dimension))
        if n_delta and total_queries:
            for index in range(n_delta):
                delta[index] = queries[index % total_queries] + rng.normal(
                    0.0, 0.01, size=dimension
                )
        correlation_note = (
            "intentional stress: every Delta vector is generated near a validation/test "
            "query; this is not claimed to be a natural time-ordered workload"
        )
    elif kind == "outlier_radius":
        center_count = max(1, min(clusters, n_base))
        centers = rng.normal(0.0, 4.0, size=(center_count, dimension))
        base_labels = rng.integers(0, center_count, size=n_base)
        delta_labels = rng.integers(0, center_count, size=n_delta)
        query_labels = rng.integers(0, center_count, size=total_queries)
        base = centers[base_labels] + rng.normal(0.0, 0.4, size=(n_base, dimension))
        delta = centers[delta_labels] + rng.normal(0.0, 0.4, size=(n_delta, dimension))
        queries = centers[query_labels] + rng.normal(
            0.0, 0.4, size=(total_queries, dimension)
        )
        outliers = max(1, n_delta // 20) if n_delta else 0
        if outliers:
            delta[-outliers:] += rng.normal(0.0, 80.0, size=(outliers, dimension))
        correlation_note = "synthetic far outliers intentionally inflate assigned radii"
    else:
        raise ValueError(f"unknown synthetic kind: {kind}")

    base_ids = np.arange(n_base, dtype=np.int64)
    delta_ids = np.arange(n_base, n_base + n_delta, dtype=np.int64)
    query_ids = np.arange(total_queries, dtype=np.int64)
    return DatasetSplit(
        name=f"synthetic-{kind}",
        base=_readonly_float32(base),
        delta=_readonly_float32(delta),
        validation_queries=_readonly_float32(queries[:n_validation]),
        test_queries=_readonly_float32(queries[n_validation:]),
        base_ids=_readonly_int64(base_ids),
        delta_ids=_readonly_int64(delta_ids),
        validation_query_ids=_readonly_int64(query_ids[:n_validation]),
        test_query_ids=_readonly_int64(query_ids[n_validation:]),
        metadata={
            "kind": kind,
            "seed": seed,
            "split_policy": "generated disjoint arrays with recorded ID ranges",
            "center_training_source": "base only",
            "query_update_correlation": correlation_note,
            "static_split_is_time_ordered": False,
        },
    )


def read_fvecs(path: str | Path, *, limit: int | None = None, offset: int = 0) -> np.ndarray:
    """Memory-map a TEXMEX fvecs file and copy only the requested rows."""

    location = Path(path)
    raw = np.memmap(location, dtype="<i4", mode="r")
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {location}")
    dimension = int(raw[0])
    if dimension <= 0 or raw.size % (dimension + 1):
        raise ValueError(f"invalid fvecs layout: {location}")
    rows = raw.reshape(-1, dimension + 1)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError(f"inconsistent fvecs dimensions: {location}")
    stop = rows.shape[0] if limit is None else min(rows.shape[0], offset + limit)
    if offset < 0 or offset > stop:
        raise ValueError("invalid fvecs offset")
    payload = np.ascontiguousarray(rows[offset:stop, 1:]).view("<f4")
    return _readonly_float32(payload)


def read_bvecs(path: str | Path, *, limit: int | None = None, offset: int = 0) -> np.ndarray:
    """Memory-map a TEXMEX bvecs file and convert requested rows to float32."""

    location = Path(path)
    raw = np.memmap(location, dtype=np.uint8, mode="r")
    if raw.size < 4:
        raise ValueError(f"invalid bvecs file: {location}")
    dimension = int(np.frombuffer(raw[:4], dtype="<i4")[0])
    width = dimension + 4
    if dimension <= 0 or raw.size % width:
        raise ValueError(f"invalid bvecs layout: {location}")
    rows = raw.reshape(-1, width)
    headers = rows[:, :4].copy().reshape(-1).view("<i4")
    if not np.all(headers == dimension):
        raise ValueError(f"inconsistent bvecs dimensions: {location}")
    stop = rows.shape[0] if limit is None else min(rows.shape[0], offset + limit)
    if offset < 0 or offset > stop:
        raise ValueError("invalid bvecs offset")
    return _readonly_float32(rows[offset:stop, 4:])


def texmex_dataset(
    *,
    name: str,
    root: str | Path,
    n_base: int,
    n_delta: int,
    n_validation: int,
    n_test: int,
) -> DatasetSplit:
    root_path = Path(root)
    provenance_manifest = root_path.parent / "manifests"
    if name == "sift":
        vector_reader = read_fvecs
        base_path = root_path / "sift" / "sift_base.fvecs"
        query_path = root_path / "sift" / "sift_query.fvecs"
        dimension = 128
    elif name == "gist":
        vector_reader = read_fvecs
        base_path = root_path / "gist" / "gist_base.fvecs"
        query_path = root_path / "gist" / "gist_query.fvecs"
        dimension = 960
    else:
        raise ValueError(f"unknown TEXMEX dataset: {name}")
    vectors = vector_reader(base_path, limit=n_base + n_delta)
    queries = read_fvecs(query_path, limit=n_validation + n_test)
    if vectors.shape[0] < n_base + n_delta or queries.shape[0] < n_validation + n_test:
        raise ValueError(f"{name} files do not contain requested split sizes")
    if vectors.shape[1] != dimension or queries.shape[1] != dimension:
        raise ValueError(f"unexpected {name} dimension")
    provenance_path = provenance_manifest / f"{name}.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"missing dataset provenance manifest {provenance_path}; run make data"
        )
    provenance_sha256 = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    return DatasetSplit(
        name=f"texmex-{name}",
        base=_readonly_float32(vectors[:n_base]),
        delta=_readonly_float32(vectors[n_base : n_base + n_delta]),
        validation_queries=_readonly_float32(queries[:n_validation]),
        test_queries=_readonly_float32(queries[n_validation : n_validation + n_test]),
        base_ids=_readonly_int64(np.arange(n_base, dtype=np.int64)),
        delta_ids=_readonly_int64(np.arange(n_base, n_base + n_delta, dtype=np.int64)),
        validation_query_ids=_readonly_int64(np.arange(n_validation, dtype=np.int64)),
        test_query_ids=_readonly_int64(
            np.arange(n_validation, n_validation + n_test, dtype=np.int64)
        ),
        metadata={
            "source": "http://corpus-texmex.irisa.fr/",
            "provenance_manifest": str(provenance_path),
            "provenance_manifest_sha256": provenance_sha256,
            "base_file": str(base_path),
            "query_file": str(query_path),
            "split_seed": None,
            "split_seed_reason": (
                "No random splitter is used: immutable prefix row ranges are recorded."
            ),
            "selection": (
                f"base rows [0,{n_base}), Delta rows [{n_base},{n_base+n_delta}), "
                f"validation queries [0,{n_validation}), test queries "
                f"[{n_validation},{n_validation+n_test})"
            ),
            "distributed_ground_truth_reused": False,
            "static_split_is_time_ordered": False,
        },
    )


def load_dataset(specification: Mapping[str, Any]) -> DatasetSplit:
    kind = specification.get("type")
    if kind == "synthetic":
        return synthetic_dataset(
            kind=str(specification["kind"]),
            n_base=int(specification["n_base"]),
            n_delta=int(specification["n_delta"]),
            n_validation=int(specification["n_validation"]),
            n_test=int(specification["n_test"]),
            dimension=int(specification["dimension"]),
            seed=int(specification["seed"]),
            clusters=int(specification.get("clusters", 16)),
        )
    if kind == "texmex":
        return texmex_dataset(
            name=str(specification["name"]),
            root=str(specification.get("root", "data/raw")),
            n_base=int(specification["n_base"]),
            n_delta=int(specification["n_delta"]),
            n_validation=int(specification["n_validation"]),
            n_test=int(specification["n_test"]),
        )
    raise ValueError(f"unsupported dataset type: {kind}")
