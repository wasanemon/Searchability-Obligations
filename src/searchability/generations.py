"""Crash-safe immutable Faiss generation artifacts.

This module deliberately does *not* make a generation authoritative.  It writes
and validates immutable files; :mod:`searchability.store` publishes a completed
artifact in SQLite only after this module has fsynced and renamed its directory.
SQLite version rows remain the source of truth for MVCC intervals and vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence
import uuid

import faiss
import numpy as np

from .base import BaseIndex
from .models import VectorRecord, as_float32_matrix


FaultInjector = Callable[[str], None]
NUMERIC_CONTRACT_VERSION = "l2-interval-fraction-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> bytes:
    """Atomically install one JSON file and return its canonical payload."""

    payload = _canonical_json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return payload


@dataclass(frozen=True, slots=True)
class GenerationArtifact:
    generation_id: str
    covered_commit_seq: int
    directory_name: str
    manifest_sha256: str
    vector_count: int


@dataclass(frozen=True, slots=True)
class ValidatedGeneration:
    generation_id: str
    covered_commit_seq: int
    directory: Path
    manifest: Mapping[str, object]
    ordinal_keys: tuple[tuple[int, int], ...]


class GenerationValidationError(RuntimeError):
    """Raised when a published generation cannot prove its own integrity."""


class GenerationManager:
    """Build, verify, and load immutable generation directories.

    Catalog publication is intentionally outside this class.  A caller first
    calls :meth:`build`, then atomically inserts the returned artifact into its
    durable catalog.  A complete but unreferenced final directory is therefore
    a harmless orphan after a crash.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        m: int = 16,
        ef_construction: int = 80,
        ef_search: int = 64,
        threads: int = 1,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.root)
        if min(m, ef_construction, ef_search, threads) <= 0:
            raise ValueError("generation HNSW parameters must be positive")
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.threads = int(threads)
        self._fault_injector = fault_injector

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    def build(
        self,
        records: Sequence[VectorRecord],
        *,
        covered_commit_seq: int,
        dimension: int,
        generation_id: str | None = None,
    ) -> GenerationArtifact:
        """Write, fsync, and atomically rename a new generation directory."""

        if covered_commit_seq < 0:
            raise ValueError("covered_commit_seq must be non-negative")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if any(record.vector.shape != (dimension,) for record in records):
            raise ValueError("generation vector dimension mismatch")
        if len({record.key for record in records}) != len(records):
            raise ValueError("generation contains duplicate version keys")

        identifier = generation_id or (
            f"generation-{covered_commit_seq:020d}-{uuid.uuid4().hex}"
        )
        if (
            not identifier
            or identifier in {".", ".."}
            or identifier.startswith(".tmp-")
            or Path(identifier).name != identifier
        ):
            raise ValueError("generation_id must be one safe path component")
        final_directory = self.root / identifier
        if final_directory.exists():
            raise FileExistsError(final_directory)
        temporary = self.root / f".tmp-{identifier}-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            base = BaseIndex(
                records,
                dimension=dimension,
                generation_id=identifier,
                covered_commit_seq=covered_commit_seq,
                m=self.m,
                ef_construction=self.ef_construction,
                ef_search=self.ef_search,
                threads=self.threads,
            )
            index_path = temporary / "index.faiss"
            faiss.write_index(base.index, str(index_path))
            _fsync_file(index_path)

            ordinal_map = [
                {"logical_id": record.logical_id, "version_id": record.version_id}
                for record in records
            ]
            ordinal_payload = _canonical_json(ordinal_map)
            metadata = {
                "generation_id": identifier,
                "covered_commit_seq": covered_commit_seq,
                "dimension": dimension,
                "dtype": "float32",
                "ordinal_map": ordinal_map,
                "ordinal_map_sha256": _sha256_bytes(ordinal_payload),
            }
            metadata_path = temporary / "metadata.json"
            metadata_payload = _atomic_json(metadata_path, metadata)
            self._fault("generation_after_index_metadata")

            build_config = {
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "threads": self.threads,
            }
            manifest = {
                "generation_id": identifier,
                "covered_commit_seq": covered_commit_seq,
                "build_snapshot": covered_commit_seq,
                "metric": "L2",
                "faiss_distance_representation": "squared_L2",
                "dimension": dimension,
                "dtype": "float32",
                "numeric_contract_version": NUMERIC_CONTRACT_VERSION,
                "hnsw_parameters": build_config,
                "vector_count": len(records),
                "ordinal_map_hash": _sha256_bytes(ordinal_payload),
                "index_hash": _file_sha256(index_path),
                "metadata_hash": _sha256_bytes(metadata_payload),
                "index_bytes": index_path.stat().st_size,
                "metadata_bytes": len(metadata_payload),
                "build_config_hash": _sha256_bytes(_canonical_json(build_config)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            # Only the Faiss file and ordinal metadata live in the build
            # directory at this boundary.  They and the directory entry have
            # been flushed, but nothing is published in SQLite.
            self._fault("generation_after_files")
            _fsync_directory(temporary)

            os.replace(temporary, final_directory)
            _fsync_directory(self.root)
            self._fault("generation_after_rename")

            # Install the self-validating manifest atomically *after* the
            # directory rename.  A crash before/during this step leaves only an
            # unreferenced, invalid orphan.  Catalog publication still happens
            # later in one SQLite transaction owned by LifecycleStore.
            manifest_path = final_directory / "manifest.json"
            manifest_payload = _atomic_json(manifest_path, manifest)
            self._fault("generation_after_manifest")
            return GenerationArtifact(
                generation_id=identifier,
                covered_commit_seq=covered_commit_seq,
                directory_name=identifier,
                manifest_sha256=_sha256_bytes(manifest_payload),
                vector_count=len(records),
            )
        except BaseException:
            # With an injected os._exit this cleanup is never run, which is the
            # intended on-disk crash state.  Ordinary exceptions do get a tidy
            # best-effort cleanup without touching a renamed final directory.
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def validate(
        self,
        directory_name: str,
        *,
        expected_manifest_sha256: str | None = None,
        expected_generation_id: str | None = None,
        expected_coverage: int | None = None,
    ) -> ValidatedGeneration:
        if directory_name.startswith(".tmp-") or Path(directory_name).name != directory_name:
            raise GenerationValidationError("unsafe generation directory name")
        directory = self.root / directory_name
        manifest_path = directory / "manifest.json"
        metadata_path = directory / "metadata.json"
        index_path = directory / "index.faiss"
        try:
            manifest_payload = manifest_path.read_bytes()
            metadata_payload = metadata_path.read_bytes()
            manifest = json.loads(manifest_payload)
            metadata = json.loads(metadata_payload)
        except (OSError, ValueError, TypeError) as exc:
            raise GenerationValidationError(f"unreadable generation: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            raise GenerationValidationError("manifest and metadata must be objects")
        if expected_manifest_sha256 is not None and (
            _sha256_bytes(manifest_payload) != expected_manifest_sha256
        ):
            raise GenerationValidationError("manifest checksum mismatch")

        required = {
            "generation_id",
            "covered_commit_seq",
            "build_snapshot",
            "metric",
            "faiss_distance_representation",
            "dimension",
            "dtype",
            "numeric_contract_version",
            "hnsw_parameters",
            "vector_count",
            "ordinal_map_hash",
            "index_hash",
            "metadata_hash",
            "index_bytes",
            "metadata_bytes",
            "build_config_hash",
            "created_at",
        }
        if not required.issubset(manifest):
            raise GenerationValidationError("generation manifest is incomplete")
        identifier = manifest["generation_id"]
        coverage = manifest["covered_commit_seq"]
        dimension = manifest["dimension"]
        vector_count = manifest["vector_count"]
        if identifier != directory_name:
            raise GenerationValidationError("generation/directory identity mismatch")
        if expected_generation_id is not None and identifier != expected_generation_id:
            raise GenerationValidationError("unexpected generation identity")
        if not isinstance(coverage, int) or coverage < 0:
            raise GenerationValidationError("invalid generation coverage")
        if expected_coverage is not None and coverage != expected_coverage:
            raise GenerationValidationError("generation coverage mismatch")
        if manifest["build_snapshot"] != coverage:
            raise GenerationValidationError("build snapshot differs from coverage")
        if manifest["metric"] != "L2":
            raise GenerationValidationError("unsupported metric")
        if manifest["faiss_distance_representation"] != "squared_L2":
            raise GenerationValidationError("Faiss metric representation mismatch")
        if manifest["dtype"] != "float32":
            raise GenerationValidationError("unsupported generation dtype")
        if manifest["numeric_contract_version"] != NUMERIC_CONTRACT_VERSION:
            raise GenerationValidationError("numeric contract mismatch")
        if not isinstance(dimension, int) or dimension <= 0:
            raise GenerationValidationError("invalid dimension")
        if not isinstance(vector_count, int) or vector_count < 0:
            raise GenerationValidationError("invalid vector count")
        if index_path.stat().st_size != manifest["index_bytes"]:
            raise GenerationValidationError("index length mismatch")
        if len(metadata_payload) != manifest["metadata_bytes"]:
            raise GenerationValidationError("metadata length mismatch")
        if _file_sha256(index_path) != manifest["index_hash"]:
            raise GenerationValidationError("index checksum mismatch")
        if _sha256_bytes(metadata_payload) != manifest["metadata_hash"]:
            raise GenerationValidationError("metadata checksum mismatch")

        ordinal_map = metadata.get("ordinal_map")
        if not isinstance(ordinal_map, list) or len(ordinal_map) != vector_count:
            raise GenerationValidationError("ordinal map length mismatch")
        ordinal_payload = _canonical_json(ordinal_map)
        ordinal_hash = _sha256_bytes(ordinal_payload)
        if ordinal_hash != manifest["ordinal_map_hash"]:
            raise GenerationValidationError("ordinal map checksum mismatch")
        if metadata.get("ordinal_map_sha256") != ordinal_hash:
            raise GenerationValidationError("metadata ordinal checksum mismatch")
        if metadata.get("generation_id") != identifier:
            raise GenerationValidationError("metadata identity mismatch")
        if metadata.get("covered_commit_seq") != coverage:
            raise GenerationValidationError("metadata coverage mismatch")
        if metadata.get("dimension") != dimension or metadata.get("dtype") != "float32":
            raise GenerationValidationError("metadata numeric description mismatch")

        keys: list[tuple[int, int]] = []
        for item in ordinal_map:
            if not isinstance(item, dict):
                raise GenerationValidationError("invalid ordinal entry")
            logical_id = item.get("logical_id")
            version_id = item.get("version_id")
            if (
                isinstance(logical_id, bool)
                or not isinstance(logical_id, int)
                or logical_id < 0
                or isinstance(version_id, bool)
                or not isinstance(version_id, int)
                or version_id < 0
            ):
                raise GenerationValidationError("invalid ordinal version key")
            keys.append((logical_id, version_id))
        if len(set(keys)) != len(keys):
            raise GenerationValidationError("duplicate ordinal version key")

        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise GenerationValidationError(f"Faiss index is unreadable: {exc}") from exc
        if index.d != dimension or index.ntotal != vector_count:
            raise GenerationValidationError("Faiss index shape mismatch")
        if getattr(index, "metric_type", faiss.METRIC_L2) != faiss.METRIC_L2:
            raise GenerationValidationError("Faiss index metric mismatch")
        return ValidatedGeneration(
            generation_id=str(identifier),
            covered_commit_seq=coverage,
            directory=directory,
            manifest=manifest,
            ordinal_keys=tuple(keys),
        )

    def load_base(
        self,
        validated: ValidatedGeneration,
        records_by_key: Mapping[tuple[int, int], VectorRecord],
    ) -> BaseIndex:
        """Load Faiss while taking vectors and MVCC intervals from SQLite."""

        try:
            records = tuple(records_by_key[key] for key in validated.ordinal_keys)
        except KeyError as exc:
            raise GenerationValidationError(
                f"generation refers to missing retained version {exc.args[0]}"
            ) from exc
        manifest = validated.manifest
        dimension = int(manifest["dimension"])
        if any(record.vector.shape != (dimension,) for record in records):
            raise GenerationValidationError("retained vector dimension mismatch")
        index = faiss.read_index(str(validated.directory / "index.faiss"))

        # BaseIndex has no persistence constructor.  Fill its documented fields
        # without rebuilding the graph; the graph remains immutable thereafter.
        base = BaseIndex.__new__(BaseIndex)
        base.records = records
        base.vectors = as_float32_matrix(
            np.stack([record.vector for record in records])
            if records
            else np.empty((0, dimension), dtype=np.float32),
            name="loaded base vectors",
        )
        base.dimension = dimension
        base.generation_id = validated.generation_id
        base.covered_commit_seq = validated.covered_commit_seq
        parameters = manifest["hnsw_parameters"]
        if not isinstance(parameters, dict):
            raise GenerationValidationError("invalid HNSW parameter manifest")
        base.m = int(parameters["m"])
        base.ef_construction = int(parameters["ef_construction"])
        base.ef_search = int(parameters["ef_search"])
        # Thread count is a runtime resource setting, not immutable index
        # provenance. Respect the current store/manager configuration even when
        # this artifact was built under another thread count.
        base.threads = self.threads
        base.index = index
        base.index.hnsw.efSearch = base.ef_search
        base.initialize_integrity_metadata()
        faiss.omp_set_num_threads(base.threads)
        return base

    def directories(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return complete-looking final directories and temporary leftovers."""

        finals: list[str] = []
        temporaries: list[str] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".tmp-"):
                temporaries.append(child.name)
            else:
                finals.append(child.name)
        return tuple(sorted(finals)), tuple(sorted(temporaries))
