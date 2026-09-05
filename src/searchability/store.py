"""Small durable SQLite lifecycle reference for Searchability Obligations.

The implementation favors an auditable safety argument over write throughput.
SQLite supplies the local transaction primitive; that existing ACID property is
not a research contribution of this project.  Vector/version rows, intervals,
and obligation rows are retained indefinitely.  Generation and group data are
derived accelerators, never the source of truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import sqlite3
import struct
import threading
import time
from typing import Iterable, Iterator, Mapping, Sequence
import uuid

import numpy as np

from .base import BaseIndex
from .generations import (
    FaultInjector,
    GenerationManager,
    GenerationValidationError,
)
from .groups import DeltaStore, Group, GroupBuildStats, GroupDirectory
from .models import SearchResult, VectorRecord, as_float32_vector
from .search import SearchEngine


GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION = (
    "GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION"
)

FAULT_POINTS = frozenset(
    {
        "commit_before_sql_commit",
        "commit_after_sql_commit",
        "commit_insert_after_version_before_obligation",
        "commit_update_after_versions_before_obligation",
        "commit_delete_after_old_end_before_obligation",
        "group_before_catalog_commit",
        "group_after_catalog_commit",
        "generation_after_index_metadata",
        "generation_after_files",
        "generation_after_rename",
        "generation_after_manifest",
        "generation_before_catalog_commit",
        "generation_after_catalog_commit",
    }
)


class SnapshotError(ValueError):
    """The requested snapshot is outside the retained commit prefix."""


class PinnedViewError(RuntimeError):
    """A view is no longer pinned by this store instance."""


class PinnedGenerationError(RuntimeError):
    """Collection was refused because an active query view exists."""


@dataclass(frozen=True, slots=True)
class CommitOperation:
    kind: str
    logical_id: int
    vector: np.ndarray | Sequence[float] | None = None

    @classmethod
    def insert(
        cls, logical_id: int, vector: np.ndarray | Sequence[float]
    ) -> "CommitOperation":
        return cls("insert", logical_id, vector)

    @classmethod
    def update(
        cls, logical_id: int, vector: np.ndarray | Sequence[float]
    ) -> "CommitOperation":
        return cls("update", logical_id, vector)

    @classmethod
    def delete(cls, logical_id: int) -> "CommitOperation":
        return cls("delete", logical_id, None)


@dataclass(frozen=True, slots=True)
class PinnedView:
    snapshot_seq: int
    generation_id: str | None
    covered_commit_seq: int | None
    generation_directory: str | None
    manifest_sha256: str | None
    group_catalog_revision: int
    token: str

    @property
    def snapshot_id(self) -> int:
        return self.snapshot_seq


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    valid_generations: tuple[str, ...]
    invalid_generations: tuple[str, ...]
    orphan_generations: tuple[str, ...]
    temporary_directories: tuple[str, ...]
    current_generation: str | None


@dataclass(frozen=True, slots=True)
class _LoadedDelta:
    """One immutable Delta view plus any safe catalog-repair fallback facts."""

    delta: DeltaStore
    delta_view_hash: str
    fallback_reason: str | None = None
    fallback_vector_count: int = 0
    fallback_group_count: int = 0
    fallback_details: tuple[str, ...] = ()


def _hash_record(digest: "hashlib._Hash", record: VectorRecord) -> None:
    """Bind a retained record's identity, interval, and exact float32 payload."""

    digest.update(
        struct.pack(
            "<qqqqq",
            record.logical_id,
            record.version_id,
            record.begin_seq,
            -1 if record.end_seq is None else record.end_seq,
            -1 if record.commit_seq is None else record.commit_seq,
        )
    )
    digest.update(struct.pack("<q", record.vector.shape[0]))
    digest.update(record.vector.tobytes(order="C"))


def _delta_view_hash(view: PinnedView, delta: DeltaStore) -> str:
    """Return a deterministic content hash for the exact Delta query view."""

    digest = hashlib.sha256()
    digest.update(b"searchability-delta-view-v1\0")
    digest.update(
        struct.pack(
            "<qqq",
            view.snapshot_seq,
            -1 if view.covered_commit_seq is None else view.covered_commit_seq,
            view.group_catalog_revision,
        )
    )
    raw = sorted(delta.raw_records, key=lambda record: record.key)
    digest.update(struct.pack("<q", len(raw)))
    for record in raw:
        digest.update(b"R")
        _hash_record(digest, record)
    groups = () if delta.grouped is None else tuple(
        sorted(delta.grouped.groups, key=lambda group: group.group_id)
    )
    digest.update(struct.pack("<q", len(groups)))
    for group in groups:
        digest.update(b"G")
        digest.update(struct.pack("<qd", group.group_id, group.radius_upper))
        digest.update(struct.pack("<q", group.center.shape[0]))
        digest.update(group.center.tobytes(order="C"))
        members = sorted(group.members, key=lambda record: record.key)
        digest.update(struct.pack("<q", len(members)))
        for record in members:
            _hash_record(digest, record)
    return digest.hexdigest()


def _normalise_operation(value: object) -> CommitOperation:
    if isinstance(value, CommitOperation):
        operation = value
    elif isinstance(value, Mapping):
        kind = value.get(
            "kind", value.get("op", value.get("operation", value.get("type")))
        )
        logical_id = value.get("logical_id", value.get("id"))
        operation = CommitOperation(str(kind), logical_id, value.get("vector"))  # type: ignore[arg-type]
    elif isinstance(value, (tuple, list)) and 2 <= len(value) <= 3:
        operation = CommitOperation(
            str(value[0]), value[1], value[2] if len(value) == 3 else None  # type: ignore[arg-type]
        )
    else:
        raise TypeError("operations must be CommitOperation, mapping, or tuple")
    kind = operation.kind.lower()
    if kind not in {"insert", "update", "delete"}:
        raise ValueError(f"unsupported operation kind: {operation.kind!r}")
    if isinstance(operation.logical_id, bool) or not isinstance(
        operation.logical_id, (int, np.integer)
    ):
        raise TypeError("logical_id must be an integer")
    logical_id = int(operation.logical_id)
    if logical_id < 0:
        raise ValueError("logical_id must be non-negative")
    vector = operation.vector
    if kind == "delete":
        if vector is not None:
            raise ValueError("delete must not contain a vector")
        canonical = None
    else:
        if vector is None:
            raise ValueError(f"{kind} requires a vector")
        canonical = as_float32_vector(vector)
    return CommitOperation(kind, logical_id, canonical)


class SQLiteLifecycleStore:
    """Durable single-node lifecycle store with conservative retention.

    ``path`` may be a directory (the DB becomes ``store.sqlite3``) or an
    explicit ``.db/.sqlite/.sqlite3`` file.  Search and generation construction
    use one CPU thread by default.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dimension: int | None = None,
        artifact_directory: str | Path | None = None,
        threads: int = 1,
        hnsw_m: int = 16,
        ef_construction: int = 80,
        ef_search: int = 64,
        fault_injector: FaultInjector | None = None,
        fault_point: str | None = None,
        fault_exit_code: int = 86,
        auto_recover: bool = True,
    ) -> None:
        supplied = Path(path)
        if supplied.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            self.db_path = supplied
            self.root = supplied.parent
            default_artifacts = supplied.parent / f"{supplied.stem}-artifacts"
        else:
            self.root = supplied
            self.db_path = supplied / "store.sqlite3"
            default_artifacts = supplied / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_directory = Path(artifact_directory or default_artifacts)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self.generation_directory = self.artifact_directory / "generations"
        self._threads = int(threads)
        if self._threads <= 0:
            raise ValueError("threads must be positive")
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be positive")
        if fault_point is not None and fault_point not in FAULT_POINTS:
            raise ValueError(f"unknown fault point {fault_point!r}")
        self._user_fault_injector = fault_injector
        self._exit_fault_point = fault_point
        self._fault_exit_code = int(fault_exit_code)
        self._lock = threading.RLock()
        self._active_pins: dict[str, PinnedView] = {}
        self._closed = False
        self._connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_sqlite()
        self._create_schema()
        self._set_or_check_dimension(dimension)
        self._generations = GenerationManager(
            self.generation_directory,
            m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            threads=self._threads,
            fault_injector=self._fault,
        )
        self.last_recovery_report = (
            self.recover()
            if auto_recover
            else RecoveryReport((), (), (), (), None)
        )

    def _configure_sqlite(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            INSERT OR IGNORE INTO meta(key, value) VALUES ('commit_seq', '0');
            INSERT OR IGNORE INTO meta(key, value) VALUES ('version_seq', '0');
            INSERT OR IGNORE INTO meta(key, value) VALUES ('catalog_revision', '0');
            INSERT OR IGNORE INTO meta(key, value) VALUES ('dimension', '');

            CREATE TABLE IF NOT EXISTS commits (
                commit_seq INTEGER PRIMARY KEY,
                operation_count INTEGER NOT NULL CHECK(operation_count > 0),
                committed_unix_ns INTEGER NOT NULL
            ) STRICT;

            CREATE TABLE IF NOT EXISTS versions (
                version_id INTEGER PRIMARY KEY,
                logical_id INTEGER NOT NULL CHECK(logical_id >= 0),
                vector BLOB NOT NULL,
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                begin_seq INTEGER NOT NULL REFERENCES commits(commit_seq),
                end_seq INTEGER REFERENCES commits(commit_seq),
                CHECK(end_seq IS NULL OR end_seq > begin_seq)
            ) STRICT;
            CREATE UNIQUE INDEX IF NOT EXISTS one_current_version_per_logical
                ON versions(logical_id) WHERE end_seq IS NULL;
            CREATE INDEX IF NOT EXISTS versions_visibility
                ON versions(begin_seq, end_seq, logical_id);

            CREATE TABLE IF NOT EXISTS obligations (
                obligation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_seq INTEGER NOT NULL REFERENCES commits(commit_seq),
                kind TEXT NOT NULL CHECK(kind IN ('insert', 'update', 'delete')),
                logical_id INTEGER NOT NULL CHECK(logical_id >= 0),
                old_version_id INTEGER REFERENCES versions(version_id),
                new_version_id INTEGER REFERENCES versions(version_id),
                CHECK(
                    (kind = 'insert' AND old_version_id IS NULL AND new_version_id IS NOT NULL)
                    OR (kind = 'update' AND old_version_id IS NOT NULL AND new_version_id IS NOT NULL)
                    OR (kind = 'delete' AND old_version_id IS NOT NULL AND new_version_id IS NULL)
                )
            ) STRICT;
            CREATE INDEX IF NOT EXISTS obligations_commit
                ON obligations(commit_seq, obligation_id);

            CREATE TABLE IF NOT EXISTS catalog_revisions (
                revision INTEGER PRIMARY KEY,
                build_commit_seq INTEGER NOT NULL,
                published_unix_ns INTEGER NOT NULL
            ) STRICT;
            INSERT OR IGNORE INTO catalog_revisions
                (revision, build_commit_seq, published_unix_ns) VALUES (0, 0, 0);

            CREATE TABLE IF NOT EXISTS groups (
                revision INTEGER NOT NULL REFERENCES catalog_revisions(revision),
                group_id INTEGER NOT NULL,
                center BLOB NOT NULL,
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                radius_upper REAL NOT NULL CHECK(radius_upper >= 0),
                PRIMARY KEY(revision, group_id)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS group_members (
                revision INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                version_id INTEGER NOT NULL REFERENCES versions(version_id),
                PRIMARY KEY(revision, version_id),
                FOREIGN KEY(revision, group_id) REFERENCES groups(revision, group_id)
            ) STRICT;

            CREATE TABLE IF NOT EXISTS generations (
                generation_id TEXT PRIMARY KEY,
                covered_commit_seq INTEGER NOT NULL CHECK(covered_commit_seq >= 0),
                directory_name TEXT NOT NULL UNIQUE,
                manifest_sha256 TEXT NOT NULL,
                vector_count INTEGER NOT NULL CHECK(vector_count >= 0),
                published_unix_ns INTEGER NOT NULL,
                valid INTEGER NOT NULL DEFAULT 1 CHECK(valid IN (0, 1))
            ) STRICT;
            CREATE INDEX IF NOT EXISTS generation_coverage
                ON generations(valid, covered_commit_seq DESC, published_unix_ns DESC);
            CREATE TABLE IF NOT EXISTS generation_failures (
                failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id TEXT NOT NULL,
                observed_unix_ns INTEGER NOT NULL,
                reason TEXT NOT NULL
            ) STRICT;
            """
        )

    def _set_or_check_dimension(self, dimension: int | None) -> None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'dimension'"
        ).fetchone()
        assert row is not None
        existing = row["value"]
        if existing:
            if dimension is not None and int(existing) != int(dimension):
                raise ValueError("configured dimension differs from persisted dimension")
        elif dimension is not None:
            self._connection.execute(
                "UPDATE meta SET value = ? WHERE key = 'dimension'", (str(dimension),)
            )

    def _fault(self, point: str) -> None:
        if self._user_fault_injector is not None:
            self._user_fault_injector(point)
        if self._exit_fault_point == point:
            os._exit(self._fault_exit_code)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("store is closed")

    def _meta_int(self, key: str) -> int:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None or row["value"] == "":
            raise RuntimeError(f"store metadata {key!r} is unset")
        return int(row["value"])

    @property
    def dimension(self) -> int | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT value FROM meta WHERE key = 'dimension'"
            ).fetchone()
            assert row is not None
            return int(row["value"]) if row["value"] else None

    @property
    def current_snapshot(self) -> int:
        return self.snapshot()

    def snapshot(self) -> int:
        with self._lock:
            self._ensure_open()
            return self._meta_int("commit_seq")

    def _rollback_quietly(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass

    def commit(self, operations: Iterable[object] | object) -> int:
        """Atomically commit vector intervals and their durable obligations."""

        if isinstance(operations, (CommitOperation, Mapping)):
            supplied = [operations]
        elif isinstance(operations, tuple) and operations and isinstance(operations[0], str):
            supplied = [operations]
        else:
            try:
                supplied = list(operations)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("operations must be an operation or iterable") from exc
        canonical = tuple(_normalise_operation(value) for value in supplied)
        if not canonical:
            raise ValueError("a commit must contain at least one operation")
        logical_ids = [operation.logical_id for operation in canonical]
        if len(set(logical_ids)) != len(logical_ids):
            raise ValueError("one commit may change a logical_id at most once")
        vector_dimensions = {
            operation.vector.shape[0]
            for operation in canonical
            if isinstance(operation.vector, np.ndarray)
        }
        if len(vector_dimensions) > 1:
            raise ValueError("all vectors in a commit must have one dimension")

        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                persisted_dimension_row = self._connection.execute(
                    "SELECT value FROM meta WHERE key = 'dimension'"
                ).fetchone()
                assert persisted_dimension_row is not None
                persisted_dimension = (
                    int(persisted_dimension_row["value"])
                    if persisted_dimension_row["value"]
                    else None
                )
                incoming_dimension = next(iter(vector_dimensions), None)
                if persisted_dimension is None and incoming_dimension is None:
                    raise ValueError("cannot infer dimension from a delete-only first commit")
                if (
                    persisted_dimension is not None
                    and incoming_dimension is not None
                    and persisted_dimension != incoming_dimension
                ):
                    raise ValueError("vector/store dimension mismatch")
                dimension = persisted_dimension or incoming_dimension
                assert dimension is not None
                if persisted_dimension is None:
                    self._connection.execute(
                        "UPDATE meta SET value = ? WHERE key = 'dimension'",
                        (str(dimension),),
                    )

                commit_seq = self._meta_int("commit_seq") + 1
                next_version_id = self._meta_int("version_seq")
                self._connection.execute(
                    "INSERT INTO commits(commit_seq, operation_count, committed_unix_ns) "
                    "VALUES (?, ?, ?)",
                    (commit_seq, len(canonical), time.time_ns()),
                )
                for operation in canonical:
                    current = self._connection.execute(
                        "SELECT version_id FROM versions "
                        "WHERE logical_id = ? AND end_seq IS NULL",
                        (operation.logical_id,),
                    ).fetchone()
                    if operation.kind == "insert" and current is not None:
                        raise ValueError(
                            f"logical_id {operation.logical_id} already exists"
                        )
                    if operation.kind in {"update", "delete"} and current is None:
                        raise ValueError(
                            f"logical_id {operation.logical_id} has no current version"
                        )
                    old_version_id = (
                        int(current["version_id"]) if current is not None else None
                    )
                    if old_version_id is not None:
                        changed = self._connection.execute(
                            "UPDATE versions SET end_seq = ? "
                            "WHERE version_id = ? AND end_seq IS NULL",
                            (commit_seq, old_version_id),
                        ).rowcount
                        if changed != 1:
                            raise RuntimeError("lost current-version update race")

                    new_version_id: int | None = None
                    if operation.kind != "delete":
                        assert isinstance(operation.vector, np.ndarray)
                        next_version_id += 1
                        new_version_id = next_version_id
                        vector = np.asarray(operation.vector, dtype="<f4", order="C")
                        self._connection.execute(
                            "INSERT INTO versions(version_id, logical_id, vector, dimension, "
                            "begin_seq, end_seq) VALUES (?, ?, ?, ?, ?, NULL)",
                            (
                                new_version_id,
                                operation.logical_id,
                                sqlite3.Binary(vector.tobytes(order="C")),
                                dimension,
                                commit_seq,
                            ),
                        )
                    if operation.kind == "insert":
                        self._fault(
                            "commit_insert_after_version_before_obligation"
                        )
                    elif operation.kind == "update":
                        self._fault(
                            "commit_update_after_versions_before_obligation"
                        )
                    else:
                        self._fault(
                            "commit_delete_after_old_end_before_obligation"
                        )
                    self._connection.execute(
                        "INSERT INTO obligations(commit_seq, kind, logical_id, "
                        "old_version_id, new_version_id) VALUES (?, ?, ?, ?, ?)",
                        (
                            commit_seq,
                            operation.kind,
                            operation.logical_id,
                            old_version_id,
                            new_version_id,
                        ),
                    )
                self._connection.execute(
                    "UPDATE meta SET value = ? WHERE key = 'commit_seq'",
                    (str(commit_seq),),
                )
                self._connection.execute(
                    "UPDATE meta SET value = ? WHERE key = 'version_seq'",
                    (str(next_version_id),),
                )
                self._fault("commit_before_sql_commit")
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback_quietly()
                raise
        self._fault("commit_after_sql_commit")
        return commit_seq

    def insert(self, logical_id: int, vector: Sequence[float] | np.ndarray) -> int:
        return self.commit(CommitOperation.insert(logical_id, vector))

    def update(self, logical_id: int, vector: Sequence[float] | np.ndarray) -> int:
        return self.commit(CommitOperation.update(logical_id, vector))

    def delete(self, logical_id: int) -> int:
        return self.commit(CommitOperation.delete(logical_id))

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> VectorRecord:
        dimension = int(row["dimension"])
        raw = bytes(row["vector"])
        if len(raw) != dimension * 4:
            raise RuntimeError("retained vector byte length is corrupt")
        vector = np.frombuffer(raw, dtype="<f4", count=dimension).copy()
        return VectorRecord(
            logical_id=int(row["logical_id"]),
            version_id=int(row["version_id"]),
            vector=vector,
            begin_seq=int(row["begin_seq"]),
            end_seq=None if row["end_seq"] is None else int(row["end_seq"]),
            commit_seq=int(row["begin_seq"]),
        )

    def _visible_records(self, snapshot: int) -> tuple[VectorRecord, ...]:
        rows = self._connection.execute(
            "SELECT version_id, logical_id, vector, dimension, begin_seq, end_seq "
            "FROM versions WHERE begin_seq <= ? AND (end_seq IS NULL OR ? < end_seq) "
            "ORDER BY logical_id, version_id",
            (snapshot, snapshot),
        ).fetchall()
        records = tuple(self._record_from_row(row) for row in rows)
        if len({record.logical_id for record in records}) != len(records):
            raise RuntimeError("MVCC invariant violation: duplicate visible logical_id")
        return records

    def visible_records(self, snapshot: int | None = None) -> tuple[VectorRecord, ...]:
        with self._lock:
            self._ensure_open()
            current = self._meta_int("commit_seq")
            selected = current if snapshot is None else self._validate_snapshot(snapshot, current)
            return self._visible_records(selected)

    def visible_keys(self, snapshot: int | None = None) -> tuple[tuple[int, int], ...]:
        return tuple(record.key for record in self.visible_records(snapshot))

    @staticmethod
    def _validate_snapshot(snapshot: int, current: int) -> int:
        if isinstance(snapshot, bool) or not isinstance(snapshot, (int, np.integer)):
            raise TypeError("snapshot must be an integer commit sequence")
        selected = int(snapshot)
        if selected < 0 or selected > current:
            raise SnapshotError(
                f"snapshot {selected} is outside retained prefix [0, {current}]"
            )
        return selected

    def _select_generation_row(self, snapshot: int) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT generation_id, covered_commit_seq, directory_name, "
            "manifest_sha256 FROM generations "
            "WHERE valid = 1 AND covered_commit_seq <= ? "
            "ORDER BY covered_commit_seq DESC, published_unix_ns DESC, generation_id DESC "
            "LIMIT 1",
            (snapshot,),
        ).fetchone()

    def _capture_view(self, snapshot: int | None) -> PinnedView:
        # One SQLite read transaction makes clock/generation/catalog selection
        # one coherent view even when another process commits concurrently.
        self._connection.execute("BEGIN")
        try:
            current = self._meta_int("commit_seq")
            selected = current if snapshot is None else self._validate_snapshot(snapshot, current)
            generation = self._select_generation_row(selected)
            revision = self._meta_int("catalog_revision")
            self._connection.execute("COMMIT")
        except BaseException:
            self._rollback_quietly()
            raise
        token = uuid.uuid4().hex
        return PinnedView(
            snapshot_seq=selected,
            generation_id=(
                None if generation is None else str(generation["generation_id"])
            ),
            covered_commit_seq=(
                None if generation is None else int(generation["covered_commit_seq"])
            ),
            generation_directory=(
                None if generation is None else str(generation["directory_name"])
            ),
            manifest_sha256=(
                None if generation is None else str(generation["manifest_sha256"])
            ),
            group_catalog_revision=revision,
            token=token,
        )

    @contextmanager
    def pin(self, snapshot: int | None = None) -> Iterator[PinnedView]:
        """Pin one snapshot/generation/catalog selection for a query lifetime."""

        with self._lock:
            self._ensure_open()
            view = self._capture_view(snapshot)
            self._active_pins[view.token] = view
        try:
            yield view
        finally:
            with self._lock:
                self._active_pins.pop(view.token, None)

    def pin_view(self, snapshot: int | None = None):
        return self.pin(snapshot)

    def pin_snapshot(self, snapshot: int | None = None):
        return self.pin(snapshot)

    def pin_generation(self, snapshot: int | None = None):
        return self.pin(snapshot)

    @property
    def active_pin_count(self) -> int:
        with self._lock:
            return len(self._active_pins)

    def _assert_pinned(self, view: PinnedView) -> None:
        if self._active_pins.get(view.token) != view:
            raise PinnedViewError("view is not actively pinned by this store")

    def _invalidate_generation(self, generation_id: str, reason: str) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "UPDATE generations SET valid = 0 WHERE generation_id = ?",
                (generation_id,),
            )
            self._connection.execute(
                "INSERT INTO generation_failures(generation_id, observed_unix_ns, reason) "
                "VALUES (?, ?, ?)",
                (generation_id, time.time_ns(), reason),
            )
            self._connection.execute("COMMIT")
        except BaseException:
            self._rollback_quietly()
            raise

    def _generation_base(self, view: PinnedView) -> BaseIndex:
        assert view.generation_id is not None
        assert view.generation_directory is not None
        validated = self._generations.validate(
            view.generation_directory,
            expected_manifest_sha256=view.manifest_sha256,
            expected_generation_id=view.generation_id,
            expected_coverage=view.covered_commit_seq,
        )
        rows = self._connection.execute(
            "SELECT version_id, logical_id, vector, dimension, begin_seq, end_seq "
            "FROM versions"
        ).fetchall()
        retained = tuple(self._record_from_row(row) for row in rows)
        records = {record.key: record for record in retained}
        return self._generations.load_base(validated, records)

    def _load_delta_for_view(self, view: PinnedView) -> _LoadedDelta:
        coverage = view.covered_commit_seq
        if coverage is None:
            delta = DeltaStore(raw_records=self._visible_records(view.snapshot_seq))
            return _LoadedDelta(delta, _delta_view_hash(view, delta))
        rows = self._connection.execute(
            "SELECT v.version_id, v.logical_id, v.vector, v.dimension, "
            "v.begin_seq, v.end_seq FROM versions AS v "
            "JOIN obligations AS o ON o.new_version_id = v.version_id "
            "WHERE v.begin_seq > ? AND v.begin_seq <= ? "
            "AND (end_seq IS NULL OR ? < end_seq) ORDER BY version_id",
            (coverage, view.snapshot_seq, view.snapshot_seq),
        ).fetchall()
        pending = {int(row["version_id"]): self._record_from_row(row) for row in rows}
        if not pending:
            delta = DeltaStore(raw_records=())
            return _LoadedDelta(delta, _delta_view_hash(view, delta))

        membership_rows = self._connection.execute(
            "SELECT version_id, group_id FROM group_members WHERE revision = ?",
            (view.group_catalog_revision,),
        ).fetchall()
        membership = {
            int(row["version_id"]): int(row["group_id"])
            for row in membership_rows
            if int(row["version_id"]) in pending
        }
        raw_by_version = {
            version_id: record
            for version_id, record in pending.items()
            if version_id not in membership
        }
        if not membership:
            delta = DeltaStore(
                raw_records=tuple(
                    raw_by_version[version_id]
                    for version_id in sorted(raw_by_version)
                )
            )
            return _LoadedDelta(delta, _delta_view_hash(view, delta))

        dimension = self.dimension
        assert dimension is not None
        group_rows = self._connection.execute(
            "SELECT group_id, center, dimension, radius_upper FROM groups "
            "WHERE revision = ? ORDER BY group_id",
            (view.group_catalog_revision,),
        ).fetchall()
        groups: list[Group] = []
        represented: set[int] = set()
        fallback_versions: set[int] = set()
        fallback_groups: set[int] = set()
        fallback_details: list[str] = []
        for row in group_rows:
            group_id = int(row["group_id"])
            member_records = tuple(
                pending[version_id]
                for version_id, assigned_group in sorted(membership.items())
                if assigned_group == group_id
            )
            if not member_records:
                continue
            stored_dimension = int(row["dimension"])
            center_raw = bytes(row["center"])
            if stored_dimension != dimension or len(center_raw) != dimension * 4:
                # An invalid bound must be scanned.  Treating all affected rows
                # as raw is the safe lifecycle fallback.
                for record in member_records:
                    raw_by_version[record.version_id] = record
                    fallback_versions.add(record.version_id)
                fallback_groups.add(group_id)
                fallback_details.append(
                    f"group:{group_id}:invalid_center_shape_or_dimension"
                )
                represented.update(record.version_id for record in member_records)
                continue
            center = np.frombuffer(center_raw, dtype="<f4", count=dimension).copy()
            vectors = np.ascontiguousarray(
                np.stack([record.vector for record in member_records]), dtype=np.float32
            )
            radius = float(row["radius_upper"])
            try:
                groups.append(
                    Group(
                        group_id=group_id,
                        center=center,
                        radius_upper=radius,
                        members=member_records,
                        vectors=vectors,
                    )
                )
                represented.update(record.version_id for record in member_records)
            except (TypeError, ValueError) as exc:
                for record in member_records:
                    raw_by_version[record.version_id] = record
                    fallback_versions.add(record.version_id)
                fallback_groups.add(group_id)
                fallback_details.append(
                    f"group:{group_id}:invalid_bound:{type(exc).__name__}"
                )
                represented.update(record.version_id for record in member_records)
        # Foreign keys normally make this impossible.  If on-disk catalog
        # damage nevertheless leaves an unresolvable membership, scan it raw.
        for version_id in sorted(membership):
            if version_id in represented:
                continue
            raw_by_version[version_id] = pending[version_id]
            fallback_versions.add(version_id)
            fallback_groups.add(membership[version_id])
            fallback_details.append(
                f"group:{membership[version_id]}:missing_catalog_row"
            )
        grouped = (
            GroupDirectory(
                groups=tuple(groups),
                dimension=dimension,
                build_stats=GroupBuildStats(0, 0, 0, 0, 0),
            )
            if groups
            else None
        )
        delta = DeltaStore(
            raw_records=tuple(
                raw_by_version[version_id] for version_id in sorted(raw_by_version)
            ),
            grouped=grouped,
        )
        reason = (
            "invalid_group_catalog_entries_scanned_raw"
            if fallback_versions
            else None
        )
        return _LoadedDelta(
            delta=delta,
            delta_view_hash=_delta_view_hash(view, delta),
            fallback_reason=reason,
            fallback_vector_count=len(fallback_versions),
            fallback_group_count=len(fallback_groups),
            fallback_details=tuple(sorted(set(fallback_details))),
        )

    def _delta_for_view(self, view: PinnedView) -> DeltaStore:
        """Compatibility helper returning only the immutable Delta payload."""

        return self._load_delta_for_view(view).delta

    @staticmethod
    def _attach_store_scope(
        result: SearchResult, view: PinnedView, loaded: _LoadedDelta
    ) -> SearchResult:
        details = dict(result.receipt.certificate_details or {})
        details.update(
            {
                "group_catalog_revision": view.group_catalog_revision,
                "delta_view_hash": loaded.delta_view_hash,
                "group_catalog_fallback_vector_count": loaded.fallback_vector_count,
                "group_catalog_fallback_group_count": loaded.fallback_group_count,
                "group_catalog_fallback_details": list(loaded.fallback_details),
            }
        )
        fallback_reason = result.receipt.fallback_reason
        if loaded.fallback_reason is not None:
            fallback_reason = (
                loaded.fallback_reason
                if fallback_reason is None
                else f"{fallback_reason}|{loaded.fallback_reason}"
            )
        receipt = replace(
            result.receipt,
            group_catalog_revision=view.group_catalog_revision,
            delta_view_hash=loaded.delta_view_hash,
            fallback_reason=fallback_reason,
            certificate_details=details,
        )
        return SearchResult(result.hits, receipt)

    def _exact_fallback(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float,
        view: PinnedView,
        candidate_count: int,
        reason: str,
        audit: bool,
    ) -> SearchResult:
        dimension = self.dimension or int(np.asarray(query).shape[0])
        base = BaseIndex(
            (),
            dimension=dimension,
            generation_id="retained-exact-fallback",
            covered_commit_seq=None,
            threads=self._threads,
        )
        delta = DeltaStore(raw_records=self._visible_records(view.snapshot_seq))
        loaded = _LoadedDelta(delta, _delta_view_hash(view, delta))
        result = SearchEngine(base, loaded.delta).search_pruned(
            query,
            k=k,
            beta=beta,
            snapshot_id=view.snapshot_seq,
            candidate_count=candidate_count,
            audit=audit,
        )
        receipt = replace(
            result.receipt,
            fallback_reason=reason,
            base_generation="retained-exact-fallback",
            covered_commit_seq=None,
        )
        return self._attach_store_scope(SearchResult(result.hits, receipt), view, loaded)

    def _search_pinned(
        self,
        query: np.ndarray,
        *,
        k: int,
        beta: float,
        view: PinnedView,
        candidate_count: int,
        audit: bool,
    ) -> SearchResult:
        with self._lock:
            self._ensure_open()
            self._assert_pinned(view)
            configured_dimension = self.dimension
            q = as_float32_vector(query, name="query")
            if configured_dimension is not None and q.shape[0] != configured_dimension:
                raise ValueError("query/store dimension mismatch")
            if view.generation_id is None:
                return self._exact_fallback(
                    q,
                    k=k,
                    beta=beta,
                    view=view,
                    candidate_count=candidate_count,
                    reason="missing_compatible_generation_exact_retained_scan",
                    audit=audit,
                )
            try:
                base = self._generation_base(view)
            except (GenerationValidationError, OSError, RuntimeError, ValueError) as exc:
                self._invalidate_generation(view.generation_id, str(exc))
                return self._exact_fallback(
                    q,
                    k=k,
                    beta=beta,
                    view=view,
                    candidate_count=candidate_count,
                    reason="invalid_generation_exact_retained_scan",
                    audit=audit,
                )
            loaded = self._load_delta_for_view(view)
        # Base and Delta objects are immutable.  The pin prevents collection,
        # while SQLite remains available to commits during the numeric search.
        result = SearchEngine(base, loaded.delta).search_pruned(
            q,
            k=k,
            beta=beta,
            snapshot_id=view.snapshot_seq,
            candidate_count=candidate_count,
            audit=audit,
        )
        return self._attach_store_scope(result, view, loaded)

    def search(
        self,
        query: np.ndarray | Sequence[float],
        k: int,
        beta: float = 0.0,
        snapshot: int | None = None,
        *,
        candidate_count: int = 64,
        audit: bool = False,
        view: PinnedView | None = None,
    ) -> SearchResult:
        """Search one immutable view, using exact retained fallback if needed."""

        q = np.asarray(query)
        if view is not None:
            if snapshot is not None and snapshot != view.snapshot_seq:
                raise ValueError("explicit snapshot disagrees with pinned view")
            return self._search_pinned(
                q,
                k=k,
                beta=beta,
                view=view,
                candidate_count=candidate_count,
                audit=audit,
            )
        with self.pin(snapshot) as captured:
            return self._search_pinned(
                q,
                k=k,
                beta=beta,
                view=captured,
                candidate_count=candidate_count,
                audit=audit,
            )

    def raw_pending(self, snapshot: int | None = None) -> tuple[VectorRecord, ...]:
        """Read raw pending records from durable state, never an in-memory cache."""

        with self.pin(snapshot) as view:
            with self._lock:
                self._assert_pinned(view)
                return self._delta_for_view(view).raw_records

    def group_pending(
        self,
        centers: np.ndarray | Sequence[Sequence[float]] | None = None,
    ) -> int:
        """Build a complete immutable catalog revision, then atomically switch."""

        with self._lock:
            self._ensure_open()
            build_commit_seq = self._meta_int("commit_seq")
            dimension = self.dimension
            if dimension is None:
                return self._meta_int("catalog_revision")
            rows = self._connection.execute(
                "SELECT DISTINCT v.version_id, v.logical_id, v.vector, v.dimension, "
                "v.begin_seq, v.end_seq FROM versions AS v "
                "JOIN obligations AS o ON o.new_version_id = v.version_id "
                "WHERE v.begin_seq <= ? ORDER BY v.version_id",
                (build_commit_seq,),
            ).fetchall()
            records = tuple(self._record_from_row(row) for row in rows)
        if not records:
            with self._lock:
                return self._meta_int("catalog_revision")
        if centers is None:
            # A fixed origin center does not inspect final queries or tune on
            # Delta outcomes.  It is intentionally conservative, not fast.
            center_matrix = np.zeros((1, dimension), dtype=np.float32)
        else:
            center_matrix = np.asarray(centers)
            if (
                center_matrix.ndim != 2
                or center_matrix.shape[0] == 0
                or center_matrix.shape[1] != dimension
            ):
                raise ValueError("centers/store dimension mismatch")
            center_matrix = np.ascontiguousarray(
                np.stack(
                    [as_float32_vector(center, name="center") for center in center_matrix]
                ),
                dtype=np.float32,
            )
        directory = GroupDirectory.from_centers(
            center_matrix, records, threads=self._threads
        )
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                old_revision = self._meta_int("catalog_revision")
                revision = old_revision + 1
                self._connection.execute(
                    "INSERT INTO catalog_revisions(revision, build_commit_seq, "
                    "published_unix_ns) VALUES (?, ?, ?)",
                    (revision, build_commit_seq, time.time_ns()),
                )
                for group in directory.groups:
                    center = np.asarray(group.center, dtype="<f4", order="C")
                    self._connection.execute(
                        "INSERT INTO groups(revision, group_id, center, dimension, "
                        "radius_upper) VALUES (?, ?, ?, ?, ?)",
                        (
                            revision,
                            group.group_id,
                            sqlite3.Binary(center.tobytes(order="C")),
                            dimension,
                            group.radius_upper,
                        ),
                    )
                    self._connection.executemany(
                        "INSERT INTO group_members(revision, group_id, version_id) "
                        "VALUES (?, ?, ?)",
                        [
                            (revision, group.group_id, record.version_id)
                            for record in group.members
                        ],
                    )
                self._connection.execute(
                    "UPDATE meta SET value = ? WHERE key = 'catalog_revision'",
                    (str(revision),),
                )
                self._fault("group_before_catalog_commit")
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback_quietly()
                raise
        self._fault("group_after_catalog_commit")
        return revision

    def build_generation(self, generation_id: str | None = None) -> str:
        """Build files at a stable cutoff, rename, then publish in SQLite."""

        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN")
            try:
                cutoff = self._meta_int("commit_seq")
                dimension = self.dimension
                if dimension is None:
                    raise ValueError("dimension is unknown; configure it or insert a vector")
                records = self._visible_records(cutoff)
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback_quietly()
                raise
        artifact = self._generations.build(
            records,
            covered_commit_seq=cutoff,
            dimension=dimension,
            generation_id=generation_id,
        )
        # Re-validate before the database is allowed to point at the files.
        self._generations.validate(
            artifact.directory_name,
            expected_manifest_sha256=artifact.manifest_sha256,
            expected_generation_id=artifact.generation_id,
            expected_coverage=artifact.covered_commit_seq,
        )
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT INTO generations(generation_id, covered_commit_seq, "
                    "directory_name, manifest_sha256, vector_count, published_unix_ns, valid) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (
                        artifact.generation_id,
                        artifact.covered_commit_seq,
                        artifact.directory_name,
                        artifact.manifest_sha256,
                        artifact.vector_count,
                        time.time_ns(),
                    ),
                )
                self._fault("generation_before_catalog_commit")
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback_quietly()
                raise
        self._fault("generation_after_catalog_commit")
        return artifact.generation_id

    def recover(self) -> RecoveryReport:
        """Validate every published file reference and ignore crash leftovers."""

        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT generation_id, covered_commit_seq, directory_name, "
                "manifest_sha256, valid FROM generations "
                "ORDER BY covered_commit_seq, published_unix_ns, generation_id"
            ).fetchall()
            valid: list[str] = []
            invalid: list[str] = []
            referenced = {str(row["directory_name"]) for row in rows}
            failures: list[tuple[str, str]] = []
            for row in rows:
                identifier = str(row["generation_id"])
                if not int(row["valid"]):
                    invalid.append(identifier)
                    continue
                try:
                    self._generations.validate(
                        str(row["directory_name"]),
                        expected_manifest_sha256=str(row["manifest_sha256"]),
                        expected_generation_id=identifier,
                        expected_coverage=int(row["covered_commit_seq"]),
                    )
                except (GenerationValidationError, OSError, ValueError) as exc:
                    invalid.append(identifier)
                    failures.append((identifier, str(exc)))
                else:
                    valid.append(identifier)
            if failures:
                try:
                    self._connection.execute("BEGIN IMMEDIATE")
                    for identifier, reason in failures:
                        self._connection.execute(
                            "UPDATE generations SET valid = 0 WHERE generation_id = ?",
                            (identifier,),
                        )
                        self._connection.execute(
                            "INSERT INTO generation_failures(generation_id, "
                            "observed_unix_ns, reason) VALUES (?, ?, ?)",
                            (identifier, time.time_ns(), reason),
                        )
                    self._connection.execute("COMMIT")
                except BaseException:
                    self._rollback_quietly()
                    raise
            finals, temporaries = self._generations.directories()
            orphans = tuple(name for name in finals if name not in referenced)
            selected = self._select_generation_row(self._meta_int("commit_seq"))
            current = None if selected is None else str(selected["generation_id"])
            # PASSIVE never blocks readers; correctness does not depend on the
            # checkpoint, but running it exercises hot-WAL recovery on reopen.
            self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchall()
            return RecoveryReport(
                valid_generations=tuple(valid),
                invalid_generations=tuple(invalid),
                orphan_generations=orphans,
                temporary_directories=temporaries,
                current_generation=current,
            )

    def obligation_rows(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT obligation_id, commit_seq, kind, logical_id, old_version_id, "
                "new_version_id FROM obligations ORDER BY obligation_id"
            ).fetchall()
            return tuple(dict(row) for row in rows)

    def generation_rows(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT generation_id, covered_commit_seq, directory_name, "
                "manifest_sha256, vector_count, published_unix_ns, valid "
                "FROM generations ORDER BY covered_commit_seq, published_unix_ns"
            ).fetchall()
            return tuple(dict(row) for row in rows)

    def collect_garbage(self) -> dict[str, object]:
        """Refuse active pins; otherwise retain every historical object safely."""

        with self._lock:
            self._ensure_open()
            if self._active_pins:
                raise PinnedGenerationError(
                    "garbage collection refused while a snapshot/generation is pinned"
                )
            counts = {
                "versions_retained": self._connection.execute(
                    "SELECT COUNT(*) FROM versions"
                ).fetchone()[0],
                "obligations_retained": self._connection.execute(
                    "SELECT COUNT(*) FROM obligations"
                ).fetchone()[0],
                "generations_retained": self._connection.execute(
                    "SELECT COUNT(*) FROM generations"
                ).fetchone()[0],
            }
            return {"status": GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION, **counts}

    def gc(self) -> dict[str, object]:
        return self.collect_garbage()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_pins:
                raise PinnedViewError("cannot close a store with active pinned views")
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "SQLiteLifecycleStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


# Short aliases keep the public reference API unsurprising without requiring
# package-level import changes in this phase-sized contribution.
LifecycleStore = SQLiteLifecycleStore
VersionStore = SQLiteLifecycleStore


__all__ = [
    "CommitOperation",
    "FAULT_POINTS",
    "GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION",
    "LifecycleStore",
    "PinnedGenerationError",
    "PinnedView",
    "PinnedViewError",
    "RecoveryReport",
    "SQLiteLifecycleStore",
    "SnapshotError",
    "VersionStore",
]
