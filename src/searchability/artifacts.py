"""Small helpers for deterministic, crash-safe experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_tree_sha256(root: str | Path = ".") -> str:
    """Hash the executable research implementation used by a benchmark run.

    Git's commit alone is insufficient while an experiment is run from a dirty
    worktree.  The digest covers paths and bytes for the Python package,
    executable scripts, dependency declarations, and Makefile.  Generated
    outputs and configuration are intentionally excluded: the effective config
    has its own object hash in every run.
    """

    repository = Path(root).resolve()
    candidates: list[Path] = []
    executable_suffixes = {".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    for directory in (repository / "src", repository / "scripts"):
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in executable_suffixes
            )
    for name in ("pyproject.toml", "requirements-lock.txt", "Makefile", "setup.py"):
        path = repository / name
        if path.is_file():
            candidates.append(path)
    digest = hashlib.sha256()
    digest.update(b"searchability-implementation-tree-v1\0")
    for path in sorted(set(candidates), key=lambda item: item.relative_to(repository).as_posix()):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def native_test_tree_sha256(root: str | Path = ".") -> str:
    """Hash every Python test/support module used by the native gate."""

    repository = Path(root).resolve()
    tests = repository / "tests"
    digest = hashlib.sha256()
    digest.update(b"issue3-native-test-tree-v3\0")
    for path in sorted(
        (value for value in tests.rglob("*.py") if value.is_file()),
        key=lambda value: value.relative_to(repository).as_posix(),
    ):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)
