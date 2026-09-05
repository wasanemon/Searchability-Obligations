"""Crash-safe identity checks for the final-only HNSW reference run."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping

from .artifacts import atomic_write_json, file_sha256, object_sha256
from .native_final import NativeFinalError


_HEX = frozenset("0123456789abcdef")
_SEGMENT_STATUSES = frozenset(
    {"running", "completed", "failed_incomplete", "interrupted_before_resume"}
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise NativeFinalError(
                    f"malformed HNSW JSON row {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise NativeFinalError(f"non-object HNSW row {path}:{line_number}")
            rows.append(value)
    return rows


def _canonical_relative_path(
    value: object, *, prefix: tuple[str, ...], label: str
) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise NativeFinalError(f"{label} path is not canonical")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or len(parsed.parts) <= len(prefix)
        or tuple(parsed.parts[: len(prefix)]) != prefix
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise NativeFinalError(f"{label} path is not canonical")
    return value


def _build_records(
    manifest: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    segments = manifest.get("build_segments")
    if not isinstance(segments, list):
        raise NativeFinalError("HNSW manifest has no build-segment registry")
    by_id: dict[str, dict[str, object]] = {}
    normalized: list[dict[str, object]] = []
    for ordinal, value in enumerate(segments):
        if not isinstance(value, dict):
            raise NativeFinalError("malformed HNSW build segment")
        segment_id = value.get("segment_id")
        if (
            segment_id != f"process-{ordinal:04d}"
            or value.get("ordinal") != ordinal
            or isinstance(value.get("ordinal"), bool)
            or segment_id in by_id
            or value.get("status") not in _SEGMENT_STATUSES
            or not isinstance(value.get("started_at_utc"), str)
            or not isinstance(value.get("builds"), dict)
        ):
            raise NativeFinalError("malformed HNSW build segment identity")
        builds = value["builds"]
        for experiment_id, build in builds.items():
            if not isinstance(experiment_id, str) or not isinstance(build, dict):
                raise NativeFinalError("malformed HNSW build-segment record")
            build_identity = build.get("build_identity")
            build_sha = build.get("build_identity_sha256")
            if (
                build.get("experiment_id") != experiment_id
                or not isinstance(build_identity, dict)
                or not _is_sha256(build_sha)
                or object_sha256(build_identity) != build_sha
                or build_identity.get("experiment_id") != experiment_id
                or build_identity.get("dataset_hash") != build.get("dataset_hash")
            ):
                raise NativeFinalError("HNSW build identity/hash mismatch")
            indexes = build_identity.get("serialized_indexes")
            parameters = build_identity.get("parameters")
            populations = build_identity.get("populations")
            if (
                build_identity.get("schema_version") != 1
                or isinstance(build_identity.get("schema_version"), bool)
                or not _is_sha256(build_identity.get("dataset_hash"))
                or not isinstance(indexes, dict)
                or set(indexes) != {"base", "delta_hnsw", "full_hnsw"}
                or not isinstance(parameters, dict)
                or not isinstance(populations, dict)
            ):
                raise NativeFinalError("malformed HNSW build identity")
            for index_name, index_evidence in indexes.items():
                if (
                    not isinstance(index_name, str)
                    or not isinstance(index_evidence, dict)
                    or not _is_sha256(index_evidence.get("sha256"))
                    or isinstance(index_evidence.get("serialized_bytes"), bool)
                    or not isinstance(index_evidence.get("serialized_bytes"), int)
                    or int(index_evidence["serialized_bytes"]) < 0
                ):
                    raise NativeFinalError("malformed serialized HNSW index identity")
            for name in ("base_vectors", "delta_vectors", "full_vectors"):
                count = populations.get(name)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise NativeFinalError("malformed HNSW build population identity")
            for name in (
                "m",
                "ef_construction",
                "base_ef_search",
                "threads",
            ):
                number = parameters.get(name)
                if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                    raise NativeFinalError("malformed HNSW build parameter identity")
            if (
                parameters.get("ef_search_values") != [128, 512]
                or parameters.get("methods")
                != ["base_plus_delta_hnsw", "full_population_hnsw"]
                or not isinstance(build_identity.get("faiss_version"), str)
                or not isinstance(build_identity.get("faiss_compile_options"), str)
            ):
                raise NativeFinalError("HNSW build method/runtime identity mismatch")
        by_id[str(segment_id)] = value
        normalized.append(value)
    return by_id, normalized


def raw_build_binding(
    rows: list[dict[str, object]],
    *,
    manifest: Mapping[str, object],
    experiment_id: str,
) -> tuple[str, str]:
    try:
        segment_ids = {str(row["build_segment_id"]) for row in rows}
        build_hashes = {str(row["build_identity_sha256"]) for row in rows}
    except KeyError as error:
        raise NativeFinalError("HNSW raw row lacks build-segment identity") from error
    if len(segment_ids) != 1 or len(build_hashes) != 1:
        raise NativeFinalError("HNSW raw block mixes build-segment identities")
    segment_id = next(iter(segment_ids))
    build_sha = next(iter(build_hashes))
    segments, _ = _build_records(manifest)
    segment = segments.get(segment_id)
    build = segment.get("builds", {}).get(experiment_id) if segment else None
    if (
        not isinstance(build, dict)
        or build.get("build_identity_sha256") != build_sha
    ):
        raise NativeFinalError("HNSW raw build identity is absent from the manifest")
    return segment_id, build_sha


def verify_build_bindings(
    *,
    manifest: Mapping[str, object],
    checkpoint: Mapping[str, object],
    expected_blocks: Mapping[str, Mapping[str, object]],
    require_complete: bool,
) -> dict[str, str]:
    """Verify every block against a process segment and one index identity.

    Blocks from multiple process invocations may coexist only when the actual
    serialized Faiss indexes have the same build identity.  This permits a
    deterministic resume without silently combining measurements from a
    different graph.
    """

    segments, normalized = _build_records(manifest)
    if require_complete and any(
        segment.get("status") == "running" for segment in normalized
    ):
        raise NativeFinalError("completed HNSW run has a running build segment")
    completed = checkpoint.get("completed_blocks")
    if not isinstance(completed, dict):
        raise NativeFinalError("HNSW checkpoint has no completed block registry")
    if require_complete and set(completed) != set(expected_blocks):
        raise NativeFinalError("HNSW checkpoint does not cover every locked block")
    active: dict[str, str] = {}
    for key, entry in completed.items():
        expected = expected_blocks.get(str(key))
        if expected is None or not isinstance(entry, dict):
            raise NativeFinalError("HNSW checkpoint contains an unconfigured block")
        experiment_id = str(expected["experiment_id"])
        segment_id = entry.get("build_segment_id")
        build_sha = entry.get("build_identity_sha256")
        segment = segments.get(str(segment_id))
        build = segment.get("builds", {}).get(experiment_id) if segment else None
        if (
            not isinstance(segment_id, str)
            or not _is_sha256(build_sha)
            or not isinstance(build, dict)
            or build.get("build_identity_sha256") != build_sha
        ):
            raise NativeFinalError("HNSW checkpoint build identity is not manifest-bound")
        prior = active.setdefault(experiment_id, str(build_sha))
        if prior != build_sha:
            raise NativeFinalError(
                f"{experiment_id}: checkpoint mixes different HNSW build identities"
            )
    return active


def orphan_destination(
    source_relative: str, *, segment_id: str, raw_sha256: str
) -> str:
    source = PurePosixPath(source_relative)
    if (
        len(source.parts) != 2
        or source.parts[0] != "raw"
        or not source.name.endswith(".jsonl")
        or not _is_sha256(raw_sha256)
        or not segment_id.startswith("process-")
    ):
        raise NativeFinalError("cannot derive a canonical HNSW orphan path")
    stem = source.name[: -len(".jsonl")]
    return f"failures/orphans/{stem}.{segment_id}.{raw_sha256}.orphan.jsonl"


def _orphan_record(
    *,
    path: Path,
    root: Path,
    manifest: Mapping[str, object],
    expected_blocks: Mapping[str, Mapping[str, object]],
    identity: Mapping[str, object],
    preserved_path: str,
) -> dict[str, object]:
    raw_sha = file_sha256(path)
    rows = read_jsonl_objects(path)
    if not rows:
        raise NativeFinalError("empty uncheckpointed HNSW raw shard")
    experiment_ids = {str(row.get("experiment_id")) for row in rows}
    positions = {int(row.get("query_position", -1)) for row in rows}
    if len(experiment_ids) != 1 or not positions:
        raise NativeFinalError("malformed uncheckpointed HNSW raw identity")
    experiment_id = next(iter(experiment_ids))
    matching = [
        (key, block)
        for key, block in expected_blocks.items()
        if str(block["experiment_id"]) == experiment_id
        and positions
        == set(range(int(block["query_start"]), int(block["query_stop"])))
    ]
    if len(matching) != 1:
        raise NativeFinalError("uncheckpointed HNSW raw block is not configured")
    key, block = matching[0]
    if any(
        any(row.get(field) != value for field, value in identity.items())
        for row in rows
    ):
        raise NativeFinalError("uncheckpointed HNSW raw identity mismatch")
    expected_rows = {
        (
            experiment_id,
            position,
            int(block["query_offset"]) + position,
            method,
            int(ef_search),
        )
        for position in range(int(block["query_start"]), int(block["query_stop"]))
        for method in identity["methods"]
        for ef_search in identity["ef_search_values"]
    }
    try:
        observed_rows = {
            (
                str(row.get("experiment_id")),
                int(row.get("query_position", -1)),
                int(row.get("query_id", -1)),
                str(row.get("method")),
                int(row.get("ef_search", -1)),
            )
            for row in rows
        }
    except (TypeError, ValueError) as error:
        raise NativeFinalError("malformed uncheckpointed HNSW raw row") from error
    if len(rows) != len(expected_rows) or observed_rows != expected_rows:
        raise NativeFinalError("uncheckpointed HNSW raw coverage mismatch")
    segment_id, build_sha = raw_build_binding(
        rows, manifest=manifest, experiment_id=experiment_id
    )
    expected_preserved = orphan_destination(
        str(block["path"]), segment_id=segment_id, raw_sha256=raw_sha
    )
    if preserved_path != expected_preserved:
        raise NativeFinalError("HNSW orphan preservation path is not canonical")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise NativeFinalError("HNSW orphan path escapes the output root") from error
    if resolved != path.absolute():
        raise NativeFinalError("HNSW orphan path traverses a symlink")
    return {
        "block_key": key,
        "source_path": str(block["path"]),
        "preserved_path": preserved_path,
        "sha256": raw_sha,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "experiment_id": experiment_id,
        "query_start": int(block["query_start"]),
        "query_stop": int(block["query_stop"]),
        "build_segment_id": segment_id,
        "build_identity_sha256": build_sha,
        "reason": "raw_published_before_checkpoint",
    }


def _safe_directory(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise NativeFinalError("HNSW failure inventory path is unsafe")
    path.mkdir(parents=True, exist_ok=True)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise NativeFinalError("HNSW failure inventory escapes the output root") from error
    if resolved != path.absolute():
        raise NativeFinalError("HNSW failure inventory traverses a symlink")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preserve_uncheckpointed_raw(
    *,
    root: Path,
    manifest_path: Path,
    manifest: dict[str, object],
    checkpoint: Mapping[str, object],
    expected_blocks: Mapping[str, Mapping[str, object]],
    identity: Mapping[str, object],
) -> None:
    """Move complete-but-uncheckpointed shards into a checksum-bound archive.

    The preserved filename is derivable from the locked block, build segment,
    and payload hash.  Therefore a crash after ``os.replace`` but before the
    manifest update is reconciled on the next resume without trusting an
    unbound file.
    """

    preserved = manifest.get("preserved_orphans")
    if not isinstance(preserved, list):
        raise NativeFinalError("HNSW manifest has no preserved-orphan registry")
    raw_dir = root / "raw"
    orphan_dir = root / "failures" / "orphans"
    if orphan_dir.exists():
        if (
            orphan_dir.is_symlink()
            or not orphan_dir.is_dir()
            or orphan_dir.resolve(strict=True) != orphan_dir.absolute()
        ):
            raise NativeFinalError("HNSW orphan inventory is unsafe")
        for candidate in orphan_dir.rglob("*"):
            if candidate.is_dir() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise NativeFinalError("HNSW orphan inventory has a non-regular file")
            relative = candidate.relative_to(root).as_posix()
            record = _orphan_record(
                path=candidate,
                root=root,
                manifest=manifest,
                expected_blocks=expected_blocks,
                identity=identity,
                preserved_path=relative,
            )
            existing = [
                value
                for value in preserved
                if isinstance(value, dict)
                and value.get("preserved_path") == relative
            ]
            if existing and (len(existing) != 1 or existing[0] != record):
                raise NativeFinalError("HNSW preserved-orphan manifest mismatch")
            if not existing:
                preserved.append(record)
                preserved.sort(
                    key=lambda value: str(value.get("preserved_path", ""))
                )
                atomic_write_json(manifest_path, manifest)

    completed = checkpoint.get("completed_blocks")
    if not isinstance(completed, dict):
        raise NativeFinalError("HNSW checkpoint has no completed block registry")
    declared_paths = {
        str(entry.get("path"))
        for entry in completed.values()
        if isinstance(entry, dict)
    }
    expected_by_path = {
        str(block["path"]): (key, block) for key, block in expected_blocks.items()
    }
    actual_paths: list[str] = []
    for candidate in raw_dir.rglob("*"):
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise NativeFinalError("HNSW raw inventory contains a non-regular file")
        actual_paths.append(candidate.relative_to(root).as_posix())
    for relative in sorted(set(actual_paths) - declared_paths):
        configured = expected_by_path.get(relative)
        if configured is None or configured[0] in completed:
            raise NativeFinalError("unrecognized uncheckpointed HNSW raw shard")
        source = root.joinpath(*PurePosixPath(relative).parts)
        rows = read_jsonl_objects(source)
        experiment_id = str(configured[1]["experiment_id"])
        segment_id, _ = raw_build_binding(
            rows, manifest=manifest, experiment_id=experiment_id
        )
        destination_relative = orphan_destination(
            relative, segment_id=segment_id, raw_sha256=file_sha256(source)
        )
        destination = root.joinpath(*PurePosixPath(destination_relative).parts)
        _safe_directory(root, "failures/orphans")
        if destination.exists() or destination.is_symlink():
            raise NativeFinalError("HNSW orphan preservation target already exists")
        # Validate the source and the deterministic destination name before the
        # atomic move.  The source remains intact if validation fails.
        record = _orphan_record(
            path=source,
            root=root,
            manifest=manifest,
            expected_blocks=expected_blocks,
            identity=identity,
            preserved_path=destination_relative,
        )
        os.replace(source, destination)
        _fsync_directory(raw_dir)
        _fsync_directory(destination.parent)
        preserved.append(record)
        preserved.sort(key=lambda value: str(value.get("preserved_path", "")))
        atomic_write_json(manifest_path, manifest)


def verify_preserved_orphan_inventory(
    *,
    root: Path,
    manifest: Mapping[str, object],
    expected_blocks: Mapping[str, Mapping[str, object]],
    identity: Mapping[str, object],
) -> None:
    preserved = manifest.get("preserved_orphans")
    if not isinstance(preserved, list):
        raise NativeFinalError("HNSW manifest has no preserved-orphan registry")
    declared: set[str] = set()
    for value in preserved:
        if not isinstance(value, dict):
            raise NativeFinalError("malformed HNSW preserved-orphan record")
        relative = _canonical_relative_path(
            value.get("preserved_path"),
            prefix=("failures", "orphans"),
            label="HNSW preserved-orphan",
        )
        if relative in declared:
            raise NativeFinalError("duplicate HNSW preserved-orphan path")
        declared.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise NativeFinalError("HNSW preserved orphan is missing or unsafe")
        observed = _orphan_record(
            path=path,
            root=root,
            manifest=manifest,
            expected_blocks=expected_blocks,
            identity=identity,
            preserved_path=relative,
        )
        if value != observed:
            raise NativeFinalError("HNSW preserved-orphan evidence changed")
    orphan_dir = root / "failures" / "orphans"
    actual: set[str] = set()
    if orphan_dir.exists():
        if (
            orphan_dir.is_symlink()
            or not orphan_dir.is_dir()
            or orphan_dir.resolve(strict=True) != orphan_dir.absolute()
        ):
            raise NativeFinalError("HNSW orphan inventory is unsafe")
        for candidate in orphan_dir.rglob("*"):
            if candidate.is_dir() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise NativeFinalError("HNSW orphan inventory has a non-regular file")
            actual.add(candidate.relative_to(root).as_posix())
    if actual != declared:
        raise NativeFinalError("HNSW preserved-orphan filesystem inventory differs")


__all__ = [
    "orphan_destination",
    "preserve_uncheckpointed_raw",
    "raw_build_binding",
    "read_jsonl_objects",
    "verify_build_bindings",
    "verify_preserved_orphan_inventory",
]
