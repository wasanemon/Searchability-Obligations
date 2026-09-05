#!/usr/bin/env python3
"""Run final-only approximate HNSW references for the locked Issue #3 anchors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
import traceback

import faiss

from searchability.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from searchability.base import BaseIndex
from searchability.baselines import PreparedFaissIndex
from searchability.datasets import load_dataset
from searchability.models import VectorRecord
from searchability.native_analysis import load_native_evidence
from searchability.native_final import NativeFinalError, read_object
from searchability.native_hnsw import (
    preserve_uncheckpointed_raw,
    raw_build_binding,
    verify_build_bindings,
    verify_preserved_orphan_inventory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records(vectors: object, identifiers: object) -> tuple[VectorRecord, ...]:
    return tuple(
        VectorRecord(int(identifier), 0, vector)
        for identifier, vector in zip(identifiers.tolist(), vectors)
    )


def _serialized_index_evidence(index: faiss.Index) -> dict[str, object]:
    serialized = faiss.serialize_index(index)
    payload = memoryview(serialized).cast("B")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "serialized_bytes": int(serialized.nbytes),
    }


def _build_identity(
    *,
    experiment_id: str,
    dataset_hash: str,
    base: BaseIndex,
    delta_hnsw: PreparedFaissIndex,
    full_hnsw: PreparedFaissIndex,
    base_records: tuple[VectorRecord, ...],
    delta_records: tuple[VectorRecord, ...],
    m: int,
    ef_construction: int,
    base_ef: int,
    ef_values: list[int],
) -> dict[str, object]:
    """Bind a build attempt to the exact serialized Faiss graph bytes."""

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_hash": dataset_hash,
        "faiss_version": str(faiss.__version__),
        "faiss_compile_options": str(faiss.get_compile_options()),
        "parameters": {
            "m": m,
            "ef_construction": ef_construction,
            "base_ef_search": base_ef,
            "ef_search_values": list(ef_values),
            "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
            "threads": 1,
        },
        "populations": {
            "base_vectors": len(base_records),
            "delta_vectors": len(delta_records),
            "full_vectors": len(base_records) + len(delta_records),
        },
        "serialized_indexes": {
            "base": _serialized_index_evidence(base.index),
            "delta_hnsw": _serialized_index_evidence(delta_hnsw.index),
            "full_hnsw": _serialized_index_evidence(full_hnsw.index),
        },
    }


def _verify_raw(
    path: Path,
    entry: dict[str, object],
    *,
    root: Path,
    identity: dict[str, object],
    manifest: dict[str, object],
    experiment_id: str,
    expected_block: dict[str, object],
) -> None:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise NativeFinalError(f"HNSW raw path escapes the output root: {path}") from error
    if path.is_symlink() or not path.is_file() or file_sha256(path) != entry.get("sha256"):
        raise NativeFinalError(f"HNSW raw checksum mismatch: {path}")
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(values) != int(entry.get("rows", -1)):
        raise NativeFinalError(f"HNSW raw row-count mismatch: {path}")
    if any(
        not isinstance(row, dict)
        or row.get("experiment_id") != experiment_id
        or any(row.get(field) != value for field, value in identity.items())
        for row in values
    ):
        raise NativeFinalError(f"HNSW raw identity mismatch: {path}")
    start = int(expected_block["query_start"])
    stop = int(expected_block["query_stop"])
    query_offset = int(expected_block["query_offset"])
    expected_rows = {
        (experiment_id, position, query_offset + position, method, int(ef_search))
        for position in range(start, stop)
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
            for row in values
        }
    except (TypeError, ValueError) as error:
        raise NativeFinalError(f"HNSW raw row key is malformed: {path}") from error
    if len(values) != len(expected_rows) or observed_rows != expected_rows:
        raise NativeFinalError(f"HNSW raw block coverage mismatch: {path}")
    segment_id, build_sha = raw_build_binding(
        values, manifest=manifest, experiment_id=experiment_id
    )
    if (
        entry.get("build_segment_id") != segment_id
        or entry.get("build_identity_sha256") != build_sha
    ):
        raise NativeFinalError(f"HNSW raw/checkpoint build identity mismatch: {path}")


def _verify_storage_inventory(
    *,
    root: Path,
    checkpoint: dict[str, object],
    manifest: dict[str, object],
    expected_blocks: dict[str, dict[str, object]],
    identity: dict[str, object],
) -> None:
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root.absolute()
    ):
        raise NativeFinalError("HNSW output root is missing or traverses a symlink")
    completed = checkpoint.get("completed_blocks")
    if not isinstance(completed, dict) or not set(completed).issubset(expected_blocks):
        raise NativeFinalError("HNSW checkpoint contains an unconfigured block")
    raw_dir = root / "raw"
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise NativeFinalError("HNSW raw directory is missing or is a symlink")
    actual_files: set[str] = set()
    for candidate in raw_dir.rglob("*"):
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise NativeFinalError("HNSW raw inventory contains a non-regular file")
        actual_files.add(candidate.relative_to(root).as_posix())
    declared_files: set[str] = set()
    for key, entry in completed.items():
        if not isinstance(entry, dict):
            raise NativeFinalError("malformed HNSW checkpoint entry")
        expected = expected_blocks[key]
        relative = entry.get("path")
        if not isinstance(relative, str) or "\\" in relative:
            raise NativeFinalError("HNSW checkpoint path is not canonical")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != relative
            or len(parsed.parts) != 2
            or parsed.parts[0] != "raw"
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or relative != expected["path"]
            or relative in declared_files
            or entry.get("experiment_id") != expected["experiment_id"]
            or entry.get("query_start") != expected["query_start"]
            or entry.get("query_stop") != expected["query_stop"]
        ):
            raise NativeFinalError("HNSW checkpoint entry differs from its locked block")
        declared_files.add(relative)
        _verify_raw(
            root.joinpath(*parsed.parts),
            entry,
            root=root,
            identity=identity,
            manifest=manifest,
            experiment_id=str(expected["experiment_id"]),
            expected_block=expected,
        )
    if actual_files != declared_files:
        raise NativeFinalError("HNSW checkpoint/raw filesystem inventory differs")


def _truth_by_query(
    source_rows: object, experiment_id: str
) -> dict[int, list[list[int]]]:
    truth: dict[int, list[list[int]]] = {}
    for row in source_rows:
        if row.get("experiment_id") != experiment_id or row.get("method_role") != "O":
            continue
        query_id = int(row["query_id"])
        keys = row.get("full_visible_exact_keys")
        if not isinstance(keys, list):
            raise NativeFinalError("native final row has no full-visible exact truth")
        if query_id in truth and truth[query_id] != keys:
            raise NativeFinalError("full-visible exact truth changed within a session")
        truth[query_id] = keys
    return truth


def _recall(result: list[list[int]], truth: list[list[int]]) -> float | None:
    if not truth:
        return None
    return len({tuple(value) for value in result}.intersection(
        tuple(value) for value in truth
    )) / len(truth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pre-gate-authorization", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for variable in THREAD_VARIABLES:
        if os.environ.get(variable) != "1":
            raise NativeFinalError(f"{variable} must be exactly 1")

    decision = read_object(args.decision)
    if decision.get("final_status") == "NOT_RUN_GATE_NOT_PASSED":
        print("validation NOT_PASSED: HNSW references not started")
        return 0
    if decision.get("final_status") not in {
        "AUTHORIZED_NOT_YET_RUN",
        "COMPLETED_PASSED",
        "COMPLETED_NOT_PASSED",
    }:
        raise NativeFinalError("HNSW reference run is not authorized")
    lock = read_object(args.lock)
    config = read_object(args.config)
    lock_sha = file_sha256(args.lock)
    config_sha = file_sha256(args.config)
    implementation_hash = implementation_tree_sha256(REPOSITORY_ROOT)
    if (
        decision.get("validation_gate_status") != "PASSED"
        or lock.get("validation_gate_status") != "PASSED"
        or lock.get("status") != "LOCKED_FOR_FRESH_FINAL"
        or decision.get("final_lock_sha256") != lock_sha
        or decision.get("final_config_file_sha256") != config_sha
        or lock.get("implementation_tree_sha256") != implementation_hash
    ):
        raise NativeFinalError("HNSW authorization identity mismatch")
    hnsw_policy = config.get("hnsw_references")
    if (
        not isinstance(hnsw_policy, dict)
        or hnsw_policy != lock.get("hnsw_references")
        or hnsw_policy.get("run_only_after_fresh_final_performance_pass") is not True
        or "run_only_after_validation_pass" in hnsw_policy
    ):
        raise NativeFinalError("HNSW post-performance-gate policy is missing")
    if not isinstance(lock.get("locked_execution"), dict) or config.get(
        "locked_execution"
    ) != lock.get("locked_execution"):
        raise NativeFinalError("HNSW kernel/method/fallback policy mismatch")
    if Path(str(hnsw_policy.get("output_root", ""))) != args.output:
        raise NativeFinalError("HNSW output differs from the locked final config")
    if Path(str(hnsw_policy.get("authorization_artifact", ""))) != (
        args.pre_gate_authorization
    ):
        raise NativeFinalError(
            "pre-HNSW authorization path differs from the locked final config"
        )
    if args.pre_gate_authorization.is_symlink() or not (
        args.pre_gate_authorization.is_file()
    ):
        raise NativeFinalError("pre-HNSW authorization is missing or symlinked")
    ef_values = [int(value) for value in hnsw_policy.get("ef_search_values", [])]
    if ef_values != [128, 512]:
        raise NativeFinalError("HNSW reference efSearch values must be 128 and 512")
    pre_gate_authorization = read_object(args.pre_gate_authorization)
    pre_gate = pre_gate_authorization.get("pre_hnsw_gate")
    receipt_bindings = pre_gate_authorization.get("session_receipts")
    if (
        pre_gate_authorization.get("status")
        != "AUTHORIZED_AFTER_FRESH_FINAL_PERFORMANCE_PASS"
        or pre_gate_authorization.get("performance_gate_status") != "PASSED"
        or pre_gate_authorization.get("performance_verdict")
        != "SUPPORTED_IN_TESTED_REGIME"
        or not isinstance(pre_gate, dict)
        or pre_gate.get("performance_gate_status") != "PASSED"
        or pre_gate.get("passed") is not True
        or pre_gate.get("hnsw_references_used_in_gate") is not False
        or pre_gate_authorization.get("pre_hnsw_gate_sha256")
        != object_sha256(pre_gate)
        or pre_gate_authorization.get("final_lock_sha256") != lock_sha
        or pre_gate_authorization.get("final_config_file_sha256") != config_sha
        or pre_gate_authorization.get("final_config_object_sha256")
        != object_sha256(config)
        or pre_gate_authorization.get("implementation_tree_sha256")
        != implementation_hash
        or pre_gate_authorization.get("hnsw_policy") != hnsw_policy
        or not isinstance(receipt_bindings, list)
        or len(receipt_bindings) != 3
    ):
        raise NativeFinalError("fresh performance-pass authorization is invalid")
    receipt_root = Path(str(config["output_root"])) / "session-receipts"
    bound_rows = 0
    for seed, binding in zip((0, 1, 2), receipt_bindings, strict=True):
        if not isinstance(binding, dict) or binding.get("session_seed") != seed:
            raise NativeFinalError("pre-HNSW receipt bindings are not seeds 0,1,2")
        bound_receipt_path = receipt_root / f"session-seed-{seed}.json"
        bound_receipt = read_object(bound_receipt_path)
        if (
            binding.get("receipt_sha256") != file_sha256(bound_receipt_path)
            or binding.get("session_id") != bound_receipt.get("session_id")
            or binding.get("run_id") != bound_receipt.get("run_id")
            or binding.get("completion_sha256")
            != bound_receipt.get("completion_sha256")
            or binding.get("run_manifest_sha256")
            != bound_receipt.get("run_manifest_sha256")
            or binding.get("build_manifest_sha256")
            != bound_receipt.get("build_manifest_sha256")
            or bound_receipt.get("final_lock_sha256") != lock_sha
            or bound_receipt.get("final_config_file_sha256") != config_sha
            or bound_receipt.get("source_final_config_object_sha256")
            != object_sha256(config)
        ):
            raise NativeFinalError("pre-HNSW receipt binding no longer verifies")
        bound_run_dir = Path(str(bound_receipt["run_dir"]))
        bound_evidence = load_native_evidence(bound_run_dir)
        expected_session = f"session-seed-{seed}:process-0000"
        if (
            len(bound_evidence.runs) != 1
            or bound_evidence.excluded_incomplete_runs
            or bound_evidence.runs[0].get("run_id") != binding.get("run_id")
            or {str(row.get("session_id")) for row in bound_evidence.rows}
            != {expected_session}
        ):
            raise NativeFinalError("pre-HNSW raw session binding no longer verifies")
        bound_rows += len(bound_evidence.rows)
    if bound_rows != int(pre_gate_authorization.get("raw_rows", -1)):
        raise NativeFinalError("pre-HNSW raw row count no longer verifies")
    receipt_path = Path(str(config["output_root"])) / "session-receipts" / "session-seed-0.json"
    session_receipt = read_object(receipt_path)
    if session_receipt.get("final_lock_sha256") != lock_sha:
        raise NativeFinalError("session-0 receipt is not bound to the final lock")
    session_run_dir = Path(str(session_receipt["run_dir"]))
    source_evidence = load_native_evidence(session_run_dir)
    if (
        len(source_evidence.runs) != 1
        or source_evidence.excluded_incomplete_runs
        or source_evidence.runs[0].get("run_id") != session_receipt.get("run_id")
        or source_evidence.runs[0].get("implementation_tree_sha256")
        != implementation_hash
        or session_receipt.get("completion_sha256")
        != file_sha256(session_run_dir / "COMPLETED.json")
        or session_receipt.get("run_manifest_sha256")
        != file_sha256(session_run_dir / "run_manifest.json")
    ):
        raise NativeFinalError("HNSW truth-source session receipt does not verify")

    if args.output.is_symlink() or (args.output.exists() and not args.output.is_dir()):
        raise NativeFinalError("HNSW output root is not a regular directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve(strict=True) != args.output.absolute():
        raise NativeFinalError("HNSW output root traverses a symlink")
    raw_dir = args.output / "raw"
    if raw_dir.is_symlink() or (raw_dir.exists() and not raw_dir.is_dir()):
        raise NativeFinalError("HNSW raw path is not a regular directory")
    raw_dir.mkdir(exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    checkpoint_path = args.output / "checkpoint.json"
    completion_path = args.output / "COMPLETED.json"
    for metadata_path in (manifest_path, checkpoint_path, completion_path):
        if metadata_path.is_symlink() or (
            metadata_path.exists() and not metadata_path.is_file()
        ):
            raise NativeFinalError("HNSW metadata path is not a regular file")
    identity = {
        "schema_version": 2,
        "study_id": "issue-3-native-recheck",
        "evidence_role": "final_approximate_reference",
        "final_lock_sha256": lock_sha,
        "final_config_file_sha256": config_sha,
        "final_config_object_sha256": object_sha256(config),
        "implementation_tree_sha256": implementation_hash,
        "truth_source_session_run_id": session_receipt["run_id"],
        "truth_source_completion_sha256": session_receipt["completion_sha256"],
        "pre_hnsw_authorization_sha256": file_sha256(
            args.pre_gate_authorization
        ),
        "ef_search_values": ef_values,
        "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
        "gate_eligible": False,
    }
    manifest_existed = manifest_path.exists()
    checkpoint_existed = checkpoint_path.exists()
    if manifest_existed != checkpoint_existed or (
        completion_path.exists() and not manifest_existed
    ):
        raise NativeFinalError("HNSW metadata file set is incomplete")
    if manifest_existed and not args.resume and not completion_path.exists():
        raise NativeFinalError("incomplete HNSW reference exists; use --resume")
    manifest = read_object(manifest_path) if manifest_existed else {
        **identity,
        "identity_sha256": object_sha256(identity),
        "status": "running",
        "started_at_utc": _utc_now(),
        "build_segments": [],
        "preserved_orphans": [],
        "failures": [],
    }
    if manifest.get("identity_sha256") != object_sha256(identity):
        raise NativeFinalError("HNSW reference identity changed on resume")
    checkpoint = read_object(checkpoint_path) if checkpoint_existed else {
        "schema_version": 1,
        "identity_sha256": object_sha256(identity),
        "completed_blocks": {},
    }
    if checkpoint.get("identity_sha256") != object_sha256(identity):
        raise NativeFinalError("HNSW checkpoint identity changed on resume")
    expected_blocks: dict[str, dict[str, object]] = {}
    block_size = int(config.get("query_block_size", 25))
    for specification in config["experiments"]:
        experiment_id = str(specification["id"])
        if (
            not experiment_id
            or experiment_id in {".", ".."}
            or "/" in experiment_id
            or "\\" in experiment_id
        ):
            raise NativeFinalError("HNSW experiment ID is not path-safe")
        query_count = int(specification["dataset"]["n_test"])
        for start in range(0, query_count, block_size):
            stop = min(query_count, start + block_size)
            key = f"{experiment_id}:{start:08d}:{stop:08d}"
            expected_blocks[key] = {
                "path": f"raw/{experiment_id}.q{start:08d}-{stop:08d}.jsonl",
                "experiment_id": experiment_id,
                "query_start": start,
                "query_stop": stop,
                "query_offset": int(
                    specification["dataset"]["test_query_offset"]
                ),
            }
    if not completion_path.exists():
        preserve_uncheckpointed_raw(
            root=args.output,
            manifest_path=manifest_path,
            manifest=manifest,
            checkpoint=checkpoint,
            expected_blocks=expected_blocks,
            identity=identity,
        )
    _verify_storage_inventory(
        root=args.output,
        checkpoint=checkpoint,
        manifest=manifest,
        expected_blocks=expected_blocks,
        identity=identity,
    )
    active_build_identities = verify_build_bindings(
        manifest=manifest,
        checkpoint=checkpoint,
        expected_blocks=expected_blocks,
        require_complete=completion_path.exists(),
    )
    verify_preserved_orphan_inventory(
        root=args.output,
        manifest=manifest,
        expected_blocks=expected_blocks,
        identity=identity,
    )
    if completion_path.exists():
        completion = read_object(completion_path)
        if (
            set(checkpoint["completed_blocks"]) != set(expected_blocks)
            or manifest.get("status") != "completed"
            or completion.get("status") != "completed"
            or completion.get("identity_sha256") != object_sha256(identity)
            or completion.get("manifest_sha256") != file_sha256(manifest_path)
            or completion.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            or int(completion.get("raw_shards", -1)) != len(expected_blocks)
            or completion.get("build_segments_sha256")
            != object_sha256(manifest["build_segments"])
            or completion.get("build_identity_by_experiment")
            != active_build_identities
            or completion.get("build_segment_count")
            != len(manifest["build_segments"])
            or completion.get("preserved_orphans_sha256")
            != object_sha256(manifest["preserved_orphans"])
            or completion.get("preserved_orphan_files")
            != len(manifest["preserved_orphans"])
        ):
            raise NativeFinalError("completed HNSW reference identity no longer verifies")
        print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    build_segments = manifest.get("build_segments")
    if not isinstance(build_segments, list):
        raise NativeFinalError("HNSW manifest has no build-segment registry")
    resumed_at = _utc_now()
    for prior_segment in build_segments:
        if isinstance(prior_segment, dict) and prior_segment.get("status") == "running":
            prior_segment["status"] = "interrupted_before_resume"
            prior_segment["interruption_observed_at_utc"] = resumed_at
    current_segment = {
        "segment_id": f"process-{len(build_segments):04d}",
        "ordinal": len(build_segments),
        "status": "running",
        "started_at_utc": resumed_at,
        "resume_requested": bool(args.resume),
        "builds": {},
    }
    build_segments.append(current_segment)
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(checkpoint_path, checkpoint)
    faiss.omp_set_num_threads(1)
    expected_block_keys = set(expected_blocks)

    try:
        for specification in config["experiments"]:
            experiment_id = str(specification["id"])
            experiment_block_keys = {
                key
                for key, block in expected_blocks.items()
                if block["experiment_id"] == experiment_id
            }
            if experiment_block_keys.issubset(checkpoint["completed_blocks"]):
                continue
            split = load_dataset(specification["dataset"])
            dataset_hash = split.sha256()
            truth = _truth_by_query(source_evidence.rows, experiment_id)
            expected_ids = {int(value) for value in split.test_query_ids.tolist()}
            if set(truth) != expected_ids:
                raise NativeFinalError(
                    f"HNSW truth/query coverage mismatch for {experiment_id}"
                )
            base_records = _records(split.base, split.base_ids)
            delta_records = _records(split.delta, split.delta_ids)
            full_records = base_records + delta_records
            m = int(specification.get("hnsw_m", 32))
            ef_construction = int(specification.get("hnsw_ef_construction", 200))
            base_ef = int(specification.get("base_hnsw_ef_search", 128))
            build_start = time.perf_counter_ns()
            base = BaseIndex(
                base_records,
                dimension=split.dimension,
                generation_id=f"native-final-hnsw-{experiment_id}",
                covered_commit_seq=0,
                m=m,
                ef_construction=ef_construction,
                ef_search=base_ef,
                threads=1,
            )
            delta_hnsw = PreparedFaissIndex(
                delta_records,
                dimension=split.dimension,
                kind="hnsw",
                m=m,
                ef_construction=ef_construction,
                ef_search=max(ef_values),
                threads=1,
            )
            full_hnsw = PreparedFaissIndex(
                full_records,
                dimension=split.dimension,
                kind="hnsw",
                m=m,
                ef_construction=ef_construction,
                ef_search=max(ef_values),
                threads=1,
            )
            build_identity = _build_identity(
                experiment_id=experiment_id,
                dataset_hash=dataset_hash,
                base=base,
                delta_hnsw=delta_hnsw,
                full_hnsw=full_hnsw,
                base_records=base_records,
                delta_records=delta_records,
                m=m,
                ef_construction=ef_construction,
                base_ef=base_ef,
                ef_values=ef_values,
            )
            build_identity_sha = object_sha256(build_identity)
            current_segment["builds"][experiment_id] = {
                "experiment_id": experiment_id,
                "dataset_hash": dataset_hash,
                "build_identity": build_identity,
                "build_identity_sha256": build_identity_sha,
                "build_wall_ns": time.perf_counter_ns() - build_start,
                "base": {
                    "vectors": len(base_records),
                    "m": m,
                    "ef_construction": ef_construction,
                    "ef_search": base_ef,
                },
                "delta_hnsw": asdict(delta_hnsw.build_stats),
                "full_hnsw": asdict(full_hnsw.build_stats),
            }
            atomic_write_json(manifest_path, manifest)
            prior_build_identity = active_build_identities.get(experiment_id)
            if (
                prior_build_identity is not None
                and prior_build_identity != build_identity_sha
            ):
                raise NativeFinalError(
                    f"{experiment_id}: rebuilt HNSW identity differs from "
                    "checkpointed blocks; preserve this run and start clean"
                )
            block_size = int(config.get("query_block_size", 25))
            for start in range(0, len(split.test_queries), block_size):
                stop = min(len(split.test_queries), start + block_size)
                key = f"{experiment_id}:{start:08d}:{stop:08d}"
                filename = f"{experiment_id}.q{start:08d}-{stop:08d}.jsonl"
                raw_path = raw_dir / filename
                prior = checkpoint["completed_blocks"].get(key)
                if prior is not None:
                    _verify_raw(
                        raw_path,
                        prior,
                        root=args.output,
                        identity=identity,
                        manifest=manifest,
                        experiment_id=experiment_id,
                        expected_block=expected_blocks[key],
                    )
                    continue
                rows: list[dict[str, object]] = []
                for position in range(start, stop):
                    query = split.test_queries[position]
                    query_id = int(split.test_query_ids[position])
                    exact_keys = truth[query_id]
                    candidates_start = time.perf_counter_ns()
                    candidates = base.prepare_candidates(
                        query,
                        snapshot_id=0,
                        k=int(specification["k"]),
                        candidate_count=int(specification["candidate_count"]),
                    )
                    base_ns = time.perf_counter_ns() - candidates_start
                    for ef_search in ef_values:
                        delta_hnsw.set_ef_search(ef_search)
                        start_ns = time.perf_counter_ns()
                        delta_result = delta_hnsw.search_and_merge_candidates(
                            query,
                            k=int(specification["k"]),
                            base_candidates=candidates,
                            ann_candidate_count=max(
                                int(specification["k"]),
                                int(specification["candidate_count"]),
                            ),
                            source=f"final_delta_hnsw_ef{ef_search}",
                        )
                        delta_micro_ns = time.perf_counter_ns() - start_ns
                        delta_keys = [list(hit.key) for hit in delta_result.hits]
                        rows.append(
                            {
                                **identity,
                                "build_segment_id": current_segment["segment_id"],
                                "build_identity_sha256": build_identity_sha,
                                "experiment_id": experiment_id,
                                "final_roles": specification["final_roles"],
                                "independence_label": specification[
                                    "final_independence_label"
                                ],
                                "dataset_hash": dataset_hash,
                                "query_id": query_id,
                                "query_position": position,
                                "method": "base_plus_delta_hnsw",
                                "ef_search": ef_search,
                                "base_ef_search": base_ef,
                                "micro_latency_ns": delta_micro_ns,
                                "api_wall_latency_ns": base_ns + delta_micro_ns,
                                "result_keys": delta_keys,
                                "full_visible_exact_keys": exact_keys,
                                "full_visible_exact_recall": _recall(delta_keys, exact_keys),
                                "approximate_not_certified": True,
                            }
                        )
                        full_hnsw.set_ef_search(ef_search)
                        start_ns = time.perf_counter_ns()
                        full_result = full_hnsw.search(
                            query, k=int(specification["k"])
                        )
                        full_ns = time.perf_counter_ns() - start_ns
                        full_keys = [list(hit.key) for hit in full_result.hits]
                        rows.append(
                            {
                                **identity,
                                "build_segment_id": current_segment["segment_id"],
                                "build_identity_sha256": build_identity_sha,
                                "experiment_id": experiment_id,
                                "final_roles": specification["final_roles"],
                                "independence_label": specification[
                                    "final_independence_label"
                                ],
                                "dataset_hash": dataset_hash,
                                "query_id": query_id,
                                "query_position": position,
                                "method": "full_population_hnsw",
                                "ef_search": ef_search,
                                "base_ef_search": None,
                                "micro_latency_ns": full_ns,
                                "api_wall_latency_ns": full_ns,
                                "result_keys": full_keys,
                                "full_visible_exact_keys": exact_keys,
                                "full_visible_exact_recall": _recall(full_keys, exact_keys),
                                "approximate_not_certified": True,
                            }
                        )
                atomic_write_jsonl(raw_path, rows)
                checkpoint["completed_blocks"][key] = {
                    "path": f"raw/{filename}",
                    "sha256": file_sha256(raw_path),
                    "rows": len(rows),
                    "experiment_id": experiment_id,
                    "query_start": start,
                    "query_stop": stop,
                    "build_segment_id": current_segment["segment_id"],
                    "build_identity_sha256": build_identity_sha,
                }
                atomic_write_json(checkpoint_path, checkpoint)
        if set(checkpoint["completed_blocks"]) != expected_block_keys:
            raise NativeFinalError("HNSW checkpoint does not exactly cover configured blocks")
        _verify_storage_inventory(
            root=args.output,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_blocks=expected_blocks,
            identity=identity,
        )
        active_build_identities = verify_build_bindings(
            manifest=manifest,
            checkpoint=checkpoint,
            expected_blocks=expected_blocks,
            require_complete=False,
        )
        verify_preserved_orphan_inventory(
            root=args.output,
            manifest=manifest,
            expected_blocks=expected_blocks,
            identity=identity,
        )
        current_segment["status"] = "completed"
        current_segment["completed_at_utc"] = _utc_now()
        active_build_identities = verify_build_bindings(
            manifest=manifest,
            checkpoint=checkpoint,
            expected_blocks=expected_blocks,
            require_complete=True,
        )
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["completed_blocks"] = len(checkpoint["completed_blocks"])
        atomic_write_json(manifest_path, manifest)
        completion = {
            **identity,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "identity_sha256": object_sha256(identity),
            "manifest_sha256": file_sha256(manifest_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "raw_shards": len(checkpoint["completed_blocks"]),
            "build_segments_sha256": object_sha256(manifest["build_segments"]),
            "build_identity_by_experiment": active_build_identities,
            "build_segment_count": len(manifest["build_segments"]),
            "preserved_orphans_sha256": object_sha256(
                manifest["preserved_orphans"]
            ),
            "preserved_orphan_files": len(manifest["preserved_orphans"]),
        }
        atomic_write_json(completion_path, completion)
        print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        if current_segment.get("status") in {"running", "completed"}:
            current_segment["status"] = "failed_incomplete"
            current_segment["failed_at_utc"] = _utc_now()
        manifest["status"] = "failed_incomplete"
        manifest["failed_at_utc"] = _utc_now()
        manifest["failures"].append(
            {
                "build_segment_id": current_segment["segment_id"],
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
