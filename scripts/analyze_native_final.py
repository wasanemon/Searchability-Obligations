#!/usr/bin/env python3
"""Verify and decide the locked three-session Issue #3 fresh final."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import statistics

import numpy as np

from searchability.artifacts import (
    atomic_write_json,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from searchability.native import native_build_info
from searchability.native_analysis import load_native_evidence
from searchability.native_final import (
    NativeFinalError,
    evaluate_final_rows,
    read_object,
    summarize_build_costs,
)
from searchability.native_hnsw import (
    raw_build_binding,
    verify_build_bindings,
    verify_preserved_orphan_inventory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _verify_current_runtime(lock: dict[str, object]) -> None:
    if implementation_tree_sha256(REPOSITORY_ROOT) != lock.get(
        "implementation_tree_sha256"
    ):
        raise NativeFinalError("current implementation differs from validation/final lock")
    if native_build_info().get("shared_object_sha256") != lock.get(
        "native_shared_object_sha256"
    ):
        raise NativeFinalError("loaded native binary differs from correctness/final lock")


def _verify_locked_source_files(
    lock: dict[str, object], config: dict[str, object]
) -> None:
    bindings = (
        ("validation_config_path", "validation_config_file_sha256"),
        ("correctness_path", "correctness_sha256"),
        ("holdout_manifest_path", "holdout_manifest_sha256"),
        ("final_policy_path", "final_policy_sha256"),
    )
    loaded: dict[str, dict[str, object]] = {}
    for path_field, hash_field in bindings:
        path = Path(str(lock.get(path_field, "")))
        if path.is_symlink() or not path.is_file() or file_sha256(path) != lock.get(
            hash_field
        ):
            raise NativeFinalError(f"current {path_field} differs from final lock")
        loaded[path_field] = read_object(path)
    if object_sha256(loaded["validation_config_path"]) != lock.get(
        "validation_config_object_sha256"
    ):
        raise NativeFinalError("current validation config object differs from final lock")
    if object_sha256(loaded["final_policy_path"]) != lock.get(
        "final_policy_object_sha256"
    ):
        raise NativeFinalError("current final policy object differs from final lock")
    correctness = loaded["correctness_path"]
    if (
        correctness.get("status") != "passed"
        or correctness.get("implementation_tree_sha256")
        != lock.get("implementation_tree_sha256")
        or correctness.get("native_shared_object_sha256")
        != lock.get("native_shared_object_sha256")
    ):
        raise NativeFinalError("current correctness evidence differs from final lock")
    if (
        config.get("required_correctness_evidence") != lock.get("correctness_path")
        or config.get("holdout_policy", {}).get("manifest")
        != lock.get("holdout_manifest_path")
    ):
        raise NativeFinalError("final config source paths differ from final lock")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NativeFinalError(f"non-object HNSW row {path}:{line_number}")
            rows.append(value)
    return rows


def _hnsw_summary(
    root: Path,
    lock_sha: str,
    config_sha: str,
    config: dict[str, object],
    pre_hnsw_authorization_sha: str,
) -> dict[str, object]:
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root.absolute()
    ):
        raise NativeFinalError("HNSW output root is missing or traverses a symlink")
    manifest_path = root / "run_manifest.json"
    checkpoint_path = root / "checkpoint.json"
    completion_path = root / "COMPLETED.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (manifest_path, checkpoint_path, completion_path)
    ):
        raise NativeFinalError("HNSW metadata is missing, non-regular, or symlinked")
    manifest = read_object(manifest_path)
    checkpoint = read_object(checkpoint_path)
    completion = read_object(completion_path)
    if (
        manifest.get("status") != "completed"
        or completion.get("status") != "completed"
        or completion.get("manifest_sha256") != file_sha256(manifest_path)
        or completion.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or completion.get("final_lock_sha256") != lock_sha
        or completion.get("final_config_file_sha256") != config_sha
        or manifest.get("pre_hnsw_authorization_sha256")
        != pre_hnsw_authorization_sha
        or completion.get("pre_hnsw_authorization_sha256")
        != pre_hnsw_authorization_sha
        or checkpoint.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
    ):
        raise NativeFinalError("HNSW completion/identity verification failed")
    identity_fields = (
        "schema_version",
        "study_id",
        "evidence_role",
        "final_lock_sha256",
        "final_config_file_sha256",
        "final_config_object_sha256",
        "implementation_tree_sha256",
        "truth_source_session_run_id",
        "truth_source_completion_sha256",
        "pre_hnsw_authorization_sha256",
        "ef_search_values",
        "methods",
        "gate_eligible",
    )
    identity = {field: manifest.get(field) for field in identity_fields}
    if (
        identity.get("schema_version") != 2
        or object_sha256(identity) != manifest.get("identity_sha256")
        or identity.get("final_config_object_sha256") != object_sha256(config)
        or identity.get("implementation_tree_sha256")
        != implementation_tree_sha256(REPOSITORY_ROOT)
        or identity.get("ef_search_values") != [128, 512]
        or identity.get("methods")
        != ["base_plus_delta_hnsw", "full_population_hnsw"]
        or identity.get("evidence_role") != "final_approximate_reference"
        or identity.get("gate_eligible") is not False
    ):
        raise NativeFinalError("HNSW manifest identity hash does not verify")
    completed = checkpoint.get("completed_blocks")
    if not isinstance(completed, dict) or len(completed) != int(
        completion.get("raw_shards", -1)
    ):
        raise NativeFinalError("HNSW checkpoint/completion block count differs")
    expected_blocks: dict[str, dict[str, object]] = {}
    block_size = int(config.get("query_block_size", 25))
    for experiment in config["experiments"]:
        experiment_id = str(experiment["id"])
        query_count = int(experiment["dataset"]["n_test"])
        for start in range(0, query_count, block_size):
            stop = min(query_count, start + block_size)
            key = f"{experiment_id}:{start:08d}:{stop:08d}"
            expected_blocks[key] = {
                "path": f"raw/{experiment_id}.q{start:08d}-{stop:08d}.jsonl",
                "experiment_id": experiment_id,
                "query_start": start,
                "query_stop": stop,
                "query_offset": int(experiment["dataset"]["test_query_offset"]),
            }
    if set(completed) != set(expected_blocks):
        raise NativeFinalError("HNSW checkpoint block coverage differs from config")
    build_identity_by_experiment = verify_build_bindings(
        manifest=manifest,
        checkpoint=checkpoint,
        expected_blocks=expected_blocks,
        require_complete=True,
    )
    verify_preserved_orphan_inventory(
        root=root,
        manifest=manifest,
        expected_blocks=expected_blocks,
        identity=identity,
    )
    build_segments = manifest.get("build_segments")
    preserved_orphans = manifest.get("preserved_orphans")
    if (
        not isinstance(build_segments, list)
        or not isinstance(preserved_orphans, list)
        or completion.get("build_segments_sha256")
        != object_sha256(build_segments)
        or completion.get("build_identity_by_experiment")
        != build_identity_by_experiment
        or completion.get("build_segment_count") != len(build_segments)
        or completion.get("preserved_orphans_sha256")
        != object_sha256(preserved_orphans)
        or completion.get("preserved_orphan_files") != len(preserved_orphans)
    ):
        raise NativeFinalError("HNSW completion does not bind build/orphan evidence")
    experiment_specs = {
        str(experiment["id"]): experiment for experiment in config["experiments"]
    }
    raw_dir = root / "raw"
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise NativeFinalError("HNSW raw directory is missing or is a symlink")
    actual_raw_paths: set[str] = set()
    for candidate in raw_dir.rglob("*"):
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise NativeFinalError("HNSW raw inventory contains a non-regular file")
        actual_raw_paths.add(candidate.relative_to(root).as_posix())
    rows: list[dict[str, object]] = []
    checkpoint_raw_paths: set[str] = set()
    for key, entry in completed.items():
        if not isinstance(entry, dict):
            raise NativeFinalError("malformed HNSW checkpoint entry")
        expected_block = expected_blocks[key]
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
            or relative in checkpoint_raw_paths
            or relative != expected_block["path"]
            or entry.get("experiment_id") != expected_block["experiment_id"]
            or entry.get("query_start") != expected_block["query_start"]
            or entry.get("query_stop") != expected_block["query_stop"]
        ):
            raise NativeFinalError(
                "HNSW checkpoint entry escapes/duplicates/differs from its locked block"
            )
        checkpoint_raw_paths.add(relative)
        path = root.joinpath(*parsed.parts)
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise NativeFinalError("HNSW checkpoint path escapes the run") from error
        if path.is_symlink() or not path.is_file():
            raise NativeFinalError("HNSW checkpoint path is not a regular file")
        if file_sha256(path) != entry.get("sha256"):
            raise NativeFinalError(f"HNSW raw checksum mismatch: {path}")
        block = _read_jsonl(path)
        if len(block) != int(entry.get("rows", -1)):
            raise NativeFinalError(f"HNSW raw row-count mismatch: {path}")
        if any(
            row.get("final_lock_sha256") != lock_sha
            or row.get("final_config_file_sha256") != config_sha
            or row.get("pre_hnsw_authorization_sha256")
            != pre_hnsw_authorization_sha
            or row.get("evidence_role") != "final_approximate_reference"
            or row.get("gate_eligible") is not False
            or row.get("approximate_not_certified") is not True
            for row in block
        ):
            raise NativeFinalError(f"HNSW raw identity/role mismatch: {path}")
        experiment_id = str(expected_block["experiment_id"])
        start = int(expected_block["query_start"])
        stop = int(expected_block["query_stop"])
        query_offset = int(
            experiment_specs[experiment_id]["dataset"]["test_query_offset"]
        )
        expected_rows = {
            (experiment_id, position, query_offset + position, method, ef_search)
            for position in range(start, stop)
            for method in ("base_plus_delta_hnsw", "full_population_hnsw")
            for ef_search in (128, 512)
        }
        observed_rows = {
            (
                str(row.get("experiment_id")),
                int(row.get("query_position", -1)),
                int(row.get("query_id", -1)),
                str(row.get("method")),
                int(row.get("ef_search", -1)),
            )
            for row in block
        }
        if len(block) != len(expected_rows) or observed_rows != expected_rows:
            raise NativeFinalError(f"HNSW raw block coverage mismatch: {path}")
        segment_id, build_sha = raw_build_binding(
            block,
            manifest=manifest,
            experiment_id=experiment_id,
        )
        if (
            entry.get("build_segment_id") != segment_id
            or entry.get("build_identity_sha256") != build_sha
        ):
            raise NativeFinalError(f"HNSW raw/checkpoint build identity mismatch: {path}")
        rows.extend(block)
    if checkpoint_raw_paths != actual_raw_paths:
        raise NativeFinalError("HNSW checkpoint/raw filesystem inventory differs")
    groups: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["experiment_id"]), str(row["method"]), int(row["ef_search"]))
            for row in rows
        }
    )
    expected_keys = {
        (str(experiment["id"]), method, ef_search)
        for experiment in config["experiments"]
        for method in ("base_plus_delta_hnsw", "full_population_hnsw")
        for ef_search in (128, 512)
    }
    if set(keys) != expected_keys:
        raise NativeFinalError("HNSW reference method/ef/experiment coverage differs")
    for experiment, method, ef_search in keys:
        selected = [
            row
            for row in rows
            if row["experiment_id"] == experiment
            and row["method"] == method
            and int(row["ef_search"]) == ef_search
        ]
        latencies = [float(row["api_wall_latency_ns"]) for row in selected]
        recalls = [float(row["full_visible_exact_recall"]) for row in selected]
        dataset = experiment_specs[experiment]["dataset"]
        expected_query_ids = set(
            range(
                int(dataset["test_query_offset"]),
                int(dataset["test_query_offset"]) + int(dataset["n_test"]),
            )
        )
        if (
            len(selected) != len(expected_query_ids)
            or {int(row["query_id"]) for row in selected} != expected_query_ids
        ):
            raise NativeFinalError("HNSW reference query coverage differs from final config")
        if any(not math.isfinite(value) or value <= 0 for value in latencies):
            raise NativeFinalError("HNSW reference has invalid latency")
        groups.append(
            {
                "experiment_id": experiment,
                "method": method,
                "ef_search": ef_search,
                "queries": len(selected),
                "api_wall_p50_ns": float(np.quantile(latencies, 0.50)),
                "api_wall_p95_ns": float(np.quantile(latencies, 0.95)),
                "exact_recall_mean": float(statistics.fmean(recalls)),
                "exact_recall_min": min(recalls),
            }
        )
    return {
        "status": "completed",
        "role": "approximate_reference_only_not_in_final_gate",
        "completion_sha256": file_sha256(completion_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "build_segments_sha256": object_sha256(build_segments),
        "build_identity_by_experiment": build_identity_by_experiment,
        "build_segment_count": len(build_segments),
        "preserved_orphans_sha256": object_sha256(preserved_orphans),
        "preserved_orphan_files": len(preserved_orphans),
        "raw_rows": len(rows),
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--hnsw-input", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--gate-output", required=True, type=Path)
    args = parser.parse_args()

    authorization = read_object(args.decision)
    if authorization.get("final_status") == "NOT_RUN_GATE_NOT_PASSED":
        print("validation NOT_PASSED: no final evidence to analyze")
        return 0
    if authorization.get("final_status") not in {
        "AUTHORIZED_NOT_YET_RUN",
        "COMPLETED_PASSED",
        "COMPLETED_NOT_PASSED",
    }:
        raise NativeFinalError("final analysis is not authorized")
    lock = read_object(args.lock)
    config = read_object(args.config)
    _verify_current_runtime(lock)
    _verify_locked_source_files(lock, config)
    lock_sha = file_sha256(args.lock)
    config_sha = file_sha256(args.config)
    if (
        authorization.get("validation_gate_status") != "PASSED"
        or lock.get("validation_gate_status") != "PASSED"
        or lock.get("status") != "LOCKED_FOR_FRESH_FINAL"
        or authorization.get("final_lock_sha256") != lock_sha
        or authorization.get("final_config_file_sha256") != config_sha
    ):
        raise NativeFinalError("final analyzer authorization identity mismatch")
    hnsw_policy = config.get("hnsw_references")
    if (
        not isinstance(hnsw_policy, dict)
        or hnsw_policy != lock.get("hnsw_references")
        or hnsw_policy.get("run_only_after_fresh_final_performance_pass") is not True
    ):
        raise NativeFinalError("final analyzer HNSW policy mismatch")
    if not isinstance(lock.get("locked_execution"), dict) or config.get(
        "locked_execution"
    ) != lock.get("locked_execution"):
        raise NativeFinalError("final analyzer kernel/method/fallback policy mismatch")
    pre_hnsw_authorization_path = Path(
        str(hnsw_policy.get("authorization_artifact", ""))
    )

    receipt_root = Path(str(config["output_root"])) / "session-receipts"
    rows: list[dict[str, object]] = []
    session_records: list[dict[str, object]] = []
    expected_experiments = {str(value["id"]): value for value in config["experiments"]}
    for session_seed in (0, 1, 2):
        receipt_path = receipt_root / f"session-seed-{session_seed}.json"
        receipt = read_object(receipt_path)
        run_dir = Path(str(receipt["run_dir"]))
        evidence = load_native_evidence(run_dir)
        if len(evidence.runs) != 1 or evidence.excluded_incomplete_runs:
            raise NativeFinalError("final session evidence verification failed")
        run = evidence.runs[0]
        effective = read_object(run_dir / "effective_config.json")
        run_manifest = read_object(run_dir / "run_manifest.json")
        build_manifest_path = run_dir / "build_manifest.json"
        build_costs = summarize_build_costs(read_object(build_manifest_path))
        expected_measurement_session = f"session-seed-{session_seed}:process-0000"
        process_segments = run_manifest.get("process_segments")
        if (
            receipt.get("session_seed") != session_seed
            or receipt.get("session_id") != f"session-seed-{session_seed}"
            or receipt.get("final_lock_sha256") != lock_sha
            or receipt.get("final_config_file_sha256") != config_sha
            or receipt.get("completion_sha256")
            != file_sha256(run_dir / "COMPLETED.json")
            or receipt.get("run_manifest_sha256")
            != file_sha256(run_dir / "run_manifest.json")
            or receipt.get("build_manifest_sha256")
            != file_sha256(build_manifest_path)
            or receipt.get("build_costs") != build_costs
            or run.get("implementation_tree_sha256")
            != lock.get("implementation_tree_sha256")
            or effective.get("source_config_object_hash") != object_sha256(config)
            or effective.get("session_seed") != session_seed
            or effective.get("method_order_seed") != session_seed
            or effective.get("final_lock_sha256") != lock_sha
            or not isinstance(process_segments, list)
            or len(process_segments) != 1
            or not isinstance(process_segments[0], dict)
            or process_segments[0].get("measurement_session_id")
            != expected_measurement_session
            or {str(row.get("session_id")) for row in evidence.rows}
            != {expected_measurement_session}
        ):
            raise NativeFinalError("final session receipt/effective identity mismatch")
        actual_experiments = {str(value["id"]): value for value in effective["experiments"]}
        if set(actual_experiments) != set(expected_experiments):
            raise NativeFinalError("final experiment set changed across sessions")
        if set(build_costs) != set(expected_experiments):
            raise NativeFinalError("final build evidence does not cover every experiment")
        for identifier, experiment in actual_experiments.items():
            if int(experiment.get("group_seed", -1)) != session_seed:
                raise NativeFinalError(
                    f"{identifier}: group-build seed differs from process seed"
                )
        rows.extend(dict(row) for row in evidence.rows)
        session_records.append(
            {
                **receipt,
                "receipt_sha256": file_sha256(receipt_path),
                "effective_config_sha256": file_sha256(
                    run_dir / "effective_config.json"
                ),
            }
        )

    summary, gate = evaluate_final_rows(
        rows,
        final_config=config,
        lock=lock,
        session_records=session_records,
        hnsw_reference={
            "status": "pre_gate_not_run",
            "role": "approximate_reference_only_not_in_final_gate",
        },
        require_post_gate_reference=False,
    )
    if gate["performance_gate_status"] == "PASSED":
        if pre_hnsw_authorization_path.is_symlink() or not (
            pre_hnsw_authorization_path.is_file()
        ):
            raise NativeFinalError("pre-HNSW authorization is missing or symlinked")
        pre_hnsw_authorization = read_object(pre_hnsw_authorization_path)
        expected_receipt_bindings = [
            {
                "session_seed": record["session_seed"],
                "session_id": record["session_id"],
                "run_id": record["run_id"],
                "receipt_sha256": record["receipt_sha256"],
                "completion_sha256": record["completion_sha256"],
                "run_manifest_sha256": record["run_manifest_sha256"],
                "build_manifest_sha256": record["build_manifest_sha256"],
            }
            for record in session_records
        ]
        if (
            pre_hnsw_authorization.get("status")
            != "AUTHORIZED_AFTER_FRESH_FINAL_PERFORMANCE_PASS"
            or pre_hnsw_authorization.get("performance_gate_status") != "PASSED"
            or pre_hnsw_authorization.get("pre_hnsw_gate") != gate
            or pre_hnsw_authorization.get("pre_hnsw_gate_sha256")
            != object_sha256(gate)
            or pre_hnsw_authorization.get("final_lock_sha256") != lock_sha
            or pre_hnsw_authorization.get("final_config_file_sha256") != config_sha
            or pre_hnsw_authorization.get("final_config_object_sha256")
            != object_sha256(config)
            or pre_hnsw_authorization.get("session_receipts")
            != expected_receipt_bindings
            or pre_hnsw_authorization.get("raw_rows") != len(rows)
            or pre_hnsw_authorization.get("hnsw_policy") != hnsw_policy
        ):
            raise NativeFinalError("pre-HNSW performance-pass authorization mismatch")
        pre_hnsw_authorization_sha = file_sha256(pre_hnsw_authorization_path)
        hnsw = _hnsw_summary(
            args.hnsw_input,
            lock_sha,
            config_sha,
            config,
            pre_hnsw_authorization_sha,
        )
        hnsw["pre_hnsw_authorization_sha256"] = pre_hnsw_authorization_sha
    else:
        if pre_hnsw_authorization_path.exists():
            raise NativeFinalError(
                "performance-negative final must not have pre-HNSW authorization"
            )
        if args.hnsw_input.exists():
            raise NativeFinalError(
                "performance-negative final must not have HNSW reference evidence"
            )
        hnsw = {
            "status": "not_run_performance_gate_not_passed",
            "role": "approximate_reference_only_not_in_final_gate",
            "reason": "fresh F/A/P performance gate did not pass",
        }
    gate["hnsw_reference_status"] = hnsw["status"]
    summary["hnsw_references"] = hnsw
    summary.update(
        {
            "final_lock_sha256": lock_sha,
            "final_config_file_sha256": config_sha,
            "final_config_object_sha256": object_sha256(config),
            "validation_gate_sha256": lock["gate_sha256"],
            "validation_summary_sha256": lock["validation_summary_sha256"],
        }
    )
    gate.update(
        {
            "final_lock_sha256": lock_sha,
            "final_config_file_sha256": config_sha,
            "session_receipt_sha256": [
                value["receipt_sha256"] for value in session_records
            ],
        }
    )
    if hnsw["status"] == "completed":
        gate["hnsw_completion_sha256"] = hnsw["completion_sha256"]
        gate["pre_hnsw_authorization_sha256"] = hnsw[
            "pre_hnsw_authorization_sha256"
        ]
    if args.summary_output.exists() and read_object(args.summary_output) != summary:
        raise NativeFinalError("refusing to rewrite a different final summary")
    if args.gate_output.exists() and read_object(args.gate_output) != gate:
        raise NativeFinalError("refusing to rewrite a different final gate")
    atomic_write_json(args.summary_output, summary)
    atomic_write_json(args.gate_output, gate)
    final_status = "COMPLETED_PASSED" if gate["passed"] else "COMPLETED_NOT_PASSED"
    completed_at = authorization.get("completed_at_utc") or datetime.now(
        timezone.utc
    ).isoformat()
    decision = {
        **authorization,
        "final_status": final_status,
        "performance_gate_status": gate["performance_gate_status"],
        "performance_verdict": gate["performance_verdict"],
        "verdict": gate["verdict"],
        "engineering_decision": gate["engineering_decision"],
        "large_final_started": True,
        "fresh_sift_holdout_loaded": True,
        "completed_at_utc": completed_at,
        "final_summary_sha256": file_sha256(args.summary_output),
        "final_gate_sha256": file_sha256(args.gate_output),
        "hnsw_reference_status": hnsw["status"],
    }
    if hnsw["status"] == "completed":
        decision["hnsw_completion_sha256"] = hnsw["completion_sha256"]
        decision["pre_hnsw_authorization_sha256"] = hnsw[
            "pre_hnsw_authorization_sha256"
        ]
    if str(authorization.get("final_status", "")).startswith("COMPLETED_"):
        if authorization != decision:
            raise NativeFinalError("completed final decision no longer reproduces")
    else:
        atomic_write_json(args.decision, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
