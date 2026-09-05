#!/usr/bin/env python3
"""Run the locked Issue #3 final in three independent process sessions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from searchability.artifacts import (
    atomic_write_json,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from searchability.native import native_build_info
from searchability.native_analysis import load_native_evidence
from searchability.native_benchmark import (
    apply_native_resource_limits,
    run_native_benchmark,
)
from searchability.native_final import (
    NativeFinalError,
    evaluate_final_rows,
    read_object,
    session_effective_config,
    summarize_build_costs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _verify_authorization(
    decision_path: Path, lock_path: Path, config_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]] | None:
    decision = read_object(decision_path)
    if decision.get("final_status") == "NOT_RUN_GATE_NOT_PASSED":
        if lock_path.exists() or config_path.exists():
            raise NativeFinalError("negative validation unexpectedly has a final lock/config")
        return None
    if decision.get("final_status") not in {
        "AUTHORIZED_NOT_YET_RUN",
        "COMPLETED_PASSED",
        "COMPLETED_NOT_PASSED",
    }:
        raise NativeFinalError("fresh final has not been authorized")
    lock = read_object(lock_path)
    config = read_object(config_path)
    if (
        decision.get("validation_gate_status") != "PASSED"
        or lock.get("validation_gate_status") != "PASSED"
        or lock.get("status") != "LOCKED_FOR_FRESH_FINAL"
    ):
        raise NativeFinalError("final lock is not active")
    if decision.get("final_lock_sha256") != file_sha256(lock_path):
        raise NativeFinalError("decision/final-lock identity mismatch")
    if decision.get("final_config_file_sha256") != file_sha256(config_path):
        raise NativeFinalError("decision/final-config identity mismatch")
    authorization = config.get("authorization")
    if not isinstance(authorization, dict) or authorization.get(
        "final_lock_sha256"
    ) != file_sha256(lock_path):
        raise NativeFinalError("locked final config has the wrong authorization")
    implementation_hash = implementation_tree_sha256(REPOSITORY_ROOT)
    if implementation_hash != lock.get("implementation_tree_sha256"):
        raise NativeFinalError("implementation changed after validation")
    if native_build_info().get("shared_object_sha256") != lock.get(
        "native_shared_object_sha256"
    ):
        raise NativeFinalError("native binary changed after validation")
    hnsw_policy = config.get("hnsw_references")
    if (
        not isinstance(hnsw_policy, dict)
        or hnsw_policy != lock.get("hnsw_references")
        or hnsw_policy.get("run_only_after_fresh_final_performance_pass") is not True
        or "run_only_after_validation_pass" in hnsw_policy
    ):
        raise NativeFinalError("locked HNSW post-performance-gate policy mismatch")
    if not isinstance(lock.get("locked_execution"), dict) or config.get(
        "locked_execution"
    ) != lock.get("locked_execution"):
        raise NativeFinalError("locked kernel/method/fallback policy mismatch")
    return decision, lock, config


def _completed_matches(output_root: Path, config_hash: str) -> list[Path]:
    matches: list[Path] = []
    for manifest_path in output_root.glob("*/run_manifest.json"):
        try:
            manifest = read_object(manifest_path)
        except NativeFinalError:
            continue
        if (
            manifest.get("config_hash") == config_hash
            and (manifest_path.parent / "COMPLETED.json").is_file()
        ):
            matches.append(manifest_path.parent)
    return sorted(matches)


def _receipt(
    *,
    run_dir: Path,
    session_seed: int,
    config_hash: str,
    lock_path: Path,
    config_path: Path,
) -> dict[str, object]:
    evidence = load_native_evidence(run_dir)
    if len(evidence.runs) != 1 or evidence.excluded_incomplete_runs:
        raise NativeFinalError("completed final session failed evidence verification")
    run = evidence.runs[0]
    if run.get("config_hash") != config_hash:
        raise NativeFinalError("final session effective-config hash mismatch")
    if run.get("session_id") != f"session-seed-{session_seed}":
        raise NativeFinalError("final session ID does not match its registered seed")
    manifest = read_object(run_dir / "run_manifest.json")
    expected_measurement_session = f"session-seed-{session_seed}:process-0000"
    segments = manifest.get("process_segments")
    if (
        not isinstance(segments, list)
        or len(segments) != 1
        or not isinstance(segments[0], dict)
        or segments[0].get("measurement_session_id") != expected_measurement_session
    ):
        raise NativeFinalError(
            "a final timing run must complete in one process segment; resumed segments "
            "are retained but cannot enter final evidence"
        )
    raw_sessions = {str(row.get("session_id")) for row in evidence.rows}
    if raw_sessions != {expected_measurement_session}:
        raise NativeFinalError("final raw rows do not have exactly one process session")
    final_config = read_object(config_path)
    build_manifest_path = run_dir / "build_manifest.json"
    build_costs = summarize_build_costs(read_object(build_manifest_path))
    expected_experiments = {
        str(experiment["id"]) for experiment in final_config.get("experiments", [])
    }
    if not expected_experiments or set(build_costs) != expected_experiments:
        raise NativeFinalError(
            "final build-cost evidence does not cover the locked experiment set"
        )
    return {
        "schema_version": 1,
        "study_id": "issue-3-native-recheck",
        "status": "completed",
        "session_seed": session_seed,
        "session_id": f"session-seed-{session_seed}",
        "run_id": run["run_id"],
        "run_dir": str(run_dir),
        "config_hash": config_hash,
        "source_final_config_object_sha256": object_sha256(read_object(config_path)),
        "final_config_file_sha256": file_sha256(config_path),
        "final_lock_sha256": file_sha256(lock_path),
        "implementation_tree_sha256": run["implementation_tree_sha256"],
        "completion_sha256": file_sha256(run_dir / "COMPLETED.json"),
        "run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
        # load_native_evidence above first verifies that this manifest is in
        # COMPLETED.json's immutable ancillary inventory.
        "build_manifest_sha256": file_sha256(build_manifest_path),
        "build_costs": build_costs,
    }


def _pre_hnsw_gate(
    *,
    lock: dict[str, object],
    config: dict[str, object],
    lock_path: Path,
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Re-verify all timing sessions and decide whether HNSW is authorized."""

    rows: list[dict[str, object]] = []
    session_records: list[dict[str, object]] = []
    for session_seed in (0, 1, 2):
        effective = apply_native_resource_limits(
            session_effective_config(
                config,
                session_seed=session_seed,
                final_config_sha256=file_sha256(config_path),
                final_lock_sha256=file_sha256(lock_path),
            )
        )
        config_hash = object_sha256(effective)
        output_root = Path(str(effective["output_root"]))
        matches = _completed_matches(output_root, config_hash)
        if len(matches) != 1:
            raise NativeFinalError(
                "pre-HNSW gate needs exactly one completed run per session seed"
            )
        receipt_path = (
            output_root / "session-receipts" / f"session-seed-{session_seed}.json"
        )
        expected_receipt = _receipt(
            run_dir=matches[0],
            session_seed=session_seed,
            config_hash=config_hash,
            lock_path=lock_path,
            config_path=config_path,
        )
        if read_object(receipt_path) != expected_receipt:
            raise NativeFinalError("pre-HNSW final session receipt no longer verifies")
        evidence = load_native_evidence(matches[0])
        rows.extend(dict(row) for row in evidence.rows)
        session_records.append(
            {
                **expected_receipt,
                "receipt_sha256": file_sha256(receipt_path),
            }
        )
    _, gate = evaluate_final_rows(
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
    receipt_bindings = [
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
    authorization = {
        "schema_version": 1,
        "study_id": "issue-3-native-recheck",
        "status": "AUTHORIZED_AFTER_FRESH_FINAL_PERFORMANCE_PASS",
        "performance_gate_status": gate["performance_gate_status"],
        "performance_verdict": gate["performance_verdict"],
        "pre_hnsw_gate_sha256": object_sha256(gate),
        "pre_hnsw_gate": gate,
        "final_lock_sha256": file_sha256(lock_path),
        "final_config_file_sha256": file_sha256(config_path),
        "final_config_object_sha256": object_sha256(config),
        "implementation_tree_sha256": lock["implementation_tree_sha256"],
        "session_receipts": receipt_bindings,
        "raw_rows": len(rows),
        "hnsw_policy": lock["hnsw_references"],
    }
    return gate, authorization


def _run_one(
    *, decision_path: Path, lock_path: Path, config_path: Path, session_seed: int
) -> int:
    verified = _verify_authorization(decision_path, lock_path, config_path)
    if verified is None:
        return 0
    _, _, config = verified
    effective = session_effective_config(
        config,
        session_seed=session_seed,
        final_config_sha256=file_sha256(config_path),
        final_lock_sha256=file_sha256(lock_path),
    )
    effective = apply_native_resource_limits(effective)
    config_hash = object_sha256(effective)
    output_root = Path(str(effective["output_root"]))
    receipt_path = output_root / "session-receipts" / f"session-seed-{session_seed}.json"
    matches = _completed_matches(output_root, config_hash)
    if len(matches) > 1:
        raise NativeFinalError("multiple completed runs exist for one final session")
    if receipt_path.exists():
        prior = read_object(receipt_path)
        if len(matches) != 1 or prior != _receipt(
            run_dir=matches[0],
            session_seed=session_seed,
            config_hash=config_hash,
            lock_path=lock_path,
            config_path=config_path,
        ):
            raise NativeFinalError("final session receipt no longer verifies")
        print(json.dumps(prior, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if matches:
        run_dir = matches[0]
    else:
        # Do not resume an interrupted timing process: doing so would mix
        # process segments in per-query repetitions. The incomplete run stays
        # preserved and a new immutable run ID is created.
        outcome = run_native_benchmark(effective, resume=False)
        if not outcome.completed:
            return 2
        run_dir = outcome.run_dir
    receipt = _receipt(
        run_dir=run_dir,
        session_seed=session_seed,
        config_hash=config_hash,
        lock_path=lock_path,
        config_path=config_path,
    )
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--session-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--hnsw-script", type=Path)
    parser.add_argument("--hnsw-output", type=Path)
    args = parser.parse_args()
    for variable in THREAD_VARIABLES:
        if os.environ.get(variable) != "1":
            raise NativeFinalError(f"{variable} must be exactly 1")
    verified = _verify_authorization(args.decision, args.lock, args.config)
    if verified is None:
        print("validation NOT_PASSED: fresh final and HNSW references were not started")
        return 0
    if args.session_seed is not None:
        return _run_one(
            decision_path=args.decision,
            lock_path=args.lock,
            config_path=args.config,
            session_seed=args.session_seed,
        )

    _, lock, config = verified
    for seed in (0, 1, 2):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--decision",
            str(args.decision),
            "--lock",
            str(args.lock),
            "--config",
            str(args.config),
            "--session-seed",
            str(seed),
        ]
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    pre_gate, pre_hnsw_authorization = _pre_hnsw_gate(
        lock=lock,
        config=config,
        lock_path=args.lock,
        config_path=args.config,
    )
    hnsw_policy = config["hnsw_references"]
    assert isinstance(hnsw_policy, dict)
    authorization_path = Path(str(hnsw_policy["authorization_artifact"]))
    if pre_gate.get("performance_gate_status") != "PASSED":
        if authorization_path.exists():
            raise NativeFinalError(
                "performance-negative final has stale pre-HNSW authorization"
            )
        if args.hnsw_output is not None and args.hnsw_output.exists():
            raise NativeFinalError(
                "performance-negative final has stale or prematurely run HNSW evidence"
            )
        print(
            "fresh performance gate NOT_PASSED: post-gate HNSW references were not started"
        )
        return 0
    if args.hnsw_script is None or args.hnsw_output is None:
        raise NativeFinalError(
            "performance-passed final requires the HNSW reference command"
        )
    if (
        authorization_path.exists()
        and read_object(authorization_path) != pre_hnsw_authorization
    ):
        raise NativeFinalError("pre-HNSW authorization no longer reproduces")
    atomic_write_json(authorization_path, pre_hnsw_authorization)
    subprocess.run(
        [
            sys.executable,
            str(args.hnsw_script),
            "--decision",
            str(args.decision),
            "--lock",
            str(args.lock),
            "--config",
            str(args.config),
            "--output",
            str(args.hnsw_output),
            "--pre-gate-authorization",
            str(authorization_path),
            "--resume",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
