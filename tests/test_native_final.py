from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from searchability.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from searchability.native import native_build_info
from searchability.native_final import (
    NativeFinalError,
    evaluate_final_rows,
    prepare_final_authorization,
    read_object,
    session_effective_config,
    summarize_build_costs,
)
from searchability.final_policy import issue_fixed_decision_values, validate_final_policy
from searchability.native_hnsw import (
    preserve_uncheckpointed_raw,
    verify_build_bindings,
    verify_preserved_orphan_inventory,
)
from scripts.run_native_hnsw_references import _verify_storage_inventory
from scripts.run_native_final import _receipt
import scripts.analyze_native_final as final_analyzer


REPOSITORY = Path(__file__).resolve().parents[1]


def _policy_identity() -> dict[str, str]:
    path = REPOSITORY / "configs" / "native_recheck_final_policy.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": "configs/native_recheck_final_policy.json",
        "file_sha256": file_sha256(path),
        "object_sha256": object_sha256(value),
    }


def _write(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _hnsw_build_identity(
    experiment_id: str = "sift", *, suffix: str = "a"
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_hash": suffix * 64,
        "faiss_version": "unit-test",
        "faiss_compile_options": "unit-test",
        "parameters": {
            "m": 32,
            "ef_construction": 200,
            "base_ef_search": 128,
            "ef_search_values": [128, 512],
            "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
            "threads": 1,
        },
        "populations": {
            "base_vectors": 10,
            "delta_vectors": 2,
            "full_vectors": 12,
        },
        "serialized_indexes": {
            name: {"sha256": suffix * 64, "serialized_bytes": 1}
            for name in ("base", "delta_hnsw", "full_hnsw")
        },
    }


def _hnsw_build_segment(
    ordinal: int,
    *,
    experiment_id: str = "sift",
    suffix: str = "a",
    status: str = "completed",
) -> dict[str, object]:
    identity = _hnsw_build_identity(experiment_id, suffix=suffix)
    return {
        "segment_id": f"process-{ordinal:04d}",
        "ordinal": ordinal,
        "status": status,
        "started_at_utc": "2026-09-06T00:00:00+00:00",
        "builds": {
            experiment_id: {
                "experiment_id": experiment_id,
                "dataset_hash": identity["dataset_hash"],
                "build_identity": identity,
                "build_identity_sha256": object_sha256(identity),
            }
        },
    }


def _validation_config() -> dict[str, object]:
    dataset = {
        "type": "texmex",
        "name": "sift",
        "root": "data/raw",
        "n_base": 100000,
        "n_delta": 10000,
        "n_validation": 200,
        "n_test": 200,
        "validation_query_offset": 0,
        "test_query_offset": 200,
    }
    return {
        "schema_version": 2,
        "study_id": "issue-3-native-recheck",
        "holdout_policy": {
            "manifest": "unused-in-unit-fixture",
            "pre_registered_final_policy": _policy_identity(),
        },
        "defaults": {
            "k": 10,
            "candidate_count": 64,
            "n_groups": 128,
            "center_training_size": 100000,
            "group_seed": 0,
            "kmeans_iterations": 20,
            "positive_beta_factors": [0.05],
            "positive_beta_min": 1e-12,
            "base_hnsw_ef_search": 128,
            "hnsw_m": 32,
            "hnsw_ef_construction": 200,
        },
        "experiments": [
            {
                "id": "sift-initial",
                "axis": "initial",
                "dataset": dataset,
                "raw_pending_count": 100,
                "positive_beta_factors": [0.05],
            },
            {
                "id": "gist-initial",
                "axis": "dataset_anchor",
                "dataset": {
                    **dataset,
                    "name": "gist",
                    "n_base": 50000,
                    "n_delta": 5000,
                },
                "center_training_size": 50000,
                "raw_pending_count": 50,
            },
            {
                "id": "sift-delta-100000",
                "axis": "delta_size",
                "dataset": {**dataset, "n_delta": 100000},
                "raw_pending_count": 1000,
            },
        ],
    }


def _ids_digest(start: int, stop: int) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in range(start, stop)).encode()
    ).hexdigest()


def _authorization_inputs(tmp_path: Path, *, status: str = "PASSED") -> dict[str, object]:
    config = _validation_config()
    config_path = tmp_path / "validation.json"
    _write(config_path, config)
    implementation = implementation_tree_sha256(REPOSITORY)
    binary = str(native_build_info()["shared_object_sha256"])
    selected = {
        "experiment_id": "sift-initial",
        "proposed_method": "native_P_beta0",
        "requested_beta_l2": 0.0,
        "passed": True,
    }
    gate = {
        "schema_version": 1,
        "study_id": "issue-3-native-recheck",
        "gate_status": status,
        "selected_candidate": selected if status == "PASSED" else None,
    }
    summary = {
        "study_id": "issue-3-native-recheck",
        "validation_gate": gate,
        "source_integrity": {
            "validation_gate_source_passed": True,
            "non_validation_runs": [],
            "implementation_tree_sha256": [implementation],
            "source_config_object_hash": [object_sha256(config)],
            "holdout_manifest_sha256": [],
        },
        "integrity_checks": {
            "native_backend_calls": {
                "passed": True,
                "native_binary_sha256": [binary],
            }
        },
    }
    correctness = {
        "status": "passed",
        "fixed_seed_cases": 10000,
        "implementation_tree_sha256": implementation,
        "native_shared_object_sha256": binary,
    }
    holdout = {
        "pre_registered_final_policy": {
            **_policy_identity(),
            "issue_fixed_decision_values": issue_fixed_decision_values(),
        },
        "selection_integrity": {
            "validation_source_config_object_sha256": object_sha256(config),
            "validation_source_config_file_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "final_policy_path": _policy_identity()["path"],
            "final_policy_file_sha256": _policy_identity()["file_sha256"],
            "final_policy_object_sha256": _policy_identity()["object_sha256"],
        },
        "sift": {
            "pre_registered_fresh_final_holdout": {
                "interval": [1200, 2200],
                "set_sha256": _ids_digest(1200, 2200),
                "disjoint_from_historically_used_union": True,
                "loaded_by_native_validation": False,
                "selected_before_native_validation_timing": True,
            }
        },
        "gist": {
            "independence_label": "reused_non_independent",
            "fresh_holdout_claim_permitted": False,
        },
    }
    policy_path = REPOSITORY / "configs" / "native_recheck_final_policy.json"
    paths = {
        "gate_path": tmp_path / "gate.json",
        "summary_path": tmp_path / "summary.json",
        "validation_config_path": config_path,
        "correctness_path": tmp_path / "correctness.json",
        "holdout_path": tmp_path / "holdout.json",
        "policy_path": policy_path,
    }
    for name, value in (
        ("gate_path", gate),
        ("correctness_path", correctness),
        ("holdout_path", holdout),
    ):
        _write(paths[name], value)
    summary["source_integrity"]["holdout_manifest_sha256"] = [
        file_sha256(paths["holdout_path"])
    ]
    _write(paths["summary_path"], summary)
    return {
        "gate": gate,
        "summary": summary,
        "validation_config": config,
        "correctness": correctness,
        "holdout": holdout,
        "policy": json.loads(policy_path.read_text(encoding="utf-8")),
        **paths,
        "decided_at_utc": "2026-09-06T00:00:00+00:00",
    }


def test_negative_validation_never_constructs_final_lock_or_config(tmp_path: Path) -> None:
    inputs = _authorization_inputs(tmp_path, status="NOT_PASSED")
    decision, lock, config = prepare_final_authorization(**inputs)

    assert decision["final_status"] == "NOT_RUN_GATE_NOT_PASSED"
    assert decision["fresh_sift_holdout_loaded"] is False
    assert lock is None
    assert config is None


def test_negative_validation_with_broken_source_is_not_a_performance_result(
    tmp_path: Path,
) -> None:
    inputs = _authorization_inputs(tmp_path, status="NOT_PASSED")
    inputs["summary"]["source_integrity"]["validation_gate_source_passed"] = False
    _write(inputs["summary_path"], inputs["summary"])

    with pytest.raises(NativeFinalError, match="source-integrity"):
        prepare_final_authorization(**inputs)


def test_passed_validation_locks_fresh_primary_anchors_and_three_seeds(
    tmp_path: Path,
) -> None:
    decision, lock, config = prepare_final_authorization(
        **_authorization_inputs(tmp_path)
    )
    assert lock is not None and config is not None
    assert decision["final_status"] == "AUTHORIZED_NOT_YET_RUN"
    assert lock["process_session_seeds"] == [0, 1, 2]
    experiments = {row["id"]: row for row in config["experiments"]}
    assert set(experiments) == {
        "sift-initial",
        "gist-initial",
        "sift-delta-100000",
    }
    assert set(experiments["sift-initial"]["final_roles"]) == {
        "locked_primary",
        "fixed_anchor_sift_initial",
    }
    assert experiments["sift-initial"]["dataset"]["test_query_offset"] == 1200
    assert experiments["sift-initial"]["dataset"]["n_test"] == 1000
    assert experiments["sift-delta-100000"]["dataset"]["test_query_offset"] == 1200
    assert experiments["gist-initial"]["dataset"]["test_query_offset"] == 0
    assert experiments["gist-initial"]["dataset"]["n_test"] == 1000
    assert experiments["gist-initial"]["dataset"]["n_validation"] == 0
    assert experiments["gist-initial"]["final_gate_eligible"] is False
    assert experiments["gist-initial"]["final_independence_label"] == (
        "reused_non_independent"
    )
    assert config["hnsw_references"] == lock["hnsw_references"]
    assert config["hnsw_references"][
        "run_only_after_fresh_final_performance_pass"
    ] is True
    assert "run_only_after_validation_pass" not in config["hnsw_references"]
    assert config["hnsw_references"]["authorization_artifact"].endswith(
        "pre_hnsw_authorization.json"
    )
    effective = session_effective_config(
        config,
        session_seed=2,
        final_config_sha256="1" * 64,
        final_lock_sha256="2" * 64,
    )
    assert effective["session_id"] == "session-seed-2"
    assert effective["method_order_seed"] == 2
    assert {row["group_seed"] for row in effective["experiments"]} == {2}


def _final_row(role: str, query: int, session: int) -> dict[str, object]:
    method = {"F": "native_F", "A": "faiss_A", "P": "native_P_beta0"}[role]
    latency = {"F": 120.0, "A": 110.0, "P": 100.0}[role]
    row: dict[str, object] = {
        "run_id": f"run-{session}",
        "session_id": f"session-seed-{session}:process-0000",
        "experiment_id": "sift-initial",
        "dataset_type": "texmex",
        "dataset_name": "texmex-sift",
        "n_delta": 10000,
        "test_query_count": 1000,
        "query_id": query,
        "query_position": query - 1200,
        "repetition": 0,
        "method": method,
        "method_role": role,
        "micro_latency_ns": latency,
        "composed_e2e_latency_ns": latency,
        "api_wall_latency_ns": latency,
        "delta_influence": query == 1200,
        "requested_beta_l2": 0.0,
        "baseline_quality_match": True,
        "baseline_validation_failure": False,
    }
    if role in {"F", "P"}:
        row.update(
            {
                "execution_backend": "pybind11_cpp17",
                "native_call_count": 2,
                "native_call_index": query,
                "native_binary_sha256": "a" * 64,
                "python_fallback_used": False,
                "contract_violation": False,
                "same_frozen_candidate_object_used": True,
                "api_candidate_hash_match": True,
                "candidate_set_id_or_hash": f"candidate-{query}",
                "receipt_candidate_set_id_or_hash": f"candidate-{query}",
                "receipt_requested_beta_l2": 0.0,
                "same_c_order_match": True,
            }
        )
    return row


def test_final_gate_uses_fresh_query_level_three_session_speedup_and_p95() -> None:
    rows = [
        _final_row(role, query, session)
        for query in range(1200, 2200)
        for session in range(3)
        for role in ("F", "A", "P")
    ]
    config = {
        "final_primary_experiment_id": "sift-initial",
        "final_primary_method": "native_P_beta0",
        "experiments": [{"id": "sift-initial"}],
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 13,
        "final_gate": {
            "required_primary_queries": 1000,
            "minimum_geometric_mean_speedup_vs_each_baseline": 1.10,
            "maximum_p95_latency_ratio_vs_each_baseline": 1.05,
            "require_A_quality_match": True,
        },
    }
    lock = {"selected_operating_point": {"requested_beta_l2": 0.0}}
    sessions = [
        {
            "session_seed": seed,
            "run_id": f"run-{seed}",
            "build_manifest_sha256": "b" * 64,
            "build_costs": {
                "sift-initial": {
                    "base_build_wall_ns": 1,
                    "base_input_vector_bytes": 1,
                    "group_build_wall_ns": 1,
                    "group_member_bytes": 1,
                    "group_metadata_bytes": 1,
                    "group_total_bytes": 2,
                    "native_packed_build_ns": 1,
                    "native_packed_python_owned_bytes": 1,
                    "native_packed_native_owned_bytes": 1,
                    "native_packed_bytes": 2,
                }
            },
        }
        for seed in range(3)
    ]
    summary, gate = evaluate_final_rows(
        rows,
        final_config=config,
        lock=lock,
        session_records=sessions,
        hnsw_reference={"status": "completed"},
    )

    assert gate["gate_status"] == "PASSED"
    assert gate["verdict"] == "SUPPORTED_IN_TESTED_REGIME"
    assert gate["engineering_decision"] == "GO"
    assert gate["comparison_outcomes"]["F"]["geometric_mean_speedup"] == pytest.approx(
        1.2
    )
    assert gate["comparison_outcomes"]["A"]["geometric_mean_speedup"] == pytest.approx(
        1.1
    )
    assert summary["gist_interpretation"]["gate_eligible"] is False

    rows[0]["session_id"] = "session-seed-0:process-0001"
    with pytest.raises(NativeFinalError, match="exactly one process-0000"):
        evaluate_final_rows(
            rows,
            final_config=config,
            lock=lock,
            session_records=sessions,
            hnsw_reference={"status": "completed"},
        )

    rows[0]["session_id"] = "session-seed-0:process-0000"
    for row in rows:
        if row["method_role"] == "A":
            row["api_wall_latency_ns"] = 105.0
            row["micro_latency_ns"] = 105.0
            row["composed_e2e_latency_ns"] = 105.0
    _, no_go_gate = evaluate_final_rows(
        rows,
        final_config=config,
        lock=lock,
        session_records=sessions,
        hnsw_reference={"status": "completed"},
    )
    assert no_go_gate["performance_verdict"] == "SUPPORTED_IN_TESTED_REGIME"
    assert no_go_gate["engineering_decision"] == "NO_GO"
    assert "speedup_or_p95_requirement_failed_vs_A" in no_go_gate[
        "engineering_reasons"
    ]

    for row in rows:
        if row["method_role"] == "P":
            row["api_wall_latency_ns"] = 130.0
            row["micro_latency_ns"] = 130.0
            row["composed_e2e_latency_ns"] = 130.0
    _, negative_gate = evaluate_final_rows(
        rows,
        final_config=config,
        lock=lock,
        session_records=sessions,
        hnsw_reference={
            "status": "not_run_performance_gate_not_passed",
            "role": "approximate_reference_only_not_in_final_gate",
        },
    )
    assert negative_gate["performance_gate_status"] == "NOT_PASSED"
    assert negative_gate["hnsw_reference_status"] == (
        "not_run_performance_gate_not_passed"
    )
    with pytest.raises(NativeFinalError, match="must record that HNSW was not run"):
        evaluate_final_rows(
            rows,
            final_config=config,
            lock=lock,
            session_records=sessions,
            hnsw_reference={"status": "completed"},
        )


def test_build_cost_summary_is_component_complete_and_fail_closed() -> None:
    manifest = {
        "sift-initial": {
            "base": {"build_wall_ns": 11},
            "groups": {
                "center_training_ns": 2,
                "assignment_ns": 3,
                "packing_and_radius_ns": 5,
                "member_bytes": 7,
                "metadata_bytes": 13,
            },
            "native_packed_view": {
                "build_ns": 17,
                "python_owned_bytes": 19,
                "native_owned_bytes": 23,
            },
            "memory": {
                "numpy_base_bytes": 29,
                "rss_change_bytes": 31,
                "peak_rss_bytes_after": 37,
            },
        }
    }
    costs = summarize_build_costs(manifest)["sift-initial"]
    assert costs["group_build_wall_ns"] == 10
    assert costs["group_total_bytes"] == 20
    assert costs["native_packed_bytes"] == 42
    assert costs["base_input_vector_bytes"] == 29

    manifest["sift-initial"]["groups"].pop("member_bytes")
    with pytest.raises(NativeFinalError, match="groups.member_bytes"):
        summarize_build_costs(manifest)


def test_hnsw_storage_inventory_rejects_extra_and_noncanonical_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hnsw"
    raw = root / "raw"
    raw.mkdir(parents=True)
    identity = {
        "study_id": "issue-3-native-recheck",
        "gate_eligible": False,
        "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
        "ef_search_values": [128, 512],
    }
    segment = _hnsw_build_segment(0)
    build_sha = segment["builds"]["sift"]["build_identity_sha256"]
    manifest = {"build_segments": [segment], "preserved_orphans": []}
    shard = raw / "sift.q00000000-00000001.jsonl"
    atomic_write_jsonl(
        shard,
        [
            {
                **identity,
                "build_segment_id": "process-0000",
                "build_identity_sha256": build_sha,
                "experiment_id": "sift",
                "query_position": 0,
                "query_id": 1200,
                "method": method,
                "ef_search": ef_search,
            }
            for method in identity["methods"]
            for ef_search in identity["ef_search_values"]
        ],
    )
    key = "sift:00000000:00000001"
    expected = {
        key: {
            "path": "raw/sift.q00000000-00000001.jsonl",
            "experiment_id": "sift",
            "query_start": 0,
            "query_stop": 1,
            "query_offset": 1200,
        }
    }
    checkpoint = {
        "completed_blocks": {
            key: {
                **expected[key],
                "sha256": file_sha256(shard),
                "rows": 4,
                "build_segment_id": "process-0000",
                "build_identity_sha256": build_sha,
            }
        }
    }
    _verify_storage_inventory(
        root=root,
        checkpoint=checkpoint,
        manifest=manifest,
        expected_blocks=expected,
        identity=identity,
    )

    atomic_write_jsonl(raw / "unlisted.jsonl", [{"unlisted": True}])
    with pytest.raises(NativeFinalError, match="filesystem inventory differs"):
        _verify_storage_inventory(
            root=root,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_blocks=expected,
            identity=identity,
        )
    (raw / "unlisted.jsonl").unlink()
    checkpoint["completed_blocks"][key]["path"] = "../outside.jsonl"
    with pytest.raises(NativeFinalError, match="differs from its locked block"):
        _verify_storage_inventory(
            root=root,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_blocks=expected,
            identity=identity,
        )

    target = tmp_path / "symlink-target"
    (target / "raw").mkdir(parents=True)
    alias = tmp_path / "symlink-root"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(NativeFinalError, match="traverses a symlink"):
        _verify_storage_inventory(
            root=alias,
            checkpoint={"completed_blocks": {}},
            manifest={"build_segments": [], "preserved_orphans": []},
            expected_blocks={},
            identity=identity,
        )


def test_hnsw_resume_preserves_uncheckpointed_raw_with_checksum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hnsw"
    raw = root / "raw"
    raw.mkdir(parents=True)
    identity = {
        "schema_version": 2,
        "study_id": "issue-3-native-recheck",
        "evidence_role": "final_approximate_reference",
        "ef_search_values": [128, 512],
        "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
        "gate_eligible": False,
    }
    segment = _hnsw_build_segment(0, status="failed_incomplete")
    build_sha = segment["builds"]["sift"]["build_identity_sha256"]
    manifest = {
        **identity,
        "build_segments": [segment],
        "preserved_orphans": [],
    }
    manifest_path = root / "run_manifest.json"
    _write(manifest_path, manifest)
    expected = {
        "sift:00000000:00000001": {
            "path": "raw/sift.q00000000-00000001.jsonl",
            "experiment_id": "sift",
            "query_start": 0,
            "query_stop": 1,
            "query_offset": 1200,
        }
    }
    orphan = raw / "sift.q00000000-00000001.jsonl"
    atomic_write_jsonl(
        orphan,
        [
            {
                **identity,
                "build_segment_id": "process-0000",
                "build_identity_sha256": build_sha,
                "experiment_id": "sift",
                "query_position": 0,
                "query_id": 1200,
                "method": method,
                "ef_search": ef_search,
            }
            for method in identity["methods"]
            for ef_search in identity["ef_search_values"]
        ],
    )
    orphan_sha = file_sha256(orphan)
    checkpoint = {"completed_blocks": {}}

    preserve_uncheckpointed_raw(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        checkpoint=checkpoint,
        expected_blocks=expected,
        identity=identity,
    )

    assert not orphan.exists()
    assert len(manifest["preserved_orphans"]) == 1
    record = manifest["preserved_orphans"][0]
    preserved = root / record["preserved_path"]
    assert preserved.is_file()
    assert record["sha256"] == orphan_sha == file_sha256(preserved)
    assert read_object(manifest_path)["preserved_orphans"] == [record]
    # A crash after the move but around manifest publication is recoverable;
    # repeated reconciliation must be idempotent and retain the same evidence.
    preserve_uncheckpointed_raw(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        checkpoint=checkpoint,
        expected_blocks=expected,
        identity=identity,
    )
    assert manifest["preserved_orphans"] == [record]
    verify_preserved_orphan_inventory(
        root=root,
        manifest=manifest,
        expected_blocks=expected,
        identity=identity,
    )


def test_hnsw_checkpoint_rejects_mixed_rebuilt_index_identities() -> None:
    first = _hnsw_build_segment(0, suffix="a")
    changed = _hnsw_build_segment(1, suffix="b")
    expected = {
        "sift:00000000:00000001": {
            "experiment_id": "sift",
            "path": "raw/first.jsonl",
        },
        "sift:00000001:00000002": {
            "experiment_id": "sift",
            "path": "raw/second.jsonl",
        },
    }
    checkpoint = {
        "completed_blocks": {
            "sift:00000000:00000001": {
                "build_segment_id": "process-0000",
                "build_identity_sha256": first["builds"]["sift"][
                    "build_identity_sha256"
                ],
            },
            "sift:00000001:00000002": {
                "build_segment_id": "process-0001",
                "build_identity_sha256": changed["builds"]["sift"][
                    "build_identity_sha256"
                ],
            },
        }
    }
    with pytest.raises(NativeFinalError, match="mixes different HNSW build"):
        verify_build_bindings(
            manifest={"build_segments": [first, changed]},
            checkpoint=checkpoint,
            expected_blocks=expected,
            require_complete=True,
        )

    deterministic_rebuild = _hnsw_build_segment(1, suffix="a")
    checkpoint["completed_blocks"]["sift:00000001:00000002"][
        "build_identity_sha256"
    ] = deterministic_rebuild["builds"]["sift"]["build_identity_sha256"]
    assert verify_build_bindings(
        manifest={"build_segments": [first, deterministic_rebuild]},
        checkpoint=checkpoint,
        expected_blocks=expected,
        require_complete=True,
    ) == {
        "sift": first["builds"]["sift"]["build_identity_sha256"]
    }


def test_hnsw_analysis_requires_completion_bound_build_segments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hnsw"
    raw = root / "raw"
    raw.mkdir(parents=True)
    lock_sha = "1" * 64
    config_sha = "2" * 64
    pre_gate_sha = "3" * 64
    config = {
        "query_block_size": 1,
        "experiments": [
            {"id": "sift", "dataset": {"test_query_offset": 1200, "n_test": 1}}
        ],
    }
    identity = {
        "schema_version": 2,
        "study_id": "issue-3-native-recheck",
        "evidence_role": "final_approximate_reference",
        "final_lock_sha256": lock_sha,
        "final_config_file_sha256": config_sha,
        "final_config_object_sha256": object_sha256(config),
        "implementation_tree_sha256": implementation_tree_sha256(REPOSITORY),
        "truth_source_session_run_id": "run-0",
        "truth_source_completion_sha256": "6" * 64,
        "pre_hnsw_authorization_sha256": pre_gate_sha,
        "ef_search_values": [128, 512],
        "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
        "gate_eligible": False,
    }
    segment = _hnsw_build_segment(0)
    build_sha = segment["builds"]["sift"]["build_identity_sha256"]
    shard = raw / "sift.q00000000-00000001.jsonl"
    atomic_write_jsonl(
        shard,
        [
            {
                **identity,
                "build_segment_id": "process-0000",
                "build_identity_sha256": build_sha,
                "experiment_id": "sift",
                "query_position": 0,
                "query_id": 1200,
                "method": method,
                "ef_search": ef_search,
                "api_wall_latency_ns": 10,
                "full_visible_exact_recall": 1.0,
                "approximate_not_certified": True,
            }
            for method in identity["methods"]
            for ef_search in identity["ef_search_values"]
        ],
    )
    key = "sift:00000000:00000001"
    manifest = {
        **identity,
        "identity_sha256": object_sha256(identity),
        "status": "completed",
        "build_segments": [segment],
        "preserved_orphans": [],
    }
    checkpoint = {
        "schema_version": 1,
        "identity_sha256": object_sha256(identity),
        "completed_blocks": {
            key: {
                "path": "raw/sift.q00000000-00000001.jsonl",
                "sha256": file_sha256(shard),
                "rows": 4,
                "experiment_id": "sift",
                "query_start": 0,
                "query_stop": 1,
                "build_segment_id": "process-0000",
                "build_identity_sha256": build_sha,
            }
        },
    }
    manifest_path = root / "run_manifest.json"
    checkpoint_path = root / "checkpoint.json"
    _write(manifest_path, manifest)
    _write(checkpoint_path, checkpoint)
    completion = {
        **identity,
        "status": "completed",
        "identity_sha256": object_sha256(identity),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "raw_shards": 1,
        "build_segments_sha256": object_sha256(manifest["build_segments"]),
        "build_identity_by_experiment": {"sift": build_sha},
        "build_segment_count": 1,
        "preserved_orphans_sha256": object_sha256([]),
        "preserved_orphan_files": 0,
    }
    completion_path = root / "COMPLETED.json"
    _write(completion_path, completion)
    result = final_analyzer._hnsw_summary(
        root, lock_sha, config_sha, config, pre_gate_sha
    )
    assert result["build_identity_by_experiment"] == {"sift": build_sha}
    assert result["build_segment_count"] == 1

    completion["build_segments_sha256"] = "0" * 64
    _write(completion_path, completion)
    with pytest.raises(NativeFinalError, match="does not bind build/orphan"):
        final_analyzer._hnsw_summary(
            root, lock_sha, config_sha, config, pre_gate_sha
        )


def test_direct_hnsw_invocation_rejects_negative_pre_gate_authorization(
    tmp_path: Path,
) -> None:
    output = tmp_path / "hnsw"
    pre_gate_path = tmp_path / "pre-hnsw.json"
    hnsw_policy = {
        "run_only_after_fresh_final_performance_pass": True,
        "ef_search_values": [128, 512],
        "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
        "role": "approximate_reference_only_not_in_final_gate",
        "output_root": str(output),
        "authorization_artifact": str(pre_gate_path),
    }
    locked_execution = json.loads(
        (REPOSITORY / "configs" / "native_recheck_final_policy.json").read_text(
            encoding="utf-8"
        )
    )["locked_execution"]
    lock = {
        "validation_gate_status": "PASSED",
        "status": "LOCKED_FOR_FRESH_FINAL",
        "implementation_tree_sha256": implementation_tree_sha256(REPOSITORY),
        "hnsw_references": hnsw_policy,
        "locked_execution": locked_execution,
    }
    config = {
        "hnsw_references": hnsw_policy,
        "locked_execution": locked_execution,
    }
    lock_path = tmp_path / "lock.json"
    config_path = tmp_path / "config.json"
    decision_path = tmp_path / "decision.json"
    _write(lock_path, lock)
    _write(config_path, config)
    _write(
        decision_path,
        {
            "validation_gate_status": "PASSED",
            "final_status": "AUTHORIZED_NOT_YET_RUN",
            "final_lock_sha256": file_sha256(lock_path),
            "final_config_file_sha256": file_sha256(config_path),
        },
    )
    _write(
        pre_gate_path,
        {
            "status": "NOT_AUTHORIZED_PERFORMANCE_GATE_NOT_PASSED",
            "performance_gate_status": "NOT_PASSED",
        },
    )
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "run_native_hnsw_references.py"),
            "--decision",
            str(decision_path),
            "--lock",
            str(lock_path),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--pre-gate-authorization",
            str(pre_gate_path),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "fresh performance-pass authorization is invalid" in completed.stderr
    assert not output.exists()


def test_final_session_receipt_hash_binds_all_experiment_build_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "final-config.json"
    lock_path = tmp_path / "lock.json"
    config = {
        "experiments": [{"id": "sift-initial"}, {"id": "gist-initial"}]
    }
    _write(config_path, config)
    _write(lock_path, {"status": "LOCKED_FOR_FRESH_FINAL"})
    _write(
        run_dir / "run_manifest.json",
        {
            "process_segments": [
                {"measurement_session_id": "session-seed-0:process-0000"}
            ]
        },
    )
    _write(run_dir / "COMPLETED.json", {"status": "completed"})

    def build(experiment_id: str) -> dict[str, object]:
        return {
            "experiment_id": experiment_id,
            "base": {"build_wall_ns": 11},
            "groups": {
                "center_training_ns": 2,
                "assignment_ns": 3,
                "packing_and_radius_ns": 5,
                "member_bytes": 7,
                "metadata_bytes": 13,
            },
            "native_packed_view": {
                "build_ns": 17,
                "python_owned_bytes": 19,
                "native_owned_bytes": 23,
            },
            "memory": {
                "numpy_base_bytes": 29,
                "rss_change_bytes": 31,
                "peak_rss_bytes_after": 37,
            },
        }

    _write(
        run_dir / "build_manifest.json",
        {identifier: build(identifier) for identifier in ("sift-initial", "gist-initial")},
    )
    fake_evidence = SimpleNamespace(
        runs=(
            {
                "config_hash": "c" * 64,
                "session_id": "session-seed-0",
                "run_id": "run-0",
                "implementation_tree_sha256": "i" * 64,
            },
        ),
        excluded_incomplete_runs=(),
        rows=({"session_id": "session-seed-0:process-0000"},),
    )
    monkeypatch.setattr(
        "scripts.run_native_final.load_native_evidence", lambda _: fake_evidence
    )
    receipt = _receipt(
        run_dir=run_dir,
        session_seed=0,
        config_hash="c" * 64,
        lock_path=lock_path,
        config_path=config_path,
    )
    assert receipt["build_manifest_sha256"] == file_sha256(
        run_dir / "build_manifest.json"
    )
    assert set(receipt["build_costs"]) == {"sift-initial", "gist-initial"}
    assert receipt["build_costs"]["gist-initial"]["native_packed_bytes"] == 42


def test_authorization_fails_if_implementation_identity_changed(tmp_path: Path) -> None:
    inputs = _authorization_inputs(tmp_path)
    inputs["summary"]["source_integrity"]["implementation_tree_sha256"] = ["0" * 64]
    _write(inputs["summary_path"], inputs["summary"])
    inputs["correctness"]["implementation_tree_sha256"] = "0" * 64
    _write(inputs["correctness_path"], inputs["correctness"])

    with pytest.raises(NativeFinalError, match="implementation changed after validation"):
        prepare_final_authorization(**inputs)


def test_final_policy_schema_rejects_threshold_quality_anchor_and_bool_changes() -> None:
    path = REPOSITORY / "configs" / "native_recheck_final_policy.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    validate_final_policy(original)

    changed = copy.deepcopy(original)
    changed["final_gate"]["minimum_geometric_mean_speedup_vs_each_baseline"] = 1.09
    with pytest.raises(NativeFinalError, match="decision thresholds"):
        try:
            validate_final_policy(changed)
        except ValueError as error:
            raise NativeFinalError(str(error)) from error

    changed = copy.deepcopy(original)
    changed["final_gate"]["require_A_quality_match"] = False
    with pytest.raises(ValueError, match="decision thresholds"):
        validate_final_policy(changed)

    changed = copy.deepcopy(original)
    changed["fixed_anchors"][0]["query_interval"] = [1201, 2201]
    with pytest.raises(ValueError, match="anchors"):
        validate_final_policy(changed)

    changed = copy.deepcopy(original)
    changed["final_gate"]["required_primary_queries"] = True
    with pytest.raises(ValueError, match="decision thresholds"):
        validate_final_policy(changed)

    changed = copy.deepcopy(original)
    changed["repetitions_per_session"] = True
    with pytest.raises(ValueError, match="repetitions"):
        validate_final_policy(changed)


def test_authorization_rejects_final_policy_hash_tampering(tmp_path: Path) -> None:
    inputs = _authorization_inputs(tmp_path)
    inputs["holdout"]["pre_registered_final_policy"]["file_sha256"] = "0" * 64
    _write(inputs["holdout_path"], inputs["holdout"])
    inputs["summary"]["source_integrity"]["holdout_manifest_sha256"] = [
        file_sha256(inputs["holdout_path"])
    ]
    _write(inputs["summary_path"], inputs["summary"])

    with pytest.raises(NativeFinalError, match="pre-validation registration"):
        prepare_final_authorization(**inputs)


def test_final_analyzer_rejects_changed_current_code_or_loaded_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = {
        "implementation_tree_sha256": implementation_tree_sha256(REPOSITORY),
        "native_shared_object_sha256": native_build_info()["shared_object_sha256"],
    }
    final_analyzer._verify_current_runtime(lock)

    monkeypatch.setattr(
        final_analyzer, "implementation_tree_sha256", lambda _: "0" * 64
    )
    with pytest.raises(NativeFinalError, match="current implementation"):
        final_analyzer._verify_current_runtime(lock)
    monkeypatch.setattr(
        final_analyzer,
        "implementation_tree_sha256",
        lambda _: lock["implementation_tree_sha256"],
    )
    monkeypatch.setattr(
        final_analyzer,
        "native_build_info",
        lambda: {"shared_object_sha256": "0" * 64},
    )
    with pytest.raises(NativeFinalError, match="loaded native binary"):
        final_analyzer._verify_current_runtime(lock)
