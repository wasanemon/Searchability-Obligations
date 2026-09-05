from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from searchability.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    file_sha256,
    native_test_tree_sha256,
)
from searchability.benchmark import BenchmarkConfigurationError, CompletedRunError
from searchability.native_analysis import load_native_evidence
import searchability.native_benchmark as native_benchmark_module
from searchability.native_benchmark import (
    apply_native_resource_limits,
    run_native_benchmark,
)
from searchability.models import SearchHit, VectorRecord


def _config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "study_id": "issue-3-native-recheck",
        "run_name": "native-runner-test",
        "phase": "smoke",
        "evidence_role": "smoke",
        "output_root": str(root),
        "threads": 1,
        "warmup_queries": 1,
        "repetitions": 1,
        "query_block_size": 1,
        "method_order_seed": 17,
        "independent_oracle_queries": 1,
        "lb_audit_queries": 1,
        "base_cache_ablation_queries": 1,
        "defaults": {
            "k": 3,
            "candidate_count": 8,
            "n_groups": 3,
            "raw_pending_count": 1,
            "center_training_size": 32,
            "kmeans_iterations": 2,
            "positive_beta_factors": [0.05],
            "positive_beta_min": 1e-12,
            "base_hnsw_ef_search": 8,
            "hnsw_m": 4,
            "hnsw_ef_construction": 12,
        },
        "experiments": [
            {
                "id": "tiny",
                "axis": "test",
                "kernel_ablations": True,
                "dataset": {
                    "type": "synthetic",
                    "kind": "delta_near_queries",
                    "n_base": 32,
                    "n_delta": 8,
                    "n_validation": 2,
                    "n_test": 2,
                    "dimension": 5,
                    "seed": 19,
                },
            }
        ],
    }


def test_native_runner_writes_checksum_bound_rows_and_immutable_completion(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs")
    outcome = run_native_benchmark(config)
    assert outcome.completed
    evidence = load_native_evidence(outcome.run_dir)
    assert len(evidence.runs) == 1
    assert len(evidence.rows) == 2 * (1 + 9)
    native_rows = [row for row in evidence.rows if row["method_role"] in {"F", "N", "P"}]
    assert native_rows
    assert all(row["native_call_count"] == 2 for row in native_rows)
    assert all(row["native_binary_sha256"] for row in native_rows)
    assert all(row["api_candidate_hash_match"] for row in native_rows)
    assert all(row["contract_valid"] for row in native_rows)
    assert {row["method_role"] for row in evidence.rows} >= {"O", "F", "A", "N", "P"}
    assert any(row["method_role"] == "A-reference" for row in evidence.rows)
    adaptive_a_rows = [row for row in evidence.rows if row["method_role"] == "A"]
    assert adaptive_a_rows
    assert all(
        isinstance(row["exact_boundary_rechecks"], int)
        and row["exact_boundary_rechecks"] >= 0
        for row in adaptive_a_rows
    )

    with pytest.raises(FileExistsError, match="immutable"):
        run_native_benchmark(config, run_id=outcome.run_id)


def test_native_runner_controlled_stop_resumes_same_identity(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    stopped = run_native_benchmark(config, stop_after_blocks=1)
    assert not stopped.completed
    manifest = json.loads((stopped.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "incomplete_controlled_stop"
    assert not (stopped.run_dir / "COMPLETED.json").exists()

    resumed = run_native_benchmark(
        config, resume=True, run_id=stopped.run_id
    )
    assert resumed.completed
    evidence = load_native_evidence(resumed.run_dir)
    assert len(evidence.rows) == 2 * (1 + 9)
    assert {row["session_id"] for row in evidence.rows} == {
        "session-0:process-0000",
        "session-0:process-0001",
    }


def test_native_resource_limits_materialize_defaults_and_relabel() -> None:
    source = _config(Path("unused"))
    effective = apply_native_resource_limits(
        source,
        max_base=16,
        max_delta=0,
        max_validation_queries=1,
        max_test_queries=1,
        max_repetitions=1,
    )
    experiment = effective["experiments"][0]
    assert experiment["dataset"]["n_base"] == 16
    assert experiment["dataset"]["n_delta"] == 0
    assert experiment["raw_pending_count"] == 0
    assert experiment["beta_factors"] == [0.0, 0.05]
    assert effective["phase"] == "smoke"
    assert effective["evidence_role"] == "calibration"


def test_formal_validation_requires_exactly_one_process_session(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    config.update(
        {
            "phase": "validation",
            "evidence_role": "validation",
            "process_sessions": 2,
        }
    )
    with pytest.raises(
        BenchmarkConfigurationError, match="process_sessions == 1"
    ):
        run_native_benchmark(config)


def test_validation_resume_rejects_duplicate_completed_pool(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    config = {
        "output_root": str(root),
        "run_name": "formal-validation",
        "phase": "validation",
    }
    config_hash = "c" * 64
    implementation_hash = "i" * 64
    run_dir = root / "completed-run"
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
        },
    )
    atomic_write_json(run_dir / "COMPLETED.json", {"status": "completed"})

    with pytest.raises(CompletedRunError, match="duplicate evidence pool"):
        native_benchmark_module._select_run_dir(
            config,
            config_hash,
            implementation_hash,
            resume=True,
            run_id=None,
        )


def test_validation_resume_rejects_failed_existing_lb_audit(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    stopped = run_native_benchmark(config, stop_after_blocks=1)
    audit_path = stopped.run_dir / "audits" / "tiny.native-lb.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["passed"] = False
    audit["failures"] = [{"tampered": True}]
    atomic_write_json(audit_path, audit)

    with pytest.raises(RuntimeError, match="LB audit identity/query-count/pass"):
        run_native_benchmark(config, resume=True, run_id=stopped.run_id)


def test_resume_preserves_uncheckpointed_orphan_without_disqualifying_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs")
    stopped = run_native_benchmark(config, stop_after_blocks=1)
    orphan = stopped.run_dir / "raw" / "tiny.q00000001-00000002.jsonl"
    atomic_write_bytes(orphan, b'{"partial":true}\n')

    resumed = run_native_benchmark(config, resume=True, run_id=stopped.run_id)
    assert resumed.completed
    manifest = json.loads((resumed.run_dir / "run_manifest.json").read_text())
    assert {
        failure["kind"] for failure in manifest["failures"]
    } == {"uncheckpointed_raw_preserved_on_resume"}


def test_auxiliary_test_change_invalidates_correctness_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    helper = repository / "tests" / "helpers" / "native_fixture.py"
    test_file = repository / "tests" / "test_native_example.py"
    atomic_write_bytes(helper, b"VALUE = 1\n")
    atomic_write_bytes(test_file, b"def test_example():\n    assert True\n")
    original_test_hash = native_test_tree_sha256(repository)
    junit = tmp_path / "native.xml"
    atomic_write_bytes(junit, b"<testsuites/>\n")
    evidence_path = tmp_path / "correctness.json"
    implementation_hash = "e" * 64
    binary_hash = str(
        native_benchmark_module.native_build_info()["shared_object_sha256"]
    )
    atomic_write_json(
        evidence_path,
        {
            "status": "passed",
            "pytest_exit_code": 0,
            "fixed_seed_cases": 10000,
            "implementation_tree_sha256": implementation_hash,
            "native_shared_object_sha256": binary_hash,
            "junit_path": str(junit),
            "junit_sha256": file_sha256(junit),
            "test_tree_sha256": original_test_hash,
        },
    )
    atomic_write_bytes(helper, b"VALUE = 2\n")
    monkeypatch.setattr(native_benchmark_module, "REPOSITORY_ROOT", repository)

    with pytest.raises(RuntimeError, match="test_tree_hash_mismatch"):
        native_benchmark_module._verify_correctness_prerequisite(
            {
                "phase": "validation",
                "required_correctness_evidence": str(evidence_path),
            },
            implementation_hash,
        )


def test_native_contract_rejects_misordered_underfilled_result() -> None:
    records = {
        (1, 0): VectorRecord(1, 0, np.asarray([1.0], dtype=np.float32)),
        (2, 0): VectorRecord(2, 0, np.asarray([2.0], dtype=np.float32)),
    }
    reference = (
        SearchHit(1, 0, 1.0, "reference"),
        SearchHit(2, 0, 2.0, "reference"),
    )
    measured = native_benchmark_module._Measured(
        hits=(
            SearchHit(2, 0, 2.0, "native"),
            SearchHit(1, 0, 1.0, "native"),
        ),
        micro_ns=1,
        api_wall_ns=1,
        api_candidate_hash="candidate",
        native_calls=2,
        backend="native",
        receipt={
            "certified_beta": None,
            "requested_beta": 0.0,
            "candidate_set_id_or_hash": "candidate",
            "tau_returned": None,
        },
        components={},
    )

    contract = native_benchmark_module._native_contract(
        native_benchmark_module._Method("F", "F", native_mode="F"),
        measured,
        reference,
        query=np.asarray([0.0], dtype=np.float32),
        records_by_key=records,
        population=2,
        k=3,
    )

    assert contract["same_c_order_match"] is False
    assert contract["contract_valid"] is False
    assert contract["contract_violation"] is True
